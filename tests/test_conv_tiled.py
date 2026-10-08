"""Tests for ``TiledMemristorConv2d`` (im2col on a ``TiledMemristorLinear``).

1. Reference fidelity: with a fine ADC the tiled conv must compute the same
   convolution as ``Conv2dQuant`` (``F.conv2d``), for one tile and for
   ``C*k0*k1`` spanning several input tiles, signed and unsigned activations,
   4- and 8-bit weights. Same 0.2 relative-error bound as test_conv_reference.
2. Tiling equivalence: single-tile ``TiledMemristorConv2d`` and the untiled
   ``MemristorConv2d`` both match the float reference with the fine ADC.
3. Saturation: with the recipe ADC (adc_max = 128 full-scale cells) a 576-row
   conv with large positive dot products clips in ``MemristorConv2d`` but not
   in ``TiledMemristorConv2d(memristor_inputs=128)``.
"""

from functools import partial

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from numpy import float32

import synaptogen_ml.synaptogen as syn
from synaptogen_ml.memristor_modules.conv import (
    MemristorConv2d,
    TiledMemristorConv2d,
)
from synaptogen_ml.memristor_modules.memristor import DacAdcHardwareSettings
from synaptogen_ml.quant_modules import ActivationQuantizer, Conv2dQuant

# fine ADC so quantization does not mask structure (as in test_conv_reference)
_HW_FINE = dict(
    input_bits=8,
    output_precision_bits=8,
    output_range_bits=8,
    hardware_input_vmax=0.6,
    hardware_output_current_scaling=8020.0,
)
# the recipe's ADC: adc_max = 2 ** (4 + 4 - 1) = 128 full-scale cells
_HW_RECIPE = dict(
    input_bits=8,
    output_precision_bits=4,
    output_range_bits=4,
    hardware_input_vmax=0.6,
    hardware_output_current_scaling=8020.0,
)


def _seed(s=0):
    torch.manual_seed(s)
    np.random.seed(s)
    syn.rng = np.random.default_rng(s)
    syn.randn = partial(syn.rng.standard_normal, dtype=float32)
    syn.rand = partial(syn.rng.random, dtype=float32)


def _relative_error(out, ref):
    return ((out - ref).norm() / ref.norm()).item()


def _act_quant(dtype):
    return ActivationQuantizer(
        bit_precision=8,
        dtype=dtype,
        method="per_tensor_symmetric" if dtype == torch.qint8 else "per_tensor",
        channel_axis=None,
        moving_avrg=None,
    )


def _conv_quant(in_c, out_c, kernel_size, weight_precision, stride, padding, bias):
    return Conv2dQuant(
        in_channels=in_c,
        out_channels=out_c,
        kernel_size=kernel_size,
        weight_bit_prec=weight_precision,
        weight_quant_dtype=torch.qint8,
        weight_quant_method="per_tensor_symmetric",
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=1,
        groups=1,
    )


def _calibrate_and_reference(conv, act, x):
    """Populate the observers on x and return the fake-quantized reference."""
    conv.train()
    act.train()
    conv(act(x))
    conv.eval()
    act.eval()
    with torch.no_grad():
        return conv(act(x))


def _build_tiled(conv, act, hw, memristor_inputs, memristor_outputs, bias):
    mem = TiledMemristorConv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        groups=1,
        weight_precision=conv.weight_bit_prec,
        converter_hardware_settings=DacAdcHardwareSettings(**hw),
        memristor_inputs=memristor_inputs,
        memristor_outputs=memristor_outputs,
        bias=bias,
    )
    mem.init_from_conv_quant(act, conv, num_cycles_init=0, correction_settings=None)
    return mem.eval()


def _build_untiled(conv, act, hw, bias):
    mem = MemristorConv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        groups=1,
        weight_precision=conv.weight_bit_prec,
        converter_hardware_settings=DacAdcHardwareSettings(**hw),
        bias=bias,
    )
    mem.init_from_conv_quant(act, conv, num_cycles_init=0, correction_settings=None)
    return mem.eval()


def _input(in_c, dtype, shape=(2, 7, 5)):
    x = torch.randn(shape[0], in_c, shape[1], shape[2])
    if dtype == torch.quint8:
        x = x.abs()  # unsigned activations (e.g. after ReLU)
    return x


@pytest.mark.fidelity
@pytest.mark.parametrize("weight_precision", [4, 8], ids=["w4", "w8"])
@pytest.mark.parametrize("dtype", [torch.qint8, torch.quint8], ids=["qint8", "quint8"])
@pytest.mark.parametrize(
    "in_c, out_c, memristor_inputs",
    [(4, 6, 36), (8, 6, 16)],
    ids=["in4_single_tile", "in8_5_tiles"],
)
def test_tiled_conv2d_matches_reference(in_c, out_c, memristor_inputs, dtype, weight_precision):
    _seed()
    act = _act_quant(dtype)
    conv = _conv_quant(in_c, out_c, 3, weight_precision, stride=1, padding=1, bias=True)
    x = _input(in_c, dtype)
    ref = _calibrate_and_reference(conv, act, x)

    mem = _build_tiled(conv, act, _HW_FINE, memristor_inputs, 4, bias=True)
    assert mem.linear.input_tiling == -(-in_c * 9 // memristor_inputs)
    with torch.no_grad():
        out = mem(x)
    assert tuple(out.shape) == tuple(ref.shape), (out.shape, ref.shape)
    assert _relative_error(out, ref) < 0.2


@pytest.mark.fidelity
def test_tiled_conv2d_row_order_matches_unfold():
    """The flattened weight rows must line up with F.unfold's (c, i, j) rows:
    a conv with a single non-zero weight tap must pick exactly that input tap."""
    _seed()
    in_c, k = 3, (3, 5)
    act = _act_quant(torch.qint8)
    conv = _conv_quant(in_c, 1, k, 8, stride=(2, 1), padding=0, bias=False)
    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[0, 2, 1, 4] = 1.0  # channel 2, kernel offset (1, 4)
    x = _input(in_c, torch.qint8, shape=(1, 9, 11))
    ref = _calibrate_and_reference(conv, act, x)
    mem = _build_tiled(conv, act, _HW_FINE, 16, 8, bias=False)
    with torch.no_grad():
        out = mem(x)
        xq = act(x)
    assert tuple(out.shape) == tuple(ref.shape)
    # the 44 zero taps still leak (HRS), so compare against every single-tap
    # hypothesis: the programmed tap must be the closest one by a clear margin
    errors = {}
    for c in range(in_c):
        for i in range(k[0]):
            for j in range(k[1]):
                w = torch.zeros_like(conv.weight)
                w[0, c, i, j] = conv.weight[0, 2, 1, 4]
                errors[(c, i, j)] = _relative_error(
                    out, F.conv2d(xq, w, stride=(2, 1))
                )
    assert min(errors, key=errors.get) == (2, 1, 4), errors
    assert errors[(2, 1, 4)] < 0.5, errors[(2, 1, 4)]
    others = min(e for tap, e in errors.items() if tap != (2, 1, 4))
    assert others > 0.7, others


@pytest.mark.fidelity
def test_tiled_conv2d_stride_and_same_padding():
    _seed()
    in_c, out_c = 3, 5
    act = _act_quant(torch.qint8)
    conv = _conv_quant(in_c, out_c, 3, 4, stride=2, padding=(1, 1), bias=False)
    x = _input(in_c, torch.qint8, shape=(2, 12, 10))
    ref = _calibrate_and_reference(conv, act, x)
    mem = TiledMemristorConv2d(
        in_channels=in_c,
        out_channels=out_c,
        kernel_size=3,
        stride=2,
        padding="same",  # k // 2 = 1, same as the (1, 1) of the reference
        groups=1,
        weight_precision=4,
        converter_hardware_settings=DacAdcHardwareSettings(**_HW_FINE),
        memristor_inputs=8,
        memristor_outputs=2,
        bias=False,
    )
    mem.init_from_conv_quant(act, conv, num_cycles_init=0, correction_settings=None)
    with torch.no_grad():
        out = mem.eval()(x)
    assert tuple(out.shape) == tuple(ref.shape) == (2, out_c, 6, 5)
    assert _relative_error(out, ref) < 0.2


@pytest.mark.fidelity
def test_single_tile_and_untiled_both_match_reference():
    """One tile holding all C*k0*k1 rows and the untiled MemristorConv2d both
    match the float reference with the fine ADC (different programmed objects,
    so they are compared to the reference, not to each other)."""
    _seed()
    in_c, out_c = 4, 6
    act = _act_quant(torch.qint8)
    conv = _conv_quant(in_c, out_c, 3, 4, stride=1, padding=1, bias=True)
    x = _input(in_c, torch.qint8)
    ref = _calibrate_and_reference(conv, act, x)

    tiled = _build_tiled(conv, act, _HW_FINE, in_c * 9, out_c, bias=True)
    assert len(tiled.linear.memristors) == 3  # one tile per bit plane
    untiled = _build_untiled(conv, act, _HW_FINE, bias=True)
    with torch.no_grad():
        err_tiled = _relative_error(tiled(x), ref)
        err_untiled = _relative_error(untiled(x), ref)
    print(f"\nsingle tile vs untiled relative error: {err_tiled:.3f} / {err_untiled:.3f}")
    assert err_tiled < 0.2
    assert err_untiled < 0.2


def _adc_range_in_cells(hw: DacAdcHardwareSettings) -> float:
    """Clip point of DacAdcPair.adc in full-scale cells: the integer range
    adc_max times the LSB 1 / 2**output_precision_bits = 2**(range_bits - 1)."""
    return hw.adc_max / 2**hw.output_precision_bits


@pytest.mark.fidelity
def test_untiled_saturates_tiled_does_not_extreme():
    """576 rows, positive weights (all bit planes set) and inputs around half
    scale: ~288 full-scale units per output and plane. The recipe ADC clips at
    2**(4-1) = 8 units per analog read, so the untiled conv clips hard, and so
    do 128-row tiles (~64 units); 16-row tiles (~8 units) are fine."""
    _seed()
    in_c, out_c = 64, 4
    act = _act_quant(torch.quint8)
    conv = _conv_quant(in_c, out_c, 3, 4, stride=1, padding=1, bias=False)
    with torch.no_grad():
        conv.weight.fill_(0.1)  # constant -> every int weight is 7 = 0b111
    x = torch.rand(1, in_c, 4, 4)  # uniform [0, 1), mean half scale
    ref = _calibrate_and_reference(conv, act, x)

    assert _adc_range_in_cells(DacAdcHardwareSettings(**_HW_RECIPE)) == 8
    untiled = _build_untiled(conv, act, _HW_RECIPE, bias=False)
    tiled_128 = _build_tiled(conv, act, _HW_RECIPE, 128, out_c, bias=False)
    tiled_16 = _build_tiled(conv, act, _HW_RECIPE, 16, out_c, bias=False)
    with torch.no_grad():
        err_untiled = _relative_error(untiled(x), ref)
        err_128 = _relative_error(tiled_128(x), ref)
        err_16 = _relative_error(tiled_16(x), ref)
    print(
        f"\nsaturation extreme (576 rows, ~288 units/plane, ADC range 8 units):"
        f" untiled MemristorConv2d = {err_untiled:.3f},"
        f" TiledMemristorConv2d(128) = {err_128:.3f},"
        f" TiledMemristorConv2d(16) = {err_16:.3f} relative error"
    )
    assert err_untiled > 0.5
    assert err_128 > 0.5  # 64 units per tile and plane still exceed the 8-unit range
    assert err_16 < 0.1


@pytest.mark.fidelity
@pytest.mark.parametrize("dtype", [torch.qint8, torch.quint8], ids=["qint8", "quint8"])
def test_untiled_saturates_tiled_does_not_realistic(dtype):
    """576 rows with random signed weights and random (signed or ReLU-like
    unsigned) inputs, recipe ADC: 128-row tiles stay mostly within the 8-unit
    range, the single 576-row read of the untiled conv clips in the tails."""
    _seed()
    in_c, out_c = 64, 4
    act = _act_quant(dtype)
    conv = _conv_quant(in_c, out_c, 3, 4, stride=1, padding=1, bias=False)
    if dtype == torch.qint8:
        x = torch.randn(1, in_c, 4, 4)
    else:
        x = torch.rand(1, in_c, 4, 4)  # uniform [0, 1), mean half scale
    ref = _calibrate_and_reference(conv, act, x)

    untiled = _build_untiled(conv, act, _HW_RECIPE, bias=False)
    tiled = _build_tiled(conv, act, _HW_RECIPE, 128, out_c, bias=False)
    with torch.no_grad():
        err_untiled = _relative_error(untiled(x), ref)
        err_tiled = _relative_error(tiled(x), ref)
    print(
        f"\nsaturation realistic ({dtype}, 576 rows, ADC range 8 units):"
        f" untiled MemristorConv2d = {err_untiled:.3f},"
        f" TiledMemristorConv2d(128) = {err_tiled:.3f} relative error"
    )
    assert err_tiled < 0.15
    assert err_tiled <= err_untiled
    if dtype == torch.quint8:  # signed inputs rarely clip a 576-row read, unsigned ones do
        assert err_untiled > 0.15
        assert err_tiled < 0.75 * err_untiled
