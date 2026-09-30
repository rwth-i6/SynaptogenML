"""
Per-layer programming-fidelity tests.

Each test builds a ``*Quant`` layer with random weights, calibrates it on a random
input, programs the matching ``Memristor*`` layer from it (the two-phase workflow)
and compares the memristor output against the quantized reference on the same
input. The relative error bounds are loose enough for the stochastic cell
programming and readout noise, but tight enough to catch wrong sign handling,
bit-plane weighting, kernel orientation or scaling: a broken layer lands far
above them (relative error near 1), a healthy one around 0.1.
"""

from functools import partial

import numpy as np
import pytest
import torch

import synaptogen_ml.synaptogen as syn
from synaptogen_ml.memristor_modules.config import CycleCorrectionSettings
from synaptogen_ml.memristor_modules.conv import MemristorConv1d, MemristorConv2d
from synaptogen_ml.memristor_modules.linear import (
    MemristorLinear,
    TiledMemristorLinear,
)
from synaptogen_ml.memristor_modules.memristor import DacAdcHardwareSettings
from synaptogen_ml.quant_modules import (
    ActivationQuantizer,
    Conv1DQuant,
    Conv2dQuant,
    LinearQuant,
)

WEIGHT_PRECISION = 3

# ``hardware_output_current_scaling`` is the empirical current -> weight factor
# from ``compute_correction_factor``. It depends on how the cells are programmed:
# stochastic programming (default) and ideal programming (r set directly) reach
# different conductance states, so each needs its own factor.
HW_STOCHASTIC = dict(
    input_bits=8,
    output_precision_bits=2,
    output_range_bits=6,
    hardware_input_vmax=0.6,
    hardware_output_current_scaling=8020.0,  # compute_correction_factor(ideal=False)
)
HW_IDEAL = dict(HW_STOCHASTIC, hardware_output_current_scaling=5476.0)  # ideal=True

QUANT_KW = dict(
    weight_bit_prec=WEIGHT_PRECISION,
    weight_quant_dtype=torch.qint8,
    weight_quant_method="per_tensor_symmetric",
    bias=False,
)

IDEAL = CycleCorrectionSettings(
    num_cycles=None,
    test_input_value=None,
    relative_deviation=None,
    ideal_programming=True,
)
WRITE_VERIFY = CycleCorrectionSettings(
    num_cycles=2, test_input_value=0.2, relative_deviation=0.2, ideal_programming=False
)


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    syn.rng = np.random.default_rng(seed)
    syn.randn = partial(syn.rng.standard_normal, dtype=np.float32)
    syn.rand = partial(syn.rng.random, dtype=np.float32)


def act_quant() -> ActivationQuantizer:
    return ActivationQuantizer(
        bit_precision=8,
        dtype=torch.qint8,
        method="per_tensor_symmetric",
        channel_axis=None,
        moving_avrg=None,
    )


def calibrate(quant_layer, act, x) -> torch.Tensor:
    """Populate the observers with one training-mode pass; return the quantized
    reference output in eval mode."""
    quant_layer.train()
    act.train()
    quant_layer(act(x))
    quant_layer.eval()
    act.eval()
    with torch.no_grad():
        return quant_layer(act(x))


def relative_error(out: torch.Tensor, ref: torch.Tensor) -> float:
    assert out.shape == ref.shape, (out.shape, ref.shape)
    return ((out - ref).norm() / ref.norm()).item()


def memristor_output(mem, x) -> torch.Tensor:
    mem.eval()
    with torch.no_grad():
        return mem(x)


@pytest.mark.fidelity
def test_linear():
    seed_all(0)
    act = act_quant()
    x = torch.randn(32, 784)
    lin = LinearQuant(in_features=784, out_features=64, **QUANT_KW)
    ref = calibrate(lin, act, x)

    mem = MemristorLinear(
        in_features=784,
        out_features=64,
        weight_precision=WEIGHT_PRECISION,
        converter_hardware_settings=DacAdcHardwareSettings(**HW_STOCHASTIC),
    )
    mem.init_from_linear_quant(act, lin)
    assert relative_error(memristor_output(mem, x), ref) < 0.2


@pytest.mark.fidelity
@pytest.mark.parametrize(
    "num_cycles_init, correction, hw, bound",
    [
        (0, None, HW_STOCHASTIC, 0.2),
        (1, WRITE_VERIFY, HW_STOCHASTIC, 0.2),
        (0, IDEAL, HW_IDEAL, 0.15),
    ],
    ids=["stochastic", "wear+write_verify", "ideal"],
)
def test_tiled_linear(num_cycles_init, correction, hw, bound):
    seed_all(0)
    act = act_quant()
    x = torch.randn(32, 256)
    lin = LinearQuant(in_features=256, out_features=128, **QUANT_KW)
    ref = calibrate(lin, act, x)

    mem = TiledMemristorLinear(
        in_features=256,
        out_features=128,
        weight_precision=WEIGHT_PRECISION,
        converter_hardware_settings=DacAdcHardwareSettings(**hw),
        memristor_inputs=128,
        memristor_outputs=128,
    )
    mem.init_from_linear_quant(
        act, lin, num_cycles_init=num_cycles_init, correction_settings=correction
    )
    assert relative_error(memristor_output(mem, x), ref) < bound


@pytest.mark.fidelity
@pytest.mark.parametrize(
    "correction, hw, bound",
    [(None, HW_STOCHASTIC, 0.3), (IDEAL, HW_IDEAL, 0.3)],
    ids=["stochastic", "ideal"],
)
def test_conv1d_depthwise(correction, hw, bound):
    channels, kernel_size = 32, 5
    seed_all(0)
    act = act_quant()
    x = torch.randn(4, channels, 28)
    conv = Conv1DQuant(
        in_channels=channels,
        out_channels=channels,
        kernel_size=kernel_size,
        stride=1,
        padding="same",
        dilation=1,
        groups=channels,
        **QUANT_KW,
    )
    ref = calibrate(conv, act, x)

    mem = MemristorConv1d(
        in_channels=channels,
        out_channels=channels,
        kernel_size=kernel_size,
        stride=1,
        padding="same",
        groups=channels,
        weight_precision=WEIGHT_PRECISION,
        converter_hardware_settings=DacAdcHardwareSettings(**hw),
    )
    mem.init_from_conv_quant(
        act, conv, num_cycles_init=0, correction_settings=correction
    )
    assert relative_error(memristor_output(mem, x), ref) < bound


@pytest.mark.fidelity
@pytest.mark.xfail(
    strict=True,
    reason="MemristorConv2d applies a spatially transposed (H<->W) kernel; "
    "the non-square kernel here exposes it. Remove this marker with the fix.",
)
def test_conv2d():
    # Non-square kernel and padding: only a correctly oriented kernel reproduces
    # the reference shape and values.
    kernel_size, padding = (3, 2), (1, 0)
    seed_all(0)
    act = act_quant()
    x = torch.randn(4, 1, 12, 12)
    conv = Conv2dQuant(
        in_channels=1,
        out_channels=8,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
        dilation=1,
        groups=1,
        **QUANT_KW,
    )
    ref = calibrate(conv, act, x)

    mem = MemristorConv2d(
        in_channels=1,
        out_channels=8,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
        groups=1,
        weight_precision=WEIGHT_PRECISION,
        converter_hardware_settings=DacAdcHardwareSettings(**HW_STOCHASTIC),
    )
    mem.init_from_conv_quant(act, conv, num_cycles_init=0, correction_settings=None)
    assert relative_error(memristor_output(mem, x), ref) < 0.3
