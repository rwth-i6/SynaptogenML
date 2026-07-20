"""
Shared builders for benchmarking and equivalence-testing the memristor inference
modules.

Each builder returns ``(name, module, example_input)`` with the module already
initialized (weights "programmed" onto cell arrays) and in ``eval`` mode, so a
single ``module(example_input)`` reproduces an inference forward pass.

Reproducibility: ``seed_all`` reseeds *both* the torch RNG (used for readout noise
in the forward) and the numpy generator inside ``synaptogen_ml.synaptogen`` (used
when programming the cells during init). Builders seed before constructing the
module so the programmed state is deterministic; callers should additionally call
``torch.manual_seed(seed)`` immediately before each forward to fix the readout
noise draw when comparing two implementations.
"""

from functools import partial

import numpy as np
import torch
from numpy import float32

import synaptogen_ml.synaptogen as syn
from synaptogen_ml.quant_modules import (
    ActivationQuantizer,
    Conv1DQuant,
    Conv2dQuant,
    LinearQuant,
)
from synaptogen_ml.memristor_modules.memristor import DacAdcHardwareSettings
from synaptogen_ml.memristor_modules.linear import (
    MemristorLinear,
    TiledMemristorLinear,
)
from synaptogen_ml.memristor_modules.conv import (
    MemristorConv1d,
    MemristorConv2d,
    SingleKernelMemristorConv2d,
)

# Matches the hardware settings used in tests/test_mnist_linear.py
DEFAULT_HW = dict(
    input_bits=8,
    output_precision_bits=2,
    output_range_bits=6,
    hardware_input_vmax=0.6,
    hardware_output_current_scaling=8020.0,
)

WEIGHT_PRECISION = 3


def seed_all(seed: int = 0) -> None:
    """Reseed torch + numpy + the synaptogen cell-array generator."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    syn.rng = np.random.default_rng(seed)
    syn.randn = partial(syn.rng.standard_normal, dtype=float32)
    syn.rand = partial(syn.rng.random, dtype=float32)


def _hw() -> DacAdcHardwareSettings:
    return DacAdcHardwareSettings(**DEFAULT_HW)


def _calibrate(quant_layer, act_quant, example_input) -> None:
    """Populate the weight/activation observers so scales are defined."""
    quant_layer.train()
    act_quant.train()
    quant_layer(act_quant(example_input))
    quant_layer.eval()
    act_quant.eval()


def _act_quant() -> ActivationQuantizer:
    return ActivationQuantizer(
        bit_precision=8,
        dtype=torch.qint8,
        method="per_tensor_symmetric",
        channel_axis=None,
        moving_avrg=None,
    )


def build_linear(*, batch=64, in_features=784, out_features=512, device="cpu", seed=0):
    seed_all(seed)
    lin = LinearQuant(
        in_features=in_features,
        out_features=out_features,
        weight_bit_prec=WEIGHT_PRECISION,
        weight_quant_dtype=torch.qint8,
        weight_quant_method="per_tensor_symmetric",
        bias=False,
    )
    act = _act_quant()
    x = torch.randn(batch, in_features)
    _calibrate(lin, act, x)

    mem = MemristorLinear(
        in_features=in_features,
        out_features=out_features,
        weight_precision=WEIGHT_PRECISION,
        converter_hardware_settings=_hw(),
        bias=False,
    )
    mem.init_from_linear_quant(act, lin)
    mem.to(device).eval()
    return "MemristorLinear", mem, x.to(device)


def build_tiled_linear(
    *,
    batch=64,
    in_features=784,
    out_features=512,
    tile_in=256,
    tile_out=256,
    device="cpu",
    seed=0,
):
    seed_all(seed)
    lin = LinearQuant(
        in_features=in_features,
        out_features=out_features,
        weight_bit_prec=WEIGHT_PRECISION,
        weight_quant_dtype=torch.qint8,
        weight_quant_method="per_tensor_symmetric",
        bias=True,
    )
    act = _act_quant()
    x = torch.randn(batch, in_features)
    _calibrate(lin, act, x)

    mem = TiledMemristorLinear(
        in_features=in_features,
        out_features=out_features,
        weight_precision=WEIGHT_PRECISION,
        converter_hardware_settings=_hw(),
        memristor_inputs=tile_in,
        memristor_outputs=tile_out,
        bias=True,
    )
    mem.init_from_linear_quant(act, lin, num_cycles_init=0, correction_settings=None)
    mem.to(device).eval()
    return "TiledMemristorLinear", mem, x.to(device)


def build_conv1d(
    *,
    batch=8,
    channels=256,
    time=512,
    kernel_size=9,
    device="cpu",
    seed=0,
):
    # depthwise: groups == in == out (the only supported conv1d mode)
    seed_all(seed)
    conv = Conv1DQuant(
        in_channels=channels,
        out_channels=channels,
        kernel_size=kernel_size,
        weight_bit_prec=WEIGHT_PRECISION,
        weight_quant_dtype=torch.qint8,
        weight_quant_method="per_tensor_symmetric",
        bias=False,
        stride=1,
        padding="same",
        dilation=1,
        groups=channels,
    )
    act = _act_quant()
    x = torch.randn(batch, channels, time)
    _calibrate(conv, act, x)

    mem = MemristorConv1d(
        in_channels=channels,
        out_channels=channels,
        kernel_size=kernel_size,
        stride=1,
        padding="same",
        groups=channels,
        weight_precision=WEIGHT_PRECISION,
        converter_hardware_settings=_hw(),
        bias=False,
    )
    mem.init_from_conv_quant(act, conv, num_cycles_init=0, correction_settings=None)
    mem.to(device).eval()
    return "MemristorConv1d", mem, x.to(device)


def build_conv2d(
    *,
    batch=8,
    in_channels=1,
    out_channels=32,
    height=32,
    width=32,
    kernel_size=3,
    stride=2,
    device="cpu",
    seed=0,
):
    seed_all(seed)
    conv = Conv2dQuant(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        weight_bit_prec=WEIGHT_PRECISION,
        weight_quant_dtype=torch.qint8,
        weight_quant_method="per_tensor_symmetric",
        bias=False,
        stride=stride,
        padding=1,
        dilation=1,
        groups=1,
    )
    act = _act_quant()
    x = torch.randn(batch, in_channels, height, width)
    _calibrate(conv, act, x)

    mem = MemristorConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=1,
        groups=1,
        weight_precision=WEIGHT_PRECISION,
        converter_hardware_settings=_hw(),
        bias=False,
    )
    mem.init_from_conv_quant(act, conv, num_cycles_init=0, correction_settings=None)
    mem.to(device).eval()
    return "MemristorConv2d", mem, x.to(device)


def build_single_kernel_conv2d(
    *,
    batch=8,
    in_channels=1,
    out_channels=32,
    height=32,
    width=32,
    kernel_size=3,
    stride=2,
    device="cpu",
    seed=0,
):
    seed_all(seed)
    conv = Conv2dQuant(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        weight_bit_prec=WEIGHT_PRECISION,
        weight_quant_dtype=torch.qint8,
        weight_quant_method="per_tensor_symmetric",
        bias=False,
        stride=stride,
        padding=1,
        dilation=1,
        groups=1,
    )
    act = _act_quant()
    x = torch.randn(batch, in_channels, height, width)
    _calibrate(conv, act, x)

    mem = SingleKernelMemristorConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=1,
        groups=1,
        weight_precision=WEIGHT_PRECISION,
        converter_hardware_settings=_hw(),
        bias=False,
    )
    mem.init_from_conv_quant(act, conv, num_cycles_init=0, correction_settings=None)
    mem.to(device).eval()
    return "SingleKernelMemristorConv2d", mem, x.to(device)


# name -> builder. Small default sizes keep CPU correctness runs cheap; pass
# overrides (e.g. larger batch/time) for GPU profiling.
#
# NOTE: ``single_kernel_conv2d`` is intentionally excluded from the default set:
# ``SingleKernelMemristorConv2d.forward`` has a pre-existing crash (it calls
# ``result.permute(0, 3, 1, 2)`` on a 3-D tensor without first reshaping the
# flattened spatial axis back to 2-D, conv.py:714). It is untested by any pytest
# marker. Fixing it would change functionality, so it is left as-is and kept out
# of the runnable harness. The builder remains available for when it is repaired.
BUILDERS = {
    "linear": build_linear,
    "tiled_linear": build_tiled_linear,
    "conv1d": build_conv1d,
    "conv2d": build_conv2d,
}
