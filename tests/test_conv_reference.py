"""Structural-correctness regression tests for the memristor conv2d variants.

Each memristor conv2d must compute the SAME convolution as the reference
``Conv2dQuant`` (i.e. ``F.conv2d``), not a spatially-transposed one, and must
weight its bit-planes correctly. An asymmetric kernel (3x5) on a non-square input
(11x9) makes any H<->W swap show up as a wrong output shape. With a fine ADC the
relative error of a correct implementation is about 0.1 (same as the linear and
conv1d layers); a transposed kernel or a double-counted bit-plane pushes it far
above the 0.2 bound.
"""

from functools import partial

import numpy as np
import pytest
import torch
from numpy import float32

import synaptogen_ml.synaptogen as syn
from synaptogen_ml.memristor_modules.conv import (
    MemristorConv2d,
    SingleKernelMemristorConv2d,
)
from synaptogen_ml.memristor_modules.memristor import DacAdcHardwareSettings
from synaptogen_ml.quant_modules import ActivationQuantizer, Conv2dQuant

# fine ADC so quantization does not mask structure
_HW = dict(
    input_bits=8,
    output_precision_bits=8,
    output_range_bits=8,
    hardware_input_vmax=0.6,
    hardware_output_current_scaling=8020.0,
)
KERNEL_SIZE = (3, 5)


def _seed(s=0):
    torch.manual_seed(s)
    np.random.seed(s)
    syn.rng = np.random.default_rng(s)
    syn.randn = partial(syn.rng.standard_normal, dtype=float32)
    syn.rand = partial(syn.rng.random, dtype=float32)


def _relative_error(out, ref):
    return ((out - ref).norm() / ref.norm()).item()


def _run(mem_cls, in_c, out_c, groups):
    _seed()
    act = ActivationQuantizer(
        bit_precision=8,
        dtype=torch.qint8,
        method="per_tensor_symmetric",
        channel_axis=None,
        moving_avrg=None,
    )
    conv = Conv2dQuant(
        in_channels=in_c,
        out_channels=out_c,
        kernel_size=KERNEL_SIZE,
        weight_bit_prec=3,
        weight_quant_dtype=torch.qint8,
        weight_quant_method="per_tensor_symmetric",
        bias=False,
        stride=1,
        padding=0,
        dilation=1,
        groups=groups,
    )
    x = torch.randn(2, in_c, 11, 9)  # H=11, W=9 (non-square)
    conv.train()
    act.train()
    conv(act(x))
    conv.eval()
    act.eval()
    with torch.no_grad():
        ref = conv(act(x))  # F.conv2d -> [2, out_c, 9, 5]

    mem = mem_cls(
        in_channels=in_c,
        out_channels=out_c,
        kernel_size=KERNEL_SIZE,
        stride=1,
        padding=0,
        groups=groups,
        weight_precision=3,
        converter_hardware_settings=DacAdcHardwareSettings(**_HW),
        bias=False,
    )
    mem.init_from_conv_quant(act, conv, num_cycles_init=0, correction_settings=None)
    mem.eval()
    with torch.no_grad():
        return mem(x), ref


@pytest.mark.conv2d
@pytest.mark.fidelity
@pytest.mark.parametrize(
    "mem_cls, in_c, out_c, groups",
    [
        (MemristorConv2d, 3, 4, 1),
        (MemristorConv2d, 4, 4, 4),
        (SingleKernelMemristorConv2d, 3, 4, 1),
    ],
    ids=["conv2d_groups1", "conv2d_depthwise", "single_kernel"],
)
def test_conv2d_matches_reference(mem_cls, in_c, out_c, groups):
    out, ref = _run(mem_cls, in_c, out_c, groups)
    assert tuple(out.shape) == tuple(ref.shape), (out.shape, ref.shape)
    assert _relative_error(out, ref) < 0.2
