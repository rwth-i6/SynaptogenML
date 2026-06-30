"""Structural-correctness regression tests for the memristor conv2d variants.

Each memristor conv2d must compute the SAME convolution as the reference
``Conv2dQuant`` (i.e. ``F.conv2d``), not a spatially-transposed one. An asymmetric
kernel (3x5) on a non-square input (11x9) makes any H<->W swap show up as a wrong
output shape; the device/ADC scatter keeps the correlation below 1 but well above
0.9 for a correct implementation (a transposed kernel drops it to ~0.6).
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


def _seed(s=0):
    torch.manual_seed(s)
    np.random.seed(s)
    syn.rng = np.random.default_rng(s)
    syn.randn = partial(syn.rng.standard_normal, dtype=float32)
    syn.rand = partial(syn.rng.random, dtype=float32)


def _corr(a, b):
    return torch.corrcoef(torch.stack([a.flatten().double(), b.flatten().double()]))[
        0, 1
    ].item()


def _run(mem_cls, in_c, out_c, groups):
    _seed()
    act = ActivationQuantizer(8, torch.qint8, "per_tensor_symmetric", None, None)
    conv = Conv2dQuant(
        in_c,
        out_c,
        (3, 5),
        3,
        torch.qint8,
        "per_tensor_symmetric",
        False,
        1,
        0,
        1,
        groups,
    )
    x = torch.randn(2, in_c, 11, 9)  # H=11, W=9 (non-square)
    conv.train()
    act.train()
    conv(act(x))
    conv.eval()
    act.eval()
    ref = conv(x).detach()  # F.conv2d -> [2, out_c, 9, 5]

    mem = mem_cls(
        in_channels=in_c,
        out_channels=out_c,
        kernel_size=(3, 5),
        stride=1,
        padding=0,
        groups=groups,
        weight_precision=3,
        converter_hardware_settings=DacAdcHardwareSettings(**_HW),
        bias=False,
    )
    mem.init_from_conv_quant(act, conv, num_cycles_init=0, correction_settings=None)
    mem.eval()
    return mem(x).detach(), ref


@pytest.mark.conv2d
def test_memristor_conv2d_groups1_matches_reference():
    out, ref = _run(MemristorConv2d, 3, 4, 1)
    assert tuple(out.shape) == tuple(ref.shape), (out.shape, ref.shape)
    assert _corr(out, ref) > 0.9


@pytest.mark.conv2d
def test_memristor_conv2d_depthwise_matches_reference():
    out, ref = _run(MemristorConv2d, 4, 4, 4)
    assert tuple(out.shape) == tuple(ref.shape), (out.shape, ref.shape)
    assert _corr(out, ref) > 0.9


@pytest.mark.conv2d
def test_single_kernel_conv2d_matches_reference():
    out, ref = _run(SingleKernelMemristorConv2d, 3, 4, 1)
    assert tuple(out.shape) == tuple(ref.shape), (out.shape, ref.shape)
    assert _corr(out, ref) > 0.9
