"""
Tests for the device-backend seam (``synaptogen_ml.backends``).

``backend=None`` on a ``Memristor*`` module is the historical Synaptogen code
path. ``SynaptogenBackend`` routes the same operations through the backend
interface; under identical seeds the two must produce bit-identical modules
and outputs for every crossbar module type. This pins the seam so that later
backends can be added without touching the default path.
"""

from functools import partial

import numpy as np
import pytest
import torch

import synaptogen_ml.synaptogen as syn
from synaptogen_ml.backends import CellBackend, SynaptogenBackend
from synaptogen_ml.memristor_modules.config import CycleCorrectionSettings
from synaptogen_ml.memristor_modules.conv import (
    MemristorConv1d,
    MemristorConv2d,
    SingleKernelMemristorConv2d,
)
from synaptogen_ml.memristor_modules.linear import TiledMemristorLinear
from synaptogen_ml.memristor_modules.memristor import (
    DacAdcHardwareSettings,
    PairedMemristorArrayV2,
)
from synaptogen_ml.memristor_modules.util import compute_correction_factor
from synaptogen_ml.quant_modules import (
    ActivationQuantizer,
    Conv1DQuant,
    Conv2dQuant,
    LinearQuant,
)

WEIGHT_PRECISION = 3
HW = DacAdcHardwareSettings(
    input_bits=8,
    output_precision_bits=2,
    output_range_bits=6,
    hardware_input_vmax=0.6,
    hardware_output_current_scaling=8020.0,
)
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


def calibrate(quant_layer, act, x) -> None:
    quant_layer.train()
    act.train()
    quant_layer(act(x))
    quant_layer.eval()
    act.eval()


# Each case: (trained QAT layer + activation quantizer + input, a factory that
# builds the matching memristor module for a backend, the init method name).
def _tiled_linear_case():
    act = act_quant()
    x = torch.randn(4, 40)
    lin = LinearQuant(in_features=40, out_features=24, **QUANT_KW)
    calibrate(lin, act, x)

    def build(backend):
        return TiledMemristorLinear(
            in_features=40,
            out_features=24,
            weight_precision=WEIGHT_PRECISION,
            converter_hardware_settings=HW,
            memristor_inputs=16,
            memristor_outputs=16,
            backend=backend,
        )

    return act, lin, x, build, "init_from_linear_quant"


def _conv1d_case():
    channels, kernel_size = 8, 3
    act = act_quant()
    x = torch.randn(2, channels, 12)
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
    calibrate(conv, act, x)

    def build(backend):
        return MemristorConv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
            groups=channels,
            weight_precision=WEIGHT_PRECISION,
            converter_hardware_settings=HW,
            backend=backend,
        )

    return act, conv, x, build, "init_from_conv_quant"


def _conv2d_quant(kernel_size, padding):
    return Conv2dQuant(
        in_channels=2,
        out_channels=4,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
        dilation=1,
        groups=1,
        **QUANT_KW,
    )


def _conv2d_case():
    kernel_size, padding = (3, 2), (1, 0)
    act = act_quant()
    x = torch.randn(2, 2, 8, 8)
    conv = _conv2d_quant(kernel_size, padding)
    calibrate(conv, act, x)

    def build(backend):
        return MemristorConv2d(
            in_channels=2,
            out_channels=4,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=1,
            weight_precision=WEIGHT_PRECISION,
            converter_hardware_settings=HW,
            backend=backend,
        )

    return act, conv, x, build, "init_from_conv_quant"


def _single_kernel_conv2d_case():
    kernel_size, padding = (3, 3), (1, 1)
    act = act_quant()
    x = torch.randn(2, 2, 8, 8)
    conv = _conv2d_quant(kernel_size, padding)
    calibrate(conv, act, x)

    def build(backend):
        return SingleKernelMemristorConv2d(
            in_channels=2,
            out_channels=4,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=1,
            weight_precision=WEIGHT_PRECISION,
            converter_hardware_settings=HW,
            backend=backend,
        )

    return act, conv, x, build, "init_from_conv_quant"


CASES = {
    "tiled_linear": _tiled_linear_case,
    "conv1d": _conv1d_case,
    "conv2d": _conv2d_case,
    "single_kernel_conv2d": _single_kernel_conv2d_case,
}
# write-verify is only implemented on the default path for the tiled linear and
# the depthwise conv1d module; conv2d asserts it away unless a backend is set
CORRECTIONS = {
    "tiled_linear": [None, IDEAL, WRITE_VERIFY],
    "conv1d": [None, IDEAL, WRITE_VERIFY],
    "conv2d": [None],
    "single_kernel_conv2d": [None],
}
PARAMS = [
    pytest.param(
        case,
        correction,
        id=f"{case}-{getattr(correction, 'ideal_programming', 'none')}",
    )
    for case in CASES
    for correction in CORRECTIONS[case]
]


def _program(build, backend, act, quant_layer, init_name, correction, seed):
    seed_all(seed)
    mem = build(backend)
    getattr(mem, init_name)(
        act, quant_layer, num_cycles_init=1, correction_settings=correction
    )
    return mem


def _forward(mem, x, seed):
    torch.manual_seed(seed)
    mem.eval()
    with torch.no_grad():
        return mem(x)


@pytest.mark.backends
def test_synaptogen_backend_satisfies_protocol():
    backend = SynaptogenBackend()
    assert isinstance(backend, CellBackend)
    array = backend.make_paired_array(5, 3, additional_axes=(2, 4))
    assert isinstance(array, PairedMemristorArrayV2)
    assert tuple(array.pos.r.shape) == (2, 4, 5, 3)
    assert tuple(array.neg.r.shape) == (2, 4, 5, 3)


@pytest.mark.backends
@pytest.mark.parametrize("case, correction", PARAMS)
def test_synaptogen_backend_is_bit_identical_to_default(case, correction):
    seed_all(0)
    act, quant_layer, x, build, init_name = CASES[case]()

    reference = _program(build, None, act, quant_layer, init_name, correction, 1)
    routed = _program(
        build, SynaptogenBackend(), act, quant_layer, init_name, correction, 1
    )

    ref_state = reference.state_dict()
    routed_state = routed.state_dict()
    assert ref_state.keys() == routed_state.keys()
    for key in ref_state:
        assert torch.equal(ref_state[key], routed_state[key]), key

    assert torch.equal(_forward(reference, x, 2), _forward(routed, x, 2))


@pytest.mark.backends
def test_default_path_does_not_hold_a_backend():
    act, lin, x, build, _ = _tiled_linear_case()
    assert build(None).backend is None
    backend = SynaptogenBackend()
    assert build(backend).backend is backend


@pytest.mark.backends
def test_compute_correction_factor_returns_both_factors():
    seed_all(0)
    paired, single = compute_correction_factor(ideal=False, trials=2)
    assert np.isfinite(paired) and paired > 0
    assert np.isfinite(single) and single > 0
    # the paired factor is the one the test configurations use (~8020)
    assert 4000 < paired < 16000


@pytest.mark.backends
def test_suggested_current_scaling():
    seed_all(0)
    backend = SynaptogenBackend(calibration_trials=2)
    assert 4000 < backend.suggested_current_scaling(0.6) < 16000
    with pytest.raises(ValueError):
        backend.suggested_current_scaling(0.3)
