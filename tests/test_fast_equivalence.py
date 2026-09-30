"""
Regression tests: the opt-in fast inference path must stay numerically equivalent
to the default (bit-exact) eager path.

The eager-vs-fast comparison runs the fast core *without* torch.compile
(``set_fast_compile(False)``) so it is quick and has no compiler dependency in CI.
That still guards the parts most likely to regress: Horner polynomial evaluation
and the eager noise draw that reproduces ``randn_broadcast``'s layout.
"""

import pytest
import torch

import synaptogen_ml
from synaptogen_ml.memristor_modules.conv import MemristorConv1d, MemristorConv2d
from synaptogen_ml.memristor_modules.linear import (
    MemristorLinear,
    TiledMemristorLinear,
)
from synaptogen_ml.memristor_modules.memristor import (
    DacAdcHardwareSettings,
    _cuda_capability_needs_jit_fallback,
    _fused_memristor_forward,
    _get_jit_fused_core,
)
from synaptogen_ml.memristor_modules.util import poly_mul_horner
from synaptogen_ml.quant_modules import (
    ActivationQuantizer,
    Conv1DQuant,
    Conv2dQuant,
    LinearQuant,
)

WEIGHT_PRECISION = 3
FORWARD_SEED = 1234


def _hw() -> DacAdcHardwareSettings:
    # Matches the hardware settings used in tests/test_mnist_linear.py
    return DacAdcHardwareSettings(
        input_bits=8,
        output_precision_bits=2,
        output_range_bits=6,
        hardware_input_vmax=0.6,
        hardware_output_current_scaling=8020.0,
    )


def _act_quant() -> ActivationQuantizer:
    return ActivationQuantizer(
        bit_precision=8,
        dtype=torch.qint8,
        method="per_tensor_symmetric",
        channel_axis=None,
        moving_avrg=None,
    )


def _calibrate(quant_layer, act_quant, x) -> None:
    """Populate the weight/activation observers so scales are defined."""
    quant_layer.train()
    act_quant.train()
    quant_layer(act_quant(x))
    quant_layer.eval()
    act_quant.eval()


_QUANT_KW = dict(
    weight_bit_prec=WEIGHT_PRECISION,
    weight_quant_dtype=torch.qint8,
    weight_quant_method="per_tensor_symmetric",
    bias=False,
)


def build_linear():
    lin = LinearQuant(in_features=96, out_features=48, **_QUANT_KW)
    act = _act_quant()
    x = torch.randn(8, 96)
    _calibrate(lin, act, x)
    mem = MemristorLinear(
        in_features=96,
        out_features=48,
        weight_precision=WEIGHT_PRECISION,
        converter_hardware_settings=_hw(),
        bias=False,
    )
    mem.init_from_linear_quant(act, lin)
    return mem.eval(), x


def build_tiled_linear():
    lin = LinearQuant(in_features=96, out_features=48, **_QUANT_KW)
    act = _act_quant()
    x = torch.randn(8, 96)
    _calibrate(lin, act, x)
    mem = TiledMemristorLinear(
        in_features=96,
        out_features=48,
        weight_precision=WEIGHT_PRECISION,
        converter_hardware_settings=_hw(),
        memristor_inputs=32,
        memristor_outputs=32,
        bias=False,
    )
    mem.init_from_linear_quant(act, lin, num_cycles_init=0, correction_settings=None)
    return mem.eval(), x


def build_conv1d():
    channels, kernel_size = 24, 5
    conv = Conv1DQuant(  # depthwise: the only supported conv1d mode
        in_channels=channels,
        out_channels=channels,
        kernel_size=kernel_size,
        stride=1,
        padding="same",
        dilation=1,
        groups=channels,
        **_QUANT_KW,
    )
    act = _act_quant()
    x = torch.randn(2, channels, 40)
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
    return mem.eval(), x


def build_conv2d():
    conv = Conv2dQuant(
        in_channels=1,
        out_channels=8,
        kernel_size=3,
        stride=2,
        padding=1,
        dilation=1,
        groups=1,
        **_QUANT_KW,
    )
    act = _act_quant()
    x = torch.randn(2, 1, 12, 12)
    _calibrate(conv, act, x)
    mem = MemristorConv2d(
        in_channels=1,
        out_channels=8,
        kernel_size=3,
        stride=2,
        padding=1,
        groups=1,
        weight_precision=WEIGHT_PRECISION,
        converter_hardware_settings=_hw(),
        bias=False,
    )
    mem.init_from_conv_quant(act, conv, num_cycles_init=0, correction_settings=None)
    return mem.eval(), x


@pytest.mark.fast
@pytest.mark.parametrize(
    "build", [build_linear, build_tiled_linear, build_conv1d, build_conv2d]
)
def test_fast_path_matches_eager_no_compile(build):
    """Run the eager and the fast path on the same programmed instance, reseeding
    torch before each forward so the readout-noise draw is identical."""
    torch.manual_seed(0)
    module, x = build()

    synaptogen_ml.set_fast_compile(False)
    try:
        synaptogen_ml.set_fast_inference(False)
        torch.manual_seed(FORWARD_SEED)
        with torch.no_grad():
            eager = module(x)

        synaptogen_ml.set_fast_inference(True)
        torch.manual_seed(FORWARD_SEED)
        with torch.no_grad():
            fast = module(x)
    finally:
        synaptogen_ml.set_fast_compile(True)
        synaptogen_ml.set_fast_inference(False)

    # post-ADC quantization can turn ~1e-6 analog drift into rare 1-step flips
    assert torch.allclose(fast, eager, atol=1e-4, rtol=1e-3)


@pytest.mark.fast
def test_jit_fused_core_matches_eager():
    """The TorchScript(NNC) fallback core (used on pre-Triton GPUs) must match
    the eager fused forward within the same tolerance as the compiled path."""
    torch.manual_seed(0)
    batch, in_features, out_features = 4, 8, 6
    low_poly = torch.randn(7)
    high_poly = torch.randn(6)
    r = torch.rand(in_features, out_features)
    inputs = torch.randn(batch, in_features)
    noise_sample = torch.randn(out_features).expand(batch, in_features, out_features)
    kBT, BW, e, noise_min = 1.380649e-23 * 300, 1e-8, 2.718281828, 1e-12

    eager = _fused_memristor_forward(
        low_poly, high_poly, r, inputs, noise_sample, kBT, BW, e, noise_min
    )

    result_low = poly_mul_horner(low_poly, inputs).unsqueeze(-1)
    result_high = poly_mul_horner(high_poly, inputs).unsqueeze(-1)
    abs_in = torch.abs(inputs.unsqueeze(-1)) + noise_min
    jit = _get_jit_fused_core()(
        result_low, result_high, r, abs_in, noise_sample, kBT, BW, e
    )

    assert torch.allclose(eager, jit, atol=1e-6, rtol=1e-5)


@pytest.mark.fast
def test_cuda_capability_needs_jit_fallback_selection(monkeypatch):
    _cuda_capability_needs_jit_fallback.cache_clear()
    assert _cuda_capability_needs_jit_fallback(torch.device("cpu")) is False

    cuda = torch.device("cuda", 0)
    for capability, expected in [((6, 1), True), ((7, 5), False), ((8, 0), False)]:
        monkeypatch.setattr(
            torch.cuda, "get_device_capability", lambda d, c=capability: c
        )
        _cuda_capability_needs_jit_fallback.cache_clear()
        assert _cuda_capability_needs_jit_fallback(cuda) is expected
    _cuda_capability_needs_jit_fallback.cache_clear()
