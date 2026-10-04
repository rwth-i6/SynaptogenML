"""
Readout-noise models of ``MemristorArray`` (``synaptogen_ml.set_readout_noise_model``):

- "legacy" (default) must stay bit-identical to the historical eager arithmetic
  (e = Euler's number, BW = 1e-8), so existing results reproduce.
- "physical" must use the upstream Synaptogen constants (same as the numpy model).
- "off" must be deterministic and must not consume the RNG stream.
- The fast path must match the eager path in every model.
"""

import os
import subprocess
import sys

import numpy as np
import pytest
import torch

import synaptogen_ml
from synaptogen_ml import synaptogen
from synaptogen_ml.memristor_modules.memristor import (
    MemristorArray,
    _fused_memristor_forward_noiseless,
    _get_jit_fused_core_noiseless,
)
from synaptogen_ml.memristor_modules.util import poly_mul_horner, randn_broadcast
from test_fast_equivalence import build_conv1d, build_linear

FORWARD_SEED = 1234
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# exactly the historical objects (numpy float64 for e), see _READOUT_NOISE_CONSTANTS
LEGACY = (1e-8, np.exp(1))
PHYSICAL = (1e8, 1.602176634e-19)


@pytest.fixture(autouse=True)
def _restore_switches():
    yield
    synaptogen_ml.set_readout_noise_model("legacy")
    synaptogen_ml.set_fast_inference(False)
    synaptogen_ml.set_fast_compile(True)


def _array(in_features=16, out_features=8):
    """A small programmed-looking array: uA-range currents at sub-volt inputs."""
    torch.manual_seed(0)
    arr = MemristorArray(in_features, out_features)
    arr.resistance_weighted_poly_low.data = torch.randn(7) * 1e-5
    arr.resistance_weighted_poly_high.data = torch.randn(6) * 1e-5
    arr.r.data = torch.rand(in_features, out_features)
    x = (torch.rand(4, in_features) * 2 - 1) * 0.6
    return arr.eval(), x


def _reference(arr, x, BW, e):
    """The pre-model eager arithmetic, written out with explicit constants."""
    with torch.no_grad():
        result_raw = arr.compute_raw_output(x)
        abs_raw = torch.abs(result_raw)
        johnson_noise = (
            4
            * arr.kBT
            * BW
            * (abs_raw / (torch.abs(x.unsqueeze(-1)) + arr.noise_minimum_voltage))
        )
        shot_noise = 2 * e * abs_raw * BW
        sigma_total = torch.sqrt(johnson_noise + shot_noise)
        torch.manual_seed(FORWARD_SEED)
        noise = randn_broadcast(
            result_raw.shape, arr.broadcast_noise_dims, device=x.device
        )
        result_raw += noise * sigma_total
        return torch.sum(result_raw, dim=-2)


def _forward(module, x):
    torch.manual_seed(FORWARD_SEED)
    with torch.no_grad():
        return module(x)


@pytest.mark.readout
def test_default_model_is_legacy():
    assert synaptogen_ml.readout_noise_model() == "legacy"
    assert synaptogen_ml.has_readout_noise()


@pytest.mark.readout
def test_legacy_bit_identical_to_historical_eager():
    arr, x = _array()
    assert torch.equal(_forward(arr, x), _reference(arr, x, *LEGACY))


@pytest.mark.readout
def test_physical_uses_upstream_constants():
    synaptogen_ml.set_readout_noise_model("physical")
    arr, x = _array()
    assert (arr.BW, arr.e) == PHYSICAL
    # same constants as the numpy cell model (float32 there)
    assert np.isclose(arr.e, synaptogen.e, rtol=1e-6)
    assert torch.equal(_forward(arr, x), _reference(arr, x, *PHYSICAL))


@pytest.mark.readout
def test_physical_noise_far_below_legacy():
    arr, x = _array()
    with torch.no_grad():
        noiseless = torch.sum(arr.compute_raw_output(x), dim=-2)
    legacy_dev = (_forward(arr, x) - noiseless).abs().mean()
    synaptogen_ml.set_readout_noise_model("physical")
    physical_dev = (_forward(arr, x) - noiseless).abs().mean()
    assert physical_dev < legacy_dev / 10


@pytest.mark.readout
def test_model_is_read_at_every_forward():
    arr, x = _array()
    for model, constants in [
        ("legacy", LEGACY),
        ("physical", PHYSICAL),
        ("legacy", LEGACY),
    ]:
        synaptogen_ml.set_readout_noise_model(model)
        assert torch.equal(_forward(arr, x), _reference(arr, x, *constants))


@pytest.mark.readout
@pytest.mark.parametrize("fast", [False, True])
def test_off_is_deterministic_and_draws_nothing(fast):
    arr, x = _array()
    synaptogen_ml.set_readout_noise_model("off")
    synaptogen_ml.set_fast_compile(False)
    synaptogen_ml.set_fast_inference(fast)
    with torch.no_grad():
        noiseless = torch.sum(arr.compute_raw_output(x), dim=-2)
        rng_state = torch.get_rng_state()
        out = arr(x)
        assert torch.equal(torch.get_rng_state(), rng_state)
    if fast:  # Horner evaluation, ~1e-6 relative drift
        assert torch.allclose(out, noiseless, atol=1e-12, rtol=1e-5)
    else:
        assert torch.equal(out, noiseless)


@pytest.mark.readout
@pytest.mark.parametrize("model", ["legacy", "physical", "off"])
@pytest.mark.parametrize("build", [build_linear, build_conv1d])
def test_fast_path_matches_eager_per_model(model, build):
    torch.manual_seed(0)
    module, x = build()
    synaptogen_ml.set_readout_noise_model(model)
    synaptogen_ml.set_fast_compile(False)
    synaptogen_ml.set_fast_inference(False)
    eager = _forward(module, x)
    synaptogen_ml.set_fast_inference(True)
    fast = _forward(module, x)
    # post-ADC quantization can turn ~1e-6 analog drift into rare 1-step flips
    assert torch.allclose(fast, eager, atol=1e-4, rtol=1e-3)


@pytest.mark.readout
def test_jit_noiseless_core_matches_eager():
    torch.manual_seed(0)
    low_poly, high_poly = torch.randn(7), torch.randn(6)
    r = torch.rand(8, 6)
    inputs = torch.randn(4, 8)
    eager = _fused_memristor_forward_noiseless(low_poly, high_poly, r, inputs)
    result_low = poly_mul_horner(low_poly, inputs).unsqueeze(-1)
    result_high = poly_mul_horner(high_poly, inputs).unsqueeze(-1)
    jit = _get_jit_fused_core_noiseless()(result_low, result_high, r)
    assert torch.allclose(eager, jit, atol=1e-6, rtol=1e-5)


@pytest.mark.readout
def test_set_readout_noise_shorthand():
    synaptogen_ml.set_readout_noise(False)
    assert synaptogen_ml.readout_noise_model() == "off"
    assert not synaptogen_ml.has_readout_noise()
    synaptogen_ml.set_readout_noise(True)
    assert synaptogen_ml.readout_noise_model() == "legacy"
    synaptogen_ml.set_readout_noise_model("physical")
    synaptogen_ml.set_readout_noise(True)
    assert synaptogen_ml.readout_noise_model() == "physical"


@pytest.mark.readout
def test_unknown_model_raises():
    with pytest.raises(ValueError):
        synaptogen_ml.set_readout_noise_model("physics")
    assert synaptogen_ml.readout_noise_model() == "legacy"


@pytest.mark.readout
@pytest.mark.parametrize(
    "value, expected", [("", "legacy"), ("physical", "physical"), (" OFF ", "off")]
)
def test_env_var_selects_model(value, expected):
    code = "import synaptogen_ml; print(synaptogen_ml.readout_noise_model())"
    env = {**os.environ, "SYN_READOUT_NOISE_MODEL": value}
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == expected


@pytest.mark.readout
def test_env_var_unknown_model_fails_import():
    code = "import synaptogen_ml"
    env = {**os.environ, "SYN_READOUT_NOISE_MODEL": "physics"}
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, cwd=REPO_ROOT, capture_output=True
    )
    assert out.returncode != 0
