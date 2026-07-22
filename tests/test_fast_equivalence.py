"""
Regression test: the opt-in fast inference path must stay numerically equivalent
to the default (bit-exact) eager path.

This runs the fast core *without* torch.compile (``set_fast_compile(False)``) so it
is fast and has no compiler dependency in CI. That still guards the parts most
likely to regress: Horner polynomial evaluation and the eager noise draw that
reproduces ``randn_broadcast``'s layout. The compiled path is validated on GPU via
``benchmarks/run_gpu_bench.sbatch``.
"""

import torch
import pytest

import synaptogen_ml
from benchmarks.check_equivalence import compare_paths
from synaptogen_ml.memristor_modules.memristor import (
    MemristorArray,
    _cuda_capability_needs_jit_fallback,
    _fused_memristor_forward,
    _jit_fused_core,
)


@pytest.mark.fast
def test_fast_path_matches_eager_no_compile():
    synaptogen_ml.set_fast_compile(False)
    try:
        ok = compare_paths(device="cpu", atol=1e-4, rtol=1e-3)
    finally:
        synaptogen_ml.set_fast_compile(True)
        synaptogen_ml.set_fast_inference(False)
    assert ok, "fast path diverged from eager beyond tolerance"


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
    kBT, BW, e, noise_min = 1.380649e-23 * 300, 1e8, 1.602176634e-19, 1e-12

    eager = _fused_memristor_forward(
        low_poly, high_poly, r, inputs, noise_sample, kBT, BW, e, noise_min
    )

    from synaptogen_ml.memristor_modules.util import poly_mul_horner

    result_low = poly_mul_horner(low_poly, inputs).unsqueeze(-1)
    result_high = poly_mul_horner(high_poly, inputs).unsqueeze(-1)
    abs_in = torch.abs(inputs.unsqueeze(-1)) + noise_min
    jit = _jit_fused_core(result_low, result_high, r, abs_in, noise_sample, kBT, BW, e)

    assert torch.allclose(eager, jit, atol=1e-6, rtol=1e-5)


@pytest.mark.fast
def test_noise_constants_match_upstream():
    """Guard against the port typo returning: e must be the elementary charge
    and BW the 1e8 Hz readout bandwidth (upstream synaptogen.py L33/L328;
    previously e = np.exp(1) and BW = 1e-8)."""
    arr = MemristorArray(4, 3)
    assert arr.e == 1.602176634e-19
    assert arr.BW == 1e8


def _small_initialized_array():
    torch.manual_seed(0)
    arr = MemristorArray(8, 6)
    arr.resistance_weighted_poly_low.data = torch.randn(7) * 1e-4
    arr.resistance_weighted_poly_high.data = torch.randn(6) * 1e-4
    arr.r.data = torch.rand(8, 6)
    inputs = torch.rand(4, 8) * 0.6
    return arr, inputs


@pytest.mark.fast
def test_noise_free_readout_deterministic_and_paths_match():
    """set_readout_noise(False): forward is deterministic, equals the raw cell
    current sum, matches across eager / fast(no-compile) / jit-core paths, and
    consumes no RNG (seed-comparability with noisy runs)."""
    arr, inputs = _small_initialized_array()
    synaptogen_ml.set_readout_noise(False)
    try:
        out1 = arr(inputs)
        out2 = arr(inputs)
        assert torch.equal(out1, out2), "noise-free forward must be deterministic"

        raw_sum = torch.sum(arr.compute_raw_output(inputs), dim=-2)
        assert torch.equal(out1, raw_sum)

        torch.manual_seed(123)
        rng_reference = torch.randn(3)
        torch.manual_seed(123)
        _ = arr(inputs)
        assert torch.equal(torch.randn(3), rng_reference), (
            "noise-free forward must not consume the RNG stream"
        )

        synaptogen_ml.set_fast_inference(True)
        synaptogen_ml.set_fast_compile(False)
        fast = arr(inputs)
        assert torch.allclose(out1, fast, atol=1e-6, rtol=1e-5)
    finally:
        synaptogen_ml.set_readout_noise(True)
        synaptogen_ml.set_fast_inference(False)
        synaptogen_ml.set_fast_compile(True)


@pytest.mark.fast
def test_jit_noiseless_core_matches_eager_noiseless():
    from synaptogen_ml.memristor_modules.memristor import (
        _fused_memristor_forward_noiseless,
        _jit_fused_core_noiseless,
    )
    from synaptogen_ml.memristor_modules.util import poly_mul_horner

    torch.manual_seed(1)
    low_poly, high_poly = torch.randn(7), torch.randn(6)
    r = torch.rand(8, 6)
    inputs = torch.randn(4, 8)

    eager = _fused_memristor_forward_noiseless(low_poly, high_poly, r, inputs)
    result_low = poly_mul_horner(low_poly, inputs).unsqueeze(-1)
    result_high = poly_mul_horner(high_poly, inputs).unsqueeze(-1)
    jit = _jit_fused_core_noiseless(result_low, result_high, r)
    assert torch.allclose(eager, jit, atol=1e-6, rtol=1e-5)


@pytest.mark.fast
def test_cuda_capability_needs_jit_fallback_selection(monkeypatch):
    cpu = torch.device("cpu")
    assert _cuda_capability_needs_jit_fallback(cpu) is False

    cuda = torch.device("cuda", 0)
    for capability, expected in [((6, 1), True), ((7, 5), False), ((8, 0), False)]:
        monkeypatch.setattr(
            torch.cuda, "get_device_capability", lambda d, c=capability: c
        )
        assert _cuda_capability_needs_jit_fallback(cuda) is expected
