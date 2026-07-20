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
    kBT, BW, e, noise_min = 1.380649e-23 * 300, 1e-8, 2.718281828, 1e-12

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
def test_cuda_capability_needs_jit_fallback_selection(monkeypatch):
    cpu = torch.device("cpu")
    assert _cuda_capability_needs_jit_fallback(cpu) is False

    cuda = torch.device("cuda", 0)
    for capability, expected in [((6, 1), True), ((7, 5), False), ((8, 0), False)]:
        monkeypatch.setattr(
            torch.cuda, "get_device_capability", lambda d, c=capability: c
        )
        assert _cuda_capability_needs_jit_fallback(cuda) is expected
