"""
Regression test: the opt-in fast inference path must stay numerically equivalent
to the default (bit-exact) eager path.

This runs the fast core *without* torch.compile (``set_fast_compile(False)``) so it
is fast and has no compiler dependency in CI. That still guards the parts most
likely to regress: Horner polynomial evaluation and the eager noise draw that
reproduces ``randn_broadcast``'s layout. The compiled path is validated on GPU via
``benchmarks/run_gpu_bench.sbatch``.
"""

import pytest

import synaptogen_ml
from benchmarks.check_equivalence import compare_paths


@pytest.mark.fast
def test_fast_path_matches_eager_no_compile():
    synaptogen_ml.set_fast_compile(False)
    try:
        ok = compare_paths(device="cpu", atol=1e-4, rtol=1e-3)
    finally:
        synaptogen_ml.set_fast_compile(True)
        synaptogen_ml.set_fast_inference(False)
    assert ok, "fast path diverged from eager beyond tolerance"
