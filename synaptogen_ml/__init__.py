"""SynaptogenML: memristor-array simulation for PyTorch.

Exposes a process-wide switch for the opt-in *fast inference* path. With it off
(the default) the memristor modules run a bit-exact eager forward. With it on,
forwards use a fused implementation that preserves the noise model and the
random-draw layout but may differ from the eager path by ~1e-6 (kernel fusion +
Horner polynomial evaluation). The fused backend is chosen automatically per
device: ``torch.compile`` (Triton) on CPU or CUDA capability >= 7.0, or a
TorchScript(NNC)-fused fallback on older CUDA devices (e.g. GTX 1080 / Pascal)
where Triton is unavailable — no separate flag needed.
"""

import os


def _truthy(value: str) -> bool:
    return value.strip().lower() not in ("", "0", "false", "no", "off")


# Allow enabling process-wide without code changes (handy for SLURM batch jobs):
#   SYN_FAST=1 python -m ...
_FAST_INFERENCE = _truthy(os.environ.get("SYN_FAST", ""))

# The fast path uses torch.compile by default. SYN_NO_COMPILE=1 keeps the fast
# core (Horner + restructured noise draw) but skips compilation — a fallback for
# environments where torch.compile is unavailable/problematic, and a cheap way to
# validate the fast core's numerics without paying compile cost.
_FAST_COMPILE = not _truthy(os.environ.get("SYN_NO_COMPILE", ""))


def set_fast_inference(enabled: bool = True) -> None:
    """Enable (default) or disable the opt-in fast inference path.

    Off by default → bit-exact eager path. Callers (e.g. a recognition pipeline)
    flip this once to trade ~1e-6 numerical drift for a faster, fused forward,
    without editing the modules themselves.
    """
    global _FAST_INFERENCE
    _FAST_INFERENCE = bool(enabled)


def is_fast_inference() -> bool:
    """Whether the fast inference path is currently enabled."""
    return _FAST_INFERENCE


def set_fast_compile(enabled: bool = True) -> None:
    """Whether the fast path should torch.compile its forward (default True)."""
    global _FAST_COMPILE
    _FAST_COMPILE = bool(enabled)


def fast_uses_compile() -> bool:
    return _FAST_COMPILE


# Opt-in parallel *programming* (cell conversion) path. 0 (default) = serial,
# behavior-identical to the historical code. N > 0 programs the independent
# (bit-plane, tile) cell-array pairs of a layer in N worker processes, each
# with its own freshly-seeded RNG streams: the physical model and every draw
# distribution are unchanged, but the individual random numbers differ from a
# serial run -- exactly as two serial runs differ from each other, since the
# programming RNG is unseeded. Enable without code changes via:
#   SYN_FAST_PROG=8 python -m ...
_FAST_PROGRAMMING_WORKERS = int(os.environ.get("SYN_FAST_PROG", "0") or "0")


def set_fast_programming(workers: int) -> None:
    """Enable the opt-in parallel programming path with ``workers`` processes
    (0 disables it -> serial default path). See benchmarks/check_programming.py
    for the statistical-equivalence demonstration."""
    global _FAST_PROGRAMMING_WORKERS
    _FAST_PROGRAMMING_WORKERS = int(workers)


def fast_programming_workers() -> int:
    """Number of parallel programming workers (0 = serial default path)."""
    return _FAST_PROGRAMMING_WORKERS
