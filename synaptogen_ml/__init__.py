"""SynaptogenML: memristor-array simulation for PyTorch.

Exposes a process-wide switch for the opt-in *fast inference* path. With it off
(the default) the memristor modules run a bit-exact eager forward. With it on,
forwards use a fused implementation that preserves the noise model and the
random-draw layout but may differ from the eager path by ~1e-6 (kernel fusion +
Horner polynomial evaluation). The fused backend is chosen automatically per
device: ``torch.compile`` (Triton) on CUDA capability >= 7.0, a
TorchScript(NNC)-fused fallback on older CUDA devices (e.g. GTX 1080 / Pascal)
where Triton is unavailable, and plain eager execution of the fused function on
CPU (same ``has_triton()`` guard as ``poly_mul``) — no separate flag needed.

Note: the first compiled forward raises ``torch._dynamo.config.cache_size_limit``
process-wide (to at least 64) so all input ranks stay cached.

A second switch selects the readout-noise model (``set_readout_noise_model``):
``"legacy"`` (default, historical constants), ``"physical"`` (upstream Synaptogen
constants) or ``"off"``. See the comment above ``READOUT_NOISE_MODELS``.
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
    """Whether the fast path torch.compiles its forward (see ``set_fast_compile``)."""
    return _FAST_COMPILE


# Readout-noise model of ``MemristorArray`` (Johnson + shot noise at inference):
#   "legacy"   (default) historical constants: electron charge e = Euler's number and
#              bandwidth BW = 1e-8, a port typo of upstream Synaptogen. The Johnson term
#              is ~0 and the shot-noise sigma ~37x the physical value. Kept as default
#              so existing results reproduce bit-exactly.
#   "physical" upstream Synaptogen constants (e = 1.602176634e-19 C, BW = 1e8 Hz), the
#              same values as the numpy cell model (``synaptogen.Iread``).
#   "off"      no readout noise: deterministic forward, no random draws at all.
# Programming variability (device-to-device, cycle-to-cycle) is unaffected by all modes.
# Select without code changes via:
#   SYN_READOUT_NOISE_MODEL=physical python -m ...
READOUT_NOISE_MODELS = ("legacy", "physical", "off")


def _check_readout_noise_model(model: str) -> str:
    if model not in READOUT_NOISE_MODELS:
        raise ValueError(
            f"Unknown readout noise model {model!r}, expected one of {READOUT_NOISE_MODELS}"
        )
    return model


_READOUT_NOISE_MODEL = _check_readout_noise_model(
    os.environ.get("SYN_READOUT_NOISE_MODEL", "").strip().lower() or "legacy"
)


def set_readout_noise_model(model: str) -> None:
    """Select the readout-noise model (``"legacy"``, ``"physical"`` or ``"off"``).

    Process-wide and read at every forward, so it may be changed between forwards
    of an already constructed/programmed model.
    """
    global _READOUT_NOISE_MODEL
    _READOUT_NOISE_MODEL = _check_readout_noise_model(model)


def readout_noise_model() -> str:
    """The currently selected readout-noise model (default ``"legacy"``)."""
    return _READOUT_NOISE_MODEL


def set_readout_noise(enabled: bool = True) -> None:
    """Shorthand: ``False`` selects ``"off"``; ``True`` re-enables the noise, going
    back to ``"legacy"`` if it was off (an explicitly chosen ``"physical"`` stays)."""
    global _READOUT_NOISE_MODEL
    if not enabled:
        _READOUT_NOISE_MODEL = "off"
    elif _READOUT_NOISE_MODEL == "off":
        _READOUT_NOISE_MODEL = "legacy"


def has_readout_noise() -> bool:
    """Whether the readout noise is applied (any model except ``"off"``)."""
    return _READOUT_NOISE_MODEL != "off"


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
    (0 disables it -> serial default path). The workers are spawned, so the
    calling script needs an ``if __name__ == "__main__":`` guard. See
    ``synaptogen_ml.programming`` for details."""
    global _FAST_PROGRAMMING_WORKERS
    _FAST_PROGRAMMING_WORKERS = int(workers)


def fast_programming_workers() -> int:
    """Number of parallel programming workers (0 = serial default path)."""
    return _FAST_PROGRAMMING_WORKERS
