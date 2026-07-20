"""
Timing / peak-memory profiler for the memristor inference modules (plan Step 1).

Examples:
  # quick CPU sanity (tiny sizes)
  python -m benchmarks.profile_modules --device cpu --iters 5 --scale tiny

  # GPU profiling (submit via SLURM sbatch / srun)
  python -m benchmarks.profile_modules --device cuda --iters 50 --scale large

  # baseline vs fast path on the same sizes
  python -m benchmarks.profile_modules --device cuda --scale large            # baseline
  SYN_FAST=1 python -m benchmarks.profile_modules --device cuda --scale large # fast path

Reports per-module wall time (median over iters) and, on CUDA, peak allocated
memory. Sizes are placeholders until the real recognition-model dimensions are
provided (see the plan's "Remaining input needed").
"""

import argparse
import statistics
import time

import torch

from benchmarks import _builders

# Per-module kwargs for each scale. The Conformer-realistic scales use the dims the
# user gave: feed-forward linears are 512 -> 2048, attention/pointwise projections
# are 512 -> 512. Here ``batch`` is the effective B*T (frames) the linear sees.
# Batches are chosen so the EAGER baseline fits a 24 GB GPU; the fast (fused) path
# should reach a much lower peak memory at the same batch.
SCALES = {
    "tiny": {
        "linear": dict(batch=4, in_features=64, out_features=32),
        "tiled_linear": dict(
            batch=4, in_features=64, out_features=32, tile_in=32, tile_out=32
        ),
        "conv1d": dict(batch=2, channels=16, time=32, kernel_size=3),
        "conv2d": dict(batch=2, in_channels=1, out_channels=8, height=16, width=16),
    },
    "test": {
        "linear": dict(batch=64, in_features=784, out_features=512),
        "tiled_linear": dict(batch=64, in_features=784, out_features=512),
        "conv1d": dict(batch=8, channels=256, time=512, kernel_size=9),
        "conv2d": dict(batch=8, in_channels=1, out_channels=32, height=32, width=32),
    },
    "large": {  # Conformer feed-forward linears: 512 -> 2048
        "linear": dict(batch=512, in_features=512, out_features=2048),
        "tiled_linear": dict(
            batch=512, in_features=512, out_features=2048, tile_in=256, tile_out=256
        ),
        "conv1d": dict(batch=16, channels=512, time=500, kernel_size=31),
        "conv2d": dict(batch=8, in_channels=1, out_channels=32, height=200, width=80),
    },
    "proj": {  # Conformer attention / pointwise projections: 512 -> 512
        "linear": dict(batch=2048, in_features=512, out_features=512),
        "tiled_linear": dict(
            batch=2048, in_features=512, out_features=512, tile_in=256, tile_out=256
        ),
        "conv1d": dict(batch=16, channels=512, time=500, kernel_size=31),
        "conv2d": dict(batch=8, in_channels=1, out_channels=32, height=200, width=80),
    },
}


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def profile_one(key, kwargs, *, device, iters, warmup):
    name, module, x = _builders.BUILDERS[key](device=device, **kwargs)
    # warmup (also triggers any lazy compile on the fast path)
    with torch.no_grad():
        for _ in range(warmup):
            module(x)
    _sync(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    times = []
    with torch.no_grad():
        for _ in range(iters):
            _sync(device)
            t0 = time.perf_counter()
            module(x)
            _sync(device)
            times.append(time.perf_counter() - t0)

    peak_mb = (
        torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else float("nan")
    )
    return name, statistics.median(times) * 1e3, peak_mb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--scale", default="tiny", choices=list(SCALES))
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--only", nargs="*", help="subset of module keys")
    args = ap.parse_args()

    import os

    print(
        f"device={args.device} scale={args.scale} iters={args.iters} "
        f"fast={'1' if os.environ.get('SYN_FAST') else '0'}"
    )
    print(f"{'module':28s} {'median_ms':>12s} {'peak_MB':>12s}")
    keys = args.only or list(SCALES[args.scale])
    for key in keys:
        name, ms, peak = profile_one(
            key,
            SCALES[args.scale][key],
            device=args.device,
            iters=args.iters,
            warmup=args.warmup,
        )
        print(f"{name:28s} {ms:12.3f} {peak:12.1f}")


if __name__ == "__main__":
    main()
