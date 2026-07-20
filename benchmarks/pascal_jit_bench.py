"""Bench: TorchScript(NNC)-fused memristor forward for pre-Volta GPUs.

torch.compile's CUDA backend is Triton, which requires CUDA capability >= 7.0 and
therefore crashes/disables on the gtx_1080 (sm_61) fleet. TorchScript's NNC
(TensorExpr) fuser generates fused elementwise CUDA kernels via NVRTC and works on
Pascal. This bench measures how much of the compiled-fast-path win NNC can recover
there, exercising the real shipped code
(``memristor._jit_fused_core`` / ``_cuda_capability_needs_jit_fallback``), not a
private copy, so this benchmark can't drift from what the package actually runs.

Run (on the target GPU):
    python -m benchmarks.pascal_jit_bench --device cuda
"""

import argparse
import statistics
import time

import torch

from benchmarks import _builders
from synaptogen_ml import set_fast_compile, set_fast_inference
from synaptogen_ml.memristor_modules.memristor import (
    MemristorArray,
    _cuda_capability_needs_jit_fallback,
    _jit_fused_core,
)
from synaptogen_ml.memristor_modules.util import poly_mul_horner


def _jit_forward(self: MemristorArray, inputs: torch.Tensor) -> torch.Tensor:
    # identical draw layout to eager/_forward_fast (one randn of the trailing shape)
    raw_shape = torch.broadcast_shapes(inputs.shape + (1,), self.r.shape)
    trailing_shape = raw_shape[self.broadcast_noise_dims :]
    noise_sample = torch.randn(trailing_shape, device=inputs.device)

    result_low = poly_mul_horner(self.resistance_weighted_poly_low, inputs).unsqueeze(
        -1
    )
    result_high = poly_mul_horner(self.resistance_weighted_poly_high, inputs).unsqueeze(
        -1
    )
    abs_in = torch.abs(inputs.unsqueeze(-1)) + self.noise_minimum_voltage
    return _jit_fused_core(
        result_low,
        result_high,
        self.r,
        abs_in,
        noise_sample,
        float(self.kBT),
        float(self.BW),
        float(self.e),
    )


# batches ~2.5x smaller than profile_modules "large"/"proj" so the eager baseline
# fits a 11 GB gtx_1080
PASCAL_SCALE = {
    "linear": dict(batch=192, in_features=512, out_features=2048),
    "tiled_linear": dict(
        batch=192, in_features=512, out_features=2048, tile_in=256, tile_out=256
    ),
    "conv1d": dict(batch=8, channels=512, time=500, kernel_size=31),
    "conv2d": dict(batch=8, in_channels=1, out_channels=32, height=200, width=80),
}


def _builders_tiny():
    return {
        "linear": dict(batch=4, in_features=64, out_features=32),
        "tiled_linear": dict(
            batch=4, in_features=64, out_features=32, tile_in=32, tile_out=32
        ),
        "conv1d": dict(batch=2, channels=16, time=32, kernel_size=3),
        "conv2d": dict(batch=2, in_channels=1, out_channels=8, height=16, width=16),
    }


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def run_once(module, x, device, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        out = module(x)
    _sync(device)
    return out


def profile(module, x, device, iters, warmup):
    with torch.no_grad():
        for _ in range(warmup):
            module(x)
    _sync(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times = []
    with torch.no_grad():
        for _ in range(iters):
            t0 = time.perf_counter()
            module(x)
            _sync(device)
            times.append((time.perf_counter() - t0) * 1000)
    peak = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.0
    return statistics.median(times), peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tiny", action="store_true", help="smoke-test sizes (CPU)")
    args = ap.parse_args()

    if args.tiny:
        for key, kwargs in _builders_tiny().items():
            PASCAL_SCALE[key] = kwargs

    if args.device == "cuda":
        cap = torch.cuda.get_device_capability()
        needs_jit = _cuda_capability_needs_jit_fallback(torch.device("cuda", 0))
        print(
            f"gpu={torch.cuda.get_device_name(0)} capability={cap[0]}.{cap[1]} "
            f"automatic-dispatch-picks-jit-fallback={needs_jit}"
        )

    eager_forward = MemristorArray.forward
    results = []
    for key, kwargs in PASCAL_SCALE.items():
        name, module, x = _builders.BUILDERS[key](device=args.device, **kwargs)

        # --- correctness: identical seeds -> identical draws in all paths
        MemristorArray.forward = eager_forward
        set_fast_inference(False)
        out_eager = run_once(module, x, args.device, args.seed)
        eager_ms, eager_mb = profile(module, x, args.device, args.iters, args.warmup)

        MemristorArray.forward = _jit_forward
        out_jit = run_once(module, x, args.device, args.seed)
        jit_ms, jit_mb = profile(module, x, args.device, args.iters, args.warmup)
        MemristorArray.forward = eager_forward

        set_fast_inference(True)
        set_fast_compile(False)
        out_nc = run_once(module, x, args.device, args.seed)
        nc_ms, nc_mb = profile(module, x, args.device, args.iters, args.warmup)
        set_fast_inference(False)

        d_jit = (out_jit - out_eager).abs().max().item()
        d_nc = (out_nc - out_eager).abs().max().item()
        n_jit = (out_jit != out_eager).sum().item()
        results.append(
            (name, eager_ms, eager_mb, jit_ms, jit_mb, nc_ms, nc_mb, d_jit, n_jit, d_nc)
        )
        print(
            f"{name:28s} eager {eager_ms:8.1f}ms/{eager_mb:7.0f}MB  "
            f"jit {jit_ms:8.1f}ms/{jit_mb:7.0f}MB (x{eager_ms / jit_ms:4.1f})  "
            f"fast-nocompile {nc_ms:8.1f}ms/{nc_mb:7.0f}MB (x{eager_ms / nc_ms:4.1f})  "
            f"jit_maxdiff {d_jit:.2e} ndiff {n_jit}  nc_maxdiff {d_nc:.2e}",
            flush=True,
        )

    # fusion evidence: NNC fusion groups in the specialized graph
    try:
        graph = torch.jit.last_executed_optimized_graph()
        n_groups = str(graph).count("TensorExprGroup")
        print(f"TensorExprGroup nodes in last optimized graph: {n_groups}")
    except Exception as exc:  # noqa: BLE001
        print(f"(could not inspect optimized graph: {exc})")

    print("\nsummary (median ms):")
    for name, e_ms, _, j_ms, _, n_ms, _, d_jit, n_jit, _ in results:
        print(
            f"  {name:28s} eager {e_ms:8.1f}  jit {j_ms:8.1f} (x{e_ms / j_ms:4.1f})  "
            f"nocompile {n_ms:8.1f} (x{e_ms / n_ms:4.1f})  maxdiff {d_jit:.2e}"
        )


if __name__ == "__main__":
    main()
