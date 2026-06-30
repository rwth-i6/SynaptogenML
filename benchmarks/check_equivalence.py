"""
Equivalence harness for the memristor modules.

Workflow:
  1. Before changing anything, snapshot the current outputs:
        python -m benchmarks.check_equivalence --save golden.pt
  2. After a change, compare against the snapshot:
        # bit-exact (Tier 1A): outputs must match exactly
        python -m benchmarks.check_equivalence --check golden.pt --exact
        # fast path (Tier 1B): allow tiny fusion/reduction drift
        SYN_FAST=1 python -m benchmarks.check_equivalence --check golden.pt --atol 1e-5 --rtol 1e-5

Each module is built with a fixed build seed (deterministic cell programming and
example input) and the torch RNG is reseeded immediately before the forward so the
readout-noise draw is identical between runs.
"""

import argparse

import torch

from benchmarks._builders import BUILDERS

BUILD_SEED = 0
FORWARD_SEED = 1234

# Small sizes: correctness is size-independent, so keep the shared node cheap.
SIZES = {
    "linear": dict(batch=8, in_features=96, out_features=48),
    "tiled_linear": dict(
        batch=8, in_features=96, out_features=48, tile_in=32, tile_out=32
    ),
    "conv1d": dict(batch=2, channels=24, time=40, kernel_size=5),
    "conv2d": dict(batch=2, in_channels=1, out_channels=8, height=12, width=12),
}


def compute_outputs(device: str = "cpu", only=None) -> dict:
    outputs = {}
    for key, builder in BUILDERS.items():
        if only and key not in only:
            continue
        name, module, x = builder(device=device, seed=BUILD_SEED, **SIZES[key])
        torch.manual_seed(FORWARD_SEED)
        with torch.no_grad():
            outputs[key] = module(x).detach().cpu()
    return outputs


def compare_paths(device="cpu", only=None, atol=1e-4, rtol=1e-3) -> bool:
    """Self-contained eager-vs-fast comparison (no golden file needed).

    For each module, run the default (eager, bit-exact) path and the fast path on
    the same instance, reseeding torch beforehand so the readout-noise draw is
    identical, and compare. Reports max abs diff and how many elements differ
    (post-ADC quantization can turn ~1e-6 analog drift into rare 1-step flips).
    """
    import synaptogen_ml

    all_ok = True
    for key, builder in BUILDERS.items():
        if only and key not in only:
            continue
        synaptogen_ml.set_fast_inference(False)
        name, module, x = builder(device=device, seed=BUILD_SEED, **SIZES[key])
        torch.manual_seed(FORWARD_SEED)
        with torch.no_grad():
            eager = module(x).detach().cpu()

        synaptogen_ml.set_fast_inference(True)
        torch.manual_seed(FORWARD_SEED)
        with torch.no_grad():
            fast = module(x).detach().cpu()
        synaptogen_ml.set_fast_inference(False)

        ok = torch.allclose(fast, eager, atol=atol, rtol=rtol)
        max_abs = (fast - eager).abs().max().item()
        n_diff = int((fast != eager).sum().item())
        print(
            f"{key:24s} {'allclose' if ok else 'DIFFERS':9s} "
            f"max_abs_diff={max_abs:.3e} n_diff={n_diff}/{eager.numel()}"
        )
        all_ok = all_ok and ok
    print("ALL OK" if all_ok else "MISMATCH")
    return all_ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", metavar="FILE", help="snapshot current outputs to FILE")
    ap.add_argument("--check", metavar="FILE", help="compare current outputs to FILE")
    ap.add_argument("--exact", action="store_true", help="require torch.equal")
    ap.add_argument("--atol", type=float, default=1e-5)
    ap.add_argument("--rtol", type=float, default=1e-5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--only", nargs="*", help="subset of module keys to check")
    ap.add_argument(
        "--compare-paths",
        action="store_true",
        help="compare eager vs fast in one process (no golden file)",
    )
    args = ap.parse_args()

    if args.compare_paths:
        ok = compare_paths(args.device, only=args.only, atol=args.atol, rtol=args.rtol)
        raise SystemExit(0 if ok else 1)

    outputs = compute_outputs(args.device, only=args.only)

    if args.save:
        torch.save(outputs, args.save)
        for k, v in outputs.items():
            print(f"saved {k:24s} shape={tuple(v.shape)}")
        print(f"-> {args.save}")
        return

    if args.check:
        golden = torch.load(args.check)
        all_ok = True
        for k, v in outputs.items():
            g = golden[k]
            if args.exact:
                ok = torch.equal(v, g)
                detail = "equal" if ok else "DIFFERS"
            else:
                ok = torch.allclose(v, g, atol=args.atol, rtol=args.rtol)
                detail = "allclose" if ok else "DIFFERS"
            max_abs = (v - g).abs().max().item() if v.shape == g.shape else float("nan")
            print(f"{k:24s} {detail:9s} max_abs_diff={max_abs:.3e}")
            all_ok = all_ok and ok
        print("ALL OK" if all_ok else "MISMATCH")
        raise SystemExit(0 if all_ok else 1)

    ap.error("pass --save or --check")


if __name__ == "__main__":
    main()
