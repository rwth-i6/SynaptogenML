"""Timing + statistical-equivalence check for the programming path.

Programs the same set of independent pos/neg weight pairs three times:

* serial run A (default path),
* serial run B (default path again -- the natural "same distribution,
  different draws" reference, since the programming RNG is unseeded),
* parallel run with ``set_fast_programming(workers)``.

and then compares (1) wall time, (2) the distribution of the programmed
device states r per target-weight group via moments and a two-sample KS
test, judged against the serial-vs-serial baseline, and (3) shows that the
individual draws differ elementwise in all three runs, as expected.

Torch-free on purpose -- runs with any Python that has numpy.

Usage:
    python benchmarks/check_programming.py [--pairs 24] [--cells 16384]
        [--wear 3] [--workers 8] [--density 0.5] [--correction-cycles 0]
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import synaptogen_ml  # noqa: E402
from synaptogen_ml.programming import program_pairs  # noqa: E402


def make_jobs(num_pairs, cells, density, seed=0x5EED):
    """Ternary weights like a real bit-plane: P(+1) = P(-1) = density / 2."""
    gen = np.random.default_rng(seed)
    jobs = []
    for _ in range(num_pairs):
        w = gen.choice([-1, 0, 1], size=cells, p=[density / 2, 1 - density, density / 2])
        pos = (w > 0).astype(np.float32)
        neg = (w < 0).astype(np.float32)
        jobs.append((pos, neg))
    return jobs


def run(jobs, wear, workers, correction):
    synaptogen_ml.set_fast_programming(workers)
    start = time.perf_counter()
    results = program_pairs(jobs, wear, correction, 1.0)
    elapsed = time.perf_counter() - start
    synaptogen_ml.set_fast_programming(0)
    r = np.concatenate(
        [np.asarray(pos.r) for pos, _ in results]
        + [np.asarray(neg.r) for _, neg in results]
    )
    targets = np.concatenate([p for p, _ in jobs] + [n for _, n in jobs])
    return r, targets, elapsed


def ks_2samp(a, b):
    """Two-sample Kolmogorov-Smirnov statistic + asymptotic p-value."""
    a = np.sort(a)
    b = np.sort(b)
    both = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, both, side="right") / len(a)
    cdf_b = np.searchsorted(b, both, side="right") / len(b)
    d = float(np.max(np.abs(cdf_a - cdf_b)))
    en = np.sqrt(len(a) * len(b) / (len(a) + len(b)))
    t = (en + 0.12 + 0.11 / en) * d
    p = 2.0 * sum((-1) ** (k - 1) * np.exp(-2.0 * (k * t) ** 2) for k in range(1, 101))
    return d, float(min(max(p, 0.0), 1.0))


def moments(x):
    q = np.quantile(x, [0.01, 0.5, 0.99])
    return (
        f"n={len(x):>9d} mean={x.mean():+.6f} std={x.std():.6f} "
        f"p1={q[0]:+.4f} p50={q[1]:+.4f} p99={q[2]:+.4f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=24)
    parser.add_argument("--cells", type=int, default=16384)
    parser.add_argument("--wear", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--density", type=float, default=0.5)
    parser.add_argument("--correction-cycles", type=int, default=0)
    args = parser.parse_args()

    correction = None
    if args.correction_cycles > 0:
        # torch-free stand-in for CycleCorrectionSettings (duck-typed)
        from types import SimpleNamespace

        correction = SimpleNamespace(
            num_cycles=args.correction_cycles,
            test_input_value=0.2,
            relative_deviation=0.2,
            ideal_programming=False,
        )

    jobs = make_jobs(args.pairs, args.cells, args.density)
    total_cells = 2 * args.pairs * args.cells
    print(
        f"programming {args.pairs} pairs x {args.cells} cells "
        f"({total_cells} cells total), wear={args.wear} "
        f"(-> {args.wear * 15 * 2 + 4} applyVoltage calls per pair), "
        f"correction_cycles={args.correction_cycles}, workers={args.workers}"
    )

    r_a, targets, t_a = run(jobs, args.wear, 0, correction)
    r_b, _, t_b = run(jobs, args.wear, 0, correction)
    # first parallel run pays the one-time worker spawn; second is steady state
    r_p, _, t_p_cold = run(jobs, args.wear, args.workers, correction)
    r_p2, _, t_p = run(jobs, args.wear, args.workers, correction)

    print(f"\nwall time: serial A {t_a:.2f}s | serial B {t_b:.2f}s | "
          f"parallel cold {t_p_cold:.2f}s | parallel warm {t_p:.2f}s "
          f"| speedup (warm vs serial A) x{t_a / t_p:.1f}")

    print("\nprogrammed state r by target-weight group:")
    ks_rows = []
    for label, mask in [("set (w=1)", targets == 1.0), ("reset (w=0)", targets == 0.0)]:
        print(f"  group {label}:")
        print(f"    serial A : {moments(r_a[mask])}")
        print(f"    serial B : {moments(r_b[mask])}")
        print(f"    parallel : {moments(r_p[mask])}")
        d_bb, p_bb = ks_2samp(r_a[mask], r_b[mask])
        d_pp, p_pp = ks_2samp(r_a[mask], r_p[mask])
        ks_rows.append((label, d_bb, p_bb, d_pp, p_pp))
        print(f"    KS serial-vs-serial   D={d_bb:.5f} p={p_bb:.3f}   (baseline)")
        print(f"    KS serial-vs-parallel D={d_pp:.5f} p={p_pp:.3f}")

    print("\nindividual draws (elementwise):")
    print(f"  serial A == serial B : {bool(np.array_equal(r_a, r_b))} "
          f"(max |diff| {np.max(np.abs(r_a - r_b)):.4f})")
    print(f"  serial A == parallel : {bool(np.array_equal(r_a, r_p))} "
          f"(max |diff| {np.max(np.abs(r_a - r_p)):.4f})")
    print(f"  parallel == parallel rerun: {bool(np.array_equal(r_p, r_p2))} "
          f"(max |diff| {np.max(np.abs(r_p - r_p2)):.4f})")

    ok = all(
        p_pp > 0.01 or d_pp <= 3 * max(d_bb, 1e-12)
        for _, d_bb, p_bb, d_pp, p_pp in ks_rows
    )
    print(
        "\nVERDICT: "
        + (
            "PASS -- parallel differs from serial no more than a serial re-run "
            "differs from another serial re-run (same distributions, different draws)"
            if ok
            else "FAIL -- serial-vs-parallel KS distance is far outside the "
            "serial-vs-serial baseline; investigate before using fast programming"
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
