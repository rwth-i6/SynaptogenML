"""Analyze partner IV measurement files (``*.dat`` from the Keithley-style
``I(V)`` export) and print per-device statistics used to build the empirical
device backend.

File format (as delivered 2026-11, ``data/measurements/*.dat``):
  * one ASCII block: header lines, then a 4-column tab-separated table
    ``Voltage Source (V) | Voltage Probe (V) | Current Probe (A) | Resistance (Ohm)``
  * all devices are concatenated into that single table; each device is one
    quasi-static loop ``0 -> -1.5 V -> +1.5 V -> 0`` in 30 mV steps
    (~201 rows). A new device starts at the ``0.0`` sample that is followed by
    a negative sample.
  * the loop is the *first* IV sweep of each (pristine) device.

Usage:
  python tools/analyze_iv_dat.py data/measurements/W_WOx_PCMO.dat [more.dat ...]
      [--plots data/measurements/plots] [--vmax 0.6]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


# ----------------------------------------------------------------- parsing ---
# the parser lives in the package so the empirical backend and this tool agree
from synaptogen_ml.empirical.adapters.keithley import (  # noqa: E402
    branches,
    load_table,
    split_devices,
)


def current_at(branch, v):
    vs, ip = branch[:, 0], branch[:, 2]
    if v < vs.min() or v > vs.max():
        return np.nan
    o = np.argsort(vs)
    return float(np.interp(v, vs[o], ip[o]))


def switching_voltage(branch):
    """Voltage of the steepest increase of log|I| along the branch (|V| > 0.3)."""
    vs, ip = branch[:, 0], np.abs(branch[:, 2]) + 1e-13
    m = np.abs(vs) > 0.3
    if m.sum() < 3:
        return np.nan
    v, li = vs[m], np.log10(ip[m])
    dl = np.diff(li) / np.diff(v)
    k = int(np.argmax(np.abs(dl)))
    return float(0.5 * (v[k] + v[k + 1]))


# ------------------------------------------------------------- fit checks ---
def fit_poly_zero_intercept(v, i, deg):
    A = np.stack([v**p for p in range(1, deg + 1)], axis=1)
    coeffs, *_ = np.linalg.lstsq(A, i, rcond=None)
    return coeffs


def eval_poly_zero_intercept(coeffs, v):
    return sum(c * v ** (p + 1) for p, c in enumerate(coeffs))


def rel_rms(pred, i):
    return float(np.sqrt(np.mean((pred - i) ** 2)) / np.sqrt(np.mean(i**2)))


# ---------------------------------------------------------------- report ---
def pct(col):
    col = np.asarray(col, dtype=float)
    col = col[np.isfinite(col)]
    return np.median(col), np.percentile(col, 10), np.percentile(col, 90)


def analyze_file(path, vmax):
    devs = split_devices(load_table(path))
    name = os.path.basename(path)
    print(
        f"\n=== {name}: {len(devs)} devices, points per device {sorted({len(d) for d in devs})}"
    )
    vmins = [d[:, 0].min() for d in devs]
    vmaxs = [d[:, 0].max() for d in devs]
    print(
        f"sweep range: Vmin {min(vmins):.2f}..{max(vmins):.2f}, Vmax {min(vmaxs):.2f}..{max(vmaxs):.2f}"
    )

    recs = []
    for d in devs:
        b0, b1, b2 = branches(d)
        recs.append(
            dict(
                I_pristine_m02=current_at(b0, -0.2),
                I_pristine_m05=current_at(b0, -0.5),
                I_lrs_m02=current_at(b1, -0.2),
                I_lrs_m05=current_at(b1, -0.5),
                I_lrs_p02=current_at(b1, 0.2),
                I_lrs_p05=current_at(b1, 0.5),
                I_post_p02=current_at(b2, 0.2),
                I_post_p05=current_at(b2, 0.5),
                I_at_vmin=float(d[np.argmin(d[:, 0]), 2]),
                I_at_vmax=float(d[np.argmax(d[:, 0]), 2]),
                V_set=switching_voltage(b0),
                V_reset=switching_voltage(b1[b1[:, 0] > 0]),
            )
        )

    print(
        f"\n{'quantity':16s} {'median':>11s} {'p10':>11s} {'p90':>11s}   R = V/I(median)"
    )
    for key in recs[0]:
        med, p10, p90 = pct([r[key] for r in recs])
        extra = ""
        if key.endswith("m02") or key.endswith("p02"):
            extra = f"  {0.2 / abs(med):.2e} Ohm"
        elif key.endswith("m05") or key.endswith("p05"):
            extra = f"  {0.5 / abs(med):.2e} Ohm"
        print(f"{key:16s} {med:11.3e} {p10:11.3e} {p90:11.3e}{extra}")

    lrs = np.array([r["I_lrs_m02"] for r in recs])
    pri = np.array([r["I_pristine_m02"] for r in recs])
    lrs_p = np.array([r["I_lrs_p02"] for r in recs])
    print(
        f"\nLRS / pristine current ratio at -0.2 V: median {np.median(lrs / pri):.1f}"
    )
    print(
        f"LRS rectification |I(-0.2)| / |I(+0.2)|: median {np.median(np.abs(lrs / lrs_p)):.1f}"
    )
    print(
        "LRS nonlinearity I(-0.5)/I(-0.2): median "
        f"{np.median([r['I_lrs_m05'] / r['I_lrs_m02'] for r in recs]):.1f} (ohmic = 2.5)"
    )
    print(
        f"device-to-device spread of log10|I_LRS(-0.2)|: {np.std(np.log10(np.abs(lrs))):.3f} dex"
    )
    print(
        f"device-to-device spread of log10|I_pristine(-0.2)|: {np.std(np.log10(np.abs(pri))):.3f} dex"
    )
    floor = [
        np.std(branches(d)[0][np.abs(branches(d)[0][:, 0]) <= 0.09, 2]) for d in devs
    ]
    print(
        f"instrument noise floor (std of I for |V| < 0.1 V, pristine): median {np.median(floor):.2e} A"
    )

    # Fit quality of a zero-intercept polynomial inside the DAC window
    print(f"\nfit quality inside |V| <= {vmax} V (relative RMS, median over devices):")
    for bname, bidx in (("LRS (branch 1)", 1), ("pristine (branch 0)", 0)):
        for window, wlabel in (
            ((-vmax, vmax), "bipolar"),
            ((-vmax, 0.0), "negative only"),
        ):
            res = {}
            for d in devs:
                b = branches(d)[bidx]
                m = (b[:, 0] >= window[0]) & (b[:, 0] <= window[1])
                v, i = b[m, 0], b[m, 2]
                if len(v) < 10:
                    continue
                for deg in (5, 7, 9):
                    c = fit_poly_zero_intercept(v, i, deg)
                    res.setdefault(f"poly{deg}", []).append(
                        rel_rms(eval_poly_zero_intercept(c, v), i)
                    )
                o = np.argsort(v)
                res.setdefault("LUT 60 mV", []).append(
                    rel_rms(np.interp(v[o], v[o][::2], i[o][::2]), i[o])
                )
            line = ", ".join(f"{k} {np.median(vals):.3f}" for k, vals in res.items())
            print(f"  {bname:20s} {wlabel:14s}: {line}")
    return devs


def plot(all_devs, out_dir, vmax):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    n = len(all_devs)
    fig, axes = plt.subplots(2, n, figsize=(7 * n, 10), squeeze=False)
    for c, (name, devs) in enumerate(all_devs.items()):
        for d in devs:
            for b, color in zip(branches(d), ("C0", "C1", "C2")):
                axes[0, c].semilogy(
                    b[:, 0], np.abs(b[:, 2]) + 1e-13, color=color, lw=0.5, alpha=0.5
                )
                m = np.abs(b[:, 0]) <= vmax + 0.05
                axes[1, c].plot(b[m, 0], b[m, 2] * 1e9, color=color, lw=0.6, alpha=0.5)
        axes[0, c].set_title(
            f"{name} ({len(devs)} devices)\nblue: 0->Vmin pristine, orange: Vmin->Vmax LRS, green: Vmax->0 post-reset"
        )
        axes[0, c].set_ylabel("|I| [A]")
        axes[1, c].set_title(f"read window |V| <= {vmax} V, linear scale")
        axes[1, c].set_ylabel("I [nA]")
        axes[1, c].set_xlabel("V source [V]")
        for ax in axes[:, c]:
            ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(out_dir, "iv_branches_overview.png")
    fig.savefig(out, dpi=110)
    print("\nplot written:", out)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("files", nargs="+")
    ap.add_argument(
        "--plots", default=None, help="directory for the overview plot (optional)"
    )
    ap.add_argument(
        "--vmax", type=float, default=0.6, help="DAC read window half-width in V"
    )
    args = ap.parse_args()
    all_devs = {os.path.basename(p): analyze_file(p, args.vmax) for p in args.files}
    if args.plots:
        plot(all_devs, args.plots, args.vmax)


if __name__ == "__main__":
    main()
