"""End-to-end (torch-module level) comparison of serial vs parallel programming.

For each workload the SAME calibrated quant layer is programmed onto three
independent memristor modules:

* serial A  -- default path (``set_fast_programming(0)``), cell RNG seed 0
* serial B  -- default path again with a different seed: the natural
               "same distribution, different draws" baseline
* parallel  -- ``set_fast_programming(N)`` worker processes

We then compare (1) init wall time, (2) the programmed r-state distributions
(two-sample KS, serial-vs-parallel judged against the serial-vs-serial
baseline), and (3) the module OUTPUTS on identical inputs with identical
readout-noise seeds, both against each other (MSE) and against the QAT
reference output (relative error -- programming fidelity).

Run on a GPU node (forwards run on cuda when available):
    sbatch -o <shared-path>/prog_torch_%j.out benchmarks/run_programming_torch.sbatch
"""

import time

import numpy as np
import torch

import synaptogen_ml
from benchmarks._builders import (
    WEIGHT_PRECISION,
    _act_quant,
    _calibrate,
    _hw,
    seed_all,
)
from benchmarks.check_programming import ks_2samp
from synaptogen_ml.memristor_modules.config import CycleCorrectionSettings
from synaptogen_ml.memristor_modules.conv import MemristorConv1d
from synaptogen_ml.memristor_modules.linear import TiledMemristorLinear
from synaptogen_ml.quant_modules import Conv1DQuant, LinearQuant

FORWARD_SEED = 1234
WORKERS = 8


def gather_r(module):
    return np.concatenate(
        [
            p.detach().cpu().numpy().ravel()
            for name, p in module.named_parameters()
            if name.endswith(".r")
        ]
    )


def timed_init(build_mem, init_fn, workers, seed):
    seed_all(seed)
    mem = build_mem()
    synaptogen_ml.set_fast_programming(workers)
    start = time.perf_counter()
    init_fn(mem)
    elapsed = time.perf_counter() - start
    synaptogen_ml.set_fast_programming(0)
    return mem, elapsed


def forward_out(mem, x, device):
    mem.to(device).eval()
    torch.manual_seed(FORWARD_SEED)
    with torch.no_grad():
        y = mem(x.to(device)).detach().cpu()
    mem.cpu()
    return y


def run_case(label, build_mem, init_fn, x, reference, device):
    mem_a, t_a = timed_init(build_mem, init_fn, 0, seed=0)
    mem_b, t_b = timed_init(build_mem, init_fn, 0, seed=1)
    mem_p, t_p_cold = timed_init(build_mem, init_fn, WORKERS, seed=2)
    mem_p2, t_p = timed_init(build_mem, init_fn, WORKERS, seed=3)

    r_a, r_b, r_p = gather_r(mem_a), gather_r(mem_b), gather_r(mem_p)
    d_bb, p_bb = ks_2samp(r_a, r_b)
    d_pp, p_pp = ks_2samp(r_a, r_p)

    y_a = forward_out(mem_a, x, device)
    y_b = forward_out(mem_b, x, device)
    y_p = forward_out(mem_p, x, device)
    del mem_p2

    ref_norm = float(reference.pow(2).mean().sqrt())
    mse_ab = float((y_a - y_b).pow(2).mean())
    mse_ap = float((y_a - y_p).pow(2).mean())
    rel = [
        float((y - reference).pow(2).mean().sqrt()) / ref_norm for y in (y_a, y_b, y_p)
    ]

    ks_ok = p_pp > 0.01 or d_pp <= 3 * max(d_bb, 1e-12)
    out_ok = mse_ap <= 3 * max(mse_ab, 1e-12)
    fid_ok = abs(rel[2] - rel[0]) <= 3 * max(abs(rel[1] - rel[0]), 1e-12) + 1e-3
    verdict = "PASS" if (ks_ok and out_ok and fid_ok) else "FAIL"

    print(
        f"| {label} | {t_a:.2f} / {t_b:.2f} | {t_p:.2f} (cold {t_p_cold:.2f}) "
        f"| x{t_a / t_p:.1f} "
        f"| {d_bb:.4f} (p={p_bb:.2f}) | {d_pp:.4f} (p={p_pp:.2f}) "
        f"| {mse_ab:.4g} | {mse_ap:.4g} "
        f"| {rel[0]:.4f} / {rel[1]:.4f} / {rel[2]:.4f} | {verdict} |"
    )
    return verdict == "PASS"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device for forwards: {device}, workers: {WORKERS}, "
          f"weight_precision: {WEIGHT_PRECISION}")
    print(
        "\n| workload | init serial A/B [s] | init parallel [s] | speedup "
        "| KS r A-vs-B (baseline) | KS r A-vs-parallel "
        "| out MSE A-B | out MSE A-P "
        "| rel.err vs QAT A / B / P | verdict |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")

    ok = True

    # --- tiled linear 512x512, 128x128 tiles, wear=6 (cluster-realistic) -----
    seed_all(0)
    lin = LinearQuant(
        in_features=512,
        out_features=512,
        weight_bit_prec=WEIGHT_PRECISION,
        weight_quant_dtype=torch.qint8,
        weight_quant_method="per_tensor_symmetric",
        bias=False,
    )
    act = _act_quant()
    x_lin = torch.randn(64, 512)
    _calibrate(lin, act, x_lin)
    lin.eval()
    with torch.no_grad():
        ref_lin = lin(act(x_lin)).detach().cpu()

    def build_tiled():
        return TiledMemristorLinear(
            in_features=512,
            out_features=512,
            weight_precision=WEIGHT_PRECISION,
            converter_hardware_settings=_hw(),
            memristor_inputs=128,
            memristor_outputs=128,
            bias=False,
        )

    ok &= run_case(
        "tiled_linear 512x512 t128 wear=6",
        build_tiled,
        lambda m: m.init_from_linear_quant(
            act, lin, num_cycles_init=6, correction_settings=None
        ),
        x_lin,
        ref_lin,
        device,
    )

    ok &= run_case(
        "tiled_linear 512x512 t128 wear=6 corr=3",
        build_tiled,
        lambda m: m.init_from_linear_quant(
            act,
            lin,
            num_cycles_init=6,
            correction_settings=CycleCorrectionSettings(
                num_cycles=3,
                test_input_value=0.2,
                relative_deviation=0.2,
                ideal_programming=False,
            ),
        ),
        x_lin,
        ref_lin,
        device,
    )

    # --- depthwise conv1d 256ch k9, wear=6 -----------------------------------
    seed_all(0)
    conv = Conv1DQuant(
        in_channels=256,
        out_channels=256,
        kernel_size=9,
        weight_bit_prec=WEIGHT_PRECISION,
        weight_quant_dtype=torch.qint8,
        weight_quant_method="per_tensor_symmetric",
        bias=False,
        stride=1,
        padding="same",
        dilation=1,
        groups=256,
    )
    act_c = _act_quant()
    x_conv = torch.randn(8, 256, 128)
    _calibrate(conv, act_c, x_conv)
    conv.eval()
    with torch.no_grad():
        ref_conv = conv(act_c(x_conv)).detach().cpu()

    def build_conv():
        return MemristorConv1d(
            in_channels=256,
            out_channels=256,
            kernel_size=9,
            stride=1,
            padding="same",
            groups=256,
            weight_precision=WEIGHT_PRECISION,
            converter_hardware_settings=_hw(),
            bias=False,
        )

    ok &= run_case(
        "conv1d dw256 k9 wear=6",
        build_conv,
        lambda m: m.init_from_conv_quant(
            act_c, conv, num_cycles_init=6, correction_settings=None
        ),
        x_conv,
        ref_conv,
        device,
    )

    print(
        "\nOVERALL: "
        + (
            "PASS -- parallel programming matches serial in distribution and "
            "output fidelity; only the individual draws differ"
            if ok
            else "FAIL -- see table"
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
