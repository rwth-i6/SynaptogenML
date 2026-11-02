"""First accuracy numbers with the partner PCMO devices.

Trains the small QAT classifier from ``tests/test_end_to_end.py`` (Gaussian
blobs, two 3-bit linear layers) once, then programs it onto crossbars with
different device backends and reports the memristor-inference accuracy:

* Synaptogen (the built-in generative model, amplitude encoding, 0.6 V DAC)
* W/WOx/PCMO and Ta/WOx/PCMO measured devices (``EmpiricalBackend``,
  pulse-count encoding at -0.2 V, states LRS / pristine), with the readout
  noise off and with physical Johnson+shot noise at two bandwidths

Each configuration is programmed with several seeds; mean and spread are
reported. Read the disclaimers at the end of the report before quoting the
numbers.

Usage:
  python tools/pcmo_first_numbers.py [--seeds 5] [--out data/measurements/first_numbers.md]
"""

import argparse
import os
import sys
from functools import partial

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import synaptogen_ml  # noqa: E402
import synaptogen_ml.synaptogen as syn  # noqa: E402
from synaptogen_ml.empirical import EmpiricalBackend  # noqa: E402
from synaptogen_ml.empirical.adapters import from_keithley_iv_dat  # noqa: E402
from synaptogen_ml.memristor_modules.linear import TiledMemristorLinear  # noqa: E402
from synaptogen_ml.memristor_modules.memristor import DacAdcHardwareSettings  # noqa: E402
from synaptogen_ml.quant_modules import ActivationQuantizer, LinearQuant  # noqa: E402

INPUT_DIM, HIDDEN_DIM, NUM_CLASSES = 32, 64, 5
NUM_TRAIN, NUM_TEST = 2000, 1000
DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "measurements"
)
HW_SYNAPTOGEN = DacAdcHardwareSettings(
    input_bits=8,
    output_precision_bits=2,
    output_range_bits=6,
    hardware_input_vmax=0.6,
    hardware_output_current_scaling=8020.0,
)


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    syn.rng = np.random.default_rng(seed)
    syn.randn = partial(syn.rng.standard_normal, dtype=np.float32)
    syn.rand = partial(syn.rng.random, dtype=np.float32)


def act_quant() -> ActivationQuantizer:
    return ActivationQuantizer(
        bit_precision=8,
        dtype=torch.qint8,
        method="per_tensor_symmetric",
        channel_axis=None,
        moving_avrg=None,
        reduce_range=False,
    )


class QatModel(nn.Module):
    def __init__(self):
        super().__init__()
        quant_kw = dict(
            weight_bit_prec=3,
            weight_quant_dtype=torch.qint8,
            weight_quant_method="per_tensor_symmetric",
            bias=False,
        )
        self.linear_1 = LinearQuant(INPUT_DIM, HIDDEN_DIM, **quant_kw)
        self.linear_2 = LinearQuant(HIDDEN_DIM, NUM_CLASSES, **quant_kw)
        self.act_1_in, self.act_1_out = act_quant(), act_quant()
        self.act_2_in, self.act_2_out = act_quant(), act_quant()

    def forward(self, x):
        h = torch.tanh(self.act_1_out(self.linear_1(self.act_1_in(x))))
        return self.act_2_out(self.linear_2(self.act_2_in(h)))


class MemristorModel(nn.Module):
    """The trained QAT model programmed onto crossbars of one backend."""

    def __init__(
        self, qat: QatModel, backend, hw: DacAdcHardwareSettings, num_cycles_init: int
    ):
        super().__init__()
        self.qat = qat
        self.memristor_1 = TiledMemristorLinear(
            INPUT_DIM,
            HIDDEN_DIM,
            3,
            hw,
            memristor_inputs=32,
            memristor_outputs=32,
            backend=backend,
        )
        self.memristor_2 = TiledMemristorLinear(
            HIDDEN_DIM,
            NUM_CLASSES,
            3,
            hw,
            memristor_inputs=32,
            memristor_outputs=32,
            backend=backend,
        )
        self.memristor_1.init_from_linear_quant(
            qat.act_1_in,
            qat.linear_1,
            num_cycles_init=num_cycles_init,
            correction_settings=None,
        )
        self.memristor_2.init_from_linear_quant(
            qat.act_2_in,
            qat.linear_2,
            num_cycles_init=num_cycles_init,
            correction_settings=None,
        )

    def forward(self, x):
        h = torch.tanh(self.qat.act_1_out(self.memristor_1(x)))
        return self.qat.act_2_out(self.memristor_2(h))


def make_data():
    centers = torch.randn(NUM_CLASSES, INPUT_DIM)
    labels = torch.randint(0, NUM_CLASSES, (NUM_TRAIN + NUM_TEST,))
    inputs = centers[labels] + 1.5 * torch.randn(NUM_TRAIN + NUM_TEST, INPUT_DIM)
    return (
        inputs[:NUM_TRAIN],
        labels[:NUM_TRAIN],
        inputs[NUM_TRAIN:],
        labels[NUM_TRAIN:],
    )


def accuracy(model, x, y) -> float:
    model.eval()
    with torch.no_grad():
        return (model(x).argmax(-1) == y).float().mean().item()


def train(seed: int):
    seed_all(seed)
    x_train, y_train, x_test, y_test = make_data()
    model = QatModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    model.train()
    for _ in range(300):
        idx = torch.randint(0, NUM_TRAIN, (64,))
        loss = nn.functional.cross_entropy(model(x_train[idx]), y_train[idx])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return model, (x_test, y_test)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--seeds", type=int, default=5, help="programming seeds per configuration"
    )
    ap.add_argument("--train-seed", type=int, default=0)
    ap.add_argument(
        "--out", default=None, help="write the markdown report here as well"
    )
    args = ap.parse_args(argv)

    qat, (x_test, y_test) = train(args.train_seed)
    quant_acc = accuracy(qat, x_test, y_test)
    lines = [
        "# First PCMO numbers (small blob classifier, two 3-bit linear layers)",
        "",
        f"Quantized (QAT, no memristors) test accuracy: **{quant_acc:.3f}**. "
        f"{args.seeds} programming seeds per row; mean +- std over seeds.",
        "",
        "| backend | encoding | readout noise | accuracy | drop vs QAT |",
        "|---|---|---|---|---|",
    ]

    datasets = {
        "W/WOx/PCMO": from_keithley_iv_dat(
            os.path.join(DATA, "W_WOx_PCMO.dat"), device_family="W/WOx/PCMO"
        ),
        "Ta/WOx/PCMO": from_keithley_iv_dat(
            os.path.join(DATA, "Ta_WOx_PCMO.dat"), device_family="Ta/WOx/PCMO"
        ),
    }
    configs = [
        ("Synaptogen", None, "amplitude 0.6 V", "legacy"),
        ("Synaptogen", None, "amplitude 0.6 V", "off"),
    ]
    for family in datasets:
        for noise, label in (
            ("off", "off"),
            ("physical", "physical, BW 1e8 Hz"),
            ("physical", "physical, BW 1e6 Hz"),
        ):
            configs.append(
                (family, label, "pulse count at -0.2 V (LRS / pristine)", noise)
            )

    descriptions = {}
    for family, noise_label, encoding, noise_model in configs:
        accs = []
        for seed in range(args.seeds):
            synaptogen_ml.set_readout_noise_model(noise_model)
            if family == "Synaptogen":
                seed_all(1000 + seed)
                backend, hw, label = None, HW_SYNAPTOGEN, noise_model
            else:
                bandwidth = 1e6 if "1e6" in noise_label else 1e8
                backend = EmpiricalBackend(
                    datasets[family],
                    read_voltage=-0.2,
                    state_for_weight={1: "LRS", 0: "pristine"},
                    seed=1000 + seed,
                    readout_bandwidth=bandwidth,
                )
                hw = backend.suggested_hardware_settings(8, 2, 6)
                label = noise_label if noise_model != "off" else "off"
                descriptions[family] = backend.describe()
                torch.manual_seed(1000 + seed)
            model = MemristorModel(qat, backend, hw, num_cycles_init=0)
            accs.append(accuracy(model, x_test, y_test))
        synaptogen_ml.set_readout_noise_model("legacy")
        mean, std = float(np.mean(accs)), float(np.std(accs))
        lines.append(
            f"| {family} | {encoding} | {label} | {mean:.3f} +- {std:.3f} | {quant_acc - mean:+.3f} |"
        )
        print(lines[-1], flush=True)

    lines += ["", "## Device pools", ""]
    for family, info in descriptions.items():
        lines.append(
            f"- **{family}**: {info['devices_with_on_state']} devices, {info['cycles']} cycle each; "
            f"median I_on {info['median_on_current']:.3g} A, I_off {info['median_off_current']:.3g} A "
            f"at {info['read_voltage']} V (on/off {info['on_off_ratio']:.1f}); device-to-device spread of "
            f"I_on {info['on_spread_dex']:.2f} dex; unit-weight conductance {info['delta_conductance']:.3g} S."
        )
    lines += [
        "",
        "Disclaimers: first-loop data only, pristine branch as the off state, no "
        "cycle-to-cycle data, 76 / 12 devices reused across all cells, no retention "
        "or read disturb, instrument floor not modelled.",
    ]
    report = "\n".join(lines)
    print("\n" + report)
    if args.out:
        with open(args.out, "w") as f:
            f.write(report + "\n")
        print("\nwritten:", args.out)


if __name__ == "__main__":
    main()
