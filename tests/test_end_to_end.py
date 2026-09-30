"""
End-to-end smoke test of the two-phase workflow on a synthetic task: train a small
QAT model, program its weights onto memristor layers, and check that memristor
inference keeps most of the quantized model's accuracy. Replaces the former MNIST
integration tests; runs in seconds and needs no download.
"""

from functools import partial

import numpy as np
import pytest
import torch
from torch import nn

import synaptogen_ml.synaptogen as syn
from synaptogen_ml.memristor_modules.linear import TiledMemristorLinear
from synaptogen_ml.memristor_modules.memristor import DacAdcHardwareSettings
from synaptogen_ml.quant_modules import ActivationQuantizer, LinearQuant

INPUT_DIM, HIDDEN_DIM, NUM_CLASSES = 32, 64, 5
NUM_TRAIN, NUM_TEST = 2000, 1000


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


class Model(nn.Module):
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

        hw = DacAdcHardwareSettings(
            input_bits=8,
            output_precision_bits=2,
            output_range_bits=6,
            hardware_input_vmax=0.6,
            hardware_output_current_scaling=8020.0,
        )
        self.memristor_1 = TiledMemristorLinear(
            in_features=INPUT_DIM,
            out_features=HIDDEN_DIM,
            weight_precision=3,
            converter_hardware_settings=hw,
            memristor_inputs=32,
            memristor_outputs=32,
        )
        self.memristor_2 = TiledMemristorLinear(
            in_features=HIDDEN_DIM,
            out_features=NUM_CLASSES,
            weight_precision=3,
            converter_hardware_settings=hw,
            memristor_inputs=32,
            memristor_outputs=32,
        )

    def forward(self, x, use_memristor=False):
        h = self.memristor_1(x) if use_memristor else self.linear_1(self.act_1_in(x))
        h = torch.tanh(self.act_1_out(h))
        o = self.memristor_2(h) if use_memristor else self.linear_2(self.act_2_in(h))
        return self.act_2_out(o)

    def prepare_memristor(self, num_cycles_init: int):
        self.memristor_1.init_from_linear_quant(
            self.act_1_in,
            self.linear_1,
            num_cycles_init=num_cycles_init,
            correction_settings=None,
        )
        self.memristor_2.init_from_linear_quant(
            self.act_2_in,
            self.linear_2,
            num_cycles_init=num_cycles_init,
            correction_settings=None,
        )


def make_data():
    """Gaussian blobs with overlap: separable, but not trivially so."""
    centers = torch.randn(NUM_CLASSES, INPUT_DIM)
    labels = torch.randint(0, NUM_CLASSES, (NUM_TRAIN + NUM_TEST,))
    inputs = centers[labels] + 1.5 * torch.randn(NUM_TRAIN + NUM_TEST, INPUT_DIM)
    return (
        inputs[:NUM_TRAIN],
        labels[:NUM_TRAIN],
        inputs[NUM_TRAIN:],
        labels[NUM_TRAIN:],
    )


def accuracy(model, x, y, use_memristor):
    model.eval()
    with torch.no_grad():
        return (
            (model(x, use_memristor=use_memristor).argmax(-1) == y)
            .float()
            .mean()
            .item()
        )


@pytest.mark.end_to_end
def test_train_program_infer():
    seed_all(0)
    x_train, y_train, x_test, y_test = make_data()
    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    model.train()
    for _ in range(300):
        idx = torch.randint(0, NUM_TRAIN, (64,))
        loss = nn.functional.cross_entropy(model(x_train[idx]), y_train[idx])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    quant_acc = accuracy(model, x_test, y_test, use_memristor=False)
    assert quant_acc > 0.7, f"QAT model did not train: acc {quant_acc:.3f}"

    model.prepare_memristor(num_cycles_init=1)
    memristor_acc = accuracy(model, x_test, y_test, use_memristor=True)
    assert memristor_acc > quant_acc - 0.1, (
        f"memristor inference lost too much accuracy: {memristor_acc:.3f} "
        f"vs quantized {quant_acc:.3f}"
    )
