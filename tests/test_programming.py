"""
Tests for ``synaptogen_ml.programming``: the shared serial programming helper and
the opt-in parallel programming path.
"""

import copy
from functools import partial

import numpy as np
import pytest
import torch

import synaptogen_ml
import synaptogen_ml.synaptogen as syn
from synaptogen_ml.memristor_modules.config import CycleCorrectionSettings
from synaptogen_ml.memristor_modules.linear import TiledMemristorLinear
from synaptogen_ml.memristor_modules.memristor import DacAdcHardwareSettings
from synaptogen_ml.programming import (
    ProgrammedCells,
    _zero_voltage_is_noop,
    program_pair,
    program_pairs,
)
from synaptogen_ml.quant_modules import ActivationQuantizer, LinearQuant

WEIGHT_PRECISION = 3
WORKERS = 2


def seed_all(seed: int) -> None:
    """Reseed torch, the legacy numpy stream (wear voltages) and the synaptogen
    cell generator (device variation, VAR draws)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    syn.rng = np.random.default_rng(seed)
    syn.randn = partial(syn.rng.standard_normal, dtype=np.float32)
    syn.rand = partial(syn.rng.random, dtype=np.float32)


def _ternary_weights(size: int, seed: int):
    gen = np.random.default_rng(seed)
    w = gen.choice([-1, 0, 1], size=size)
    return (w > 0).astype(np.float32), (w < 0).astype(np.float32)


def _build_tiled_linear():
    lin = LinearQuant(
        in_features=64,
        out_features=64,
        weight_bit_prec=WEIGHT_PRECISION,
        weight_quant_dtype=torch.qint8,
        weight_quant_method="per_tensor_symmetric",
        bias=False,
    )
    act = ActivationQuantizer(
        bit_precision=8,
        dtype=torch.qint8,
        method="per_tensor_symmetric",
        channel_axis=None,
        moving_avrg=None,
    )
    x = torch.randn(8, 64)
    lin.train()
    act.train()
    lin(act(x))
    lin.eval()
    act.eval()

    mem = TiledMemristorLinear(
        in_features=64,
        out_features=64,
        weight_precision=WEIGHT_PRECISION,
        converter_hardware_settings=DacAdcHardwareSettings(
            input_bits=8,
            output_precision_bits=2,
            output_range_bits=6,
            hardware_input_vmax=0.6,
            hardware_output_current_scaling=8020.0,
        ),
        memristor_inputs=32,
        memristor_outputs=32,
        bias=False,
    )
    return mem, act, lin, x


def _r_state(module):
    return torch.cat(
        [p.detach().flatten() for n, p in module.named_parameters() if n.endswith(".r")]
    )


@pytest.mark.programming
def test_zero_voltage_noop_leaves_state_and_rng_untouched():
    """When ``_zero_voltage_is_noop`` says a zero pulse cannot act, applying it
    must change neither the cell state nor the generator state. This is what
    makes the early exit in the write-verify loop bit-exact."""
    seed_all(0)
    positive, negative = _ternary_weights(512, seed=1)
    cells, _ = program_pair(positive, negative, 1, None, 1.0)
    assert _zero_voltage_is_noop(cells)

    r_before = cells.r.copy()
    rng_before = copy.deepcopy(syn.rng.bit_generator.state)
    legacy_before = np.random.get_state()[1].copy()

    cells.applyVoltage(np.zeros_like(positive))

    assert np.array_equal(cells.r, r_before)
    assert syn.rng.bit_generator.state == rng_before
    assert np.array_equal(np.random.get_state()[1], legacy_before)


@pytest.mark.programming
def test_serial_program_pairs_is_deterministic_under_seed():
    correction = CycleCorrectionSettings(
        num_cycles=2,
        test_input_value=0.2,
        relative_deviation=0.2,
        ideal_programming=False,
    )
    jobs = [_ternary_weights(256, seed=s) for s in range(3)]
    synaptogen_ml.set_fast_programming(0)

    seed_all(0)
    first = program_pairs(jobs, 1, correction, 8020.0)
    seed_all(0)
    second = program_pairs(jobs, 1, correction, 8020.0)

    for (pos_a, neg_a), (pos_b, neg_b) in zip(first, second):
        assert np.array_equal(pos_a.r, pos_b.r)
        assert np.array_equal(neg_a.r, neg_b.r)
        assert np.all((0.0 <= pos_a.r) & (pos_a.r <= 1.0))


@pytest.mark.programming
def test_parallel_program_pairs_matches_serial_in_distribution():
    """The parallel path draws different random numbers but from the same
    distributions: per target-weight group, its mean r must sit within the
    spread seen between two serial runs (plus a small margin)."""
    jobs = [_ternary_weights(1024, seed=s) for s in range(4)]
    targets = np.concatenate([p for p, _ in jobs] + [n for _, n in jobs])

    def gather(results):
        return np.concatenate(
            [np.asarray(p.r) for p, _ in results]
            + [np.asarray(n.r) for _, n in results]
        )

    synaptogen_ml.set_fast_programming(0)
    seed_all(0)
    serial_a = gather(program_pairs(jobs, 1, None, 8020.0))
    seed_all(1)
    serial_b = gather(program_pairs(jobs, 1, None, 8020.0))

    synaptogen_ml.set_fast_programming(WORKERS)
    try:
        results = program_pairs(jobs, 1, None, 8020.0)
    finally:
        synaptogen_ml.set_fast_programming(0)
    assert all(isinstance(p, ProgrammedCells) for p, _ in results)
    parallel = gather(results)

    assert not np.array_equal(serial_a, parallel)
    for group in (0.0, 1.0):
        mask = targets == group
        baseline = abs(serial_a[mask].mean() - serial_b[mask].mean())
        deviation = abs(serial_a[mask].mean() - parallel[mask].mean())
        assert deviation <= 3 * baseline + 0.01, (group, baseline, deviation)


@pytest.mark.programming
def test_parallel_ideal_programming_equals_serial():
    ideal = CycleCorrectionSettings(
        num_cycles=None,
        test_input_value=None,
        relative_deviation=None,
        ideal_programming=True,
    )
    seed_all(0)
    mem_serial, act, lin, x = _build_tiled_linear()
    mem_parallel = copy.deepcopy(mem_serial)

    synaptogen_ml.set_fast_programming(0)
    mem_serial.init_from_linear_quant(
        act, lin, num_cycles_init=0, correction_settings=ideal
    )
    synaptogen_ml.set_fast_programming(WORKERS)
    try:
        mem_parallel.init_from_linear_quant(
            act, lin, num_cycles_init=0, correction_settings=ideal
        )
    finally:
        synaptogen_ml.set_fast_programming(0)

    assert torch.equal(_r_state(mem_serial), _r_state(mem_parallel))


@pytest.mark.programming
def test_parallel_tiled_linear_smoke():
    """End to end: a tiled linear layer programmed in worker processes must
    produce a valid device state and a forward output close to the serial one."""
    seed_all(0)
    mem_serial, act, lin, x = _build_tiled_linear()
    mem_parallel = copy.deepcopy(mem_serial)
    correction = CycleCorrectionSettings(
        num_cycles=2,
        test_input_value=0.2,
        relative_deviation=0.2,
        ideal_programming=False,
    )

    synaptogen_ml.set_fast_programming(0)
    mem_serial.init_from_linear_quant(
        act, lin, num_cycles_init=1, correction_settings=correction
    )
    synaptogen_ml.set_fast_programming(WORKERS)
    try:
        mem_parallel.init_from_linear_quant(
            act, lin, num_cycles_init=1, correction_settings=correction
        )
    finally:
        synaptogen_ml.set_fast_programming(0)

    r = _r_state(mem_parallel)
    assert torch.all((0.0 <= r) & (r <= 1.0))

    mem_serial.eval()
    mem_parallel.eval()
    with torch.no_grad():
        torch.manual_seed(1234)
        y_serial = mem_serial(x)
        torch.manual_seed(1234)
        y_parallel = mem_parallel(x)
    rel = ((y_parallel - y_serial).norm() / y_serial.norm()).item()
    assert rel < 0.5, rel
