"""Shared cell-programming helpers for the memristor modules.

The per-pair programming sequence (optional wear cycles -> full reset -> set ->
optional write-verify correction) used to be duplicated inline in
``memristor_modules/linear.py`` and ``memristor_modules/conv.py``. It lives
here so that

* the default serial path stays step-for-step identical to the old inline code
  (same operations in the same order, same RNG draw order -> bit-identical
  results), and
* the opt-in *fast programming* path (``synaptogen_ml.set_fast_programming``)
  can run the same sequence for the many independent (bit-plane, tile) pairs
  of a layer in parallel worker processes.

Fast programming changes which random numbers each cell draws -- every pair is
programmed from an independent, freshly-seeded RNG stream -- but not the
distributions those numbers are drawn from. Statistically the result is
indistinguishable from a serial re-run (and the default programming RNG is
unseeded, so two serial runs never were bit-identical either). See
``benchmarks/check_programming.py`` for the empirical demonstration.

This module must stay importable without torch: worker processes import only
this module (numpy + the Synaptogen cell model).
"""

import os
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from multiprocessing import get_context
from typing import List, Optional, Tuple

import numpy as np

from . import fast_programming_workers
from .synaptogen import CellArray, CellArrayCPU, default_params


@dataclass
class ProgrammedCells:
    """Slim programming result: exactly what
    ``MemristorArray.init_from_cell_array_input_major`` consumes (r + params).

    Returned instead of a full ``CellArray`` when the cell state was produced
    in a worker process (avoids pickling the whole VAR/GMM state) or when
    ideal programming makes the stochastic cell state irrelevant.
    """

    r: np.ndarray
    params: object


# Picklable, torch-free mirror of CycleCorrectionSettings (the real dataclass
# lives in memristor_modules.config, which cannot be imported in the torch-free
# worker processes). Field names match the attributes program_pair reads.
_CorrectionParams = namedtuple(
    "_CorrectionParams",
    ["num_cycles", "test_input_value", "relative_deviation", "ideal_programming"],
)


def _zero_voltage_is_noop(cells: CellArray) -> bool:
    """Whether ``applyVoltage(0.0)`` provably cannot change any cell state or
    consume RNG.

    Set requires ``0 <= US`` on a cell not in LRS, reset requires ``0 > UR`` on
    a cell not in HRS. Both thresholds are drawn from Gaussians, so the "wrong"
    sign is possible in the distribution tails -- hence a runtime check instead
    of an assumption about threshold signs.
    """
    if (~cells.inLRS & (cells.get_US() >= 0)).any():
        return False
    return not (~cells.inHRS & (cells.UR < 0)).any()


def program_pair(
    positive_weights: np.ndarray,
    negative_weights: np.ndarray,
    num_cycles_init: int,
    correction_settings,
    current_scaling: float,
    skip_stochastic_init_for_ideal: bool = False,
):
    """Program one pos/neg cell-array pair. Serial-path behavior is identical
    to the former inline code in linear.py/conv.py, with two bit-exact
    optimizations in the write-verify loop: the constant test-input vector is
    hoisted out of the loop, and rounds stop early once both deviation masks
    are empty and a zero-voltage pulse is provably a no-op (state and RNG
    untouched, so all remaining rounds would be no-ops too).

    :param correction_settings: CycleCorrectionSettings-like object (duck-typed)
        or None.
    :param skip_stochastic_init_for_ideal: under ideal programming the
        stochastic CellArray state is fully overwritten and never read, so the
        GMM/VAR init draws are pure waste; skipping them shifts the RNG stream
        position, which is why this is only done on the opt-in fast path.
    """
    size = positive_weights.shape[0]

    if correction_settings is not None and correction_settings.ideal_programming:
        if skip_stochastic_init_for_ideal:
            return (
                ProgrammedCells(
                    np.ones_like(positive_weights) - positive_weights, default_params
                ),
                ProgrammedCells(
                    np.ones_like(negative_weights) - negative_weights, default_params
                ),
            )
        positive_cells = CellArrayCPU(size)
        negative_cells = CellArrayCPU(size)
        positive_cells.r = np.ones_like(positive_weights) - positive_weights
        negative_cells.r = np.ones_like(negative_weights) - negative_weights
        return positive_cells, negative_cells

    positive_cells = CellArrayCPU(size)
    negative_cells = CellArrayCPU(size)

    for _ in range(num_cycles_init * 15):
        positive_cells.applyVoltage(np.random.uniform(-2.0, 2.0))
        negative_cells.applyVoltage(np.random.uniform(-2.0, 2.0))

    positive_cells.applyVoltage(2.0)
    negative_cells.applyVoltage(2.0)
    positive_cells.applyVoltage(positive_weights * -2.0)
    negative_cells.applyVoltage(negative_weights * -2.0)

    if correction_settings is not None:
        tensor = np.ones_like(positive_weights) * correction_settings.test_input_value
        for _ in range(correction_settings.num_cycles):
            pos = positive_cells.I(tensor) * current_scaling
            neg = negative_cells.I(tensor) * current_scaling
            pos_dev = np.abs(pos - positive_weights)
            neg_dev = np.abs(neg - negative_weights)
            pos_mask = pos_dev > correction_settings.relative_deviation
            neg_mask = neg_dev > correction_settings.relative_deviation
            pos_active = pos_mask.any() or not _zero_voltage_is_noop(positive_cells)
            neg_active = neg_mask.any() or not _zero_voltage_is_noop(negative_cells)
            if not pos_active and not neg_active:
                break
            if pos_active:
                positive_cells.applyVoltage(pos_mask * positive_weights * 2.0)
                positive_cells.applyVoltage(pos_mask * positive_weights * -2.0)
                positive_cells.applyVoltage(pos_mask * (1 - positive_weights) * -2.0)
                positive_cells.applyVoltage(pos_mask * (1 - positive_weights) * 2.0)
            if neg_active:
                negative_cells.applyVoltage(neg_mask * negative_weights * 2.0)
                negative_cells.applyVoltage(neg_mask * negative_weights * -2.0)
                negative_cells.applyVoltage(neg_mask * (1 - negative_weights) * -2.0)
                negative_cells.applyVoltage(neg_mask * (1 - negative_weights) * 2.0)

    return positive_cells, negative_cells


def _worker_program_pair(args):
    """Executed in a worker process: give this task its own RNG streams (both
    the synaptogen Generator and the legacy np.random stream used for the wear
    voltages), then run the standard programming sequence and return only the
    programmed r vectors (the rest of the cell state is never read again)."""
    (
        generator_seed,
        legacy_seed,
        positive_weights,
        negative_weights,
        num_cycles_init,
        correction_params,
        current_scaling,
    ) = args

    from . import synaptogen as _synaptogen

    generator = np.random.default_rng(generator_seed)
    _synaptogen.rng = generator
    _synaptogen.randn = partial(generator.standard_normal, dtype=np.float32)
    _synaptogen.rand = partial(generator.random, dtype=np.float32)
    np.random.seed(legacy_seed)

    positive_cells, negative_cells = program_pair(
        positive_weights,
        negative_weights,
        num_cycles_init,
        correction_params,
        current_scaling,
    )
    return np.asarray(positive_cells.r), np.asarray(negative_cells.r)


_pool: Optional[ProcessPoolExecutor] = None
_pool_size = 0

# each worker gets a single-threaded BLAS: the per-cell numpy work is
# ufunc-bound (no benefit from threaded BLAS), and N workers x threaded BLAS
# oversubscribes the node and can erase the whole parallel speedup
_BLAS_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _noop(_):
    return None


def _get_pool(workers: int) -> ProcessPoolExecutor:
    global _pool, _pool_size
    if _pool is None or _pool_size != workers:
        if _pool is not None:
            _pool.shutdown(wait=False)
        # spawn (not fork): workers only import this torch-free module, and
        # spawning avoids forking a parent that may hold torch thread pools.
        # The single-thread BLAS env is set only while the workers spawn (they
        # inherit it); the parent's env is restored afterwards.
        saved = {var: os.environ.get(var) for var in _BLAS_ENV_VARS}
        for var in _BLAS_ENV_VARS:
            os.environ[var] = "1"
        try:
            _pool = ProcessPoolExecutor(
                max_workers=workers, mp_context=get_context("spawn")
            )
            # force all workers to spawn now, while the env is in place
            list(_pool.map(_noop, range(workers)))
        finally:
            for var, value in saved.items():
                if value is None:
                    del os.environ[var]
                else:
                    os.environ[var] = value
        _pool_size = workers
    return _pool


def program_pairs(
    jobs: List[Tuple[np.ndarray, np.ndarray]],
    num_cycles_init: int,
    correction_settings,
    current_scaling: float,
):
    """Program a list of independent pos/neg weight pairs, in job order.

    With fast programming disabled (default) this is a plain serial loop over
    ``program_pair`` -- bit-identical to the historical inline code. With
    ``synaptogen_ml.set_fast_programming(workers)`` (or SYN_FAST_PROG=N) the
    pairs are programmed concurrently in a persistent process pool, each from
    its own freshly-seeded RNG streams; results are slim ``ProgrammedCells``.
    """
    workers = fast_programming_workers()

    if correction_settings is not None and correction_settings.ideal_programming:
        skip_init = workers > 0
        return [
            program_pair(
                positive_weights,
                negative_weights,
                num_cycles_init,
                correction_settings,
                current_scaling,
                skip_stochastic_init_for_ideal=skip_init,
            )
            for positive_weights, negative_weights in jobs
        ]

    if workers <= 0 or len(jobs) <= 1:
        return [
            program_pair(
                positive_weights,
                negative_weights,
                num_cycles_init,
                correction_settings,
                current_scaling,
            )
            for positive_weights, negative_weights in jobs
        ]

    correction_params = (
        None
        if correction_settings is None
        else _CorrectionParams(
            num_cycles=correction_settings.num_cycles,
            test_input_value=correction_settings.test_input_value,
            relative_deviation=correction_settings.relative_deviation,
            ideal_programming=correction_settings.ideal_programming,
        )
    )

    # fresh OS entropy per call; two independent child sequences per job (one
    # for the synaptogen Generator, one for the legacy wear-voltage stream)
    children = np.random.SeedSequence().spawn(len(jobs) * 2)
    task_args = [
        (
            children[2 * idx],
            int(children[2 * idx + 1].generate_state(1)[0]),
            positive_weights,
            negative_weights,
            num_cycles_init,
            correction_params,
            current_scaling,
        )
        for idx, (positive_weights, negative_weights) in enumerate(jobs)
    ]
    pool = _get_pool(workers)
    return [
        (
            ProgrammedCells(positive_r, default_params),
            ProgrammedCells(negative_r, default_params),
        )
        for positive_r, negative_r in pool.map(_worker_program_pair, task_args)
    ]
