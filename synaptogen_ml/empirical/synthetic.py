"""Synthetic ``IVDataset`` from the Synaptogen cell model.

Lets the empirical backend be tested against a known generative truth: devices
are drawn from the Synaptogen device-to-device model, every cycle is a full
RESET (``applyVoltage(2.0)``) followed by a full SET (``applyVoltage(-2.0)``),
the pulse sequence ``programming.program_pair`` uses, and the HRS/LRS read
curves are recorded with the non-mutating ``I`` (or noisy ``Iread``).

CLI::

    python -m synaptogen_ml.empirical.synthetic out.npz --devices 512 --cycles 16
"""

__all__ = ["generate_synthetic_dataset", "synaptogen_loops", "DEFAULT_GRID"]

import argparse
import contextlib
from functools import partial
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .. import synaptogen as syn
from .dataset import IVDataset

DEFAULT_GRID = np.linspace(-0.7, 0.7, 141, dtype=np.float32)


@contextlib.contextmanager
def _seeded_synaptogen_rng(seed: Optional[int]):
    """Temporarily bind the Synaptogen module RNG to a seeded generator; the
    previous generator is restored afterwards so callers' RNG streams (and the
    RNG-stream tests) are untouched. ``seed=None`` leaves the RNG alone."""
    if seed is None:
        yield
        return
    saved = (syn.rng, syn.randn, syn.rand)
    rng = np.random.default_rng(seed)
    syn.rng = rng
    syn.randn = partial(rng.standard_normal, dtype=np.float32)
    syn.rand = partial(rng.random, dtype=np.float32)
    try:
        yield
    finally:
        syn.rng, syn.randn, syn.rand = saved


def _fresh_cells(n_devices: int):
    """``CellArrayCPU`` with the reset-threshold buffer detached from ``y``.

    In ``synaptogen.CellArray.__init__`` ``UR`` is a *view* of ``y[UR_INDEX]``,
    so the first RESET (which stores the applied voltage in ``UR``) also
    overwrites the drawn threshold in ``y``; after the following SET re-reads
    ``y[UR_INDEX]`` the cell can only be RESET by a voltage *above* the last
    RESET voltage until a VAR draw replaces ``y``. ``program_pair`` applies a
    single RESET/SET so it is unaffected, but a multi-cycle recording would
    freeze every cell in the LRS from the second cycle on. Detaching the
    buffer here reproduces the intended model (thresholds drawn per cycle)
    without touching ``synaptogen.py``."""
    cells = syn.CellArrayCPU(n_devices)
    cells.UR = cells.UR.copy()
    return cells


def _record(cells, grid, read_noise: bool) -> np.ndarray:
    """``float32[M, P]`` current of every cell at every grid voltage."""
    out = np.empty((len(cells), grid.shape[0]), dtype=np.float32)
    for k, u in enumerate(grid):
        u = float(u)
        if read_noise and u != 0.0:
            out[:, k] = syn.Iread(cells, np.float32(u))
        else:
            out[:, k] = syn.I(cells, np.float32(u))
    return out


def generate_synthetic_dataset(
    n_devices: int,
    n_cycles: int,
    *,
    grid: Sequence[float] = DEFAULT_GRID,
    wear_cycles: int = 0,
    read_noise: bool = False,
    seed: Optional[int] = None,
) -> IVDataset:
    """
    :param n_devices: cells drawn from the Synaptogen device-to-device model.
    :param n_cycles: RESET/SET cycles recorded per device (``cycle_index``
        counts from ``wear_cycles``).
    :param grid: read voltages (must include the read voltage 0.2 V to be
        useful; the default does).
    :param wear_cycles: random ``+-2 V`` pulses applied before recording,
        15 per unit like ``num_cycles_init`` in the programming helpers.
    :param read_noise: record ``Iread`` (Johnson + shot noise) instead of the
        noiseless ``I``.
    :param seed: seeds a private Synaptogen RNG for the call; None uses (and
        advances) the module RNG.
    """
    grid = np.asarray(grid, dtype=np.float32)
    states = ["HRS", "LRS"]
    out = np.empty((n_devices, n_cycles, 2, grid.shape[0]), dtype=np.float32)
    with _seeded_synaptogen_rng(seed):
        cells = _fresh_cells(n_devices)
        wear_rng = np.random.default_rng(seed)
        for _ in range(wear_cycles * 15):
            cells.applyVoltage(np.float32(wear_rng.uniform(-2.0, 2.0)))
        for c in range(n_cycles):
            cells.applyVoltage(np.float32(2.0))
            out[:, c, 0] = _record(cells, grid, read_noise)
            cells.applyVoltage(np.float32(-2.0))
            out[:, c, 1] = _record(cells, grid, read_noise)
    meta = {
        "device_family": "synaptogen",
        "read_polarity": 1,
        "read_voltage": float(syn.Uread),
        "source": "synaptogen_ml.empirical.synthetic.generate_synthetic_dataset",
        "programming": "per cycle: applyVoltage(2.0) -> record HRS; applyVoltage(-2.0) -> record LRS",
        "wear_cycles": int(wear_cycles),
        "read_noise": bool(read_noise),
        "seed": seed,
    }
    return IVDataset(
        U=grid,
        I=out,
        states=states,
        cycle_index=np.arange(wear_cycles, wear_cycles + n_cycles),
        meta=meta,
    )


def synaptogen_loops(
    n_devices: int,
    *,
    vmax: float = 2.0,
    step: float = 0.05,
    seed: Optional[int] = None,
) -> List[List[Tuple[np.ndarray, np.ndarray]]]:
    """One quasi-static switching loop ``0 -> -vmax -> +vmax -> 0`` per fresh
    Synaptogen cell, in the nested ``loops[device][cycle] = (u, i)`` layout of
    ``from_hysteresis_loops`` (one cycle). Exercises the loop adapter with
    known states: ramp 2 is the LRS after the negative SET, ramp 3 the HRS
    after the positive RESET."""
    n = int(round(vmax / step))
    ramp = np.concatenate(
        [
            -np.arange(0, n + 1) * step,
            np.arange(-n + 1, n + 1) * step,
            np.arange(n - 1, -1, -1) * step,
        ]
    ).astype(np.float32)
    with _seeded_synaptogen_rng(seed):
        cells = _fresh_cells(n_devices)
        currents = np.empty((n_devices, ramp.shape[0]), dtype=np.float32)
        for k, u in enumerate(ramp):
            cells.applyVoltage(u)
            currents[:, k] = syn.I(cells, u)
    return [[(ramp.copy(), currents[d])] for d in range(n_devices)]


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("out", help="output .npz path")
    ap.add_argument("--devices", type=int, default=512)
    ap.add_argument("--cycles", type=int, default=16)
    ap.add_argument("--wear-cycles", type=int, default=0)
    ap.add_argument("--read-noise", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--vmax", type=float, default=0.7, help="grid half-width in V")
    ap.add_argument("--points", type=int, default=141, help="grid points")
    args = ap.parse_args(argv)
    grid = np.linspace(-args.vmax, args.vmax, args.points, dtype=np.float32)
    ds = generate_synthetic_dataset(
        args.devices,
        args.cycles,
        grid=grid,
        wear_cycles=args.wear_cycles,
        read_noise=args.read_noise,
        seed=args.seed,
    )
    ds.save(args.out)
    print(ds.summary())
    print("written:", args.out)


if __name__ == "__main__":
    main()
