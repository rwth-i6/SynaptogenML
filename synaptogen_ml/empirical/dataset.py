__all__ = ["IVDataset", "from_state_sweeps", "from_hysteresis_loops", "segment_loop"]

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# metadata every dataset must carry; the remaining keys are free-form
REQUIRED_META = ("device_family", "read_polarity")
DEFAULT_META = {"current_unit": "A", "voltage_unit": "V"}


def _interp_nan(u: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Linear interpolation of one curve onto ``grid``; NaN outside the
    finite support of the curve and for curves with fewer than two finite
    points."""
    finite = np.isfinite(y) & np.isfinite(u)
    if finite.sum() < 2:
        return np.full(grid.shape, np.nan, dtype=np.float32)
    order = np.argsort(u[finite], kind="stable")
    uu, yy = u[finite][order], y[finite][order]
    return np.interp(grid, uu, yy, left=np.nan, right=np.nan).astype(np.float32)


@dataclass
class IVDataset:
    """Current-voltage curves of ``D`` devices, ``C`` cycles and ``S`` states
    on a common grid of ``P`` voltages.

    :param U: ``float32[P]`` strictly increasing voltage grid (volt).
    :param I: ``float32[D, C, S, P]`` current (ampere); NaN marks a missing
        sample (not measured, outside the sweep of that device, ...).
    :param states: names of the ``S`` programmed states, e.g.
        ``["HRS", "LRS"]`` or ``["pristine", "LRS", "post_reset_pos"]``.
    :param cycle_index: ``int64[C]`` switching-cycle number of each cycle axis
        entry (0 = first/pristine loop).
    :param meta: JSON-serialisable metadata. Required: ``device_family`` (one
        dataset never mixes device families) and ``read_polarity`` (+1 or -1:
        sign of the voltage the cells are read at). Conventional optional keys:
        ``read_voltage``, ``device_ids``, ``source``, ``temperature``,
        ``programming``, ``notes``.
    """

    U: np.ndarray
    I: np.ndarray
    states: List[str]
    cycle_index: np.ndarray
    meta: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.U = np.asarray(self.U, dtype=np.float32)
        self.I = np.asarray(self.I, dtype=np.float32)
        self.states = [str(s) for s in self.states]
        self.cycle_index = np.asarray(self.cycle_index, dtype=np.int64)
        self.meta = {**DEFAULT_META, **dict(self.meta)}
        self.validate()

    # ------------------------------------------------------------ basics ---
    @property
    def n_devices(self) -> int:
        return self.I.shape[0]

    @property
    def n_cycles(self) -> int:
        return self.I.shape[1]

    @property
    def n_states(self) -> int:
        return self.I.shape[2]

    @property
    def n_points(self) -> int:
        return self.I.shape[3]

    @property
    def device_family(self) -> str:
        return self.meta["device_family"]

    @property
    def read_polarity(self) -> int:
        return int(self.meta["read_polarity"])

    def state_index(self, state: Union[str, int]) -> int:
        if isinstance(state, (int, np.integer)):
            if not 0 <= state < self.n_states:
                raise IndexError(f"state index {state} out of range")
            return int(state)
        try:
            return self.states.index(state)
        except ValueError:
            raise KeyError(
                f"unknown state {state!r}; dataset has {self.states}"
            ) from None

    def validate(self) -> None:
        if self.U.ndim != 1 or self.U.shape[0] < 2:
            raise ValueError("U must be a 1-d grid with at least two points")
        if not np.all(np.isfinite(self.U)):
            raise ValueError("U contains non-finite values")
        if not np.all(np.diff(self.U) > 0):
            raise ValueError("U must be strictly increasing")
        if self.I.ndim != 4:
            raise ValueError(
                f"I must have shape [devices, cycles, states, points], got {self.I.shape}"
            )
        if self.I.shape[3] != self.U.shape[0]:
            raise ValueError(
                f"I has {self.I.shape[3]} points but U has {self.U.shape[0]}"
            )
        if self.I.shape[2] != len(self.states):
            raise ValueError(
                f"I has {self.I.shape[2]} states but {len(self.states)} names given"
            )
        if len(set(self.states)) != len(self.states):
            raise ValueError(f"duplicate state names: {self.states}")
        if self.cycle_index.ndim != 1 or self.cycle_index.shape[0] != self.I.shape[1]:
            raise ValueError("cycle_index must have one entry per cycle axis entry")
        if np.any(np.diff(self.cycle_index) <= 0):
            raise ValueError("cycle_index must be strictly increasing")
        missing = [k for k in REQUIRED_META if k not in self.meta]
        if missing:
            raise ValueError(f"meta is missing required keys {missing}")
        if int(self.meta["read_polarity"]) not in (-1, 1):
            raise ValueError("meta['read_polarity'] must be +1 or -1")
        if "device_ids" in self.meta and len(self.meta["device_ids"]) != self.n_devices:
            raise ValueError("meta['device_ids'] must have one entry per device")
        if np.isinf(self.I).any():
            raise ValueError("I contains infinite values (use NaN for missing)")
        json.dumps(self.meta)  # must be serialisable

    # --------------------------------------------------------------- io ---
    def save(self, path: str) -> None:
        """Write an ``.npz`` (no pickled objects)."""
        np.savez(
            path,
            U=self.U,
            I=self.I,
            states=np.array(self.states, dtype=np.str_),
            cycle_index=self.cycle_index,
            meta=np.array(json.dumps(self.meta), dtype=np.str_),
        )

    @classmethod
    def load(cls, path: str) -> "IVDataset":
        with np.load(path, allow_pickle=False) as f:
            return cls(
                U=f["U"],
                I=f["I"],
                states=[str(s) for s in f["states"]],
                cycle_index=f["cycle_index"],
                meta=json.loads(str(f["meta"])),
            )

    # ---------------------------------------------------------- queries ---
    def resampled(self, grid: Sequence[float]) -> "IVDataset":
        """The same curves linearly interpolated onto ``grid`` (NaN where a
        curve does not cover the grid)."""
        grid = np.asarray(grid, dtype=np.float32)
        flat = self.I.reshape(-1, self.n_points)
        out = np.stack([_interp_nan(self.U, y, grid) for y in flat], axis=0)
        return IVDataset(
            U=grid,
            I=out.reshape(self.I.shape[:3] + (grid.shape[0],)),
            states=list(self.states),
            cycle_index=self.cycle_index.copy(),
            meta=dict(self.meta),
        )

    def current_at(
        self, u: float, state: Optional[Union[str, int]] = None
    ) -> np.ndarray:
        """Current of every curve at voltage ``u`` (linear interpolation):
        ``[D, C]`` for one state, ``[D, C, S]`` for ``state=None``."""
        sel = self.I if state is None else self.I[:, :, self.state_index(state)]
        flat = sel.reshape(-1, self.n_points)
        grid = np.asarray([u], dtype=np.float32)
        vals = np.array([_interp_nan(self.U, y, grid)[0] for y in flat])
        return vals.reshape(sel.shape[:-1])

    def conductance_at(
        self, u0: float, state: Optional[Union[str, int]] = None
    ) -> np.ndarray:
        """``I(u0) / u0`` per curve, same layout as ``current_at``."""
        if u0 == 0:
            raise ValueError("conductance is undefined at u0 = 0")
        return self.current_at(u0, state) / np.float32(u0)

    def median_curve(self, state: Union[str, int]) -> np.ndarray:
        """``float32[P]`` NaN-aware median over devices and cycles."""
        sel = self.I[:, :, self.state_index(state)].reshape(-1, self.n_points)
        with np.errstate(all="ignore"):
            return np.nanmedian(sel, axis=0).astype(np.float32)

    def coverage(
        self, state: Union[str, int], fraction: float = 1.0
    ) -> Tuple[float, float]:
        """Voltage range ``(umin, umax)`` on which at least ``fraction`` of the
        curves of ``state`` are measured (finite). ``(nan, nan)`` if none."""
        sel = self.I[:, :, self.state_index(state)].reshape(-1, self.n_points)
        ok = np.isfinite(sel).mean(axis=0) >= fraction
        if not ok.any():
            return (float("nan"), float("nan"))
        return (float(self.U[ok].min()), float(self.U[ok].max()))

    # ------------------------------------------------------- selections ---
    def select_devices(self, indices: Sequence[int]) -> "IVDataset":
        """Sub-population by device index (e.g. to split the two W/WOx/PCMO
        populations, see data/measurements/README.md)."""
        indices = np.asarray(indices, dtype=np.int64)
        meta = dict(self.meta)
        if "device_ids" in meta:
            meta["device_ids"] = [meta["device_ids"][k] for k in indices]
        return IVDataset(
            U=self.U.copy(),
            I=self.I[indices],
            states=list(self.states),
            cycle_index=self.cycle_index.copy(),
            meta=meta,
        )

    def select_cycles(self, mask: Sequence[bool]) -> "IVDataset":
        mask = np.asarray(mask, dtype=bool)
        return IVDataset(
            U=self.U.copy(),
            I=self.I[:, mask],
            states=list(self.states),
            cycle_index=self.cycle_index[mask],
            meta=dict(self.meta),
        )

    @staticmethod
    def concatenate(
        datasets: Sequence["IVDataset"], *, force: bool = False
    ) -> "IVDataset":
        """Stack datasets along the device axis. Refuses to merge different
        ``device_family`` values unless ``force=True`` (a crossbar is built
        from one device family)."""
        if not datasets:
            raise ValueError("no datasets given")
        first = datasets[0]
        families = {d.device_family for d in datasets}
        if len(families) > 1 and not force:
            raise ValueError(
                f"refusing to mix device families {sorted(families)}; pass force=True"
            )
        for d in datasets[1:]:
            if d.U.shape != first.U.shape or not np.allclose(d.U, first.U):
                raise ValueError("voltage grids differ; resample first")
            if d.states != first.states:
                raise ValueError(f"state names differ: {d.states} vs {first.states}")
            if not np.array_equal(d.cycle_index, first.cycle_index):
                raise ValueError("cycle_index differs")
            if d.read_polarity != first.read_polarity:
                raise ValueError("read_polarity differs")
        meta = dict(first.meta)
        if all("device_ids" in d.meta for d in datasets):
            meta["device_ids"] = sum((list(d.meta["device_ids"]) for d in datasets), [])
        else:
            meta.pop("device_ids", None)
        if len(families) > 1:
            meta["device_family"] = "+".join(sorted(families))
        return IVDataset(
            U=first.U.copy(),
            I=np.concatenate([d.I for d in datasets], axis=0),
            states=list(first.states),
            cycle_index=first.cycle_index.copy(),
            meta=meta,
        )

    def summary(self) -> str:
        lines = [
            f"IVDataset {self.device_family!r}: {self.n_devices} devices x "
            f"{self.n_cycles} cycles x {self.n_states} states, "
            f"{self.n_points} points on [{self.U.min():.3g}, {self.U.max():.3g}] V, "
            f"read polarity {self.read_polarity:+d}"
        ]
        for s in self.states:
            lo, hi = self.coverage(s)
            lines.append(f"  {s:16s} covered on [{lo:.3g}, {hi:.3g}] V")
        return "\n".join(lines)


# ---------------------------------------------------------------- adapters ---
def from_state_sweeps(
    U: Sequence[float],
    curves: Dict[str, np.ndarray],
    *,
    cycle_index: Optional[Sequence[int]] = None,
    meta: Dict,
) -> IVDataset:
    """Build a dataset from per-state read sweeps already on a common grid.

    :param curves: ``{state: I}`` with ``I`` of shape ``[D, C, P]`` or
        ``[D, P]`` (one cycle). All states must share ``D`` and ``C``.
    """
    arrays = []
    for name, arr in curves.items():
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[:, None, :]
        if arr.ndim != 3:
            raise ValueError(
                f"state {name!r}: expected [D, C, P] or [D, P], got {arr.shape}"
            )
        arrays.append(arr)
    shapes = {a.shape for a in arrays}
    if len(shapes) != 1:
        raise ValueError(f"states have different shapes: {shapes}")
    stacked = np.stack(arrays, axis=2)  # [D, C, S, P]
    if cycle_index is None:
        cycle_index = np.arange(stacked.shape[1])
    return IVDataset(
        U=np.asarray(U, dtype=np.float32),
        I=stacked,
        states=list(curves.keys()),
        cycle_index=cycle_index,
        meta=meta,
    )


def segment_loop(u: np.ndarray, i: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Split one switching loop into its three ramps at the global voltage
    extremes: ``start -> first extreme``, ``first -> second extreme``,
    ``second extreme -> end``. Shared endpoints are included in both
    neighbours. Works for negative-first (``0 -> Vmin -> Vmax -> 0``) and
    positive-first loops alike."""
    u = np.asarray(u, dtype=np.float64)
    i = np.asarray(i, dtype=np.float64)
    if u.shape != i.shape or u.ndim != 1 or u.shape[0] < 3:
        raise ValueError("u and i must be equal-length 1-d arrays with >= 3 samples")
    a, b = sorted((int(np.argmin(u)), int(np.argmax(u))))
    return [(u[: a + 1], i[: a + 1]), (u[a : b + 1], i[a : b + 1]), (u[b:], i[b:])]


def from_hysteresis_loops(
    loops: Sequence[Sequence[Tuple[np.ndarray, np.ndarray]]],
    *,
    state_names: Sequence[str] = ("pristine", "LRS", "post_reset_pos"),
    grid: Optional[Sequence[float]] = None,
    cycle_index: Optional[Sequence[int]] = None,
    meta: Dict,
) -> IVDataset:
    """Build a dataset from full switching loops.

    :param loops: ``loops[d][c] = (u, i)`` for device ``d`` and cycle ``c``;
        every loop is segmented with ``segment_loop`` into three ramps which
        become the three ``state_names`` (in ramp order).
    :param grid: common voltage grid; default: sorted unique voltages of all
        loops rounded to 1 µV.
    """
    if len(state_names) != 3:
        raise ValueError("state_names must name the three ramps of a loop")
    n_devices = len(loops)
    if n_devices == 0:
        raise ValueError("no loops given")
    n_cycles = len(loops[0])
    if any(len(l) != n_cycles for l in loops):
        raise ValueError("every device needs the same number of loops (cycles)")
    if grid is None:
        all_u = np.concatenate(
            [np.asarray(u, dtype=np.float64) for dev in loops for u, _ in dev]
        )
        grid = np.unique(np.round(all_u, 6))
    grid = np.asarray(grid, dtype=np.float32)
    out = np.full((n_devices, n_cycles, 3, grid.shape[0]), np.nan, dtype=np.float32)
    for d, dev in enumerate(loops):
        for c, (u, i) in enumerate(dev):
            for s, (us, is_) in enumerate(segment_loop(u, i)):
                out[d, c, s] = _interp_nan(us, is_, grid)
    if cycle_index is None:
        cycle_index = np.arange(n_cycles)
    return IVDataset(
        U=grid, I=out, states=list(state_names), cycle_index=cycle_index, meta=meta
    )
