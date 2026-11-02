"""Reader for the partner's Keithley-style ``I(V)`` ``.dat`` export
(``data/measurements/*.dat``, see ``data/measurements/README.md``).

File layout: a few ``key: value`` header lines, a blank line, a column header
and one tab-separated table ``Voltage Source (V) | Voltage Probe (V) |
Current Probe (A) | Resistance (Ohm)`` with all devices concatenated. Each
device is one quasi-static loop ``0 -> Vmin -> Vmax -> 0``; a new device starts
at a ``0.0`` source sample that is followed by a negative sample.
"""

__all__ = [
    "read_header",
    "load_table",
    "split_devices",
    "branches",
    "from_keithley_iv_dat",
]

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..dataset import IVDataset, from_hysteresis_loops, segment_loop

COLUMNS = ("voltage_source", "voltage_probe", "current", "resistance")
V_SOURCE, V_PROBE, CURRENT, RESISTANCE = range(4)


def read_header(path: str) -> Dict[str, str]:
    """The ``key: value`` lines above the table (tab-separated pairs on one
    line are split). Values are kept as strings."""
    header = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("*"):
                continue
            if line.startswith("Voltage Source"):
                break
            for part in line.split("\t"):
                if ":" in part:
                    key, value = part.split(":", 1)
                    header.setdefault(key.strip(), value.strip())
    return header


def load_table(path: str) -> np.ndarray:
    """``float64[N, 4]`` of every numeric 4-column row in the file."""
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 4:
                continue
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                continue  # column header
    if not rows:
        raise ValueError(f"no data rows found in {path}")
    return np.array(rows, dtype=np.float64)


def split_devices(table: np.ndarray) -> List[np.ndarray]:
    """Split the concatenated table into one array per device (loop): a device
    starts at a ``0.0`` source sample followed by a negative sample."""
    vs = table[:, V_SOURCE]
    starts = [0]
    for k in range(1, len(vs) - 1):
        if vs[k] == 0.0 and vs[k + 1] < 0:
            starts.append(k)
    starts.append(len(vs))
    return [table[a:b] for a, b in zip(starts[:-1], starts[1:])]


def branches(dev: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The three ramps of one device table as row slices (0: ``0 -> Vmin``
    pristine, 1: ``Vmin -> Vmax`` after SET, 2: ``Vmax -> 0`` after RESET)."""
    vs = dev[:, V_SOURCE]
    i_min, i_max = int(np.argmin(vs)), int(np.argmax(vs))
    return dev[: i_min + 1], dev[i_min : i_max + 1], dev[i_max:]


def from_keithley_iv_dat(
    path: str,
    *,
    device_family: str,
    device_indices: Optional[Sequence[int]] = None,
    grid: Optional[Sequence[float]] = None,
    state_names: Sequence[str] = ("pristine", "LRS", "post_reset_pos"),
    read_polarity: int = -1,
    extra_meta: Optional[Dict] = None,
) -> IVDataset:
    """Read one ``.dat`` file into an ``IVDataset`` with one cycle
    (``cycle_index=[0]``, the first loop of each pristine device) and the three
    ramps as states.

    :param device_family: required dataset label, e.g. ``"Ta/WOx/PCMO"``.
    :param device_indices: keep only these devices (file order), e.g. to split
        the two W/WOx/PCMO sub-populations.
    :param grid: common voltage grid (default: the file's own source voltages).
    :param read_polarity: polarity the cells will be read at (-1 for these
        rectifying PCMO devices: the negative branch conducts).
    """
    table = load_table(path)
    devices = split_devices(table)
    device_ids = list(range(len(devices)))
    if grid is None:
        # grid from the whole file, so that sub-populations selected with
        # device_indices share one grid and can be concatenated later
        grid = np.unique(np.round(table[:, V_SOURCE], 6))
    if device_indices is not None:
        device_indices = [int(k) for k in device_indices]
        devices = [devices[k] for k in device_indices]
        device_ids = device_indices
    loops = [[(dev[:, V_SOURCE], dev[:, CURRENT])] for dev in devices]
    meta = {
        "device_family": device_family,
        "read_polarity": int(read_polarity),
        "source": os.path.basename(path),
        "instrument_header": read_header(path),
        "device_ids": device_ids,
        "programming": "first quasi-static IV loop 0 -> Vmin -> Vmax -> 0 of a pristine device; "
        "states are the three ramps",
        "voltage_column": "source",
    }
    if extra_meta:
        meta.update(extra_meta)
    return from_hysteresis_loops(
        loops, state_names=state_names, grid=grid, cycle_index=[0], meta=meta
    )
