"""Bundled measurement datasets, loadable by name.

The partner ``.dat`` files live in ``data/measurements`` at the repository
root (not inside the package), so a checkout of the repository is enough and
no path has to be passed around. ``SYNAPTOGEN_MEASUREMENTS`` overrides the
directory, e.g. for an installed package without the repository.
"""

__all__ = ["BUNDLED", "measurements_dir", "load_bundled", "bundled_names"]

import os
from typing import Dict, Optional, Sequence, Tuple

from .adapters.keithley import from_keithley_iv_dat
from .dataset import IVDataset

ENV_VAR = "SYNAPTOGEN_MEASUREMENTS"

# name -> (file, device_family)
BUNDLED: Dict[str, Tuple[str, str]] = {
    "w_pcmo": ("W_WOx_PCMO.dat", "W/WOx/PCMO"),
    "ta_pcmo": ("Ta_WOx_PCMO.dat", "Ta/WOx/PCMO"),
}


def bundled_names() -> Sequence[str]:
    return tuple(BUNDLED)


def measurements_dir() -> str:
    override = os.environ.get(ENV_VAR)
    if override:
        return override
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "data", "measurements"))


def load_bundled(
    name: str, *, device_indices: Optional[Sequence[int]] = None
) -> IVDataset:
    """Load one of the bundled datasets (``bundled_names()``) as ``IVDataset``.

    :param device_indices: optional subset of devices (indices into the file
        order); the voltage grid is always derived from the whole file so
        subsets of one family stay comparable.
    """
    if name not in BUNDLED:
        raise KeyError(f"unknown bundled dataset {name!r}; known: {sorted(BUNDLED)}")
    filename, family = BUNDLED[name]
    path = os.path.join(measurements_dir(), filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found; set {ENV_VAR} to the directory holding {filename}"
        )
    return from_keithley_iv_dat(
        path, device_family=family, device_indices=device_indices
    )
