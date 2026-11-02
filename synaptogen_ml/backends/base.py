__all__ = ["CellBackend", "PairJobs", "ProgrammedPairs"]

from typing import Any, List, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np
from torch import nn

# one job per (bit-plane, tile) paired array: flat ``positive_weights`` and
# ``negative_weights`` vectors in {0, 1}, input-major, as every
# ``init_from_*_quant`` already produces them
PairJobs = List[Tuple[np.ndarray, np.ndarray]]
# one (pos, neg) pair of cell objects per job, in job order. Cell objects are
# duck-typed: whatever the backend's array module accepts in
# ``init_from_paired_cell_array_input_major``.
ProgrammedPairs = List[Tuple[Any, Any]]


@runtime_checkable
class CellBackend(Protocol):
    """What a crossbar module needs from a device model.

    Implementations carry their own data and random state; the crossbar
    modules never look inside the cell objects a backend returns, they only
    hand them to the array module the same backend created.
    """

    def make_paired_array(
        self,
        in_features: int,
        out_features: int,
        *,
        additional_axes: Sequence[int] = (),
    ) -> nn.Module:
        """A differential (pos - neg) array module with ``r`` of shape
        ``additional_axes + (in_features, out_features)`` whose
        ``init_from_paired_cell_array_input_major(pos, neg)`` consumes this
        backend's programmed cells and whose ``forward([..., I])`` returns
        ``[..., *additional_axes, O]`` currents in ampere."""
        ...

    def program_pairs(
        self,
        jobs: PairJobs,
        num_cycles_init: int,
        correction_settings,
        current_scaling: float,
    ) -> ProgrammedPairs:
        """Program every job, in job order. ``correction_settings`` is a
        ``CycleCorrectionSettings`` or None; ``current_scaling`` is the
        module's ``hardware_output_current_scaling`` (needed by write-verify
        to compare read currents against target weights)."""
        ...

    def suggested_current_scaling(self, vmax: float, ideal: bool = False) -> float:
        """Estimate of ``DacAdcHardwareSettings.hardware_output_current_scaling``
        for a DAC window of ``[-vmax, vmax]``: the factor that maps the
        differential read current of a unit weight back to 1."""
        ...
