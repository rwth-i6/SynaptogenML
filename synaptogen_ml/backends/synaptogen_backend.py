__all__ = ["SynaptogenBackend"]

from typing import Sequence

from torch import nn

from ..memristor_modules.memristor import PairedMemristorArrayV2
from ..memristor_modules.util import compute_correction_factor
from ..programming import program_pairs
from .base import CellBackend, PairJobs, ProgrammedPairs


class SynaptogenBackend(CellBackend):
    """The generative Synaptogen cell model (VAR + GMM, ``synaptogen.py``)
    behind the backend interface.

    ``backend=SynaptogenBackend()`` and the default ``backend=None`` program
    the pairs in the same order through the same helper, so under the same
    seeds they produce bit-identical modules (tested in
    ``tests/test_backends.py``). Process-wide switches (fast programming
    workers, readout-noise model, linear-IV oracle) keep applying.

    :param calibration_trials: number of 200-cell arrays averaged by
        ``suggested_current_scaling`` (``compute_correction_factor``).
    """

    def __init__(self, calibration_trials: int = 1000):
        self.calibration_trials = calibration_trials

    def make_paired_array(
        self,
        in_features: int,
        out_features: int,
        *,
        additional_axes: Sequence[int] = (),
    ) -> nn.Module:
        return PairedMemristorArrayV2(
            in_features=in_features,
            out_features=out_features,
            additional_axes=tuple(additional_axes),
        )

    def program_pairs(
        self,
        jobs: PairJobs,
        num_cycles_init: int,
        correction_settings,
        current_scaling: float,
    ) -> ProgrammedPairs:
        return program_pairs(
            jobs, num_cycles_init, correction_settings, current_scaling
        )

    def suggested_current_scaling(self, vmax: float, ideal: bool = False) -> float:
        # compute_correction_factor is calibrated on the 0.6 V window used by
        # every existing configuration (its check voltages and the /0.6
        # normalisation are hard-coded there); other windows are not supported
        # for this backend rather than silently extrapolated.
        if abs(vmax - 0.6) > 1e-9:
            raise ValueError(
                f"SynaptogenBackend.suggested_current_scaling is only calibrated "
                f"for vmax=0.6, got {vmax}"
            )
        paired, _single = compute_correction_factor(
            ideal=ideal, trials=self.calibration_trials
        )
        return float(paired)
