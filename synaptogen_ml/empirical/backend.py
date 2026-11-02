"""``EmpiricalBackend``: program crossbar cells by sampling measured devices.

Non-generative by design: every cell *is* one measured (device, cycle) curve
of the backend's ``IVDataset``, read at one fixed voltage. Nothing is fitted
or drawn from a distribution unless explicitly asked for (``jitter_sigma``).
One backend instance holds exactly one dataset of one device family;
device families are never mixed in one crossbar.

Representations
    ``"pulse_count"`` (recommended): one current per cell at ``read_voltage``,
    consumed by ``PairedPulseCountMemristorArray`` (pulse-count input
    encoding; linear by construction, see ``memristor_modules/pulse.py``).
    ``"shared_poly"``: the Synaptogen array layout -- one polynomial for the
    pool-median "on" curve, one for the "off" curve, and a scalar ``r`` per
    cell that reproduces the cell's measured conductance at ``read_voltage``
    (amplitude input encoding through the unchanged ``PairedMemristorArrayV2``;
    keeps the existing fast path and switches, loses per-device IV shape).

Sign convention: cell currents are positive in read polarity; the crossbar
never sees the raw measured sign. For ``"shared_poly"`` the curves are
mirrored accordingly (``U' = polarity * U``, ``I' = polarity * I``).

This module imports torch only inside ``make_paired_array`` /
``suggested_hardware_settings`` so that it stays importable in the torch-free
programming workers.
"""

__all__ = [
    "EmpiricalBackend",
    "EmpiricalParams",
    "PulseProgrammedCells",
    "REPRESENTATIONS",
]

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ..backends.base import CellBackend, PairJobs, ProgrammedPairs
from ..programming import ProgrammedCells
from .dataset import IVDataset
from .fit import eval_poly, fit_poly_zero_intercept, fit_quality, to_polyval_order

log = logging.getLogger(__name__)

REPRESENTATIONS = ("pulse_count", "shared_poly")
VERIFY_REFERENCES = ("median", "target")


@dataclass
class PulseProgrammedCells:
    """Programming result for the pulse-count representation (what
    ``PulseCountMemristorArray.init_from_cell_array_input_major`` consumes).

    :param I_pulse: ``float32[M]`` current per read pulse, positive in read
        polarity.
    :param device_index: ``int64[M]`` dataset device each cell was sampled from
        (-1 under ideal programming).
    :param cycle_index: ``int64[M]`` cycle axis entry each cell was sampled from.
    """

    I_pulse: np.ndarray
    device_index: np.ndarray
    cycle_index: np.ndarray


@dataclass
class EmpiricalParams:
    """Stand-in for ``CellParams`` in the shared-polynomial representation:
    what ``MemristorArray.init_resistance_states`` reads. Coefficients in
    ``numpy.polyval`` (descending) order like ``CellParams.LLRS``/``HHRS``;
    ``U0`` is the (mirrored, positive) read voltage the linear-IV secant uses."""

    LLRS: np.ndarray
    HHRS: np.ndarray
    U0: float


class EmpiricalBackend(CellBackend):
    """
    :param dataset: measured curves of one device family.
    :param read_voltage: voltage the cells are read at (volt, signed; its sign
        must equal ``dataset.read_polarity``). Every state used must be
        measured there.
    :param state_for_weight: ``{1: <on state>, 0: <off state>}`` -- which
        dataset state a programmed 1 / 0 means. Required, no default: with
        first-loop data the only "off" state may be the pristine branch, and
        that choice has to be explicit.
    :param representation: ``"pulse_count"`` or ``"shared_poly"``.
    :param degree: polynomial degree for ``"shared_poly"``.
    :param fit_window: ``(lo, hi)`` in mirrored volts for the ``"shared_poly"``
        fits (default: the full measured range of the median curves).
    :param seed: seed of the backend's private RNG. The Synaptogen module RNG
        and ``numpy.random`` are never touched.
    :param jitter_sigma: **opt-in synthetic variability** in dex (log10):
        multiplies every sampled current by ``10 ** (jitter_sigma * N(0, 1))``.
        0 (default) keeps the backend purely data-based.
    :param readout_bandwidth: Hz, noise model of the pulse-count arrays.
    :param verify_reference: write-verify criterion, ``"median"`` (deviation
        from the pool median of the target state, relative to the median "on"
        current) or ``"target"`` (the literal Synaptogen criterion against the
        integer weight, degenerate for most scalings).
    :param cycle_resample_for_verify: whether write-verify may resample a
        cycle of the same device (needs more than one cycle per device).
    """

    def __init__(
        self,
        dataset: IVDataset,
        *,
        read_voltage: float,
        state_for_weight: Dict[int, str],
        representation: str = "pulse_count",
        degree: int = 7,
        fit_window: Optional[Tuple[float, float]] = None,
        seed: Optional[int] = None,
        jitter_sigma: float = 0.0,
        readout_bandwidth: float = 1e8,
        verify_reference: str = "median",
    ):
        if representation not in REPRESENTATIONS:
            raise ValueError(f"representation must be one of {REPRESENTATIONS}")
        if verify_reference not in VERIFY_REFERENCES:
            raise ValueError(f"verify_reference must be one of {VERIFY_REFERENCES}")
        if set(state_for_weight) != {0, 1}:
            raise ValueError("state_for_weight must map exactly the weights 0 and 1")
        if read_voltage == 0 or np.sign(read_voltage) != dataset.read_polarity:
            raise ValueError(
                f"read_voltage {read_voltage} must be non-zero with the sign of the "
                f"dataset read polarity ({dataset.read_polarity:+d})"
            )
        if jitter_sigma < 0:
            raise ValueError("jitter_sigma must be >= 0")

        self.dataset = dataset
        self.read_voltage = float(read_voltage)
        self.polarity = int(dataset.read_polarity)
        self.pulse_voltage = abs(self.read_voltage)
        self.state_for_weight = {int(k): v for k, v in state_for_weight.items()}
        self.representation = representation
        self.degree = int(degree)
        self.fit_window = fit_window
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.jitter_sigma = float(jitter_sigma)
        self.readout_bandwidth = float(readout_bandwidth)
        self.verify_reference = verify_reference

        # measured current (positive in read polarity) of every (device, cycle)
        # per used state, plus the valid (finite) entries
        self.state_names = [self.state_for_weight[1], self.state_for_weight[0]]
        for name in self.state_names:
            dataset.state_index(name)  # raises for unknown states
        table = self.polarity * dataset.current_at(self.read_voltage)  # [D, C, S]
        self._current = {
            name: table[:, :, dataset.state_index(name)].astype(np.float64)
            for name in self.state_names
        }
        for name, cur in self._current.items():
            if not np.isfinite(cur).any():
                raise ValueError(
                    f"state {name!r} is not measured at {self.read_voltage} V "
                    f"(covered on {dataset.coverage(name, fraction=0.0)})"
                )
            if (cur[np.isfinite(cur)] <= 0).any():
                log.warning(
                    "state %r: %d of %d samples at %.3g V are <= 0 in read polarity "
                    "(noise floor / offset); they are kept as measured",
                    name,
                    int((cur[np.isfinite(cur)] <= 0).sum()),
                    int(np.isfinite(cur).sum()),
                    self.read_voltage,
                )
        self.median_current = {
            name: float(np.nanmedian(cur)) for name, cur in self._current.items()
        }
        self.mean_current = {
            name: float(np.nanmean(cur)) for name, cur in self._current.items()
        }
        self._warned_reuse = False
        self._warned_single_cycle = False
        self.device_use_counts = np.zeros(dataset.n_devices, dtype=np.int64)
        self.cells_programmed = 0

        if self.representation == "shared_poly":
            self.params = self._fit_shared_polys()
        else:
            self.params = None

        if self.jitter_sigma > 0:
            log.warning(
                "EmpiricalBackend: jitter_sigma=%.3g dex -> sampled currents are "
                "perturbed; results are no longer purely data-based",
                self.jitter_sigma,
            )
        log.info("EmpiricalBackend ready: %s", self.describe())

    # ----------------------------------------------------------- reporting ---
    def describe(self) -> Dict:
        on, off = self.state_names
        on_cur, off_cur = self._current[on], self._current[off]
        valid_on = np.isfinite(on_cur).any(axis=1).sum()
        valid_off = np.isfinite(off_cur).any(axis=1).sum()
        pos_on = on_cur[np.isfinite(on_cur) & (on_cur > 0)]
        return {
            "device_family": self.dataset.device_family,
            "representation": self.representation,
            "read_voltage": self.read_voltage,
            "states": {"1": on, "0": off},
            "devices_with_on_state": int(valid_on),
            "devices_with_off_state": int(valid_off),
            "cycles": int(self.dataset.n_cycles),
            "median_on_current": self.median_current[on],
            "median_off_current": self.median_current[off],
            "on_off_ratio": self.median_current[on] / self.median_current[off]
            if self.median_current[off] > 0
            else float("inf"),
            "on_spread_dex": float(np.std(np.log10(pos_on))) if pos_on.size else None,
            "delta_conductance": self.delta_conductance(),
            "jitter_sigma": self.jitter_sigma,
            "readout_bandwidth": self.readout_bandwidth,
        }

    def reuse_report(self) -> str:
        used = self.device_use_counts[self.device_use_counts > 0]
        if used.size == 0:
            return "no cells programmed yet"
        return (
            f"{self.cells_programmed} cells from {used.size} of "
            f"{self.dataset.n_devices} devices; reuse per device min {used.min()}, "
            f"median {int(np.median(used))}, max {used.max()}"
        )

    # ------------------------------------------------------------ scaling ---
    def delta_conductance(self, ideal: bool = False) -> float:
        """Differential conductance of a unit weight (on minus off) at the read
        voltage: mean over the pool (median under ideal programming), i.e. the
        estimator of ``compute_correction_factor`` applied to measured data."""
        on, off = self.state_names
        if ideal:
            delta = self.median_current[on] - self.median_current[off]
        else:
            delta = self.mean_current[on] - self.mean_current[off]
        return delta / self.pulse_voltage

    def suggested_current_scaling(self, vmax: float, ideal: bool = False) -> float:
        """``1 / (vmax * delta_G)``: maps the differential current of a unit
        weight at full-scale input (DAC voltage ``vmax``) back to 1."""
        delta = self.delta_conductance(ideal)
        if not delta > 0:
            raise ValueError(
                f"on state {self.state_names[0]!r} does not conduct more than off "
                f"state {self.state_names[1]!r} at {self.read_voltage} V"
            )
        return float(1.0 / (vmax * delta))

    def suggested_hardware_settings(
        self,
        input_bits: int,
        output_precision_bits: int,
        output_range_bits: int,
        *,
        pulses_full_scale: Optional[int] = None,
        ideal: bool = False,
    ):
        """``DacAdcHardwareSettings`` for pulse-count encoding: the DAC window is
        ``pulses_full_scale * pulse_voltage`` (default ``2 ** (input_bits - 1)``
        pulses, one per DAC step) and the current scaling is
        ``suggested_current_scaling`` for that window."""
        from ..memristor_modules.memristor import DacAdcHardwareSettings

        if pulses_full_scale is None:
            pulses_full_scale = 2 ** (input_bits - 1)
        vmax = pulses_full_scale * self.pulse_voltage
        return DacAdcHardwareSettings(
            input_bits=input_bits,
            output_precision_bits=output_precision_bits,
            output_range_bits=output_range_bits,
            hardware_input_vmax=vmax,
            hardware_output_current_scaling=self.suggested_current_scaling(vmax, ideal),
        )

    # ------------------------------------------------------------- arrays ---
    def make_paired_array(
        self,
        in_features: int,
        out_features: int,
        *,
        additional_axes: Sequence[int] = (),
    ):
        if self.representation == "pulse_count":
            from ..memristor_modules.pulse import PairedPulseCountMemristorArray

            return PairedPulseCountMemristorArray(
                in_features,
                out_features,
                additional_axes=tuple(additional_axes),
                pulse_voltage=self.pulse_voltage,
                bandwidth=self.readout_bandwidth,
            )
        from ..memristor_modules.memristor import PairedMemristorArrayV2

        return PairedMemristorArrayV2(
            in_features=in_features,
            out_features=out_features,
            additional_axes=tuple(additional_axes),
        )

    # ----------------------------------------------------------- sampling ---
    def _valid_cycles(self, state: str, num_cycles_init: int):
        """Per device the cycle-axis entries of ``state`` that are measured at
        the read voltage and have ``cycle_index >= num_cycles_init``.
        Returns ``(devices[Dv], table[Dv, Cmax], counts[Dv])``."""
        finite = np.isfinite(self._current[state])  # [D, C]
        allowed = self.dataset.cycle_index >= num_cycles_init
        if not (finite & allowed).any():
            log.warning(
                "no cycle with cycle_index >= %d for state %r; using all cycles",
                num_cycles_init,
                state,
            )
            allowed = np.ones_like(allowed)
        ok = finite & allowed
        counts = ok.sum(axis=1)
        devices = np.nonzero(counts > 0)[0]
        table = np.full((devices.size, int(counts.max())), -1, dtype=np.int64)
        for row, d in enumerate(devices):
            cyc = np.nonzero(ok[d])[0]
            table[row, : cyc.size] = cyc
        return devices, table, counts[devices]

    def _sample_state(self, state: str, size: int, num_cycles_init: int):
        devices, table, counts = self._valid_cycles(state, num_cycles_init)
        rows = self.rng.integers(0, devices.size, size=size)
        k = (self.rng.random(size) * counts[rows]).astype(np.int64)
        cyc = table[rows, k]
        dev = devices[rows]
        return self._current[state][dev, cyc], dev, cyc

    def _resample_cycles(self, state: str, dev: np.ndarray, num_cycles_init: int):
        """A new valid cycle (uniform) for the given devices, same device."""
        devices, table, counts = self._valid_cycles(state, num_cycles_init)
        row_of = np.full(self.dataset.n_devices, -1, dtype=np.int64)
        row_of[devices] = np.arange(devices.size)
        rows = row_of[dev]
        k = (self.rng.random(dev.size) * counts[rows]).astype(np.int64)
        cyc = table[rows, k]
        return self._current[state][dev, cyc], cyc

    def _sample_cells(self, weights: np.ndarray, num_cycles_init: int):
        """One measured current per cell: state by weight (0/1), device uniform
        with replacement, then cycle uniform over the device's valid cycles."""
        weights = np.asarray(weights)
        if not np.isin(weights, (0, 1)).all():
            raise ValueError("weights must be binary (bit-plane magnitudes)")
        size = weights.shape[0]
        current = np.empty(size, dtype=np.float64)
        dev = np.empty(size, dtype=np.int64)
        cyc = np.empty(size, dtype=np.int64)
        for w, state in self.state_for_weight.items():
            mask = weights == w
            if mask.any():
                current[mask], dev[mask], cyc[mask] = self._sample_state(
                    state, int(mask.sum()), num_cycles_init
                )
        return current, dev, cyc

    def _ideal_cells(self, weights: np.ndarray):
        weights = np.asarray(weights)
        current = np.where(
            weights == 1,
            self.median_current[self.state_for_weight[1]],
            self.median_current[self.state_for_weight[0]],
        ).astype(np.float64)
        return current, np.full(weights.shape[0], -1), np.full(weights.shape[0], -1)

    def _write_verify(
        self,
        weights: np.ndarray,
        current: np.ndarray,
        dev: np.ndarray,
        cyc: np.ndarray,
        num_cycles_init: int,
        correction_settings,
        current_scaling: float,
    ):
        """Iterative correction: cells whose scaled read value deviates too much
        are re-programmed, which for a measured device means drawing another
        cycle of the same device. With one cycle per device this cannot change
        anything and is skipped with a warning."""
        on, off = self.state_names
        if self.dataset.n_cycles < 2:
            if not self._warned_single_cycle:
                log.warning(
                    "write-verify requested but the dataset has one cycle per "
                    "device: re-programming cannot change a cell, skipping"
                )
                self._warned_single_cycle = True
            return current, cyc
        pulses = correction_settings.test_input_value / self.pulse_voltage
        target_state = np.where(weights == 1, on, off)
        ref_current = np.where(
            weights == 1, self.median_current[on], self.median_current[off]
        )
        for _ in range(correction_settings.num_cycles):
            value = pulses * current * current_scaling
            if self.verify_reference == "target":
                deviation = np.abs(value - weights)
            else:
                reference = pulses * ref_current * current_scaling
                scale = abs(pulses * self.median_current[on] * current_scaling)
                deviation = np.abs(value - reference) / scale
            flagged = deviation > correction_settings.relative_deviation
            if not flagged.any():
                break
            for state in (on, off):
                mask = flagged & (target_state == state)
                if mask.any():
                    current[mask], cyc[mask] = self._resample_cycles(
                        state, dev[mask], num_cycles_init
                    )
        return current, cyc

    def _program_one(
        self,
        weights: np.ndarray,
        num_cycles_init: int,
        correction_settings,
        current_scaling: float,
    ):
        weights = np.asarray(weights)
        ideal = (
            correction_settings is not None and correction_settings.ideal_programming
        )
        if ideal:
            current, dev, cyc = self._ideal_cells(weights)
        else:
            current, dev, cyc = self._sample_cells(weights, num_cycles_init)
            if correction_settings is not None:
                current, cyc = self._write_verify(
                    weights,
                    current,
                    dev,
                    cyc,
                    num_cycles_init,
                    correction_settings,
                    current_scaling,
                )
            if self.jitter_sigma > 0:
                current = current * 10.0 ** (
                    self.jitter_sigma * self.rng.standard_normal(current.shape)
                )
            self.device_use_counts += np.bincount(dev, minlength=self.dataset.n_devices)
            self.cells_programmed += weights.shape[0]
            if not self._warned_reuse and weights.shape[0] > self.dataset.n_devices:
                log.warning(
                    "EmpiricalBackend (%s): %d cells sampled from a pool of %d devices; "
                    "devices are reused (see reuse_report())",
                    self.dataset.device_family,
                    weights.shape[0],
                    self.dataset.n_devices,
                )
                self._warned_reuse = True
        return self._wrap(current, dev, cyc)

    def _wrap(self, current: np.ndarray, dev: np.ndarray, cyc: np.ndarray):
        if self.representation == "pulse_count":
            return PulseProgrammedCells(
                I_pulse=current.astype(np.float32),
                device_index=dev.astype(np.int64),
                cycle_index=cyc.astype(np.int64),
            )
        # shared_poly: r reproducing the cell's conductance at the read voltage
        g_cell = current / self.pulse_voltage
        r = (self._g_on - g_cell) / (self._g_on - self._g_off)
        if r.size and (r.min() < -0.5 or r.max() > 1.5):
            log.info(
                "shared_poly: projected r outside [0, 1]: min %.3f max %.3f",
                r.min(),
                r.max(),
            )
        return ProgrammedCells(r.astype(np.float32), self.params)

    def program_pairs(
        self,
        jobs: PairJobs,
        num_cycles_init: int,
        correction_settings,
        current_scaling: float,
    ) -> ProgrammedPairs:
        return [
            (
                self._program_one(
                    pos, num_cycles_init, correction_settings, current_scaling
                ),
                self._program_one(
                    neg, num_cycles_init, correction_settings, current_scaling
                ),
            )
            for pos, neg in jobs
        ]

    # --------------------------------------------------------- shared_poly ---
    def _fit_shared_polys(self) -> EmpiricalParams:
        """Zero-intercept polynomials of the pool-median on / off curves in
        mirrored coordinates (read polarity -> positive)."""
        on, off = self.state_names
        u = self.polarity * self.dataset.U.astype(np.float64)
        order = np.argsort(u)
        u = u[order]
        polys = {}
        for name in (on, off):
            med = self.polarity * self.dataset.median_curve(name).astype(np.float64)
            med = med[order]
            coeffs = fit_poly_zero_intercept(
                u, med, self.degree, window=self.fit_window
            )
            quality = fit_quality(coeffs, u, med, window=self.fit_window)
            log.info("shared_poly fit %r: relative RMS %.3g", name, quality)
            polys[name] = coeffs
        self._g_on = float(
            eval_poly(polys[on], self.pulse_voltage) / self.pulse_voltage
        )
        self._g_off = float(
            eval_poly(polys[off], self.pulse_voltage) / self.pulse_voltage
        )
        if not self._g_on > self._g_off:
            raise ValueError(
                "shared_poly: fitted on-state conductance does not exceed the off "
                "state at the read voltage"
            )
        return EmpiricalParams(
            LLRS=to_polyval_order(polys[on]),
            HHRS=to_polyval_order(polys[off]),
            U0=self.pulse_voltage,
        )
