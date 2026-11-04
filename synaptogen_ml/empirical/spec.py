"""``DeviceSpec``: a plain-data description of which device backend to use.

Experiment pipelines (sisyphus etc.) hash their job arguments and cannot pass
backend objects around; they pass this spec as a dict instead and build the
backend inside the worker. ``DeviceSpec.from_dict(cfg["device_spec"]).build()``
is the whole contract. A spec with ``device="synaptogen"`` gives the historical
generative backend, every other name is a bundled measurement dataset
(``synaptogen_ml.empirical.datasets.bundled_names()``) or a path to an
``IVDataset`` npz file.
"""

__all__ = ["DeviceSpec", "SYNAPTOGEN"]

import dataclasses
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from .backend import EmpiricalBackend
from .dataset import IVDataset
from .datasets import BUNDLED, load_bundled

if TYPE_CHECKING:
    from ..backends.base import CellBackend
    from ..memristor_modules.memristor import DacAdcHardwareSettings

SYNAPTOGEN = "synaptogen"


@dataclass(frozen=True)
class DeviceSpec:
    """Which device backs the crossbar, and how it is read.

    :param device: ``"synaptogen"``, a bundled dataset name (``"w_pcmo"``,
        ``"ta_pcmo"``) or a path to an ``IVDataset`` npz file.
    :param read_voltage: read voltage of the empirical backend (signed; sign
        must match the dataset's read polarity). Ignored for Synaptogen.
    :param on_state, off_state: dataset states programmed for a weight bit 1 / 0.
    :param representation: ``"pulse_count"`` or ``"shared_poly"``.
    :param readout_bandwidth: Hz, Johnson+shot noise bandwidth of the
        pulse-count arrays.
    :param seed: seed of the backend's private sampling RNG (``None`` -> OS
        entropy). ``build(seed=...)`` overrides it per instantiation.
    :param device_indices: optional subset of the dataset's devices.
    :param jitter_sigma: opt-in synthetic device variability in dex; 0 keeps
        the backend purely data-based.
    :param pulses_full_scale: DAC steps per full-scale input for the pulse-count
        hardware settings (``None`` -> one pulse per DAC step).
    """

    device: str = SYNAPTOGEN
    read_voltage: float = -0.2
    on_state: str = "LRS"
    off_state: str = "pristine"
    representation: str = "pulse_count"
    readout_bandwidth: float = 1e8
    seed: Optional[int] = None
    device_indices: Optional[Tuple[int, ...]] = None
    jitter_sigma: float = 0.0
    pulses_full_scale: Optional[int] = None

    def __post_init__(self):
        if self.device_indices is not None:
            object.__setattr__(
                self, "device_indices", tuple(int(i) for i in self.device_indices)
            )
        if not self.is_synaptogen and self.device not in BUNDLED:
            if not self.device.endswith(".npz"):
                raise ValueError(
                    f"device {self.device!r} is neither {SYNAPTOGEN!r}, a bundled "
                    f"dataset {sorted(BUNDLED)} nor an .npz path"
                )

    # ------------------------------------------------------------ plain data ---
    @property
    def is_synaptogen(self) -> bool:
        return self.device == SYNAPTOGEN

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        if d["device_indices"] is not None:
            d["device_indices"] = list(d["device_indices"])
        return d

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "DeviceSpec":
        if d is None:
            return cls()
        if isinstance(d, DeviceSpec):
            return d
        return cls(**d)

    def describe(self) -> str:
        if self.is_synaptogen:
            return SYNAPTOGEN
        parts = [
            self.device,
            f"{self.representation}@{self.read_voltage:+g}V",
            f"{self.on_state}/{self.off_state}",
            f"bw{self.readout_bandwidth:g}",
        ]
        if self.device_indices is not None:
            parts.append(f"{len(self.device_indices)}dev")
        if self.jitter_sigma:
            parts.append(f"jitter{self.jitter_sigma:g}")
        return "_".join(parts)

    # --------------------------------------------------------------- build ---
    def load_dataset(self) -> IVDataset:
        if self.is_synaptogen:
            raise ValueError("the synaptogen spec has no dataset")
        if self.device in BUNDLED:
            return load_bundled(self.device, device_indices=self.device_indices)
        ds = IVDataset.load(os.path.expanduser(self.device))
        if self.device_indices is not None:
            ds = ds.select_devices(list(self.device_indices))
        return ds

    def build(self, *, seed: Optional[int] = None) -> "CellBackend":
        """Instantiate the backend. ``seed`` overrides the spec's seed."""
        if self.is_synaptogen:
            from ..backends.synaptogen_backend import SynaptogenBackend

            return SynaptogenBackend()
        return EmpiricalBackend(
            self.load_dataset(),
            read_voltage=self.read_voltage,
            state_for_weight={1: self.on_state, 0: self.off_state},
            representation=self.representation,
            seed=self.seed if seed is None else seed,
            jitter_sigma=self.jitter_sigma,
            readout_bandwidth=self.readout_bandwidth,
        )

    def hardware_settings(
        self,
        input_bits: int,
        output_precision_bits: int,
        output_range_bits: int,
        *,
        default: Optional["DacAdcHardwareSettings"] = None,
        backend: Optional["CellBackend"] = None,
        ideal: bool = False,
    ) -> "DacAdcHardwareSettings":
        """DAC/ADC settings for this device.

        Synaptogen keeps the caller's ``default`` (the historically calibrated
        0.6 V window), which is therefore required. Empirical devices derive
        the pulse-count window and current scaling from the dataset
        (``EmpiricalBackend.suggested_hardware_settings``); pass the already
        built ``backend`` to avoid loading the dataset twice.
        """
        if self.is_synaptogen:
            if default is None:
                raise ValueError("synaptogen spec needs the default hardware settings")
            return default
        if backend is None:
            backend = self.build()
        return backend.suggested_hardware_settings(
            input_bits,
            output_precision_bits,
            output_range_bits,
            pulses_full_scale=self.pulses_full_scale,
            ideal=ideal,
        )
