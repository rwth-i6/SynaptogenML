"""Tests for ``DeviceSpec`` (plain-data backend description) and the bundled
dataset loaders."""

import os

import numpy as np
import pytest

from synaptogen_ml.backends import SynaptogenBackend
from synaptogen_ml.empirical import (
    DeviceSpec,
    EmpiricalBackend,
    bundled_names,
    load_bundled,
)
from synaptogen_ml.empirical.datasets import ENV_VAR, measurements_dir
from synaptogen_ml.empirical.synthetic import generate_synthetic_dataset
from synaptogen_ml.memristor_modules.memristor import DacAdcHardwareSettings

pytestmark = pytest.mark.empirical

DEFAULT_HW = DacAdcHardwareSettings(
    input_bits=8,
    output_precision_bits=2,
    output_range_bits=6,
    hardware_input_vmax=0.6,
    hardware_output_current_scaling=8020.0,
)


def test_bundled_loaders():
    assert set(bundled_names()) == {"w_pcmo", "ta_pcmo"}
    assert os.path.isdir(measurements_dir())
    w = load_bundled("w_pcmo")
    ta = load_bundled("ta_pcmo", device_indices=[0, 3, 5])
    assert w.n_devices == 76 and w.meta["device_family"] == "W/WOx/PCMO"
    assert ta.n_devices == 3 and ta.meta["device_family"] == "Ta/WOx/PCMO"
    with pytest.raises(KeyError):
        load_bundled("nope")


def test_measurements_dir_override(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    assert measurements_dir() == str(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_bundled("w_pcmo")


def test_spec_round_trip_and_validation():
    spec = DeviceSpec(device="ta_pcmo", seed=7, device_indices=[2, 0])
    d = spec.to_dict()
    assert d["device_indices"] == [2, 0]  # json friendly
    assert DeviceSpec.from_dict(d) == spec
    assert DeviceSpec.from_dict(None) == DeviceSpec()
    assert DeviceSpec.from_dict(spec) is spec
    assert DeviceSpec().is_synaptogen and DeviceSpec().describe() == "synaptogen"
    assert "ta_pcmo" in spec.describe() and "2dev" in spec.describe()
    with pytest.raises(ValueError):
        DeviceSpec(device="unknown_family")


def test_spec_builds_backends():
    syn = DeviceSpec().build()
    assert isinstance(syn, SynaptogenBackend)
    assert DeviceSpec().hardware_settings(8, 2, 6, default=DEFAULT_HW) is DEFAULT_HW
    with pytest.raises(ValueError):
        DeviceSpec().hardware_settings(8, 2, 6)

    spec = DeviceSpec(device="w_pcmo", seed=1, readout_bandwidth=1e6)
    emp = spec.build()
    assert isinstance(emp, EmpiricalBackend)
    assert emp.readout_bandwidth == 1e6 and emp.seed == 1
    assert emp.state_for_weight == {1: "LRS", 0: "pristine"}
    assert spec.build(seed=5).seed == 5
    hw = spec.hardware_settings(8, 2, 6, backend=emp)
    assert hw.hardware_input_vmax == pytest.approx(128 * 0.2)
    assert hw.hardware_output_current_scaling == pytest.approx(
        emp.suggested_current_scaling(hw.hardware_input_vmax)
    )
    hw2 = DeviceSpec(device="w_pcmo", pulses_full_scale=16).hardware_settings(
        8, 2, 6, backend=emp
    )
    assert hw2.hardware_input_vmax == pytest.approx(16 * 0.2)


def test_spec_from_npz_path(tmp_path):
    ds = generate_synthetic_dataset(6, 2, seed=0)
    path = tmp_path / "syn.npz"
    ds.save(str(path))
    spec = DeviceSpec(
        device=str(path),
        read_voltage=0.2,
        on_state="LRS",
        off_state="HRS",
        device_indices=[0, 1, 4],
    )
    backend = spec.build(seed=0)
    assert backend.dataset.n_devices == 3
    assert np.sign(backend.read_voltage) == backend.dataset.read_polarity
