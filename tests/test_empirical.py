"""
Tests for the measurement-based device description
(``synaptogen_ml.empirical``): the ``IVDataset`` container, the loop /
Keithley adapters, the synthetic Synaptogen generator and the fit helpers.
"""

import os

import numpy as np
import pytest

import synaptogen_ml.synaptogen as syn
from synaptogen_ml.empirical import (
    IVDataset,
    from_hysteresis_loops,
    from_state_sweeps,
    segment_loop,
)
from synaptogen_ml.empirical.adapters import from_keithley_iv_dat
from synaptogen_ml.empirical.adapters.keithley import load_table, split_devices
from synaptogen_ml.empirical.fit import (
    check_fit,
    eval_poly,
    fit_poly_zero_intercept,
    fit_quality,
    to_polyval_order,
)
from synaptogen_ml.empirical.synthetic import (
    generate_synthetic_dataset,
    synaptogen_loops,
)

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA = os.path.join(REPO, "data", "measurements")
META = {"device_family": "unit-test", "read_polarity": 1}


@pytest.fixture(scope="module")
def synthetic():
    return generate_synthetic_dataset(64, 4, seed=0)


def _toy(n_devices=3, n_cycles=2):
    U = np.linspace(-0.5, 0.5, 11, dtype=np.float32)
    g = np.arange(1, n_devices + 1, dtype=np.float32)[:, None, None] * 1e-6
    I = np.stack([g * U, 2 * g * U], axis=2)  # [D, 1, S, P]
    I = np.repeat(I, n_cycles, axis=1)
    return IVDataset(
        U=U, I=I, states=["HRS", "LRS"], cycle_index=range(n_cycles), meta=META
    )


# --------------------------------------------------------------- dataset ---
@pytest.mark.empirical
def test_dataset_validation():
    ds = _toy()
    assert (ds.n_devices, ds.n_cycles, ds.n_states, ds.n_points) == (3, 2, 2, 11)
    with pytest.raises(ValueError):
        IVDataset(
            U=ds.U[::-1],
            I=ds.I,
            states=ds.states,
            cycle_index=ds.cycle_index,
            meta=META,
        )
    with pytest.raises(ValueError):
        IVDataset(
            U=ds.U,
            I=ds.I[..., :-1],
            states=ds.states,
            cycle_index=ds.cycle_index,
            meta=META,
        )
    with pytest.raises(ValueError):
        IVDataset(
            U=ds.U, I=ds.I, states=["a", "a"], cycle_index=ds.cycle_index, meta=META
        )
    with pytest.raises(ValueError):
        IVDataset(U=ds.U, I=ds.I, states=ds.states, cycle_index=ds.cycle_index, meta={})
    with pytest.raises(KeyError):
        ds.state_index("nope")


@pytest.mark.empirical
def test_dataset_save_load_round_trip(tmp_path):
    ds = _toy()
    ds.I[0, 0, 0, 3] = np.nan
    ds.meta["device_ids"] = ["a", "b", "c"]
    path = tmp_path / "toy.npz"
    ds.save(str(path))
    back = IVDataset.load(str(path))
    assert np.array_equal(back.U, ds.U)
    assert np.array_equal(back.I, ds.I, equal_nan=True)
    assert back.states == ds.states
    assert np.array_equal(back.cycle_index, ds.cycle_index)
    assert back.meta == ds.meta


@pytest.mark.empirical
def test_dataset_queries_and_nan_handling():
    ds = _toy()
    g = ds.current_at(0.25, "LRS")  # [D, C]
    assert g.shape == (3, 2)
    np.testing.assert_allclose(g[:, 0], 2 * np.arange(1, 4) * 1e-6 * 0.25, rtol=1e-5)
    np.testing.assert_allclose(
        ds.conductance_at(0.25, "LRS")[:, 0], 2 * np.arange(1, 4) * 1e-6, rtol=1e-5
    )
    med = ds.median_curve("HRS")
    np.testing.assert_allclose(med, 2e-6 * ds.U, atol=1e-12)

    # a curve measured only on the negative half: NaN elsewhere, finite inside
    ds.I[0, :, :, ds.U > 0] = np.nan
    assert np.isnan(ds.current_at(0.25, "LRS")[0]).all()
    assert np.isfinite(ds.current_at(-0.25, "LRS")[0]).all()
    assert ds.coverage("LRS") == (-0.5, 0.0)
    assert ds.coverage("LRS", fraction=0.5) == (-0.5, 0.5)
    # resampling keeps NaN outside the support and interpolates inside
    rs = ds.resampled(np.linspace(-0.45, 0.45, 7))
    assert rs.n_points == 7
    assert np.isnan(rs.I[0, 0, 0, rs.U > 0]).all()
    np.testing.assert_allclose(rs.I[1, 0, 1], 2 * 2e-6 * rs.U, rtol=1e-5, atol=1e-15)


@pytest.mark.empirical
def test_dataset_select_and_concatenate():
    a, b = _toy(), _toy(n_devices=2)
    a.meta["device_ids"] = [10, 11, 12]
    b.meta["device_ids"] = [20, 21]
    sub = a.select_devices([2, 0])
    assert sub.n_devices == 2 and sub.meta["device_ids"] == [12, 10]
    np.testing.assert_array_equal(sub.I[0], a.I[2])
    both = IVDataset.concatenate([a, b])
    assert both.n_devices == 5 and both.meta["device_ids"] == [10, 11, 12, 20, 21]
    cyc = a.select_cycles([True, False])
    assert cyc.n_cycles == 1 and cyc.cycle_index.tolist() == [0]

    other = _toy()
    other.meta["device_family"] = "something-else"
    with pytest.raises(ValueError, match="device families"):
        IVDataset.concatenate([a, other])
    forced = IVDataset.concatenate([a, other], force=True)
    assert forced.device_family == "something-else+unit-test"


@pytest.mark.empirical
def test_from_state_sweeps():
    U = np.linspace(-0.3, 0.3, 7)
    ds = from_state_sweeps(
        U, {"HRS": np.ones((4, 7)), "LRS": 2 * np.ones((4, 7))}, meta=META
    )
    assert ds.I.shape == (4, 1, 2, 7) and ds.states == ["HRS", "LRS"]
    with pytest.raises(ValueError):
        from_state_sweeps(
            U, {"HRS": np.ones((4, 7)), "LRS": np.ones((3, 7))}, meta=META
        )


# -------------------------------------------------------------- adapters ---
@pytest.mark.empirical
def test_segment_loop_negative_and_positive_first():
    u = np.concatenate(
        [np.linspace(0, -1, 5), np.linspace(-1, 1, 9), np.linspace(1, 0, 5)]
    )
    segs = segment_loop(u, u**2)
    assert [s[0][0] for s in segs] == [0.0, -1.0, 1.0]
    assert [s[0][-1] for s in segs] == [-1.0, 1.0, 0.0]
    segs_pos = segment_loop(-u, u**2)
    assert [s[0][0] for s in segs_pos] == [0.0, 1.0, -1.0]


@pytest.mark.empirical
def test_hysteresis_adapter_on_synaptogen_loops():
    loops = synaptogen_loops(8, seed=1)
    ds = from_hysteresis_loops(loops, state_names=["initial", "LRS", "HRS"], meta=META)
    assert (
        ds.n_devices == 8
        and ds.n_cycles == 1
        and ds.states == ["initial", "LRS", "HRS"]
    )
    # ramp 2 (after the -2 V SET) must conduct more than ramp 3 (after the +2 V
    # RESET) at the Synaptogen read voltage; both are measured there
    lrs = ds.current_at(0.2, "LRS")[:, 0]
    hrs = ds.current_at(0.2, "HRS")[:, 0]
    assert np.isfinite(lrs).all() and np.isfinite(hrs).all()
    assert (lrs > hrs).all()
    # the initial ramp never reaches positive voltages
    assert np.isnan(ds.current_at(0.2, "initial")).all()


@pytest.mark.empirical
@pytest.mark.parametrize(
    "name, family, n_devices, lrs_ohm",
    [
        ("W_WOx_PCMO.dat", "W/WOx/PCMO", 76, 1.7e9),
        ("Ta_WOx_PCMO.dat", "Ta/WOx/PCMO", 12, 0.58e9),
    ],
)
def test_keithley_adapter_on_partner_files(name, family, n_devices, lrs_ohm):
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        pytest.skip("partner data not available")
    assert len(split_devices(load_table(path))) == n_devices
    ds = from_keithley_iv_dat(path, device_family=family)
    assert ds.n_devices == n_devices and ds.n_cycles == 1
    assert ds.states == ["pristine", "LRS", "post_reset_pos"]
    assert ds.read_polarity == -1 and ds.device_family == family
    assert ds.meta["device_ids"] == list(range(n_devices))
    assert "Date/Time" in ds.meta["instrument_header"]
    # the pristine ramp is negative-only, the post-reset ramp positive-only
    assert ds.coverage("pristine")[1] <= 0.0 and ds.coverage("post_reset_pos")[0] >= 0.0
    # numbers from data/measurements/README.md (median LRS resistance at -0.2 V,
    # on/off ratio about 14x)
    lrs = ds.current_at(-0.2, "LRS")[:, 0]
    pristine = ds.current_at(-0.2, "pristine")[:, 0]
    assert np.isfinite(lrs).all() and (lrs < 0).all()
    assert abs(0.2 / np.median(-lrs) / lrs_ohm - 1) < 0.1
    assert 10 < np.median(lrs / pristine) < 20
    # sub-population selection keeps file order ids
    sub = from_keithley_iv_dat(path, device_family=family, device_indices=range(3))
    assert sub.n_devices == 3 and sub.meta["device_ids"] == [0, 1, 2]
    np.testing.assert_array_equal(
        sub.I, ds.I[:3], err_msg="selection must not resample"
    )


# ------------------------------------------------------------- synthetic ---
@pytest.mark.empirical
def test_synthetic_dataset_shape_states_and_rng_isolation(synthetic):
    assert synthetic.I.shape == (64, 4, 2, 141)
    assert synthetic.states == ["HRS", "LRS"] and synthetic.read_polarity == 1
    assert synthetic.cycle_index.tolist() == [0, 1, 2, 3]
    assert np.isfinite(synthetic.I).all()
    # I(0) = 0 for the Synaptogen polynomials
    assert np.abs(synthetic.I[..., 70]).max() < 1e-12
    lrs = synthetic.current_at(0.2, "LRS")
    hrs = synthetic.current_at(0.2, "HRS")
    assert (lrs > hrs).all()
    # Synaptogen LRS/HRS at 0.2 V are of order 1e-4 / 1e-6 A
    assert 1e-5 < np.median(lrs) < 1e-3 and 1e-8 < np.median(hrs) < 1e-4

    # deterministic under seed and does not advance the module RNG
    state_before = syn.rng.bit_generator.state
    again = generate_synthetic_dataset(64, 4, seed=0)
    assert syn.rng.bit_generator.state == state_before
    assert np.array_equal(again.I, synthetic.I)
    other = generate_synthetic_dataset(64, 4, seed=1)
    assert not np.array_equal(other.I, synthetic.I)


@pytest.mark.empirical
def test_synthetic_wear_cycles_shift_cycle_index():
    ds = generate_synthetic_dataset(4, 2, wear_cycles=1, seed=3)
    assert ds.cycle_index.tolist() == [1, 2]
    noisy = generate_synthetic_dataset(4, 1, read_noise=True, seed=3)
    assert np.isfinite(noisy.I).all()


@pytest.mark.empirical
def test_synthetic_cli_round_trip(tmp_path):
    from synaptogen_ml.empirical.synthetic import main

    out = tmp_path / "syn.npz"
    main([str(out), "--devices", "8", "--cycles", "2", "--seed", "0", "--points", "15"])
    ds = IVDataset.load(str(out))
    assert ds.I.shape == (8, 2, 2, 15) and ds.meta["seed"] == 0


# ------------------------------------------------------------------- fit ---
@pytest.mark.empirical
def test_fit_zero_intercept_on_synaptogen_median_curves(synthetic):
    for state in ("HRS", "LRS"):
        med = synthetic.median_curve(state)
        coeffs = fit_poly_zero_intercept(synthetic.U, med, degree=7)
        assert coeffs[0] == 0.0 and coeffs.shape == (8,)
        assert fit_quality(coeffs, synthetic.U, med) < 1e-3
        check_fit(coeffs, 0.2, expected_sign=+1)
        with pytest.raises(ValueError):
            check_fit(coeffs, 0.2, expected_sign=-1)
    # polyval-order round trip matches numpy
    u = np.linspace(-0.5, 0.5, 5)
    np.testing.assert_allclose(
        eval_poly(coeffs, u), np.polyval(to_polyval_order(coeffs), u)
    )


@pytest.mark.empirical
def test_fit_window_weights_and_nan():
    U = np.linspace(-1, 1, 41)
    I = 3 * U + 2 * U**3
    I[5] = np.nan
    coeffs = fit_poly_zero_intercept(
        U, I, degree=3, window=(-0.5, 0.5), weights=np.ones_like(U)
    )
    np.testing.assert_allclose(coeffs, [0, 3, 0, 2], atol=1e-9)
    with pytest.raises(ValueError):
        fit_poly_zero_intercept(U[:2], I[:2], degree=5)
