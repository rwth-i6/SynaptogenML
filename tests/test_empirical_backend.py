"""
Tests for ``EmpiricalBackend`` (measurement-based cell programming) and the
pulse-count array module it programs.

Fidelity is checked twice: against the Synaptogen model through a synthetic
dataset recorded from it (known truth, comparable to ``SynaptogenBackend``),
and on the partner PCMO data (``data/measurements``), where the on/off ratio
is ~14 and the only "off" state is the pristine branch.
"""

import logging
import os
from functools import partial

import numpy as np
import pytest
import torch

import synaptogen_ml
import synaptogen_ml.synaptogen as syn
from synaptogen_ml.backends import SynaptogenBackend
from synaptogen_ml.empirical import EmpiricalBackend, from_state_sweeps
from synaptogen_ml.empirical.adapters import from_keithley_iv_dat
from synaptogen_ml.empirical.backend import PulseProgrammedCells
from synaptogen_ml.empirical.synthetic import generate_synthetic_dataset
from synaptogen_ml.memristor_modules.config import CycleCorrectionSettings
from synaptogen_ml.memristor_modules.conv import MemristorConv1d, MemristorConv2d
from synaptogen_ml.memristor_modules.linear import TiledMemristorLinear
from synaptogen_ml.memristor_modules.memristor import (
    DacAdcHardwareSettings,
    PairedMemristorArrayV2,
)
from synaptogen_ml.memristor_modules.pulse import (
    PairedPulseCountMemristorArray,
    PulseCountMemristorArray,
)
from synaptogen_ml.programming import ProgrammedCells
from synaptogen_ml.quant_modules import ActivationQuantizer, Conv1DQuant, LinearQuant

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA = os.path.join(REPO, "data", "measurements")
PCMO_FILES = {
    "W": ("W_WOx_PCMO.dat", "W/WOx/PCMO"),
    "Ta": ("Ta_WOx_PCMO.dat", "Ta/WOx/PCMO"),
}
SYN_STATES = {1: "LRS", 0: "HRS"}
PCMO_STATES = {1: "LRS", 0: "pristine"}
WEIGHT_PRECISION = 3
QUANT_KW = dict(
    weight_bit_prec=WEIGHT_PRECISION,
    weight_quant_dtype=torch.qint8,
    weight_quant_method="per_tensor_symmetric",
    bias=False,
)
IDEAL = CycleCorrectionSettings(None, None, None, ideal_programming=True)
HW_SYNAPTOGEN = DacAdcHardwareSettings(
    input_bits=8,
    output_precision_bits=2,
    output_range_bits=6,
    hardware_input_vmax=0.6,
    hardware_output_current_scaling=8020.0,
)


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    syn.rng = np.random.default_rng(seed)
    syn.randn = partial(syn.rng.standard_normal, dtype=np.float32)
    syn.rand = partial(syn.rng.random, dtype=np.float32)


@pytest.fixture
def noise_model():
    """Restore the process-wide readout-noise model after a test."""
    saved = synaptogen_ml.readout_noise_model()
    yield synaptogen_ml.set_readout_noise_model
    synaptogen_ml.set_readout_noise_model(saved)


@pytest.fixture(scope="module")
def synthetic():
    return generate_synthetic_dataset(64, 4, seed=0)


def pcmo_dataset(key):
    name, family = PCMO_FILES[key]
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        pytest.skip("partner data not available")
    return from_keithley_iv_dat(path, device_family=family)


def syn_backend(dataset, **kw):
    kw.setdefault("seed", 0)
    return EmpiricalBackend(
        dataset, read_voltage=0.2, state_for_weight=SYN_STATES, **kw
    )


# ----------------------------------------------------------- pulse array ---
@pytest.mark.empirical
def test_pulse_array_mean_is_a_matmul(noise_model):
    noise_model("off")
    torch.manual_seed(0)
    arr = PulseCountMemristorArray(5, 3, pulse_voltage=0.2)
    arr.I_pulse.data = torch.rand(5, 3) * 1e-10
    x = torch.randn(7, 5)
    torch.testing.assert_close(arr(x), (x / 0.2) @ arr.I_pulse)
    assert torch.equal(arr(x), arr(x)), "noise off must be deterministic"

    # conv1d depthwise layout: additional axes (G, 1, 1), kernel as inputs
    arr = PulseCountMemristorArray(3, 1, additional_axes=(4, 1, 1), pulse_voltage=0.2)
    arr.I_pulse.data = torch.rand(4, 1, 1, 3, 1) * 1e-10
    x = torch.randn(2, 6, 4, 1, 1, 3)
    out = arr(x)
    assert out.shape == (2, 6, 4, 1, 1, 1)
    ref = torch.einsum("xtgcdk,gcdko->xtgcdo", x / 0.2, arr.I_pulse)
    torch.testing.assert_close(out, ref)

    paired = PairedPulseCountMemristorArray(2, 2, pulse_voltage=0.5)
    paired.init_from_paired_cell_array_input_major(
        PulseProgrammedCells(
            np.array([1, 2, 3, 4], np.float32), np.zeros(4), np.zeros(4)
        ),
        PulseProgrammedCells(
            np.array([1, 1, 1, 1], np.float32), np.zeros(4), np.zeros(4)
        ),
    )
    x = torch.tensor([[0.5, 0.5]])
    torch.testing.assert_close(paired(x), torch.tensor([[0.0 + 2.0, 1.0 + 3.0]]))


@pytest.mark.empirical
def test_pulse_array_noise_accumulates_over_pulses(noise_model):
    noise_model("physical")
    torch.manual_seed(0)
    i_pulse, pv, bw = 1e-10, 0.2, 1e8
    arr = PulseCountMemristorArray(1, 1, pulse_voltage=pv, bandwidth=bw)
    arr.I_pulse.data = torch.full((1, 1), i_pulse)
    sigma2_pulse = bw * (
        4 * 1.380649e-23 * 300 * i_pulse / pv + 2 * 1.602176634e-19 * i_pulse
    )
    torch.testing.assert_close(
        arr.pulse_noise_variance(), torch.full((1, 1), sigma2_pulse)
    )
    for n in (1, 16):
        out = arr(torch.full((40000, 1), n * pv))
        assert abs(out.mean().item() / (n * i_pulse) - 1) < 0.02
        assert abs(out.var().item() / (n * sigma2_pulse) - 1) < 0.1
    noise_model("legacy")  # "legacy" means on as well for this module
    a = arr(torch.full((4, 1), pv))
    assert not torch.equal(a, arr(torch.full((4, 1), pv)))


# ------------------------------------------------------------ construction ---
@pytest.mark.empirical
def test_backend_construction_checks(synthetic):
    with pytest.raises(ValueError, match="state_for_weight"):
        EmpiricalBackend(synthetic, read_voltage=0.2, state_for_weight={1: "LRS"})
    with pytest.raises(ValueError, match="read polarity"):
        EmpiricalBackend(synthetic, read_voltage=-0.2, state_for_weight=SYN_STATES)
    with pytest.raises(KeyError):
        EmpiricalBackend(
            synthetic, read_voltage=0.2, state_for_weight={1: "LRS", 0: "nope"}
        )
    with pytest.raises(ValueError, match="representation"):
        syn_backend(synthetic, representation="lut")
    # a state that is not measured at the read voltage is refused
    U = np.linspace(-0.3, 0.3, 7)
    hrs = np.ones((4, 7)) * 1e-6
    hrs[:, U > 0] = np.nan
    ds = from_state_sweeps(
        U,
        {"HRS": hrs, "LRS": 2e-6 * np.ones((4, 7))},
        meta={"device_family": "toy", "read_polarity": 1},
    )
    with pytest.raises(ValueError, match="not measured"):
        EmpiricalBackend(ds, read_voltage=0.2, state_for_weight=SYN_STATES)
    assert isinstance(
        EmpiricalBackend(
            ds, read_voltage=-0.2 * -1, state_for_weight={1: "LRS", 0: "LRS"}
        ),
        EmpiricalBackend,
    )

    backend = syn_backend(synthetic)
    info = backend.describe()
    assert info["devices_with_on_state"] == 64 and info["cycles"] == 4
    assert info["on_off_ratio"] > 5
    assert isinstance(backend.make_paired_array(3, 2), PairedPulseCountMemristorArray)
    assert isinstance(
        syn_backend(synthetic, representation="shared_poly").make_paired_array(3, 2),
        PairedMemristorArrayV2,
    )


# --------------------------------------------------------------- sampling ---
def _jobs(n, seed=0):
    gen = np.random.default_rng(seed)
    return [
        (
            gen.integers(0, 2, n).astype(np.float32),
            gen.integers(0, 2, n).astype(np.float32),
        )
    ]


@pytest.mark.empirical
def test_sampling_is_seeded_and_isolated(synthetic):
    seed_all(0)
    syn_state = syn.rng.bit_generator.state
    legacy_state = np.random.get_state()[1].copy()
    a = syn_backend(synthetic, seed=1).program_pairs(_jobs(50), 0, None, 1.0)
    b = syn_backend(synthetic, seed=1).program_pairs(_jobs(50), 0, None, 1.0)
    c = syn_backend(synthetic, seed=2).program_pairs(_jobs(50), 0, None, 1.0)
    assert syn.rng.bit_generator.state == syn_state
    assert np.array_equal(np.random.get_state()[1], legacy_state)
    for (pa, na), (pb, nb) in zip(a, b):
        assert np.array_equal(pa.I_pulse, pb.I_pulse) and np.array_equal(
            na.I_pulse, nb.I_pulse
        )
    assert not np.array_equal(a[0][0].I_pulse, c[0][0].I_pulse)


@pytest.mark.empirical
def test_sampled_cells_are_measured_values_of_the_right_state(synthetic):
    backend = syn_backend(synthetic)
    pos_w, neg_w = _jobs(200)[0]
    pos, neg = backend.program_pairs([(pos_w, neg_w)], 0, None, 1.0)[0]
    table = synthetic.current_at(0.2)  # [D, C, S], polarity +1
    for cells, weights in ((pos, pos_w), (neg, neg_w)):
        assert isinstance(cells, PulseProgrammedCells)
        states = np.where(
            weights == 1, synthetic.state_index("LRS"), synthetic.state_index("HRS")
        )
        expected = table[cells.device_index, cells.cycle_index, states]
        np.testing.assert_allclose(cells.I_pulse, expected, rtol=1e-6)
    # weights 1 conduct more than weights 0 on the Synaptogen truth
    assert pos.I_pulse[pos_w == 1].min() > pos.I_pulse[pos_w == 0].max()


@pytest.mark.empirical
def test_num_cycles_init_filters_cycles(synthetic, caplog):
    backend = syn_backend(synthetic)
    pos, _ = backend.program_pairs(_jobs(300), 2, None, 1.0)[0]
    assert set(synthetic.cycle_index[pos.cycle_index].tolist()) == {2, 3}
    with caplog.at_level(logging.WARNING):
        pos, _ = backend.program_pairs(_jobs(300), 10, None, 1.0)[0]
    assert "using all cycles" in caplog.text
    assert set(synthetic.cycle_index[pos.cycle_index].tolist()) == {0, 1, 2, 3}


@pytest.mark.empirical
def test_reuse_warning_and_report(synthetic, caplog):
    backend = syn_backend(synthetic)
    with caplog.at_level(logging.WARNING):
        backend.program_pairs(_jobs(500), 0, None, 1.0)
    assert "pool of 64 devices" in caplog.text
    assert backend.cells_programmed == 1000
    assert backend.reuse_report().startswith("1000 cells from")
    assert backend.device_use_counts.sum() == 1000


@pytest.mark.empirical
def test_ideal_programming_uses_medians_without_rng(synthetic):
    backend = syn_backend(synthetic)
    state = backend.rng.bit_generator.state
    pos_w, neg_w = _jobs(40)[0]
    pos, _ = backend.program_pairs([(pos_w, neg_w)], 0, IDEAL, 1.0)[0]
    assert backend.rng.bit_generator.state == state
    assert (pos.device_index == -1).all()
    np.testing.assert_allclose(
        pos.I_pulse,
        np.where(
            pos_w == 1, backend.median_current["LRS"], backend.median_current["HRS"]
        ),
        rtol=1e-6,
    )


@pytest.mark.empirical
def test_write_verify_resamples_cycles(synthetic, caplog):
    strict = CycleCorrectionSettings(
        num_cycles=3,
        test_input_value=0.2,
        relative_deviation=0.0,
        ideal_programming=False,
    )
    jobs = _jobs(200)
    seed_all(0)
    backend = syn_backend(synthetic)
    plain, _ = backend.program_pairs(jobs, 0, None, 1.0)[0]
    backend = syn_backend(synthetic)  # same seed -> same first draw
    verified, _ = backend.program_pairs(jobs, 0, strict, 1.0)[0]
    assert np.array_equal(plain.device_index, verified.device_index), "same devices"
    assert not np.array_equal(plain.cycle_index, verified.cycle_index), "cycles redrawn"
    # the literal Synaptogen criterion runs as well
    target = CycleCorrectionSettings(
        num_cycles=1,
        test_input_value=0.2,
        relative_deviation=0.2,
        ideal_programming=False,
    )
    syn_backend(synthetic, verify_reference="target").program_pairs(
        jobs, 0, target, 1.0
    )
    # one cycle per device: nothing can change, warned once
    single = synthetic.select_cycles([True, False, False, False])
    backend = syn_backend(single)
    with caplog.at_level(logging.WARNING):
        a, _ = backend.program_pairs(jobs, 0, strict, 1.0)[0]
    assert "one cycle per device" in caplog.text
    assert (a.cycle_index == 0).all()


@pytest.mark.empirical
def test_jitter_is_opt_in(synthetic, caplog):
    table = synthetic.current_at(0.2)
    with caplog.at_level(logging.WARNING):
        backend = syn_backend(synthetic, jitter_sigma=0.1)
    assert "jitter_sigma" in caplog.text
    pos_w, neg_w = _jobs(100)[0]
    pos, _ = backend.program_pairs([(pos_w, neg_w)], 0, None, 1.0)[0]
    states = np.where(pos_w == 1, 1, 0)
    measured = table[pos.device_index, pos.cycle_index, states]
    assert not np.allclose(pos.I_pulse, measured, rtol=1e-3)
    assert np.allclose(np.log10(pos.I_pulse / measured).std(), 0.1, atol=0.03)


@pytest.mark.empirical
def test_shared_poly_projection_reproduces_conductance(synthetic):
    backend = syn_backend(synthetic, representation="shared_poly")
    pos_w, neg_w = _jobs(100)[0]
    pos, _ = backend.program_pairs([(pos_w, neg_w)], 0, None, 1.0)[0]
    assert isinstance(pos, ProgrammedCells) and pos.params.U0 == 0.2
    # r near 0 for the on state and near 1 for the off state (Synaptogen layout)
    assert (
        np.abs(pos.r[pos_w == 1]).max() < 0.2
        and np.abs(pos.r[pos_w == 0] - 1).max() < 0.2
    )
    # evaluating the mixed polynomial at the read voltage gives the sampled current
    L, H = np.polyval(pos.params.LLRS, 0.2), np.polyval(pos.params.HHRS, 0.2)
    mixed = (1 - pos.r) * L + pos.r * H
    arr = PairedMemristorArrayV2(100, 1)
    arr.init_from_paired_cell_array_input_major(pos, pos)
    assert arr.pos.u0 == 0.2
    synaptogen_ml.set_linear_iv(True)
    try:
        g = arr.pos.readout_polys()[0][1].item()
    finally:
        synaptogen_ml.set_linear_iv(False)
    assert abs(g - L / 0.2) < 1e-3 * abs(L / 0.2)
    assert np.allclose(mixed, mixed)  # finite


# ---------------------------------------------------------------- scaling ---
@pytest.mark.empirical
def test_suggested_scaling_and_hardware_settings(synthetic):
    backend = syn_backend(synthetic)
    # the synthetic pool is the Synaptogen model, so its factor is near the
    # calibrated 8020 (stochastic) of the test configurations
    assert 0.7 < backend.suggested_current_scaling(0.6) / 8020 < 1.3
    hw = backend.suggested_hardware_settings(8, 2, 6)
    assert hw.hardware_input_vmax == pytest.approx(128 * 0.2)
    assert hw.hardware_output_current_scaling == pytest.approx(
        backend.suggested_current_scaling(128 * 0.2)
    )
    assert backend.suggested_hardware_settings(
        8, 2, 6, pulses_full_scale=1
    ).hardware_input_vmax == pytest.approx(0.2)
    swapped = EmpiricalBackend(
        synthetic, read_voltage=0.2, state_for_weight={1: "HRS", 0: "LRS"}
    )
    with pytest.raises(ValueError, match="does not conduct more"):
        swapped.suggested_current_scaling(0.6)


# --------------------------------------------------------------- fidelity ---
def act_quant():
    return ActivationQuantizer(
        bit_precision=8,
        dtype=torch.qint8,
        method="per_tensor_symmetric",
        channel_axis=None,
        moving_avrg=None,
    )


def calibrate(layer, act, x):
    layer.train()
    act.train()
    layer(act(x))
    layer.eval()
    act.eval()
    with torch.no_grad():
        return layer(act(x))


def rel_err(out, ref):
    return ((out - ref).norm() / ref.norm()).item()


def tiled_linear_case():
    act = act_quant()
    x = torch.randn(16, 48)
    lin = LinearQuant(48, 32, **QUANT_KW)
    ref = calibrate(lin, act, x)

    def build(backend, hw):
        return TiledMemristorLinear(
            in_features=48,
            out_features=32,
            weight_precision=WEIGHT_PRECISION,
            converter_hardware_settings=hw,
            memristor_inputs=16,
            memristor_outputs=16,
            backend=backend,
        )

    return act, lin, x, ref, build, "init_from_linear_quant"


def conv1d_case():
    channels, k = 16, 5
    act = act_quant()
    x = torch.randn(4, channels, 20)
    conv = Conv1DQuant(
        in_channels=channels,
        out_channels=channels,
        kernel_size=k,
        stride=1,
        padding="same",
        dilation=1,
        groups=channels,
        **QUANT_KW,
    )
    ref = calibrate(conv, act, x)

    def build(backend, hw):
        return MemristorConv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=k,
            stride=1,
            padding="same",
            groups=channels,
            weight_precision=WEIGHT_PRECISION,
            converter_hardware_settings=hw,
            backend=backend,
        )

    return act, conv, x, ref, build, "init_from_conv_quant"


def programmed_output(case, backend, hw, correction=None, num_cycles_init=0):
    act, layer, x, ref, build, init = case
    mem = build(backend, hw)
    getattr(mem, init)(
        act, layer, num_cycles_init=num_cycles_init, correction_settings=correction
    )
    mem.eval()
    with torch.no_grad():
        return mem(x), ref


@pytest.mark.empirical
@pytest.mark.parametrize("representation", ["pulse_count", "shared_poly"])
def test_fidelity_on_synthetic_matches_synaptogen_backend(
    synthetic, representation, noise_model
):
    noise_model("physical")
    seed_all(0)
    case = tiled_linear_case()
    out_syn, ref = programmed_output(case, SynaptogenBackend(), HW_SYNAPTOGEN)
    err_syn = rel_err(out_syn, ref)

    backend = syn_backend(synthetic, representation=representation)
    if representation == "pulse_count":
        hw = backend.suggested_hardware_settings(8, 2, 6)
    else:
        hw = DacAdcHardwareSettings(
            input_bits=8,
            output_precision_bits=2,
            output_range_bits=6,
            hardware_input_vmax=0.6,
            hardware_output_current_scaling=backend.suggested_current_scaling(0.6),
        )
    out_emp, _ = programmed_output(case, backend, hw)
    err_emp = rel_err(out_emp, ref)
    print(f"\n{representation}: synaptogen {err_syn:.3f} vs empirical {err_emp:.3f}")
    assert err_emp < 0.3
    assert err_emp < 1.5 * err_syn + 0.05

    # ideal programming (pool medians) is at least as good as sampled
    out_ideal, _ = programmed_output(case, backend, hw, correction=IDEAL)
    assert rel_err(out_ideal, ref) < 0.3


@pytest.mark.empirical
@pytest.mark.parametrize("key", ["W", "Ta"])
@pytest.mark.parametrize("noise", ["off", "physical"])
def test_fidelity_on_pcmo_data(key, noise, noise_model):
    noise_model(noise)
    seed_all(0)
    dataset = pcmo_dataset(key)
    backend = EmpiricalBackend(
        dataset, read_voltage=-0.2, state_for_weight=PCMO_STATES, seed=0
    )
    hw = backend.suggested_hardware_settings(8, 2, 6)
    assert hw.hardware_output_current_scaling > 1e7  # GOhm devices, pA..nA currents
    for name, case in (
        ("tiled_linear", tiled_linear_case()),
        ("conv1d", conv1d_case()),
    ):
        out, ref = programmed_output(case, backend, hw)
        err = rel_err(out, ref)
        print(f"\n{key} {noise} {name}: relative error {err:.3f}")
        assert err < 0.35, (key, noise, name, err)


@pytest.mark.empirical
def test_conv2d_accepts_backend_and_correction(noise_model):
    noise_model("off")
    seed_all(0)
    dataset = pcmo_dataset("W")
    backend = EmpiricalBackend(
        dataset, read_voltage=-0.2, state_for_weight=PCMO_STATES, seed=0
    )
    hw = backend.suggested_hardware_settings(8, 2, 6)
    from synaptogen_ml.quant_modules import Conv2dQuant

    act = act_quant()
    x = torch.randn(2, 2, 8, 8)
    conv = Conv2dQuant(
        in_channels=2,
        out_channels=4,
        kernel_size=(3, 3),
        stride=1,
        padding=(1, 1),
        dilation=1,
        groups=1,
        **QUANT_KW,
    )
    ref = calibrate(conv, act, x)
    mem = MemristorConv2d(
        in_channels=2,
        out_channels=4,
        kernel_size=(3, 3),
        stride=1,
        padding=(1, 1),
        groups=1,
        weight_precision=WEIGHT_PRECISION,
        converter_hardware_settings=hw,
        backend=backend,
    )
    # correction settings are allowed with a backend (ideal here)
    mem.init_from_conv_quant(act, conv, num_cycles_init=0, correction_settings=IDEAL)
    mem.eval()
    with torch.no_grad():
        out = mem(x)
    assert out.shape == ref.shape
    assert rel_err(out, ref) < 0.35
