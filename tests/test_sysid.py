"""System identification: recordings, excitation, the fit, fitted configs, resume."""

from __future__ import annotations

import dataclasses
import importlib.util
import math
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest
import torch

from n_cartpole.config import load_tqc_config, replace_plant
from n_cartpole.env.dynamics import PhysicsParams
from n_cartpole.env.randomization import PlantRandomization
from n_cartpole.sysid import recording as rec_mod
from n_cartpole.sysid.config_out import dump_toml, fitted_config, write_fitted_config
from n_cartpole.sysid.excite import ExciteParams, excitation_signal
from n_cartpole.sysid.fit import (
    FitOptions,
    FitResult,
    actuator_force,
    consistency_warnings,
    fit_recordings,
)
from n_cartpole.sysid.recording import Recording, make_recording
from n_cartpole.sysid.report import write_report
from n_cartpole.sysid.synth import simulate_recording
from tests.firmware_util import REPO, RIG_TOML, make_checkpoint, rig_env

DT = 0.005


# --- recordings -------------------------------------------------------------------


def _synthetic_recording(n: int = 2, N: int = 50) -> Recording:
    rng = np.random.default_rng(0)
    rec = make_recording(
        DT,
        rng.normal(size=(N, n + 1)),
        rng.normal(size=N),
        armed=np.arange(N) % 3 != 0,
        excite=np.where(np.arange(N) > 10, 1.5, 0.0),
        truth=rng.normal(size=(N, 2 + 2 * n)),
        est=rng.normal(size=(N, 2 + 2 * n)),
        note="unit test",
    )
    rec.excite_kind = "prbs"
    rec.excite_params = {"amp": 2.0, "period": 0.05, "seed": 3.0}
    rec.model_label = "tiny@1"
    rec.fw_version = "0.1.0"
    return rec


def test_recording_layout_matches_header() -> None:
    """256-byte header, 128-byte samples, offsets as in ncp_record.h."""
    assert rec_mod.HEADER_DTYPE.itemsize == 256
    assert rec_mod.SAMPLE_DTYPE.itemsize == 128
    assert rec_mod.HEADER_DTYPE.fields is not None
    assert rec_mod.SAMPLE_DTYPE.fields is not None
    assert rec_mod.HEADER_DTYPE.fields["dt"][1] == 24
    assert rec_mod.HEADER_DTYPE.fields["start_us"][1] == 60
    assert rec_mod.HEADER_DTYPE.fields["note"][1] == 164
    assert rec_mod.SAMPLE_DTYPE.fields["cmd"][1] == 28
    assert rec_mod.SAMPLE_DTYPE.fields["truth"][1] == 80
    assert rec_mod.SAMPLE_DTYPE.fields["flags"][1] == 123


def test_recording_round_trip(tmp_path: Path) -> None:
    """Write -> read keeps every field (float32 samples)."""
    rec = _synthetic_recording()
    path = tmp_path / "r.ncpr"
    rec.write(path)
    assert path.stat().st_size == 256 + 128 * len(rec)
    back = Recording.read(path)
    assert back.to_bytes() == path.read_bytes()
    assert (back.n_links, back.dt, len(back)) == (2, DT, 50)
    for k in ("pos", "cmd", "excite", "est", "truth"):
        np.testing.assert_allclose(
            getattr(back, k), getattr(rec, k), rtol=1e-6, atol=1e-6
        )
    np.testing.assert_array_equal(back.armed, rec.armed)
    np.testing.assert_allclose(back.t, rec.t, atol=1e-6)
    assert back.has_truth and back.excite_kind == "prbs"
    assert back.excite_params["amp"] == 2.0 and back.excite_params["f0"] == 0.0
    assert (back.note, back.model_label, back.fw_version) == (
        "unit test",
        "tiny@1",
        "0.1.0",
    )
    assert back.source == str(path)
    assert not make_recording(DT, np.zeros((3, 2)), np.zeros(3)).has_truth


def test_recording_segments_and_trim() -> None:
    """Tick gaps and encoder errors split segments; trimming keeps a time range."""
    rec = _synthetic_recording(N=40)
    rec.tick = np.concatenate([np.arange(20), np.arange(25, 45)])
    rec.flags[30] |= rec_mod.FLAG_ENC_ERR
    assert rec.segments() == [(0, 20), (20, 30), (31, 40)]
    part = rec.trimmed(0.02, 0.1)
    assert len(part) == 16 and part.t[0] == pytest.approx(0.02)
    assert part.note == rec.note
    rec.flags[39] |= rec_mod.FLAG_ENC_ERR
    assert rec.segments()[-1] == (31, 39)


def test_recording_rejects_bad_files() -> None:
    """Magic, version, sample size, truncation and link count are checked."""
    data = bytearray(_synthetic_recording(N=4).to_bytes())
    with pytest.raises(ValueError, match="too short"):
        Recording.from_bytes(bytes(data[:100]))
    with pytest.raises(ValueError, match="magic"):
        Recording.from_bytes(b"XXXX" + bytes(data[4:]))
    for off, val, msg in (
        (4, 7, "version"),
        (8, 64, "sample size"),
        (32, 9, "n_links"),
    ):
        bad = bytearray(data)
        bad[off] = val
        with pytest.raises(ValueError, match=msg):
            Recording.from_bytes(bytes(bad))
    with pytest.raises(ValueError, match="truncated"):
        Recording.from_bytes(bytes(data[:-10]))


# --- excitation -------------------------------------------------------------------


def test_excitation_kinds() -> None:
    """Chirp sweeps up continuously, PRBS/steps hold, everything stops at duration."""
    chirp = np.array(
        excitation_signal(ExciteParams("chirp", 3, 1, 8, duration=2), DT, 500)
    )
    assert chirp[0] == 0.0 and np.all(chirp[400:] == 0.0)
    assert np.max(np.abs(chirp)) == pytest.approx(3.0, abs=0.01)
    assert np.max(np.abs(np.diff(chirp[:400]))) < 2 * np.pi * 8 * 3 * DT * 1.01
    zc = np.flatnonzero(np.diff(np.sign(chirp[1:400])) != 0)
    assert np.diff(zc)[0] > 3 * np.diff(zc)[-1]  # frequency rises
    prbs = np.array(
        excitation_signal(ExciteParams("prbs", 2, period=0.05, seed=4), DT, 400)
    )
    assert set(np.unique(prbs)) == {-2.0, 2.0}
    assert np.all(prbs.reshape(-1, 10) == prbs.reshape(-1, 10)[:, :1])
    steps = excitation_signal(ExciteParams("steps", 4, period=0.01), DT, 9)
    assert steps == [4, 4, 0, 0, -4, -4, 0, 0, 4]
    sine = excitation_signal(ExciteParams("sine", 1, f0=50), DT, 3)
    assert sine == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)
    assert excitation_signal(ExciteParams("none"), DT, 3) == [0, 0, 0]
    lin = excitation_signal(ExciteParams("chirp", 1, 0.0, 10, duration=1), DT, 5)
    assert lin[0] == 0.0 and lin[4] > 0.0
    with pytest.raises(ValueError, match="kind"):
        ExciteParams("square")


# --- actuator + fit ----------------------------------------------------------------


def test_actuator_force_chain() -> None:
    """Delay, then slew vs the last applied force; disarmed ticks give 0."""
    cmd = np.array([0.0, 5.0, 5.0, 5.0, -5.0, -5.0])
    out = actuator_force(cmd, np.ones(6, bool), 1, 300.0, DT)
    np.testing.assert_allclose(out, [0, 0, 1.5, 3.0, 4.5, 3.0])
    out = actuator_force(cmd, np.array([1, 1, 0, 1, 1, 1], bool), 0, None, DT)
    np.testing.assert_allclose(out, [0, 5, 0, 5, -5, -5])


def _true_plant(p: PhysicsParams) -> PhysicsParams:
    return dataclasses.replace(
        p,
        M=p.M * 1.08,
        b=p.b * 1.5,
        cart_coulomb=p.cart_coulomb * 0.7,
        com=tuple(p.link_coms(2) * [0.95, 1.04]),
        inertia=tuple(p.link_inertias(2) * [1.2, 0.85]),
        joint_friction=(0.004, 0.0012),
    )


@pytest.fixture(scope="module")
def experiment(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[list[Path], PhysicsParams]:
    """Simulate a chirp (armed, manual PD) and a ring-down of a perturbed rig."""
    p = rig_env().physics
    true = _true_plant(p)
    d = tmp_path_factory.mktemp("sysid")
    chirp = simulate_recording(
        true, 2, 4.0, excite=ExciteParams("chirp", 3.0, 0.3, 6.0, duration=4.0), seed=1
    )
    lifted = np.array([0.0, 0.0, np.pi - 1.0, 0.0, np.pi + 0.8, 0.0])
    ring = simulate_recording(true, 2, 3.0, armed=False, start=lifted, seed=2)
    paths = [d / "chirp.ncpr", d / "ring.ncpr"]
    chirp.write(paths[0])
    ring.write(paths[1])
    return paths, true


@pytest.fixture(scope="module")
def fitted(experiment: tuple[list[Path], PhysicsParams]) -> FitResult:
    """Fit the synthetic experiment against the rig config."""
    paths, _ = experiment
    env = rig_env()
    recs = [Recording.read(p) for p in paths]
    return fit_recordings(
        recs,
        env.physics,
        2,
        env.force_slew,
        nominal_delay=1,
        opts=FitOptions(delays=[0, 1, 2], substeps=1),
    )


def test_fit_recovers_known_plant(
    fitted: FitResult, experiment: tuple[list[Path], PhysicsParams]
) -> None:
    """Every perturbed parameter is recovered, far better than the config's value."""
    _, true = experiment
    truth = {
        "M": true.M,
        "b": true.b,
        "cart_coulomb": true.cart_coulomb,
        "com0": true.link_coms(2)[0],
        "com1": true.link_coms(2)[1],
        "inertia0": true.link_inertias(2)[0],
        "inertia1": true.link_inertias(2)[1],
        "joint_friction0": true.joint_frictions(2)[0],
        "joint_friction1": true.joint_frictions(2)[1],
    }
    tol = {"M": 0.01, "cart_coulomb": 0.02, "com0": 0.01, "com1": 0.01}
    for name, value in truth.items():
        e = fitted.estimate(name)
        err = abs(e.value / value - 1.0)
        assert err < tol.get(name, 0.03 if name != "b" else 0.15), (name, e, value)
        assert err < 0.5 * abs(e.nominal / value - 1.0), name  # better than nominal
        assert abs(e.value - value) < 4 * e.se + 0.01 * value, name  # SE is honest
    assert fitted.delay == 1 and fitted.delay_cost[1] < min(
        fitted.delay_cost[0], fitted.delay_cost[2]
    )
    for ch in ("x", "th0", "th1"):
        assert fitted.rms[ch] < 0.3 * fitted.rms_nominal[ch], ch
    assert fitted.rms["th0"] < 1.5 * math.sqrt(5e-4**2 + (2 * np.pi / 16384) ** 2 / 12)
    assert fitted.physics.M == pytest.approx(fitted.estimate("M").value)
    assert not fitted.warnings
    assert "delay_steps" in fitted.summary()


def test_fitted_config_loads_and_narrows(fitted: FitResult, tmp_path: Path) -> None:
    """The TOML reloads to the fitted plant, delay and narrowed randomization."""
    out = tmp_path / "fitted.toml"
    text = write_fitted_config(RIG_TOML, fitted, out)
    assert text.startswith("# Fitted by")
    cfg = load_tqc_config(out).env
    p = cfg.physics
    assert p.M == pytest.approx(fitted.physics.M)
    np.testing.assert_allclose(p.link_coms(2), fitted.physics.link_coms(2))
    np.testing.assert_allclose(p.link_inertias(2), fitted.physics.link_inertias(2))
    np.testing.assert_allclose(p.joint_frictions(2), fitted.physics.joint_frictions(2))
    assert (
        p.force_max == fitted.nominal.force_max
        and p.v_noload == fitted.nominal.v_noload
    )
    assert cfg.hardware is not None and cfg.hardware.delay_steps == 1
    src_hw = load_tqc_config(RIG_TOML).env.hardware
    assert src_hw is not None
    assert cfg.hardware.sensors == src_hw.sensors  # encoder model carried over
    src = load_tqc_config(RIG_TOML).env.randomize
    rnd = cfg.randomize
    assert src is not None and rnd is not None
    assert rnd.cart_mass_rel < src.cart_mass_rel and rnd.link_com_rel < src.link_com_rel
    assert rnd.link_mass_rel < src.link_mass_rel
    assert rnd.cart_coulomb_scale[1] < src.cart_coulomb_scale[1]
    assert rnd.force_gain_rel == src.force_gain_rel  # not fitted: unchanged
    assert rnd.link_length_rel == src.link_length_rel
    assert cfg.reward_shape == load_tqc_config(RIG_TOML).env.reward_shape
    # A config without hardware or randomization gains both.
    bare = tmp_path / "bare.toml"
    bare.write_text("[physics]\ndt = 0.005\nM = 1.0\n")
    data = fitted_config(bare, fitted)
    assert data["hardware"]["pipeline"]["delay_steps"] == 1
    assert data["randomize"]["cart_mass_rel"] <= PlantRandomization().cart_mass_rel


def test_report_html(fitted: FitResult, tmp_path: Path) -> None:
    """The report holds the table, the delay grid and the window plots."""
    path = write_report(fitted, tmp_path / "r.html", ["a.ncpr"])
    page = path.read_text()
    assert "System identification" in page and "a.ncpr" in page
    assert page.count("plotly-graph-div") >= 3 and "cdn.plot.ly" in page


def test_fit_force_gain_and_delay(tmp_path: Path) -> None:
    """With the cart mass fixed, the drive gain and a 2-step delay are found."""
    env = rig_env()
    p = env.physics
    rec = simulate_recording(
        p,
        2,
        3.0,
        excite=ExciteParams("prbs", 3.0, period=0.05, seed=2),
        delay=2,
        force_gain=0.9,
        seed=3,
    )
    res = fit_recordings(
        [rec],
        p,
        2,
        env.force_slew,
        nominal_delay=1,
        opts=FitOptions(fit=("force_gain", "b"), delays=[1, 2, 3], substeps=1),
    )
    assert res.delay == 2
    assert res.force_gain == pytest.approx(0.9, rel=0.01)
    assert res.estimate("b").value == pytest.approx(p.b, rel=0.2)
    assert math.isnan(res.estimate("com0").se)  # links fixed
    text = write_fitted_config(RIG_TOML, res, tmp_path / "g.toml")
    assert "PWM full-scale" in text
    assert tomllib.loads(text)["randomize"]["force_gain_rel"] < 0.1


def test_fit_rejects_bad_requests() -> None:
    """Unidentifiable or unknown parameter sets, link/dt mismatches, short data."""
    env = rig_env()
    p = env.physics
    rec = make_recording(DT, np.zeros((400, 3)), np.zeros(400))
    with pytest.raises(ValueError, match="cannot all be fitted"):
        fit_recordings(
            [rec], p, 2, 300.0, opts=FitOptions(fit=("M", "force_gain", "links"))
        )
    with pytest.raises(ValueError, match="unknown"):
        fit_recordings([rec], p, 2, 300.0, opts=FitOptions(fit=("mass",)))
    with pytest.raises(ValueError, match="links"):
        fit_recordings([rec], p, 1, 300.0)
    with pytest.raises(ValueError, match="dt"):
        fit_recordings([rec], dataclasses.replace(p, dt=0.01), 2, 300.0)
    short = make_recording(DT, np.zeros((30, 3)), np.zeros(30))
    with pytest.raises(ValueError, match="too short"):
        fit_recordings([short], p, 2, 300.0)


def test_consistency_warnings() -> None:
    """A COM outside its link or a negative inertia is flagged."""
    ln = np.array([0.25, 0.25])
    assert consistency_warnings(np.array([0.1, 0.2]), np.array([1e-4, 1e-4]), ln) == []
    w = consistency_warnings(np.array([0.4, 0.2]), np.array([1e-4, -1e-5]), ln)
    assert len(w) == 2 and "outside" in w[0] and "negative" in w[1]


def test_dump_toml() -> None:
    """Nested tables, arrays, strings and booleans survive a tomllib round trip."""
    data = {
        "a": 1,
        "s": 'q"x',
        "t": {
            "f": 0.1,
            "l": [1.0, 2.0],
            "b": True,
            "sub": {"n": np.int64(3), "arr": np.array([0.5])},
        },
    }
    back = tomllib.loads(dump_toml(data, "header\nline 2"))
    assert back == {
        "a": 1,
        "s": 'q"x',
        "t": {"f": 0.1, "l": [1.0, 2.0], "b": True, "sub": {"n": 3, "arr": [0.5]}},
    }
    with pytest.raises(ValueError, match="non-finite"):
        dump_toml({"x": float("nan")})
    with pytest.raises(TypeError):
        dump_toml({"x": object()})


# --- fine-tuning on a fitted plant --------------------------------------------------


def test_replace_plant_keeps_policy_settings() -> None:
    """Physics/encoders/randomization swap in; obs, actions, estimator stay."""
    env = rig_env(action_history=2)
    other = load_tqc_config(RIG_TOML).env
    other = dataclasses.replace(
        other,
        physics=dataclasses.replace(other.physics, M=1.3),
        force_slew=500.0,
        randomize=None,
    )
    new, changes = replace_plant(env, other)
    assert new.physics.M == 1.3 and new.randomize is None
    assert (
        new.action_mode == "delta" and new.goal_conditioned and new.force_slew == 300.0
    )
    assert new.hardware is not None and new.hardware.sensors is not None
    assert new.hardware.sensors.estimator == "ekf"  # policy-side, kept
    assert new.hardware.physics.M == 1.3
    assert any(c.startswith("physics.M") for c in changes)
    assert any("randomize" in c for c in changes)
    assert any("kept force_slew" in c for c in changes)
    assert replace_plant(env, env)[1] == []
    with pytest.raises(ValueError, match="links"):
        replace_plant(env, dataclasses.replace(other, n_links=3))
    with pytest.raises(ValueError, match="dt"):
        replace_plant(env, dataclasses.replace(other, physics=PhysicsParams(dt=0.01)))
    no_hw = dataclasses.replace(other, hardware=None)
    dropped, changes = replace_plant(env, no_hw)
    assert dropped.hardware is None and any(c.startswith("hardware:") for c in changes)
    hw = env.hardware
    assert hw is not None
    sysid_env = dataclasses.replace(
        env, hardware=dataclasses.replace(hw, sysid_context=True)
    )
    with pytest.raises(ValueError, match="sysID"):
        replace_plant(sysid_env, no_hw)


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_train_resume_with_fitted_config(
    tmp_path: Path, fitted: FitResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    """train.py --resume --config fine-tunes the same network on the fitted plant."""
    ckpt = make_checkpoint(tmp_path / "run" / "tqc_1234.pt")
    toml = tmp_path / "fitted.toml"
    write_fitted_config(RIG_TOML, fitted, toml)
    train = _load_script("train")
    argv = [
        "train.py",
        "--resume",
        str(ckpt),
        "--config",
        str(toml),
        "--steps",
        "1240",
        "--device",
        "cpu",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    train.main()
    out_dir = tmp_path / "run-fitted"
    saved = sorted(out_dir.glob("tqc_*.pt"))
    assert saved, list(tmp_path.rglob("*"))
    new = torch.load(saved[-1], map_location="cpu", weights_only=False)
    cfg = new["cfg"].env
    assert cfg.physics.M == pytest.approx(fitted.physics.M)
    assert cfg.hardware.sensors.estimator == "ekf"
    assert cfg.action_history == 2 and cfg.goal_conditioned
    assert new["step"] >= 1240
    old = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert (
        old["actor"]["net.in_proj.weight"].shape
        == new["actor"]["net.in_proj.weight"].shape
    )
