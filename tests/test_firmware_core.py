"""Parity of the firmware's portable C core with the Python originals.

Compiles ``firmware/components/cartpole_core`` plus the host harness
``firmware/test/host/ncp_host.c`` with the host C compiler and drives it over
stdin/stdout: the policy against torch, the dynamics against ``dynamics._ode`` /
``step`` / ``available_force``, the EKF against ``StateEKF``, the controller tick
against the training env's observation and command, and the simulator against a
Python model of the same actuator. Skipped when no C compiler is available.
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pytest
import torch

from n_cartpole.deploy import firmware_blob as fb
from n_cartpole.deploy.actor_export import actor_tensors
from n_cartpole.deploy.quantize import actor_forward, quantize_tensors
from n_cartpole.env.cartpole import NPendulumCartpole
from n_cartpole.env.dynamics import PhysicsParams, _model, _ode, available_force, step
from n_cartpole.env.estimator import StateEKF
from n_cartpole.env.goals import goal_configs, goal_labels
from n_cartpole.policy.loader import PolicyBundle, load_policy
from n_cartpole.sysid.excite import Excitation, ExciteParams, excitation_signal
from n_cartpole.sysid.fit import FitOptions, fit_recordings
from n_cartpole.sysid.recording import Recording
from tests.firmware_util import REPO, make_checkpoint, rig_env

CORE = REPO / "firmware" / "components" / "cartpole_core"
HARNESS = REPO / "firmware" / "test" / "host" / "ncp_host.c"


def _compiler() -> str | None:
    cc = os.environ.get("CC")
    if cc and shutil.which(cc):
        return cc
    for name in ("cc", "gcc", "clang"):
        if shutil.which(name):
            return name
    return None


def _build_host(tmp_path_factory: pytest.TempPathFactory, *defines: str) -> Path:
    cc = _compiler()
    if cc is None:
        pytest.skip("no C compiler found")
    out = tmp_path_factory.mktemp("ncp_host") / "ncp_host"
    srcs = sorted(str(p) for p in (CORE / "src").glob("*.c"))
    cmd = [
        cc,
        "-std=c11",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        *defines,
        f"-I{CORE / 'include'}",
        *srcs,
        str(HARNESS),
        "-lm",
        "-o",
        str(out),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0, res.stderr
    return out


@pytest.fixture(scope="module")
def host(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Compile the core + harness (double-precision EKF, as the parity tests need)."""
    return _build_host(tmp_path_factory)


@pytest.fixture(scope="module")
def host_float(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Compile the core with the single-precision EKF the firmware uses (NCP_EKF_FLOAT=1)."""
    return _build_host(tmp_path_factory, "-DNCP_EKF_FLOAT=1")


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Tiny rig-config checkpoint: rigid links, Coulomb, back-EMF, delay, EKF."""
    return make_checkpoint(tmp_path_factory.mktemp("rig") / "tqc_1234.pt")


@pytest.fixture(scope="module")
def bundle(ckpt: Path) -> PolicyBundle:
    """Load the checkpoint as the Python side sees it."""
    return load_policy(ckpt)


@pytest.fixture(scope="module")
def blob(ckpt: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Export the f32 .ncpm blob."""
    path = tmp_path_factory.mktemp("blob") / "model.ncpm"
    path.write_bytes(fb.export_blob(ckpt))
    return path


@pytest.fixture(scope="module")
def blob16(ckpt: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Export the f16 .ncpm blob."""
    path = tmp_path_factory.mktemp("blob16") / "model16.ncpm"
    path.write_bytes(fb.export_blob(ckpt, dtype="f16"))
    return path


@pytest.fixture(scope="module")
def blob8(ckpt: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Export the int8 .ncpm blob."""
    path = tmp_path_factory.mktemp("blob8") / "model8.ncpm"
    path.write_bytes(fb.export_blob(ckpt, dtype="i8"))
    return path


def _fmt(rows: Iterable[Iterable[float]]) -> str:
    return "\n".join(" ".join(repr(float(v)) for v in row) for row in rows) + "\n"


def _run(host: Path, blob: Path, *args: str, stdin: str = "") -> list[list[str]]:
    res = subprocess.run(
        [str(host), str(blob), *args],
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr
    return [line.split() for line in res.stdout.splitlines() if line.strip()]


def _floats(rows: list[list[str]]) -> np.ndarray:
    return np.array([[float(v) for v in row] for row in rows])


def _random_states(rng: np.random.Generator, n: int, count: int) -> np.ndarray:
    s = np.empty((count, 2 + 2 * n))
    s[:, 0] = rng.uniform(-0.45, 0.45, count)
    s[:, 1] = rng.uniform(-2.0, 2.0, count)
    s[:, 2::2] = rng.uniform(-8.0, 8.0, (count, n))  # unwrapped angles
    s[:, 3::2] = rng.uniform(-12.0, 12.0, (count, n))
    s[: count // 4, 1] = rng.uniform(-0.01, 0.01, count // 4)  # Coulomb's linear zone
    return s


# --- model blob -------------------------------------------------------------------


def test_info_matches_python_header(host: Path, blob: Path) -> None:
    """The C parser reads every field the Python exporter wrote; goals agree."""
    info = {
        r[0].split("=", 1)[0]: " ".join(r)[len(r[0].split("=", 1)[0]) + 1 :]
        for r in _run(host, blob, "info")
    }
    h = fb.read_header(blob.read_bytes())
    assert info["crc_ok"] == "1" and int(info["crc32"]) == h["crc32"]
    for key in ("n_links", "n_blocks", "hidden", "obs_dim", "flags", "n_goals"):
        assert int(info[key]) == h[key], key
    for key in ("dt", "force_max", "force_slew", "v_noload", "goal_tol_vel"):
        assert float(info[key]) == h[key], key
    np.testing.assert_array_equal([float(v) for v in info["com"].split()], h["com"])
    assert info["label"] == h["label"]
    assert int(info["weight_count"]) == fb.weight_count(h)
    n = h["n_links"]
    assert info["goal_labels"].split() == goal_labels(n)
    targets = np.array([float(v) for v in info["goal_targets"].split()])
    np.testing.assert_array_equal(targets.reshape(-1, n), goal_configs(n))
    # The EKF's R, as NPendulumCartpole._ekf_state builds it at realism 1.
    std = np.sqrt(
        np.array([h["x_noise_std"]] + [h["angle_noise_std"]] * n) ** 2
        + np.array([h["x_resolution"]] + [h["angle_resolution"]] * n) ** 2 / 12.0
    )
    np.testing.assert_allclose(
        [float(v) for v in info["pos_std"].split()], np.maximum(std, 1e-6), rtol=1e-15
    )


def test_c_rejects_corrupt_blobs(host: Path, blob: Path, tmp_path: Path) -> None:
    """CRC, truncation and magic errors are reported and the harness exits 2."""
    data = blob.read_bytes()
    cases = {
        "CRC": data[:600] + bytes([data[600] ^ 1]) + data[601:],
        "truncated": data[:-8],
        "magic": b"ZZZZ" + data[4:],
    }
    for word, bad in cases.items():
        path = tmp_path / f"{word}.ncpm"
        path.write_bytes(bad)
        res = subprocess.run(
            [str(host), str(path), "info"], capture_output=True, text=True, check=False
        )
        assert res.returncode == 2 and word in res.stderr, (word, res.stderr)


def test_f16_weights_decode_exactly(host: Path, blob16: Path, blob: Path) -> None:
    """C decodes float16 bit-exactly like numpy, and acts close to the f32 model."""
    c16 = _floats(_run(host, blob16, "weights"))[:, 0].astype(np.float32)
    np.testing.assert_array_equal(c16, fb.read_weights(blob16.read_bytes()))
    c32 = _floats(_run(host, blob, "weights"))[:, 0].astype(np.float32)
    np.testing.assert_array_equal(c32, fb.read_weights(blob.read_bytes()))
    rng = np.random.default_rng(3)
    obs = _fmt(rng.normal(size=(50, 12)))
    f32 = _floats(_run(host, blob, "policy", stdin=obs))[:, 0]
    f16 = _floats(_run(host, blob16, "policy", stdin=obs))[:, 0]
    np.testing.assert_allclose(f16, f32, atol=0.05)


def _info(host: Path, blob: Path) -> dict[str, str]:
    return {
        r[0].split("=", 1)[0]: r[0].split("=", 1)[1] for r in _run(host, blob, "info")
    }


def test_i8_weights_and_policy_match_python(
    host: Path, blob8: Path, blob: Path, ckpt: Path
) -> None:
    """C reads the int8 image like Python and runs the same network as actor_forward."""
    info = _info(host, blob8)
    h = fb.read_header(blob8.read_bytes())
    assert int(info["weight_bytes"]) == fb.weight_bytes(h)
    assert int(info["weight_ram_bytes"]) == fb.weight_ram_bytes(h)
    assert int(info["weight_count"]) == fb.weight_count(h)
    c8 = _floats(_run(host, blob8, "weights"))[:, 0].astype(np.float32)
    np.testing.assert_array_equal(c8, fb.read_weights(blob8.read_bytes()))
    rng = np.random.default_rng(4)
    obs = rng.normal(size=(200, h["obs_dim"])).astype(np.float32)
    obs[:50] *= 4.0
    out = _floats(_run(host, blob8, "policy", stdin=_fmt(obs)))[:, 0]
    t = quantize_tensors(
        actor_tensors(torch.load(ckpt, map_location="cpu", weights_only=False))
    )
    np.testing.assert_allclose(out, actor_forward(t, obs, h["force_max"]), atol=2e-4)
    f32 = _floats(_run(host, blob, "policy", stdin=_fmt(obs)))[:, 0]
    assert np.mean(np.abs(out - f32)) < 0.02 * h["force_max"]


def test_i8_odd_sizes_are_padded(host: Path, blob: Path, tmp_path: Path) -> None:
    """Rows x cols not a multiple of 4: the zero padding keeps every tensor aligned."""
    h = fb.read_header(blob.read_bytes())
    h.update(hidden=5, expand=3, n_blocks=2, dtype=fb.DTYPE_I8)
    rng = np.random.default_rng(5)
    tensors = [(n, rng.normal(size=s)) for n, s in fb.tensor_shapes(h)]
    payload = fb._payload(tensors, "i8")
    assert len(payload) == fb.weight_bytes(h) and len(payload) % 4 == 0
    h["total_size"] = 512 + len(payload)
    data = bytearray(fb.pack_header(h) + payload)
    data[fb.CRC_OFFSET : fb.CRC_OFFSET + 4] = fb.blob_crc(bytes(data)).to_bytes(
        4, "little"
    )
    path = tmp_path / "odd.ncpm"
    path.write_bytes(bytes(data))
    c = _floats(_run(host, path, "weights"))[:, 0].astype(np.float32)
    np.testing.assert_array_equal(c, fb.read_weights(bytes(data)))
    obs = rng.normal(size=(30, h["obs_dim"])).astype(np.float32)
    out = _floats(_run(host, path, "policy", stdin=_fmt(obs)))[:, 0]
    ref = actor_forward(quantize_tensors(tensors), obs, h["force_max"])
    np.testing.assert_allclose(out, ref, atol=2e-4)


# --- policy -----------------------------------------------------------------------


def test_policy_matches_torch(host: Path, blob: Path, bundle: PolicyBundle) -> None:
    """tanh(mean) * force_max agrees with the torch actor on random observations."""
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(200, bundle.obs_dim)).astype(np.float32)
    obs[:50] *= 4.0
    ref = bundle.select_action(bundle.normalize(obs))[:, 0]
    out = _floats(_run(host, blob, "policy", stdin=_fmt(obs)))
    np.testing.assert_allclose(out[:, 0], ref, rtol=1e-4, atol=1e-4)
    assert np.ptp(ref) > 5.0  # the test net uses the force range
    np.testing.assert_allclose(
        np.tanh(out[:, 1]) * bundle.physics.force_max, out[:, 0], rtol=1e-5, atol=1e-5
    )


# --- dynamics ---------------------------------------------------------------------


def test_deriv_matches_ode(host: Path, blob: Path, bundle: PolicyBundle) -> None:
    """ncp_dyn_deriv == dynamics._ode (rigid links, Coulomb, joint friction)."""
    p = bundle.physics
    assert p.cart_coulomb > 0 and p.com is not None
    rng = np.random.default_rng(1)
    states = _random_states(rng, 2, 300)
    forces = rng.uniform(-p.force_max, p.force_max, len(states))
    mdl = _model(p, 2)
    ref = np.array([_ode(0.0, s, f, mdl) for s, f in zip(states, forces, strict=True)])
    rows = np.column_stack([states, forces])
    out = _floats(_run(host, blob, "deriv", stdin=_fmt(rows)))
    np.testing.assert_allclose(out, ref, rtol=1e-9, atol=1e-9)


def test_step_matches_rk45(host: Path, blob: Path, bundle: PolicyBundle) -> None:
    """Fixed-step RK4 (4 substeps) agrees with dynamics.step's RK45 to ~1e-6.

    Except where the cart speed starts inside the stiff Coulomb zone (|xd| of a
    few ``cart_coulomb_vscale``): there the fixed step smooths the tanh and the
    two differ by up to ~1e-4 (RK45 adapts its step; both are approximations).
    """
    p = bundle.physics
    rng = np.random.default_rng(2)
    states = _random_states(rng, 2, 120)
    forces = rng.uniform(-1.2 * p.force_max, 1.2 * p.force_max, len(states))
    ref = np.array([step(s, f, p) for s, f in zip(states, forces, strict=True)])
    rows = np.column_stack([states, forces])
    out = _floats(_run(host, blob, "step", stdin=_fmt(rows)))
    zone = np.abs(states[:, 1]) < 0.05
    assert 0 < zone.sum() < len(zone)
    np.testing.assert_allclose(out[~zone], ref[~zone], rtol=0, atol=1e-6)
    np.testing.assert_allclose(out[zone], ref[zone], rtol=0, atol=1e-3)


def test_available_force(host: Path, blob: Path, bundle: PolicyBundle) -> None:
    """Back-EMF limit along the motion, full force when braking, clip at force_max."""
    p = bundle.physics
    assert p.v_noload is not None
    F = np.linspace(-20.0, 20.0, 21)
    xd = np.array([-3.0, -p.v_noload, -1.0, -0.2, 0.0, 0.3, 1.5, 2.9])
    grid = np.array([(f, v) for f in F for v in xd])
    ref = np.array([available_force(f, v, p) for f, v in grid])
    out = _floats(_run(host, blob, "avail", stdin=_fmt(grid)))[:, 0]
    np.testing.assert_allclose(out, ref, rtol=1e-15, atol=1e-15)


# --- estimator --------------------------------------------------------------------


def _ekf_trajectory(
    p: PhysicsParams, rng: np.random.Generator, steps: int, pos_std: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    s = np.array([0.0, 0.0, np.pi - 0.3, 0.0, np.pi + 0.2, 0.0])
    forces, readings = [], []
    for _ in range(steps):
        f = float(rng.uniform(-p.force_max, p.force_max))
        s = step(s, f, p)
        pos = np.concatenate(([s[0]], s[2::2]))
        forces.append(f)
        readings.append(pos + rng.normal(size=3) * pos_std)
    return np.array(forces), np.array(readings)


def test_ekf_matches_python(host: Path, blob: Path, bundle: PolicyBundle) -> None:
    """The C EKF tracks StateEKF step for step (reset, predict, Joseph update)."""
    p = bundle.physics
    hw = bundle.env_config.hardware
    assert hw is not None and hw.sensors is not None
    sen = hw.sensors
    noise = np.array([sen.x_noise_std] + [sen.angle_noise_std] * 2)
    res = np.array([sen.x_resolution] + [sen.angle_resolution] * 2)
    pos_std = np.maximum(np.sqrt(noise**2 + res**2 / 12.0), 1e-6)
    rng = np.random.default_rng(4)
    forces, readings = _ekf_trajectory(p, rng, 150, 20 * pos_std)
    ekf = StateEKF(
        p, 2, pos_std, force_std=sen.ekf_force_std, accel_std=sen.ekf_accel_std
    )
    ref = [ekf.reset(readings[0])]
    ref += [ekf.update(f, r) for f, r in zip(forces[:-1], readings[1:], strict=True)]
    rows = [readings[0]] + [
        np.concatenate(([f], r)) for f, r in zip(forces[:-1], readings[1:], strict=True)
    ]
    out = _floats(_run(host, blob, "ekf", stdin=_fmt(rows)))
    np.testing.assert_allclose(out, np.array(ref), rtol=1e-8, atol=1e-8)


# --- controller -------------------------------------------------------------------


def _env_episode(
    bundle: PolicyBundle, start: str, goal: str, steps: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Roll the env with the torch policy; return readings, true speeds, obs, cmds."""
    cfg = dataclasses.replace(
        bundle.env_config, realism=0.0, goal_hold_steps=None, randomize=None
    )
    env = NPendulumCartpole(cfg)
    obs, _ = env.reset(seed=seed, options={"start": start, "goal": goal})
    readings, speeds, observations, cmds = [], [], [], []
    for _ in range(steps):
        s = env.get_state()
        readings.append(np.concatenate(([s[0]], s[2::2])))
        speeds.append(s[1])
        observations.append(obs)
        act = bundle.select_action(bundle.normalize(np.asarray(obs)[None]))[0]
        # Keep going past the rail end (the env integrates regardless): the
        # untrained test policy leaves early, and high speeds exercise back-EMF.
        obs, _, _, _, _ = env.step(act)
        cmds.append(env._prev_cmd)
    return (
        np.array(readings),
        np.array(speeds),
        np.array(observations),
        np.array(cmds),
    )


def _ctrl_lines(
    goal: int, readings: np.ndarray, speeds: np.ndarray | None
) -> list[str]:
    rows = readings if speeds is None else np.column_stack([readings, speeds])
    return ["reset", f"goal {goal}", _fmt(rows)]


@pytest.mark.parametrize(
    ("start", "goal", "seed"), [("DD", "UU", 0), ("UU", "DD", 1), ("DU", "UD", 2)]
)
def test_controller_matches_env(
    host: Path, blob: Path, bundle: PolicyBundle, start: str, goal: str, seed: int
) -> None:
    """Fed the env's readings, the C tick reproduces its observation and command.

    At realism 0 the readings are exact (the EKF still runs, with its 1e-6 floor
    on R). The env evaluates the back-EMF limit at the TRUE cart speed, so the
    harness is given it for ``est_force`` here; the device uses its estimate.
    """
    readings, speeds, obs, cmds = _env_episode(bundle, start, goal, 200, seed)
    gi = goal_labels(2).index(goal)
    out = _floats(
        _run(
            host,
            blob,
            "ctrl",
            "realism=0",
            stdin="\n".join(_ctrl_lines(gi, readings, speeds)),
        )
    )
    D = bundle.obs_dim
    assert out.shape[0] == len(obs)
    np.testing.assert_allclose(out[:, :D], obs, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(out[:, D], cmds, rtol=1e-4, atol=1e-3)
    assert np.ptp(cmds) > 5.0  # the command actually moved
    # Without the true speed (as on the device): the EKF's speed is close enough
    # that the back-EMF limit, and so everything downstream, still agrees.
    dev = _floats(
        _run(
            host,
            blob,
            "ctrl",
            "realism=0",
            stdin="\n".join(_ctrl_lines(gi, readings, None)),
        )
    )
    np.testing.assert_allclose(dev[:, :D], obs, rtol=1e-3, atol=1e-3)


def test_manual_mode_and_x_ref(host: Path, blob: Path, bundle: PolicyBundle) -> None:
    """MANUAL: PD to x_ref, capped at 10 N, slewed; AI sees x - x_ref (clamped)."""
    p = bundle.physics
    slew = bundle.env_config.force_slew
    assert slew is not None
    pos = _fmt([[0.0, np.pi, np.pi]] * 30)
    out = _floats(_run(host, blob, "ctrl", stdin="mode manual\nxref 0.1\n" + pos))
    D = bundle.obs_dim
    cmd = out[:, D]
    assert np.all(np.isnan(out[:, D + 2]))  # no policy action in manual mode
    assert np.all(np.abs(cmd) <= 10.0 + 1e-12)
    assert np.all(np.abs(np.diff(np.concatenate(([0.0], cmd)))) <= slew * p.dt + 1e-9)
    assert cmd[-1] == pytest.approx(10.0)  # 150 N/m * 0.1 m, capped
    np.testing.assert_allclose(out[:, 0], -0.1, atol=1e-4)  # EKF x vs reading
    out = _floats(_run(host, blob, "ctrl", stdin="xref 1.0\n" + pos))
    np.testing.assert_allclose(out[:, 0], -p.x_lim / 2, atol=1e-4)
    assert np.all(np.isfinite(out[:, D + 2]))


# --- simulator --------------------------------------------------------------------


def _rk4(s: np.ndarray, F: float, p: PhysicsParams, substeps: int = 4) -> np.ndarray:
    mdl = _model(p, (s.size - 2) // 2)
    h = p.dt / substeps
    for _ in range(substeps):
        k1 = _ode(0.0, s, F, mdl)
        k2 = _ode(0.0, s + 0.5 * h * k1, F, mdl)
        k3 = _ode(0.0, s + 0.5 * h * k2, F, mdl)
        k4 = _ode(0.0, s + h * k3, F, mdl)
        s = s + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return s


def _sim_reference(
    s0: np.ndarray, cmds: np.ndarray, p: PhysicsParams, slew: float, delay: int
) -> tuple[np.ndarray, np.ndarray]:
    """env.step's actuator (delay -> slew -> available_force) + RK4."""
    buf = [0.0] * delay
    prev, s = 0.0, s0.copy()
    applied, states = [], []
    for c in cmds:
        a = float(np.clip(c, -p.force_max, p.force_max))
        if delay:
            buf.append(a)
            a = buf.pop(0)
        a = prev + float(np.clip(a - prev, -slew * p.dt, slew * p.dt))
        prev = a
        a = available_force(a, float(s[1]), p)
        s = _rk4(s, a, p)
        applied.append(a)
        states.append(s)
    return np.array(applied), np.array(states)


def test_sim_matches_env_actuator_and_plant(
    host: Path, blob: Path, bundle: PolicyBundle
) -> None:
    """With the encoder model off, ncp_sim is the env's actuator + nominal plant."""
    p = bundle.physics
    slew = bundle.env_config.force_slew
    hw = bundle.env_config.hardware
    assert slew is not None and hw is not None and hw.delay_steps == 1
    t = np.arange(150) * p.dt
    cmds = 14.0 * np.sign(np.sin(2 * np.pi * 1.5 * t)) + 3.0 * np.sin(17 * t)
    rows = _run(
        host, blob, "simcmd", "3", "0.05", "9", "noise=0", stdin=_fmt(cmds[:, None])
    )
    s0 = _floats(rows[:1])[0]
    assert np.all(np.abs(s0 - [0, 0, np.pi, 0, np.pi, 0]) <= 0.05)
    out = _floats(rows[1:])
    applied, states = _sim_reference(s0, cmds, p, slew, hw.delay_steps)
    np.testing.assert_allclose(out[:, 0], applied, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(out[:, 1:7], states, rtol=1e-9, atol=1e-9)
    np.testing.assert_array_equal(out[:, 7], out[:, 1])  # exact readings
    np.testing.assert_array_equal(out[:, 8:10], out[:, [3, 5]])
    assert np.any(applied != np.clip(cmds, -p.force_max, p.force_max))  # slew bit
    # One step against the reference integrator (RK45) as well.
    np.testing.assert_allclose(
        _floats(rows[1:2])[0, 1:7], step(s0, applied[0], p), atol=1e-6
    )


def test_sim_encoder_model(host: Path, blob: Path, bundle: PolicyBundle) -> None:
    """Readings are quantized to the resolution with the configured noise."""
    hw = bundle.env_config.hardware
    assert hw is not None and hw.sensors is not None
    sen = hw.sensors
    rows = _run(host, blob, "simcmd", "3", "0.0", "5", stdin=_fmt([[0.0]] * 2000))
    out = _floats(rows[1:])
    th, reading = out[:, 3], out[:, 8]
    q = reading / sen.angle_resolution
    np.testing.assert_allclose(q, np.round(q), atol=1e-6)
    err = reading - th
    expect = np.sqrt(sen.angle_noise_std**2 + sen.angle_resolution**2 / 12)
    assert 0.8 * expect < np.std(err) < 1.2 * expect
    xq = out[:, 7] / sen.x_resolution
    np.testing.assert_allclose(xq, np.round(xq), atol=1e-6)
    assert np.all(out[:, -1] == 0)  # resting at DD: on the rail


def test_sim_closed_loop_and_bench(host: Path, blob: Path) -> None:
    """The closed-loop commands run end to end; full push leaves the rail."""
    ep = _run(host, blob, "sim", "3", "3", "300", "1")[0]
    assert len(ep) == 5 and int(ep[3]) <= 300
    mat = _run(host, blob, "matrix", "1", "0.05", "0")
    assert mat[-1][0] == "overall" and len(mat) == 1 + 16 + 1
    bench = dict(r[0].split("=") for r in _run(host, blob, "bench", "50"))
    assert float(bench["ctrl_us"]) > 0 and int(bench["iters"]) == 50
    rows = _run(host, blob, "simcmd", "3", "0", "1", stdin=_fmt([[15.0]] * 400))
    assert int(rows[-1][-1]) == 1


@pytest.mark.parametrize("noise_scale", [1.0, 20.0])
def test_ekf_float_close_to_python(
    host_float: Path, blob: Path, bundle: PolicyBundle, noise_scale: float
) -> None:
    """The single-precision EKF (the firmware build) stays close to StateEKF over a long run."""
    p = bundle.physics
    hw = bundle.env_config.hardware
    assert hw is not None and hw.sensors is not None
    sen = hw.sensors
    noise = np.array([sen.x_noise_std] + [sen.angle_noise_std] * 2)
    res = np.array([sen.x_resolution] + [sen.angle_resolution] * 2)
    pos_std = np.maximum(np.sqrt(noise**2 + res**2 / 12.0), 1e-6)
    rng = np.random.default_rng(11)
    forces, readings = _ekf_trajectory(p, rng, 600, noise_scale * pos_std)
    ekf = StateEKF(
        p, 2, pos_std, force_std=sen.ekf_force_std, accel_std=sen.ekf_accel_std
    )
    ref = np.array(
        [ekf.reset(readings[0])]
        + [ekf.update(f, r) for f, r in zip(forces[:-1], readings[1:], strict=True)]
    )
    rows = [readings[0]] + [
        np.concatenate(([f], r)) for f, r in zip(forces[:-1], readings[1:], strict=True)
    ]
    out = _floats(_run(host_float, blob, "ekf", stdin=_fmt(rows)))
    # Measured max error: 2e-3 (rad/s, at 20x noise) on velocities of 3-5 rad/s.
    np.testing.assert_allclose(out, ref, rtol=1e-3, atol=5e-3)


# --- excitation, ext_force, recordings ---------------------------------------------


@pytest.mark.parametrize(
    "params",
    [
        ExciteParams("chirp", 3.0, 0.2, 6.0, duration=1.3),
        ExciteParams("chirp", 2.0, 0.0, 4.0, duration=0.5),
        ExciteParams("prbs", 2.5, period=0.05, duration=1.0, seed=7),
        ExciteParams("prbs", 1.0, period=0.012, duration=0.0, seed=0),
        ExciteParams("steps", 4.0, period=0.1, duration=0.9),
        ExciteParams("sine", 1.5, f0=2.0, duration=0.4),
        ExciteParams("none"),
    ],
)
def test_excite_matches_python(host: Path, blob: Path, params: ExciteParams) -> None:
    """ncp_excite reproduces n_cartpole.sysid.excite sample for sample."""
    N = 300
    out = _floats(
        _run(
            host,
            blob,
            "excite",
            params.kind,
            repr(params.amp),
            repr(params.f0),
            repr(params.f1),
            repr(params.period),
            repr(params.duration),
            str(params.seed),
            str(N),
        )
    )
    e = Excitation(params, 0.005)
    ref, active = [], []
    for _ in range(N):
        ref.append(e.next())
        active.append(e.active)
    np.testing.assert_allclose(out[:, 0], ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(out[:, 1].astype(bool), active)


def _ticks(lines: list[str], pos: Sequence[float], n: int) -> list[str]:
    return lines + [" ".join(repr(v) for v in pos)] * n


def test_ext_force_manual_mode(host: Path, blob: Path, bundle: PolicyBundle) -> None:
    """MANUAL: command = slew toward clip(PD) + ext, bounded by force_max."""
    p = bundle.physics
    slew = bundle.env_config.force_slew
    assert slew is not None
    rest = [0.01, np.pi, np.pi]
    lines = ["mode manual"]
    exts = [0.0] * 3 + [6.0] * 6 + [30.0] * 12 + [-2.0] * 8
    for v in exts:
        lines = _ticks([*lines, f"ext {v!r}"], rest, 1)
    out = _floats(_run(host, blob, "ctrl", stdin="\n".join(lines) + "\n"))
    D = bundle.obs_dim
    cmd, est = out[:, D], out[:, D + 4 :]
    prev = 0.0
    for t, ext in enumerate(exts):
        pd = np.clip(150.0 * (0.0 - est[t, 0]) - 20.0 * est[t, 1], -10.0, 10.0)
        step = slew * p.dt
        want = np.clip(
            prev + np.clip(pd + ext - prev, -step, step), -p.force_max, p.force_max
        )
        assert cmd[t] == pytest.approx(want, abs=1e-9), t
        prev = want
    assert cmd.max() == pytest.approx(p.force_max)  # PD + 30 N clipped
    # The action history the policy would see includes the excitation.
    np.testing.assert_allclose(out[1:, D - 2], cmd[:-1] / p.force_max, rtol=1e-6)


def test_ext_force_ai_delta_is_superposed(
    host: Path, blob: Path, bundle: PolicyBundle
) -> None:
    """AI delta: the policy integrates its own command; ext is added, not integrated."""
    p = bundle.physics
    slew = bundle.env_config.force_slew
    assert slew is not None
    exts = [0.0] * 5 + [2.0] * 5 + [-3.0] * 5 + [0.0] * 5
    lines: list[str] = []
    for v in exts:
        lines = _ticks([*lines, f"ext {v!r}"], [0.0, np.pi, np.pi], 1)
    out = _floats(_run(host, blob, "ctrl", stdin="\n".join(lines) + "\n"))
    D = bundle.obs_dim
    cmd, action = out[:, D], out[:, D + 2].astype(np.float32)
    fmax32 = np.float32(p.force_max)
    base = 0.0
    for t, ext in enumerate(exts):
        rate = float(np.clip(action[t] / fmax32, np.float32(-1), np.float32(1)))
        base = float(np.clip(base + rate * slew * p.dt, -p.force_max, p.force_max))
        want = (
            base
            if ext == 0.0
            else float(np.clip(base + ext, -p.force_max, p.force_max))
        )
        assert cmd[t] == pytest.approx(want, abs=1e-12), t
    assert np.any(np.abs(np.diff(cmd[4:6])) > slew * p.dt)  # the step is not slewed


def test_record_writes_ncpr(host: Path, blob: Path, tmp_path: Path) -> None:
    """The harness's recorder (C structs) reads back with n_cartpole.sysid.recording."""
    path = tmp_path / "prbs.ncpr"
    rows = _run(
        host,
        blob,
        "record",
        str(path),
        "3",
        "excite=prbs",
        "amp=2",
        "period=0.05",
        "eseed=5",
        "seed=3",
        "note=c test",
    )
    assert rows == [["samples=600"]]
    rec = Recording.read(path)
    h = fb.read_header(blob.read_bytes())
    assert (len(rec), rec.dt, rec.n_links) == (600, h["dt"], 2)
    assert (rec.model_label, rec.model_crc32, rec.note) == (
        h["label"],
        h["crc32"],
        "c test",
    )
    assert rec.excite_kind == "prbs" and rec.excite_params["amp"] == 2.0
    ref = excitation_signal(
        ExciteParams("prbs", 2.0, 0.2, 6.0, 0.05, 3.0, 5), rec.dt, 600
    )
    np.testing.assert_allclose(rec.excite, ref, rtol=1e-6)
    assert rec.armed.all() and rec.has_truth and rec.segments() == [(0, 600)]
    assert np.all(rec.mode == 1)
    err = rec.pos - rec.truth[:, [0, 2, 4]]
    assert np.max(np.abs(err[:, 0])) < 2e-4 and np.max(np.abs(err[:, 1:])) < 5e-3
    np.testing.assert_allclose(rec.t, np.arange(600) * rec.dt, atol=1e-6)
    ring = tmp_path / "ring.ncpr"
    _run(host, blob, "record", str(ring), "1", "arm=0", "lift0=0.8")
    r2 = Recording.read(ring)
    assert not r2.armed.any() and np.all(r2.excite == 0.0)
    assert r2.truth[0, 2] == pytest.approx(np.pi + 0.8, abs=0.06)


def test_fit_on_c_recording_finds_delay(host: Path, blob: Path, tmp_path: Path) -> None:
    """End to end: a C-simulated chirp is fitted in Python; the 1-step delay is found."""
    path = tmp_path / "chirp.ncpr"
    _run(
        host,
        blob,
        "record",
        str(path),
        "3",
        "excite=chirp",
        "amp=3",
        "f0=0.3",
        "f1=6",
        "seed=2",
    )
    env = rig_env()
    res = fit_recordings(
        [Recording.read(path)],
        env.physics,
        2,
        env.force_slew,
        nominal_delay=0,
        opts=FitOptions(fit=("b", "cart_coulomb"), delays=[0, 1, 2], substeps=1),
    )
    assert res.delay == 1
    assert res.estimate("cart_coulomb").value == pytest.approx(
        env.physics.cart_coulomb, rel=0.05
    )
