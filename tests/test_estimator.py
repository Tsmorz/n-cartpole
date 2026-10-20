"""The model-based state estimator (EKF) versus finite-difference velocities."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from n_cartpole.config import load_tqc_config
from n_cartpole.env.cartpole import EnvConfig, NPendulumCartpole
from n_cartpole.env.dynamics import step
from n_cartpole.env.estimator import StateEKF
from n_cartpole.env.hardware_config import HardwareConfig, SensorModel

RIG = Path(__file__).parent.parent / "config" / "rig.toml"
NOISE, RES = 5e-4, 2 * np.pi / 16384


def _swing(physics, steps: int, seed: int):
    """Simulate a free swing (with a few force pulses); return true states and forces."""
    rng = np.random.default_rng(seed)
    s = np.array([0.0, 0.0, 2.6, 0.0, 3.4, 0.0])
    states, forces = [s.copy()], []
    for k in range(steps):
        f = 4.0 if (k // 40) % 2 == 0 else -4.0
        s = step(s, f, physics)
        states.append(s.copy())
        forces.append(f)
    return np.array(states), np.array(forces), rng


def test_ekf_recovers_velocities_far_better_than_differencing() -> None:
    """From noisy quantized positions alone, the EKF's velocity error is much lower."""
    phys = load_tqc_config(RIG).env.physics
    states, forces, rng = _swing(phys, 400, 0)
    n = 2
    pos_true = np.concatenate([states[:, :1], states[:, 2::2]], axis=1)
    sigma = np.array([2e-5, NOISE, NOISE])
    res = np.array([1e-5, RES, RES])
    meas = pos_true + rng.normal(size=pos_true.shape) * sigma
    meas = np.round(meas / res) * res
    ekf = StateEKF(
        phys, n, np.sqrt(sigma**2 + res**2 / 12), force_std=0.5, accel_std=1.0
    )  # tight model assumption: here the filter's model IS the plant
    est = [ekf.reset(meas[0])]
    for k in range(len(forces)):
        est.append(ekf.update(forces[k], meas[k + 1]))
    est = np.array(est)
    diff_v = np.diff(meas, axis=0) / phys.dt
    vel_idx = [3, 5]  # link angular velocities in the state
    skip = 40  # let the filter settle
    ekf_err = np.sqrt(np.mean((est[skip:, vel_idx] - states[skip:, vel_idx]) ** 2))
    diff_err = np.sqrt(np.mean((diff_v[skip - 1 :, 1:] - states[skip:, vel_idx]) ** 2))
    assert ekf_err < 0.4 * diff_err
    # positions are filtered too
    assert np.sqrt(np.mean((est[skip:, 2] - states[skip:, 2]) ** 2)) < NOISE


def test_ekf_is_accurate_without_noise() -> None:
    """With negligible reading noise the estimate converges onto the true state."""
    phys = load_tqc_config(RIG).env.physics
    states, forces, _ = _swing(phys, 300, 1)
    pos_true = np.concatenate([states[:, :1], states[:, 2::2]], axis=1)
    ekf = StateEKF(phys, 2, np.full(3, 1e-6), force_std=0.5, accel_std=5.0)
    est = ekf.reset(pos_true[0])
    for k in range(len(forces)):
        est = ekf.update(forces[k], pos_true[k + 1])
    np.testing.assert_allclose(est, states[-1], atol=5e-3)


def _env(estimator: str, noise: float = NOISE) -> NPendulumCartpole:
    cfg = dataclasses.replace(
        EnvConfig(),
        hardware=HardwareConfig(
            sysid_context=False,
            sensors=SensorModel(
                angle_noise_std=noise, angle_resolution=RES, estimator=estimator
            ),
        ),
        init_random_prob=0.0,
        max_steps=500,
    )
    return NPendulumCartpole(cfg)


def test_env_uses_the_ekf_and_beats_differencing_on_velocity_error() -> None:
    """Observed link velocity is closer to the truth with the EKF than with differencing."""
    errs = {}
    for est in ("diff", "ekf"):
        env = _env(est)
        env.reset(seed=3)
        e = []
        for k in range(300):
            obs, *_ = env.step(np.array([6.0 * np.sin(k / 15)]))
            if k > 30:
                e.append(obs[4] - env.get_state()[3])  # link-0 rate in the obs
        errs[est] = float(np.sqrt(np.mean(np.square(e))))
    assert errs["ekf"] < 0.5 * errs["diff"]


def test_ekf_restarts_every_episode_and_rejects_unknown_estimators() -> None:
    """A new episode re-initialises the filter; a bad estimator name is an error."""
    env = _env("ekf")
    obs, _ = env.reset(seed=0)
    assert obs[4] == 0.0 and obs[7] == 0.0  # velocities unknown at the first reading
    for _ in range(20):
        env.step(np.array([3.0]))
    obs2, _ = env.reset(seed=1)
    assert obs2[4] == 0.0 and obs2[7] == 0.0
    with pytest.raises(ValueError, match="estimator"):
        SensorModel(estimator="kalmann")
