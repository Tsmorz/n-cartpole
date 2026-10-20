"""Observation extras (past forces, past frames) and single-reading-per-step sensing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from n_cartpole.env.cartpole import (
    EnvConfig,
    NPendulumCartpole,
    extra_obs_dim,
    obs_dim,
    obs_mirror_sign,
)
from n_cartpole.env.dynamics import PhysicsParams
from n_cartpole.env.factory import env_spec
from n_cartpole.env.hardware_config import HardwareConfig, SensorModel
from n_cartpole.policy.loader import load_policy
from n_cartpole.training.off_policy import TQCConfig, TQCTrainer

KIN = obs_dim(2)


def test_defaults_add_nothing() -> None:
    """With no extras configured the observation is exactly the original one."""
    assert extra_obs_dim(2) == 0
    env = NPendulumCartpole(EnvConfig())
    assert env.OBS_DIM == KIN and env.reset(seed=0)[0].shape == (KIN,)


def test_dimensions_and_mirror_sign_agree_everywhere() -> None:
    """env, env_spec and obs_mirror_sign all describe the same layout."""
    for goal in (False, True):
        cfg = EnvConfig(goal_conditioned=goal, action_history=3, obs_history=2)
        env = NPendulumCartpole(cfg)
        dim, sign = env_spec(cfg)
        expected = KIN + (2 if goal else 0) + 3 + 2 * KIN
        assert env.OBS_DIM == dim == sign.size == expected
        assert env.reset(seed=0)[0].shape == (expected,)
        assert env.observation_space.shape == (expected,)
    sign = obs_mirror_sign(2, action_history=2, obs_history=1)
    np.testing.assert_array_equal(sign[KIN : KIN + 2], [-1, -1])  # forces are odd
    np.testing.assert_array_equal(
        sign[KIN + 2 :], sign[:KIN]
    )  # frames mirror as kinematics


def test_action_history_holds_recent_commands_newest_first() -> None:
    """Extras are the last commanded forces / force_max, newest first, zero at reset."""
    cfg = EnvConfig(action_history=3, force_slew=None)
    env = NPendulumCartpole(cfg)
    obs, _ = env.reset(seed=0)
    np.testing.assert_array_equal(obs[KIN:], 0.0)
    fmax = cfg.physics.force_max
    sent = [0.5 * fmax, -0.25 * fmax, 0.1 * fmax, 0.9 * fmax]
    for k, f in enumerate(sent):
        obs, *_ = env.step(np.array([f]))
        want = [x / fmax for x in sent[: k + 1][::-1]][:3]
        want += [0.0] * (3 - len(want))
        np.testing.assert_allclose(obs[KIN:], want, atol=1e-6)
    obs, _ = env.reset(seed=1)  # a new episode forgets them
    np.testing.assert_array_equal(obs[KIN:], 0.0)


def test_action_history_exposes_the_command_in_flight_under_delay() -> None:
    """With a 1-step delay the newest entry is the force the actuator applies next."""
    cfg = EnvConfig(
        action_history=1,
        force_slew=None,
        hardware=HardwareConfig(sysid_context=False, delay_steps=1),
    )
    env = NPendulumCartpole(cfg)
    env.reset(seed=0)
    f = 7.0
    obs, _, _, _, info1 = env.step(np.array([f]))
    assert info1["applied_force"] == 0.0  # the delay: nothing applied yet
    assert obs[KIN] == pytest.approx(f / cfg.physics.force_max)  # but it is visible
    _, _, _, _, info2 = env.step(np.array([0.0]))
    assert info2["applied_force"] == pytest.approx(f)  # applied one step later


def test_obs_history_is_the_previous_frames() -> None:
    """Past frames equal the earlier kinematic observations, newest first."""
    cfg = EnvConfig(obs_history=2, init_random_prob=0.0)
    env = NPendulumCartpole(cfg)
    obs0, _ = env.reset(seed=0)
    # before any step, the "past" is the first frame repeated
    np.testing.assert_array_equal(obs0[KIN : 2 * KIN], obs0[:KIN])
    np.testing.assert_array_equal(obs0[2 * KIN :], obs0[:KIN])
    frames = [obs0[:KIN].copy()]
    for _ in range(5):
        obs, *_ = env.step(np.array([3.0]))
        np.testing.assert_allclose(obs[KIN : 2 * KIN], frames[-1], atol=1e-7)
        prev2 = frames[-2] if len(frames) > 1 else frames[-1]
        np.testing.assert_allclose(obs[2 * KIN :], prev2, atol=1e-7)
        frames.append(obs[:KIN].copy())


def test_one_sensor_reading_per_step_survives_a_goal_switch() -> None:
    """A goal switch must not re-read the sensors (it used to zero the velocity)."""
    sensors = SensorModel()  # no noise, no quantization: velocity = exact difference
    cfg = EnvConfig(
        goal_conditioned=True,
        goal_hold_steps=(5, 6),
        hardware=HardwareConfig(sysid_context=False, delay_steps=0, sensors=sensors),
        init_random_prob=0.0,
        max_steps=300,
    )
    env = NPendulumCartpole(cfg)
    env.reset(seed=0, options={"start": "DD", "goal": "UU"})
    dt = cfg.physics.dt
    prev_theta = env.get_state()[2]
    switches = 0
    for _ in range(80):
        obs, _, term, trunc, info = env.step(np.array([4.0]))
        if "goal" in info and obs.shape and env.get_state() is not None:
            pass
        theta = env.get_state()[2]
        # obs layout: [x, xd, cos0, sin0, th0d, ...]; th0d is the differenced rate
        assert obs[4] == pytest.approx((theta - prev_theta) / dt, abs=1e-3)
        prev_theta = theta
        switches += int(env._seg_steps == 0)
        if term or trunc:
            break
    assert switches >= 1  # the test actually crossed goal switches


def test_make_obs_without_fresh_does_not_advance_the_estimator() -> None:
    """Re-issuing the observation reuses the reading: same values, same history."""
    sensors = SensorModel(angle_noise_std=1e-3)
    cfg = EnvConfig(
        obs_history=2,
        hardware=HardwareConfig(sysid_context=False, sensors=sensors),
    )
    env = NPendulumCartpole(cfg)
    env.reset(seed=0)
    env.step(np.array([1.0]))
    a = env._make_obs(fresh=False)
    b = env._make_obs(fresh=False)
    np.testing.assert_array_equal(a, b)
    assert env._meas_prev is not None
    meas_prev = env._meas_prev.copy()
    env._make_obs(fresh=False)
    np.testing.assert_array_equal(env._meas_prev, meas_prev)
    c = env._make_obs()  # a genuinely new reading differs (fresh noise)
    assert not np.array_equal(a[:KIN], c[:KIN])


def _tiny(tmp_path: Path, **env_kw) -> TQCConfig:
    return TQCConfig(
        env=EnvConfig(
            goal_conditioned=True,
            goal_hold_steps=(20, 40),
            physics=PhysicsParams(),
            **env_kw,
        ),
        device="cpu",
        hidden=16,
        batch_size=16,
        n_quantiles=4,
        start_steps=40,
        total_steps=100,
        buffer_size=400,
        log_every=1000,
        checkpoint_every=10_000,
        checkpoint_dir=tmp_path,
    )


def test_trainer_runs_with_extras_and_relabel_leaves_them_alone(tmp_path: Path) -> None:
    """Goal relabeling only rewrites the goal block; extras and mirroring stay valid."""
    cfg = _tiny(tmp_path, action_history=2, obs_history=1)
    trainer = TQCTrainer(cfg)
    trainer.train()
    n = cfg.env.n_links
    dim = KIN + n + 2 + KIN
    assert trainer.buffer.obs.shape[1] == dim and trainer._mirror.numel() == dim
    idx = np.arange(32)
    obs, _, rew, nxt, _ = trainer.buffer.sample(32, idx)
    before = obs.copy()
    trainer.cfg.relabel_frac = 1.0
    trainer._relabel(idx, obs, rew, nxt)
    changed = np.flatnonzero(np.any(obs != before, axis=0))
    assert set(changed) <= set(range(KIN, KIN + n))  # only the goal block moves
    # the checkpoint reloads and encodes a state to the right width
    trainer.save(tmp_path / "t.pt")
    bundle = load_policy(tmp_path / "t.pt")
    assert bundle.obs_dim == dim
    enc = bundle.encode(np.zeros((3, 6)), goal="UU")
    assert enc.shape == (3, dim) and np.all(enc[:, KIN + n : KIN + n + 2] == 0)


def test_warm_start_rejects_mismatched_history(tmp_path: Path) -> None:
    """A plain checkpoint without extras cannot warm-start a run that has them."""
    plain = TQCTrainer(
        TQCConfig(
            env=EnvConfig(),
            device="cpu",
            hidden=16,
            n_quantiles=4,
            buffer_size=10,
            checkpoint_dir=tmp_path,
        )
    )
    plain.save(tmp_path / "plain.pt")
    target = TQCTrainer(_tiny(tmp_path, action_history=2))
    with pytest.raises(ValueError, match="action_history"):
        target.warm_start(tmp_path / "plain.pt")
