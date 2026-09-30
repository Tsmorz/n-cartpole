"""Tests for goal-conditioned transitions between up/down configurations."""

from __future__ import annotations

import dataclasses
import math
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from n_cartpole.env.cartpole import (
    EnvConfig,
    NPendulumCartpole,
    goal_reward,
    obs_dim,
    obs_mirror_sign,
    sysid_dim,
)
from n_cartpole.env.dynamics import PhysicsParams
from n_cartpole.env.factory import checkpoint_subdir, env_spec
from n_cartpole.env.goals import (
    FROM_OTHER,
    TransitionStats,
    goal_configs,
    goal_index,
    goal_labels,
    parse_goal_schedule,
)
from n_cartpole.env.hardware_config import HardwareConfig
from n_cartpole.policy.loader import load_policy
from n_cartpole.training.evaluate import evaluate_transitions
from n_cartpole.training.off_policy import TQCConfig, TQCTrainer


def _goal_env(**kw) -> NPendulumCartpole:
    """Double-link goal-conditioned env with overridable config."""
    return NPendulumCartpole(EnvConfig(goal_conditioned=True, **kw))


# ---------------------------------------------------------------------------
# goals.py
# ---------------------------------------------------------------------------


def test_goal_configs_and_labels() -> None:
    """2 links → UU, DU, UD, DD with 0/π targets, base link first."""
    assert goal_labels(2) == ["UU", "DU", "UD", "DD"]
    cfgs = goal_configs(2)
    np.testing.assert_allclose(cfgs[goal_index("DU", 2)], [np.pi, 0.0])
    np.testing.assert_allclose(cfgs[goal_index("UD", 2)], [0.0, np.pi])
    assert goal_index("dd", 2) == 3
    assert goal_index(2, 2) == 2


@pytest.mark.parametrize("bad", ["U", "UX", "UUU"])
def test_goal_index_rejects_bad_labels(bad: str) -> None:
    """Labels must have one U/D per link."""
    with pytest.raises(ValueError):
        goal_index(bad, 2)


def test_goal_index_rejects_out_of_range() -> None:
    """Indices must address one of the 2**n configurations."""
    with pytest.raises(ValueError):
        goal_index(4, 2)


def test_parse_goal_schedule() -> None:
    """Schedules parse to sorted (seconds, index) pairs; later goals need a time."""
    assert parse_goal_schedule("UU,DD@5,DU@2.5", 2) == [(0.0, 0), (2.5, 1), (5.0, 3)]
    with pytest.raises(ValueError):
        parse_goal_schedule("UU,DD", 2)
    with pytest.raises(ValueError):
        parse_goal_schedule(" ", 2)


def test_transition_stats_curriculum_and_matrix() -> None:
    """Failing transitions get more sampling weight; the matrix renders counts."""
    stats = TransitionStats(4, ema_rate=1.0)
    stats.record(0, 1, True)
    stats.record(0, 2, False)
    w = stats.sample_weights(0)
    assert w.sum() == pytest.approx(1.0)
    assert w[2] > w[1]
    assert stats.success_rate() == pytest.approx(0.5)
    table = stats.format_matrix(goal_labels(2))
    assert "random" not in table  # no random-start segments recorded
    stats.record(FROM_OTHER, 3, True)
    assert "random" in stats.format_matrix(goal_labels(2))
    assert math.isnan(TransitionStats(4).success_rate())


# ---------------------------------------------------------------------------
# Observation layout and symmetry
# ---------------------------------------------------------------------------


def test_goal_obs_layout() -> None:
    """Goal block (cos of targets) sits right after the kinematic obs."""
    env = _goal_env()
    obs, info = env.reset(seed=0, options={"start": "DD", "goal": "DU"})
    assert obs.shape == (obs_dim(2) + 2,)
    assert env.observation_space.contains(obs)
    np.testing.assert_allclose(obs[obs_dim(2) :], [-1.0, 1.0])
    assert info == {"goal": "DU"}
    np.testing.assert_allclose(env.get_state()[2::2], [np.pi, np.pi], atol=0.06)


def test_goal_obs_layout_with_sysid() -> None:
    """With hardware, the order is kinematics, goal, sysID; goal is noise-free."""
    hw = HardwareConfig(physics=PhysicsParams(), delay_steps=0, sensor_noise_std=0.5)
    cfg = EnvConfig(goal_conditioned=True, hardware=hw)
    env = NPendulumCartpole(cfg)
    obs, _ = env.reset(seed=0, options={"goal": "UD"})
    dim, sign = env_spec(cfg)
    assert obs.shape == (dim,) == (obs_dim(2) + 2 + sysid_dim(2),)
    np.testing.assert_array_equal(obs[obs_dim(2) : obs_dim(2) + 2], [1.0, -1.0])
    assert sign.shape == (dim,)


def test_goal_mirror_sign_is_positive() -> None:
    """Targets are 0/π, which mirroring leaves unchanged."""
    sign = obs_mirror_sign(2, with_goal=True)
    np.testing.assert_array_equal(sign[: obs_dim(2)], obs_mirror_sign(2))
    np.testing.assert_array_equal(sign[obs_dim(2) :], [1.0, 1.0])


def test_non_goal_env_unchanged() -> None:
    """Without goal conditioning the obs stays 8-D and the target is upright."""
    env = NPendulumCartpole(EnvConfig())
    obs, info = env.reset(seed=0)
    assert obs.shape == (8,)
    assert info == {}
    np.testing.assert_array_equal(env.goal_angles, [0.0, 0.0])
    with pytest.raises(RuntimeError):
        env.set_goal("DD")


def test_legacy_pickled_env_config() -> None:
    """EnvConfigs pickled before goal fields existed load as non-goal configs."""
    cfg = EnvConfig()
    state = {
        k: v
        for k, v in cfg.__dict__.items()
        if not k.startswith("goal_")  # simulate an old checkpoint
    }
    old = EnvConfig.__new__(EnvConfig)
    old.__dict__.update(state)
    restored = pickle.loads(pickle.dumps(old))  # noqa: S301 (own data)
    assert restored.goal_conditioned is False
    assert env_spec(restored)[0] == 8


def test_checkpoint_subdir() -> None:
    """Goal-conditioned runs get their own checkpoint folder."""
    assert checkpoint_subdir("tqc", False) == "tqc"
    assert checkpoint_subdir("tqc", True) == "tqc-goal"


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label", ["UU", "DU", "UD", "DD"])
def test_reward_peaks_at_goal(label: str) -> None:
    """At rest in the commanded configuration the base reward is ~1."""
    env = _goal_env()
    env.reset(seed=0, options={"goal": label})
    state = np.zeros(6)
    state[2::2] = env.goal_angles
    env._state = state
    env._prev_potential = env._angle_potential(state)
    assert env._compute_reward(state, 0.0) == pytest.approx(1.0, abs=0.01)
    # ...and far from it (every link flipped) it is ~0.
    flipped = state.copy()
    flipped[2::2] += np.pi
    env._prev_potential = env._angle_potential(flipped)
    assert env._compute_reward(flipped, 0.0) < 0.05


def test_goal_reward_matches_env_reward() -> None:
    """The vectorized relabeling reward reproduces the env reward step by step."""
    env = _goal_env(goal_hold_steps=(20, 30), init_random_prob=0.5)
    env.reset(seed=1)
    rng = np.random.default_rng(1)
    for _ in range(200):
        prev, goal = env.get_state(), env.goal_angles
        action = rng.uniform(-20, 20, (1,)).astype(np.float32)
        _, r, term, trunc, info = env.step(action)
        expected = goal_reward(
            prev,
            env.get_state(),
            info["applied_force"],
            info["dF_cmd"],
            goal,
            env.cfg.physics,
        )
        assert r == pytest.approx(float(expected), abs=1e-9)
        if term or trunc:
            env.reset()


def test_goal_switch_has_no_shaping_spike() -> None:
    """After a switch, shaping uses the new goal for both potentials."""
    env = _goal_env()
    env.reset(seed=0, options={"start": "UU", "goal": "UU"})
    env.set_goal("DD")
    # Upright is maximally misaligned with DD: potential ~0, so no big negative
    # term from the old (UU ≈ 1) potential.
    assert env._prev_potential < 0.01
    _, r, *_ = env.step(np.array([0.0], dtype=np.float32))
    assert r > -0.01


# ---------------------------------------------------------------------------
# Segments, switching, curriculum
# ---------------------------------------------------------------------------


def test_hold_down_segment_succeeds() -> None:
    """With zero force, starting and targeting DD is a successful segment."""
    env = _goal_env(goal_hold_steps=(100, 100))
    env.reset(seed=0, options={"start": "DD", "goal": "DD"})
    seg = None
    for _ in range(100):
        _, _, _, _, info = env.step(np.array([0.0], dtype=np.float32))
        seg = info.get("segment", seg)
    assert seg == (3, 3, True)
    assert env.transition_stats.count[4, 3] == 1


def test_auto_switch_updates_goal_in_obs() -> None:
    """When the hold time runs out, a new goal is drawn and shown in the obs."""
    env = _goal_env(goal_hold_steps=(5, 5), goal_curriculum=False)
    env.reset(seed=3, options={"start": "DD"})
    for _ in range(5):
        obs, _, _, _, info = env.step(np.array([0.0], dtype=np.float32))
    assert "segment" in info
    assert info["goal"] == env.goal
    np.testing.assert_allclose(obs[obs_dim(2) :], np.cos(env.goal_angles))
    assert env._seg_steps == 0


def test_termination_closes_segment_as_failure() -> None:
    """Leaving the rail ends the segment unsuccessfully."""
    p = PhysicsParams(x_lim=0.05)
    env = _goal_env(physics=p, goal_hold_steps=None)
    env.reset(seed=0, options={"start": "DD", "goal": "DD"})
    for _ in range(500):
        _, _, term, _, info = env.step(np.array([20.0], dtype=np.float32))
        if term:
            break
    assert term
    assert info["segment"] == (3, 3, False)


def test_short_segment_at_truncation_not_recorded() -> None:
    """A truncation right after a switch says nothing about that transition."""
    env = _goal_env(goal_hold_steps=(50, 50), max_steps=10)
    env.reset(seed=0)
    for _ in range(10):
        _, _, _, trunc, info = env.step(np.array([0.0], dtype=np.float32))
    assert trunc
    assert "segment" not in info


def test_set_goal_from_settled_config() -> None:
    """Commanding a goal while settled records the old goal as the start."""
    env = _goal_env(goal_hold_steps=None)
    env.reset(seed=0, options={"start": "DD", "goal": "DD"})
    for _ in range(env.cfg.goal_settle_steps + 5):
        env.step(np.array([0.0], dtype=np.float32))
    assert env.settled
    obs = env.set_goal("UU")
    assert env._seg_from == 3
    assert env.transition_stats.count[4, 3] == 1  # the DD hold was recorded
    assert env.goal == "UU"
    np.testing.assert_allclose(obs[obs_dim(2) :], [1.0, 1.0])


def test_curriculum_samples_failing_goals_more() -> None:
    """Goals with a high success EMA are drawn less often."""
    env = _goal_env()
    env.reset(seed=0)
    env.transition_stats.ema[:, :3] = 1.0  # UU, DU, UD mastered; DD not
    draws = [env._sample_goal(FROM_OTHER) for _ in range(400)]
    assert np.bincount(draws, minlength=4).argmax() == 3


def test_random_start_is_from_other() -> None:
    """A random reset marks the segment start as FROM_OTHER."""
    env = _goal_env(init_random_prob=1.0)
    env.reset(seed=0)
    assert env._seg_from == FROM_OTHER


# ---------------------------------------------------------------------------
# Trainers, loader, evaluation
# ---------------------------------------------------------------------------


def _tiny_tqc_cfg(tmp_path: Path, **kw) -> TQCConfig:
    return TQCConfig(
        env=EnvConfig(goal_conditioned=True, goal_hold_steps=(20, 40)),
        device="cpu",
        hidden=32,
        batch_size=32,
        n_quantiles=8,
        start_steps=64,
        total_steps=160,
        log_every=80,
        checkpoint_every=10_000,
        buffer_size=1000,
        checkpoint_dir=tmp_path,
        **kw,
    )


def test_tqc_relabel_recomputes_reward(tmp_path: Path) -> None:
    """Relabeled rows carry the new goal and the reward the env would give."""
    trainer = TQCTrainer(_tiny_tqc_cfg(tmp_path, relabel_frac=1.0))
    env = trainer.env
    obs, _ = env.reset(seed=0)
    for _ in range(50):
        s = env.get_state()
        nobs, r, term, trunc, info = env.step(np.array([3.0], dtype=np.float32))
        raw = (s, env.get_state(), info["applied_force"], info["dF_cmd"])
        trainer.buffer.add(obs, np.array([3.0]), r, nobs, term, raw)
        obs = nobs
    idx = np.arange(50)
    o, _, rew, no, _ = trainer.buffer.sample(50, idx)
    np.random.seed(0)
    trainer._relabel(idx, o, rew, no)
    b = trainer.buffer
    for i in range(50):
        g = np.where(o[i, trainer._goal_slice] > 0, 0.0, np.pi)
        np.testing.assert_array_equal(
            o[i, trainer._goal_slice], no[i, trainer._goal_slice]
        )
        expected = goal_reward(
            b.state[i], b.next_state[i], b.force[i], b.dforce[i], g, env.cfg.physics
        )
        assert rew[i, 0] == pytest.approx(float(expected), abs=1e-5)


def test_tqc_goal_training_smoke_and_resume(tmp_path: Path) -> None:
    """A goal-conditioned TQC run trains, saves, and reloads with its buffer."""
    cfg = _tiny_tqc_cfg(tmp_path)
    trainer = TQCTrainer(cfg)
    trainer.train()
    path = tmp_path / "tqc_latest.pt"
    assert path.exists()
    fresh = TQCTrainer(cfg)
    fresh.load(path)
    assert fresh.buffer.size == trainer.buffer.size
    np.testing.assert_array_equal(
        fresh.buffer.state[: fresh.buffer.size],
        trainer.buffer.state[: trainer.buffer.size],
    )

    bundle = load_policy(path)
    assert bundle.goal_conditioned
    assert bundle.obs_dim == obs_dim(2) + 2
    enc = bundle.encode(np.zeros((3, 6)), goal="DU")
    np.testing.assert_allclose(enc[:, obs_dim(2) :], [[-1.0, 1.0]] * 3)
    with pytest.raises(ValueError):
        bundle.encode(np.zeros((1, 6)))
    act = bundle.select_action(bundle.normalize(enc))
    assert act.shape == (3, 1)

    stats = evaluate_transitions(bundle, seconds=0.6, trials=1)
    assert stats.count[1:].sum() == 16  # every (start, goal) pair once


def test_buffer_without_raw_rejects_relabel_file(tmp_path: Path) -> None:
    """Loading a buffer that lacks raw transitions into a goal run fails loudly."""
    trainer = TQCTrainer(_tiny_tqc_cfg(tmp_path))
    plain = TQCTrainer(
        TQCConfig(device="cpu", hidden=32, n_quantiles=8, buffer_size=10)
    )
    plain.buffer.add(np.zeros(8), np.zeros(1), 0.0, np.zeros(8), False)
    plain.buffer.save(tmp_path / "b.npz")
    with pytest.raises(ValueError, match="raw transitions"):
        trainer.buffer.load(tmp_path / "b.npz")
    with pytest.raises(ValueError, match="raw="):
        trainer.buffer.add(np.zeros(10), np.zeros(1), 0.0, np.zeros(10), False)


def test_evaluate_rejects_non_goal_policy(tmp_path: Path) -> None:
    """The transition evaluator needs a goal-conditioned checkpoint."""
    trainer = TQCTrainer(
        TQCConfig(device="cpu", hidden=32, n_quantiles=8, buffer_size=10)
    )
    trainer.save(tmp_path / "t.pt")
    with pytest.raises(ValueError, match="goal-conditioned"):
        evaluate_transitions(load_policy(tmp_path / "t.pt"))


def test_play_goal_schedule(tmp_path: Path) -> None:
    """play.run_goal_episode follows the schedule and returns per-step goals."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("play", "scripts/play.py")
    assert spec is not None and spec.loader is not None
    play = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(play)

    trainer = TQCTrainer(_tiny_tqc_cfg(tmp_path))
    trainer.save(tmp_path / "t.pt")
    bundle = load_policy(tmp_path / "t.pt")
    env = NPendulumCartpole(
        dataclasses.replace(bundle.env_config, goal_hold_steps=None, max_steps=40)
    )
    sched = parse_goal_schedule("DD@0,UU@0.2", 2)
    states, actions, rewards, _, goals = play.run_goal_episode(
        bundle, env, sched, start="DD", seed=0
    )
    assert len(states) == len(goals) == 41
    np.testing.assert_allclose(goals[0], [np.pi, np.pi])
    np.testing.assert_allclose(goals[-1], [0.0, 0.0])
    assert torch.is_tensor(bundle.normalize(np.zeros((1, bundle.obs_dim))))
