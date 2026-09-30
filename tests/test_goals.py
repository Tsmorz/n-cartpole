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
from n_cartpole.env.randomization import PlantRandomization
from n_cartpole.policy.loader import load_policy
from n_cartpole.training.evaluate import (
    evaluate_transition_metrics,
    evaluate_transitions,
)
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

    stats, metrics = evaluate_transition_metrics(bundle, seconds=0.6, trials=1)
    assert stats.count[1:].sum() == 16
    assert all(len(metrics.trials[s][g]) == 1 for s in range(4) for g in range(4))
    # Peak force/speed are recorded for every pair and respect the actuator cap.
    assert np.all(metrics.peak_force() <= bundle.physics.force_max + 1e-6)
    assert np.all(metrics.peak_xdot() >= 0.0)
    assert "median time-to-settle" in metrics.format(goal_labels(2))

    # A randomized true plant is accepted and still yields metrics for every pair.
    _, noisy = evaluate_transition_metrics(
        bundle, seconds=0.6, trials=1, randomize=PlantRandomization()
    )
    assert all(len(noisy.trials[s][g]) == 1 for s in range(4) for g in range(4))


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


# ---------------------------------------------------------------------------
# Hold curriculum, warm start, entropy target
# ---------------------------------------------------------------------------


def test_hold_reset_starts_near_goal_and_holds_it() -> None:
    """A hold start sits near an equilibrium and commands that same goal."""
    env = _goal_env(hold_prob=1.0)
    env.hold_scale[:] = 0.3
    for seed in range(8):
        env.reset(seed=seed)
        dth = np.angle(np.exp(1j * (env.get_state()[2::2] - env.goal_angles)))
        assert np.all(np.abs(dth) <= 0.3 + env.cfg.init_noise)
        assert env._hold_seg and env._seg_from == FROM_OTHER


def test_pinned_reset_ignores_hold_prob() -> None:
    """Evaluation/replay resets pin start or goal and never become holds."""
    env = _goal_env(hold_prob=1.0)
    env.reset(seed=0, options={"start": "DD", "goal": "UU"})
    assert env.goal == "UU" and not env._hold_seg


def test_hold_scale_adapts_to_outcome() -> None:
    """A held segment widens that equilibrium's perturbation; a failure narrows it."""
    env = _goal_env(hold_prob=1.0, goal_hold_steps=(60, 60), hold_noise=(0.05, 1.0))
    env.hold_scale[:] = 0.1
    env.reset(seed=0)
    env._state[:] = 0.0  # force a DD hold at rest for a guaranteed success
    env._state[2::2] = np.pi
    env._begin_segment(3, FROM_OTHER)
    env._hold_seg = True
    for _ in range(60):
        _, _, _, _, info = env.step(np.array([0.0], dtype=np.float32))
    assert info["hold"] == (3, True)
    assert env.hold_scale[3] == pytest.approx(0.1 * env.HOLD_GROW)

    env._update_hold_scale(False)
    assert env.hold_scale[3] == pytest.approx(0.1 * env.HOLD_GROW / env.HOLD_SHRINK)
    for _ in range(100):
        env._update_hold_scale(False)
    assert env.hold_scale[3] == pytest.approx(0.05)  # clamped to the minimum


def test_hold_only_episode_ends_with_its_segment() -> None:
    """In hold-only mode the episode truncates when the hold segment closes."""
    env = _goal_env(goal_hold_steps=(30, 30))
    env.hold_only = True
    env.reset(seed=1)
    goal = env.goal
    env._state[:] = 0.0  # exactly at the hold equilibrium: never leaves the rail
    env._state[2::2] = env.goal_angles
    for _ in range(29):
        _, _, _, trunc, _ = env.step(np.array([0.0], dtype=np.float32))
        assert not trunc
    _, _, term, trunc, info = env.step(np.array([0.0], dtype=np.float32))
    assert trunc and not term and "segment" in info
    assert env.goal == goal  # no switch to a new goal


def test_default_target_entropy_is_in_force_units(tmp_path: Path) -> None:
    """Default target = -1 on the [-1, 1] action, converted to newtons."""
    trainer = TQCTrainer(_tiny_tqc_cfg(tmp_path))
    fmax = trainer.cfg.env.physics.force_max
    assert trainer.target_entropy == pytest.approx(math.log(fmax) - 1.0)
    trainer = TQCTrainer(_tiny_tqc_cfg(tmp_path, target_entropy=-1.0))
    assert trainer.target_entropy == -1.0


def test_goal_switch_transition_bootstraps_under_old_goal(tmp_path: Path) -> None:
    """The stored next_obs keeps the goal the transition was rewarded for."""
    trainer = TQCTrainer(_tiny_tqc_cfg(tmp_path))
    env = trainer.env
    obs, _ = env.reset(seed=0, options={"start": "DD", "goal": "DD"})
    env._seg_len = 1  # switch after the next step ...
    env._sample_goal = lambda frm: 0  # type: ignore[method-assign]  # ... to UU
    next_obs, *_, info = trainer._store_step(env, obs, np.zeros(1, np.float32))
    assert "segment" in info and env.goal == "UU"
    np.testing.assert_array_equal(next_obs[trainer._goal_slice], [1.0, 1.0])
    stored = trainer.buffer.next_obs[trainer.buffer.size - 1]
    np.testing.assert_array_equal(stored[trainer._goal_slice], [-1.0, -1.0])


def test_warm_start_reproduces_source_policy(tmp_path: Path) -> None:
    """A warm-started goal net acts like the plain source, whatever the goal."""
    src = TQCTrainer(TQCConfig(device="cpu", hidden=32, n_quantiles=8, buffer_size=10))
    src.norm.update(torch.randn(64, obs_dim(2)) * 2 + 1)
    with torch.no_grad():
        src.log_alpha.fill_(-3.0)
    src.save(tmp_path / "src.pt")

    goal = TQCTrainer(_tiny_tqc_cfg(tmp_path))
    goal.warm_start(tmp_path / "src.pt")
    assert goal.resumed and goal.alpha.item() == pytest.approx(math.exp(-3.0))

    kin = np.random.default_rng(0).normal(size=(5, obs_dim(2))).astype(np.float32)
    with torch.no_grad():
        ref_mean, _ = src.actor._mean_logstd(src.norm.normalize(torch.as_tensor(kin)))
        ref_q = src.critic(src.norm.normalize(torch.as_tensor(kin)), torch.ones(5, 1))
        for g in ([1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0]):
            o = np.concatenate([kin, np.tile(g, (5, 1))], axis=1).astype(np.float32)
            x = goal.norm.normalize(torch.as_tensor(o))
            mean, _ = goal.actor._mean_logstd(x)
            torch.testing.assert_close(mean, ref_mean)
            torch.testing.assert_close(goal.critic(x, torch.ones(5, 1)), ref_q)


def test_warm_start_rejects_mismatches(tmp_path: Path) -> None:
    """Goal checkpoints and differently shaped networks can't warm-start."""
    goal = TQCTrainer(_tiny_tqc_cfg(tmp_path))
    goal.save(tmp_path / "goal.pt")
    with pytest.raises(ValueError, match="already goal-conditioned"):
        goal.warm_start(tmp_path / "goal.pt")
    wide = TQCTrainer(TQCConfig(device="cpu", hidden=64, n_quantiles=8, buffer_size=10))
    wide.save(tmp_path / "wide.pt")
    with pytest.raises(ValueError, match="hidden"):
        goal.warm_start(tmp_path / "wide.pt")


def test_seed_buffer_stores_raw_transitions(tmp_path: Path) -> None:
    """Seeding fills the buffer (with raw states) and leaves the env's stats alone."""
    trainer = TQCTrainer(_tiny_tqc_cfg(tmp_path))
    trainer.seed_buffer(50)
    assert trainer.buffer.size == 50
    assert np.any(trainer.buffer.state[:50] != 0)
    assert trainer.env.transition_stats.count.sum() == 0
    assert int(trainer.norm.count) == 50


def test_hold_phase_switches_off(tmp_path: Path) -> None:
    """Training uses hold-only episodes until hold_phase_steps, then transitions."""
    trainer = TQCTrainer(_tiny_tqc_cfg(tmp_path, hold_phase_steps=60))
    trainer.train()
    assert not trainer.env.hold_only


def test_settle_metrics_for_held_equilibrium() -> None:
    """A do-nothing policy holding DD settles immediately with ~0 N force."""
    from n_cartpole.training.evaluate import _rollout

    class Zero:
        goal_conditioned = True
        physics = PhysicsParams()

        @staticmethod
        def normalize(obs):
            return obs

        @staticmethod
        def select_action(_obs):
            return np.zeros((1, 1))

    cfg = EnvConfig(goal_conditioned=True, goal_hold_steps=None, max_steps=300)
    env = NPendulumCartpole(cfg)
    trial, seg = _rollout(Zero(), env, start=3, goal=3, seed=0)  # type: ignore[arg-type]  # DD → DD
    assert trial.success and seg is not None and seg[2]
    assert trial.settle_time == 0.0
    assert trial.hold_rms_force == 0.0 and trial.peak_force == 0.0
    assert not trial.terminated
