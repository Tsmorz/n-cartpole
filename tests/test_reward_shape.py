"""Optional reward shaping: energy gap, rail margin, hold effort / steadiness."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from n_cartpole.config import load_tqc_config
from n_cartpole.env.cartpole import (
    EnvConfig,
    NPendulumCartpole,
    RewardShape,
    base_reward,
    energy_potential,
    goal_reward,
)
from n_cartpole.env.dynamics import PhysicsParams
from n_cartpole.training.off_policy import TQCConfig, TQCTrainer

RIG = Path(__file__).parent.parent / "config" / "rig.toml"
SHAPE = RewardShape(energy_weight=0.1, wall_weight=0.5, effort_weight=0.5, vel_k=0.5)
UU = np.zeros(2)
DD = np.full(2, np.pi)


def _state(x=0.0, xd=0.0, th=(0.0, 0.0), thd=(0.0, 0.0)) -> np.ndarray:
    return np.array([x, xd, th[0], thd[0], th[1], thd[1]], dtype=float)


def _rig_env(**kw) -> NPendulumCartpole:
    cfg = load_tqc_config(RIG).env
    return NPendulumCartpole(
        dataclasses.replace(
            cfg, goal_conditioned=True, reward_shape=SHAPE, hardware=None, **kw
        )
    )


def test_zero_weights_reproduce_the_original_reward() -> None:
    """``RewardShape()`` is indistinguishable from no shaping at all."""
    p = PhysicsParams()
    rng = np.random.default_rng(0)
    for _ in range(50):
        prev = rng.uniform(-2, 2, 6)
        cur = rng.uniform(-2, 2, 6)
        goal = rng.choice([0.0, np.pi], 2)
        F, dF = rng.uniform(-20, 20, 2)
        np.testing.assert_allclose(
            goal_reward(prev, cur, F, dF, goal, p, RewardShape()),
            goal_reward(prev, cur, F, dF, goal, p),
        )


def test_energy_potential_has_no_dead_zone_at_hanging_down() -> None:
    """Pumping energy from still-DD is rewarded; the angle potential alone is not."""
    p = PhysicsParams()
    shape = RewardShape(energy_weight=0.1)
    still = _state(th=(np.pi, np.pi))
    swinging = _state(th=(np.pi, np.pi), thd=(1.0, 1.0))
    for goal in (UU, np.array([0.0, np.pi]), np.array([np.pi, 0.0])):
        phi0 = energy_potential(still, goal, p, shape)
        phi1 = energy_potential(swinging, goal, p, shape)
        assert phi1 > phi0  # more energy, closer to the goal's energy
        # A step from still to swinging pays, where the angle term pays ~0.
        step_reward = goal_reward(still, swinging, 0.0, 0.0, goal, p, shape)
        assert step_reward > goal_reward(still, swinging, 0.0, 0.0, goal, p)
    # No energy gap at the goal pose at rest.
    assert energy_potential(_state(), UU, p, shape) == pytest.approx(0.0)
    assert energy_potential(_state(th=(np.pi, np.pi)), DD, p, shape) == pytest.approx(
        0.0
    )
    # Off -> exactly zero.
    assert energy_potential(still, UU, p, None) == 0.0


def test_energy_potential_is_bounded_and_mirror_invariant() -> None:
    """The potential stays in [-w, 0] and is even under left-right mirroring."""
    p = load_tqc_config(RIG).env.physics
    shape = RewardShape(energy_weight=0.1)
    rng = np.random.default_rng(1)
    for _ in range(100):
        s = rng.uniform(-3, 3, 6)
        goal = rng.choice([0.0, np.pi], 2)
        phi = energy_potential(s, goal, p, shape)
        assert -0.1 - 1e-12 <= phi <= 1e-12
        assert phi == pytest.approx(float(energy_potential(-s, goal, p, shape)))


def test_wall_and_effort_terms() -> None:
    """The wall penalty bites near the rail ends; effort only bites at the goal."""
    p = PhysicsParams()
    at_goal = _state()
    mid_swing = _state(th=(np.pi, np.pi))
    plain = RewardShape()
    wall = RewardShape(wall_weight=0.5)
    eff = RewardShape(effort_weight=0.5, effort_scale=2.0)
    # Wall: untouched inside the margin, reduced near the end-stop.
    assert base_reward(_state(x=0.3), 0, 0, UU, p, wall) == pytest.approx(
        base_reward(_state(x=0.3), 0, 0, UU, p, plain)
    )
    assert base_reward(_state(x=0.49), 0, 0, UU, p, wall) < 0.8 * base_reward(
        _state(x=0.49), 0, 0, UU, p, plain
    )
    # Effort: a 6 N push at the goal costs a lot; mid-transition it costs ~nothing.
    assert base_reward(at_goal, 6.0, 0, UU, p, eff) < 0.7 * base_reward(
        at_goal, 6.0, 0, UU, p, plain
    )
    far = base_reward(mid_swing, 6.0, 0, UU, p, plain)
    assert base_reward(mid_swing, 6.0, 0, UU, p, eff) == pytest.approx(
        far, rel=1e-3, abs=1e-6
    )
    # A quiet hold is nearly free.
    assert base_reward(at_goal, 0.2, 0, UU, p, eff) > 0.97 * base_reward(
        at_goal, 0.2, 0, UU, p, plain
    )


def test_shaped_base_reward_is_bounded_and_mirror_invariant() -> None:
    """With every term on, the base reward stays in [0, 1] and is mirror-even."""
    p = load_tqc_config(RIG).env.physics
    rng = np.random.default_rng(2)
    for _ in range(200):
        s = rng.uniform(-3, 3, 6)
        s[0] = rng.uniform(-p.x_lim, p.x_lim)
        goal = rng.choice([0.0, np.pi], 2)
        F, dF = rng.uniform(-p.force_max, p.force_max, 2)
        r = base_reward(s, F, dF, goal, p, SHAPE)
        assert 0.0 <= r <= 1.0
        assert r == pytest.approx(float(base_reward(-s, -F, -dF, goal, p, SHAPE)))


def test_shaped_goal_reward_matches_env_reward_with_rigid_links() -> None:
    """Relabeling's vectorized reward equals the env's, shaping on, goals switching."""
    env = _rig_env(goal_hold_steps=(20, 30), init_random_prob=0.5)
    env.reset(seed=3)
    rng = np.random.default_rng(3)
    for _ in range(300):
        prev, goal = env.get_state(), env.goal_angles
        action = rng.uniform(-15, 15, (1,)).astype(np.float32)
        _, r, term, trunc, info = env.step(action)
        expected = goal_reward(
            prev,
            env.get_state(),
            info["applied_force"],
            info["dF_cmd"],
            goal,
            env.cfg.physics,
            env.cfg.reward_shape,
        )
        assert r == pytest.approx(float(expected), abs=1e-9)
        if term or trunc:
            env.reset()


def test_goal_switch_has_no_energy_shaping_spike() -> None:
    """Re-commanding the goal re-anchors the energy potential too."""
    env = _rig_env()
    env.reset(seed=0, options={"start": "UU", "goal": "UU"})
    env.set_goal("DD")
    assert env._prev_energy_potential == pytest.approx(
        env._energy_potential(env._state)
    )
    _, r, *_ = env.step(np.array([0.0], dtype=np.float32))
    assert abs(r) < 0.05


def test_relabel_uses_the_shaped_reward(tmp_path: Path) -> None:
    """Replay relabeling recomputes rewards with the env's reward shape."""
    shape = SHAPE
    cfg = TQCConfig(
        env=EnvConfig(
            goal_conditioned=True, goal_hold_steps=(20, 40), reward_shape=shape
        ),
        device="cpu",
        hidden=32,
        batch_size=32,
        n_quantiles=8,
        start_steps=64,
        total_steps=160,
        buffer_size=1000,
        checkpoint_dir=tmp_path,
        relabel_frac=1.0,
    )
    trainer = TQCTrainer(cfg)
    env = trainer.env
    obs, _ = env.reset(seed=0)
    for _ in range(50):
        s = env.get_state()
        nobs, r, term, trunc, info = env.step(np.array([8.0], dtype=np.float32))
        raw = (s, env.get_state(), info["applied_force"], info["dF_cmd"])
        trainer.buffer.add(obs, np.array([8.0]), r, nobs, term, raw)
        obs = nobs
    idx = np.arange(50)
    o, _, rew, no, _ = trainer.buffer.sample(50, idx)
    np.random.seed(0)
    trainer._relabel(idx, o, rew, no)
    b = trainer.buffer
    for i in range(50):
        g = np.where(o[i, trainer._goal_slice] > 0, 0.0, np.pi)
        expected = goal_reward(
            b.state[i],
            b.next_state[i],
            b.force[i],
            b.dforce[i],
            g,
            env.cfg.physics,
            shape,
        )
        assert rew[i, 0] == pytest.approx(float(expected), abs=1e-5)


def test_reward_shape_from_toml_and_legacy_envconfig(tmp_path: Path) -> None:
    """``[reward]`` parses; configs without it (and old pickles) stay unshaped."""
    assert load_tqc_config(RIG).env.reward_shape is not None
    assert EnvConfig().reward_shape is None
    toml = tmp_path / "c.toml"
    toml.write_text("[reward]\nenergy_weight = 0.3\nwall_start = 0.6\n")
    shape = load_tqc_config(toml).env.reward_shape
    assert shape is not None and shape.energy_weight == 0.3 and shape.wall_start == 0.6
