"""Tests for the double cartpole Gymnasium environment."""

from __future__ import annotations

import numpy as np
import pytest

from n_cartpole.env.double_cartpole import DoublePendulumCartpole, EnvConfig
from n_cartpole.env.dynamics import PhysicsParams


@pytest.fixture()
def env() -> DoublePendulumCartpole:
    """Environment with default config."""
    return DoublePendulumCartpole()


def test_observation_space_shape(env: DoublePendulumCartpole) -> None:
    """Observation space must be 8D (cos/sin encoding)."""
    assert env.observation_space.shape == (8,)


def test_action_space_shape(env: DoublePendulumCartpole) -> None:
    """Action space must be 1D (continuous force)."""
    assert env.action_space.shape == (1,)


def test_reset_returns_obs_in_space(env: DoublePendulumCartpole) -> None:
    """reset() must return an observation inside the observation space."""
    obs, info = env.reset(seed=0)
    assert obs.shape == (8,)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)


def test_step_signature(env: DoublePendulumCartpole) -> None:
    """step() must return (obs, reward, terminated, truncated, info)."""
    env.reset(seed=0)
    action = np.array([0.0], dtype=np.float32)
    result = env.step(action)
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert obs.shape == (8,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_obs_cos_sin_bounded(env: DoublePendulumCartpole) -> None:
    """cos/sin components of obs must lie in [-1, 1]."""
    obs, _ = env.reset(seed=7)
    # obs: [x, xd, cos1, sin1, td1, cos2, sin2, td2]
    # indices 2,3 are cos/sin of theta1; indices 5,6 are cos/sin of theta2
    for idx in [2, 3, 5, 6]:
        assert -1.0 <= obs[idx] <= 1.0, f"obs[{idx}]={obs[idx]} out of [-1,1]"


def test_cos_sin_unit_circle(env: DoublePendulumCartpole) -> None:
    """cos²(θ) + sin²(θ) must equal 1 for each pendulum angle."""
    obs, _ = env.reset(seed=13)
    norm1 = obs[2] ** 2 + obs[3] ** 2
    norm2 = obs[5] ** 2 + obs[6] ** 2
    assert norm1 == pytest.approx(1.0, abs=1e-5)
    assert norm2 == pytest.approx(1.0, abs=1e-5)


def test_reward_range(env: DoublePendulumCartpole) -> None:
    """Reward = bounded multiplicative base in (0, 1] + potential-based shaping.

    The shaping term F(s,s') = γ·Φ(s') − Φ(s) (Φ = r_angle ∈ [0, 1]) can go
    slightly negative when alignment regresses, so the total reward's tight
    bound is [-SHAPING_WEIGHT, 1 + SHAPING_WEIGHT·SHAPING_GAMMA], not (0, 1].
    """
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    w, g = env.SHAPING_WEIGHT, env.SHAPING_GAMMA
    lo, hi = -w - 1e-6, 1.0 + w * g + 1e-6
    for _ in range(50):
        action = rng.uniform(-1.0, 1.0, (1,)).astype(np.float32)
        _, reward, terminated, truncated, _ = env.step(action)
        assert lo <= reward <= hi, f"reward {reward} out of [{lo}, {hi}]"
        if terminated or truncated:
            env.reset()


def test_reward_max_at_upright_centered() -> None:
    """Reward approaches 1.0 only when both poles are upright, centered, at rest.

    Sets ``_prev_potential`` to match each probed state's own potential before
    calling ``_compute_reward``, isolating the base multiplicative reward from
    the potential-based shaping term (which depends on the *previous* state and
    is ~0 at steady state, up to the (1-γ)·Φ drag of a discounted potential).
    """
    env = DoublePendulumCartpole()
    env.reset(seed=0)
    env._state = np.zeros(6)  # both upright, centered, zero velocity
    env._prev_potential = env._angle_potential(env._state)
    r_up = env._compute_reward(env._state, 0.0)
    env._state = np.array([0.0, 0.0, np.pi, 0.0, np.pi, 0.0])  # both hanging down
    env._prev_potential = env._angle_potential(env._state)
    r_down = env._compute_reward(env._state, 0.0)
    assert r_up == pytest.approx(1.0, abs=0.01)
    assert r_down < 0.05


def test_truncation_at_max_steps() -> None:
    """Episode must truncate after max_steps even if cart stays in bounds."""
    cfg = EnvConfig(max_steps=5)
    env = DoublePendulumCartpole(cfg)
    env.reset(seed=0)

    truncated = False
    for _ in range(6):
        _, _, terminated, truncated, _ = env.step(np.array([0.0], dtype=np.float32))
        if terminated or truncated:
            break

    assert truncated


def test_termination_on_cart_out_of_bounds() -> None:
    """Episode must terminate when cart exits track bounds."""
    p = PhysicsParams(x_lim=0.1, force_max=100.0, dt=0.02)
    cfg = EnvConfig(physics=p, max_steps=10000)
    env = DoublePendulumCartpole(cfg)
    env.reset(seed=0)

    terminated = False
    for _ in range(500):
        _, _, terminated, truncated, _ = env.step(np.array([100.0], dtype=np.float32))
        if terminated or truncated:
            break

    assert terminated


def test_get_state_returns_copy(env: DoublePendulumCartpole) -> None:
    """get_state() must return a copy — mutating it should not affect env."""
    env.reset(seed=0)
    state = env.get_state()
    state[0] = 999.0
    assert env.get_state()[0] != 999.0


def test_action_clipping() -> None:
    """Actions beyond force_max must be silently clipped without error."""
    p = PhysicsParams(force_max=5.0)
    env = DoublePendulumCartpole(EnvConfig(physics=p))
    env.reset(seed=0)
    # Should not raise
    env.step(np.array([1000.0], dtype=np.float32))
    env.step(np.array([-1000.0], dtype=np.float32))
