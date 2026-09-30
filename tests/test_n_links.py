"""Tests that the general n-link dynamics/env scale past two links (n = 3, 4).

The single- and double-link cases are covered term-by-term in ``test_dynamics``,
``test_env`` and ``test_single``; here we check the general machinery holds for
higher link counts — energy conservation, a positive-definite mass matrix,
friction dissipation, the gym contract, and reward bounds — plus the parameter
broadcasting and legacy-checkpoint upgrade paths.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from n_cartpole.env.cartpole import (
    EnvConfig,
    NPendulumCartpole,
    encode_obs,
    obs_dim,
    obs_mirror_sign,
)
from n_cartpole.env.dynamics import (
    PhysicsParams,
    mass_matrix,
    step,
    total_energy,
)
from n_cartpole.env.factory import env_spec, links_name, make_env


@pytest.mark.parametrize("n", [3, 4])
def test_mass_matrix_positive_definite(n: int) -> None:
    """The (n+1)x(n+1) mass matrix is symmetric PD for any configuration."""
    p = PhysicsParams()
    rng = np.random.default_rng(n)
    for _ in range(40):
        theta = rng.uniform(-np.pi, np.pi, n)
        M = mass_matrix(theta, p)
        assert M.shape == (n + 1, n + 1)
        np.testing.assert_allclose(M, M.T, atol=1e-12)
        assert np.all(np.linalg.eigvalsh(M) > 0)


@pytest.mark.parametrize("n", [3, 4])
def test_energy_conservation_frictionless(n: int) -> None:
    """Frictionless, force-free: total mechanical energy is conserved."""
    p = PhysicsParams(b=0.0, joint_friction=0.0)
    rng = np.random.default_rng(100 + n)
    state = np.zeros(2 + 2 * n)
    state[1] = 0.2
    state[2::2] = rng.uniform(-np.pi, np.pi, n)
    state[3::2] = rng.uniform(-0.5, 0.5, n)

    e0 = total_energy(state, p)
    for _ in range(300):
        state = step(state, 0.0, p)
    # RK45 with max_step=dt/4 conserves energy tightly even for chaotic n-link.
    assert abs(total_energy(state, p) - e0) < 0.02, "energy drift too large"


@pytest.mark.parametrize("n", [3, 4])
def test_friction_dissipates_energy(n: int) -> None:
    """With viscous friction and no force, mechanical energy must decrease."""
    p = PhysicsParams()  # defaults include friction
    rng = np.random.default_rng(200 + n)
    state = np.zeros(2 + 2 * n)
    state[3::2] = rng.uniform(-1.0, 1.0, n)
    state[2::2] = rng.uniform(-np.pi, np.pi, n)
    e0 = total_energy(state, p)
    for _ in range(300):
        state = step(state, 0.0, p)
    assert total_energy(state, p) < e0


@pytest.mark.parametrize("n", [3, 4])
def test_env_contract(n: int) -> None:
    """reset/step honor the gym API and the 2+3n observation space."""
    env = make_env(EnvConfig(n_links=n))
    assert isinstance(env, NPendulumCartpole)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (obs_dim(n),)
    assert env.observation_space.contains(obs)

    rng = np.random.default_rng(0)
    w, g = env.SHAPING_WEIGHT, env.SHAPING_GAMMA
    lo, hi = -w - 1e-6, 1.0 + w * g + 1e-6
    for _ in range(30):
        obs, reward, terminated, truncated, _ = env.step(
            rng.uniform(-1, 1, (1,)).astype(np.float32)
        )
        assert obs.shape == (obs_dim(n),)
        assert lo <= reward <= hi
        if terminated or truncated:
            env.reset()


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_env_spec_and_helpers(n: int) -> None:
    """obs_dim / mirror sign / env_spec agree for every link count."""
    d, sign = env_spec(EnvConfig(n_links=n))
    assert d == 2 + 3 * n
    assert sign.shape == (2 + 3 * n,)
    np.testing.assert_array_equal(sign, obs_mirror_sign(n))
    # Mirror sign keeps cos (even) and flips x, ẋ, sin, θ̇ (odd).
    assert sign[0] == -1 and sign[1] == -1
    for k in range(n):
        assert sign[2 + 3 * k] == 1  # cos
        assert sign[3 + 3 * k] == -1  # sin
        assert sign[4 + 3 * k] == -1  # θ̇


def test_reward_max_at_upright_triple() -> None:
    """Reward → 1.0 only when all three poles are upright, centered, at rest."""
    env = NPendulumCartpole(EnvConfig(n_links=3))
    env.reset(seed=0)
    env._state = np.zeros(8)  # all upright, centered, zero velocity
    env._prev_potential = env._angle_potential(env._state)
    r_up = env._compute_reward(env._state, 0.0)

    down = np.zeros(8)
    down[2::2] = np.pi
    env._state = down
    env._prev_potential = env._angle_potential(env._state)
    r_down = env._compute_reward(env._state, 0.0)

    assert r_up == pytest.approx(1.0, abs=0.01)
    assert r_down < 0.05


def test_encode_obs_layout() -> None:
    """encode_obs lays out [x, ẋ, cos, sin, θ̇] per link, in order."""
    state = np.array([0.1, 0.2, 0.0, 1.0, np.pi, -1.0])  # 2 links
    obs = encode_obs(state)
    np.testing.assert_allclose(
        obs, [0.1, 0.2, 1.0, 0.0, 1.0, -1.0, 0.0, -1.0], atol=1e-7
    )


def test_param_broadcasting() -> None:
    """Scalar specs broadcast; short sequences extend by the last value."""
    p = PhysicsParams(masses=(0.2, 0.15), lengths=0.25, joint_friction=0.002)
    # scalar → every link
    np.testing.assert_allclose(p.link_lengths(4), [0.25, 0.25, 0.25, 0.25])
    np.testing.assert_allclose(p.joint_frictions(3), [0.002, 0.002, 0.002])
    # short sequence → extend last value to fill
    np.testing.assert_allclose(p.link_masses(4), [0.2, 0.15, 0.15, 0.15])
    # exact-length sequence → used as-is
    np.testing.assert_allclose(p.link_masses(2), [0.2, 0.15])
    # longer than n → truncated to the first n (single-link reads link 0)
    np.testing.assert_allclose(p.link_masses(1), [0.2])
    # explicit per-link build
    q = PhysicsParams(masses=(0.3, 0.2, 0.1), lengths=(0.2, 0.25, 0.3))
    np.testing.assert_allclose(q.link_masses(3), [0.3, 0.2, 0.1])
    np.testing.assert_allclose(q.link_lengths(3), [0.2, 0.25, 0.3])


def test_legacy_checkpoint_params_upgrade() -> None:
    """A PhysicsParams pickled with the old m1/m2/l1/l2/c1/c2 fields still loads."""
    legacy_state = {
        "M": 1.0,
        "m1": 0.20,
        "m2": 0.15,
        "l1": 0.25,
        "l2": 0.30,
        "g": 9.81,
        "dt": 0.01,
        "force_max": 20.0,
        "x_lim": 0.5,
        "b": 0.10,
        "c1": 0.002,
        "c2": 0.003,
    }
    obj = PhysicsParams.__new__(PhysicsParams)
    obj.__setstate__(legacy_state)
    assert obj.m1 == 0.20 and obj.m2 == 0.15
    assert obj.l1 == 0.25 and obj.l2 == 0.30
    assert obj.c1 == 0.002 and obj.c2 == 0.003
    np.testing.assert_allclose(obj.link_masses(2), [0.20, 0.15])
    # round-trips through pickle in the new form
    reloaded = pickle.loads(pickle.dumps(obj))  # noqa: S301
    np.testing.assert_allclose(reloaded.link_lengths(2), [0.25, 0.30])


def test_links_name() -> None:
    """Checkpoint folder names cover the common counts and fall back cleanly."""
    assert links_name(1) == "single"
    assert links_name(2) == "double"
    assert links_name(3) == "triple"
    assert links_name(4) == "quadruple"
    assert links_name(5) == "5link"
