"""Tests for the single-pendulum cartpole env, dynamics, and factory."""

from __future__ import annotations

import numpy as np

from n_cartpole.env.cartpole import EnvConfig, NPendulumCartpole
from n_cartpole.env.dynamics import PhysicsParams
from n_cartpole.env.factory import env_spec, make_env
from n_cartpole.env.single_cartpole import SinglePendulumCartpole
from n_cartpole.env.single_dynamics import mass_matrix, total_energy


def test_factory_selects_by_links() -> None:
    """make_env / env_spec must build the env width from n_links."""
    single = make_env(EnvConfig(n_links=1))
    double = make_env(EnvConfig(n_links=2))
    assert isinstance(single, NPendulumCartpole)
    assert single.n_links == 1
    assert double.n_links == 2
    assert single.observation_space.shape == (5,)
    assert double.observation_space.shape == (8,)
    assert env_spec(EnvConfig(n_links=1))[0] == 5
    assert env_spec(EnvConfig(n_links=2))[0] == 8
    assert env_spec(EnvConfig(n_links=1))[1].shape == (5,)


def test_single_env_contract() -> None:
    """reset/step must honor the gym API and 5D observation space."""
    env = SinglePendulumCartpole(EnvConfig(n_links=1))
    obs, _ = env.reset(seed=0)
    assert obs.shape == (5,)
    assert env.observation_space.contains(obs)
    obs, reward, terminated, truncated, _ = env.step(np.array([1.0], dtype=np.float32))
    assert obs.shape == (5,)
    assert 0.0 < reward <= 1.0
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_single_cos_sin_unit_circle() -> None:
    """cos/sin obs components must lie on the unit circle."""
    env = SinglePendulumCartpole(EnvConfig(n_links=1))
    obs, _ = env.reset(seed=13)
    assert abs(obs[2] ** 2 + obs[3] ** 2 - 1.0) < 1e-5


def test_single_mass_matrix_positive_definite() -> None:
    """The 2x2 mass matrix must be PD for any angle."""
    p = PhysicsParams()
    for th in np.linspace(-np.pi, np.pi, 25):
        M = mass_matrix(float(th), p)
        assert np.allclose(M, M.T)
        assert np.all(np.linalg.eigvalsh(M) > 0)


def test_single_energy_conservation_frictionless() -> None:
    """With no friction and no force, mechanical energy is conserved."""
    from n_cartpole.env.single_dynamics import step

    p = PhysicsParams(b=0.0, joint_friction=0.0)
    state = np.array([0.0, 0.0, np.pi / 3, 0.5])
    e0 = total_energy(state, p)
    for _ in range(200):
        state = step(state, 0.0, p)
    assert abs(total_energy(state, p) - e0) < 1e-2
