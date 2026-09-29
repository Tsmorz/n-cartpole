"""Tests for the double pendulum cartpole dynamics."""

from __future__ import annotations

import numpy as np
import pytest

from n_cartpole.env.dynamics import (
    PhysicsParams,
    kinetic_energy,
    mass_matrix,
    potential_energy,
    step,
    total_energy,
)


@pytest.fixture()
def params() -> PhysicsParams:
    """Return default physics parameters."""
    return PhysicsParams()


def test_mass_matrix_shape(params: PhysicsParams) -> None:
    """Mass matrix should be 3x3."""
    M = mass_matrix(0.0, 0.0, params)
    assert M.shape == (3, 3)


def test_mass_matrix_symmetric(params: PhysicsParams) -> None:
    """Mass matrix must be symmetric."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        th1, th2 = rng.uniform(-np.pi, np.pi, 2)
        M = mass_matrix(th1, th2, params)
        np.testing.assert_allclose(M, M.T, atol=1e-12)


def test_mass_matrix_positive_definite(params: PhysicsParams) -> None:
    """Mass matrix must be positive definite (all eigenvalues > 0)."""
    rng = np.random.default_rng(1)
    for _ in range(50):
        th1, th2 = rng.uniform(-np.pi, np.pi, 2)
        M = mass_matrix(th1, th2, params)
        eigvals = np.linalg.eigvalsh(M)
        assert np.all(eigvals > 0), (
            f"Non-positive eigenvalue at th1={th1:.2f}, th2={th2:.2f}"
        )


def test_step_returns_correct_shape(params: PhysicsParams) -> None:
    """Step should return a 6-element state vector."""
    state = np.array([0.0, 0.0, np.pi, 0.0, np.pi, 0.0])
    next_state = step(state, 0.0, params)
    assert next_state.shape == (6,)


def test_energy_conservation_zero_force(params: PhysicsParams) -> None:
    """With zero applied force, total mechanical energy should be conserved."""
    rng = np.random.default_rng(42)
    state = np.array(
        [0.0, 0.2, rng.uniform(-np.pi, np.pi), 0.5, rng.uniform(-np.pi, np.pi), -0.3]
    )
    E0 = total_energy(state, params)

    for _ in range(500):
        state = step(state, 0.0, params)

    E_final = total_energy(state, params)
    # RK45 with max_step=dt/4 should conserve energy to ~1e-4 J over 10 s
    assert abs(E_final - E0) < 0.01, f"Energy drift: {abs(E_final - E0):.6f} J"


def test_hanging_equilibrium_is_stable(params: PhysicsParams) -> None:
    """Both poles at rest pointing straight down should remain stationary."""
    state = np.array([0.0, 0.0, np.pi, 0.0, np.pi, 0.0])

    for _ in range(100):
        state = step(state, 0.0, params)

    # Positions should not drift from equilibrium
    np.testing.assert_allclose(state[0], 0.0, atol=1e-6)  # x
    np.testing.assert_allclose(state[1], 0.0, atol=1e-6)  # x_dot
    np.testing.assert_allclose(state[3], 0.0, atol=1e-6)  # theta1_dot
    np.testing.assert_allclose(state[5], 0.0, atol=1e-6)  # theta2_dot


def test_upright_equilibrium_unstable(params: PhysicsParams) -> None:
    """Upright equilibrium with a tiny perturbation should grow (unstable)."""
    eps = 1e-4
    state = np.array([0.0, 0.0, eps, 0.0, eps, 0.0])
    deviation_0 = abs(state[2]) + abs(state[4])

    for _ in range(200):
        state = step(state, 0.0, params)

    deviation_final = abs(state[2]) + abs(state[4])
    # Perturbation must grow — upright is an unstable equilibrium
    assert deviation_final > deviation_0 * 10


def test_force_accelerates_cart(params: PhysicsParams) -> None:
    """Applying force should accelerate the cart in the force direction."""
    state = np.array([0.0, 0.0, np.pi, 0.0, np.pi, 0.0])
    next_pos = step(state, params.force_max, params)[0]
    assert next_pos > 0.0


def test_kinetic_energy_zero_at_rest(params: PhysicsParams) -> None:
    """With all velocities zero, kinetic energy should be zero."""
    state = np.array([0.0, 0.0, 1.2, 0.0, -0.5, 0.0])
    assert kinetic_energy(state, params) == pytest.approx(0.0, abs=1e-12)


def test_potential_energy_max_upright(params: PhysicsParams) -> None:
    """Potential energy is maximized when both pendulums are upright."""
    p = params
    state_up = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    state_down = np.array([0.0, 0.0, np.pi, 0.0, np.pi, 0.0])
    assert potential_energy(state_up, p) > potential_energy(state_down, p)
