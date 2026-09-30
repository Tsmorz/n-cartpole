"""Rigid-body links (centre of mass + inertia) in the n-link dynamics."""

from __future__ import annotations

import numpy as np
import pytest

from n_cartpole.env.dynamics import (
    PhysicsParams,
    _mu,
    mass_matrix,
    potential_energy,
    rhs,
    step,
    total_energy,
    total_energy_batch,
)


def _legacy_mass_matrix(theta: np.ndarray, p: PhysicsParams) -> np.ndarray:
    """Return the original point-mass mass matrix (mu_max(j,k) l_j l_k cos(...))."""
    n = theta.size
    m, ln = p.link_masses(n), p.link_lengths(n)
    mu = _mu(m)
    M = np.zeros((n + 1, n + 1))
    M[0, 0] = p.M + m.sum()
    for k in range(n):
        M[0, k + 1] = M[k + 1, 0] = mu[k] * ln[k] * np.cos(theta[k])
    for j in range(n):
        for k in range(n):
            M[j + 1, k + 1] = (
                mu[max(j, k)] * ln[j] * ln[k] * np.cos(theta[j] - theta[k])
            )
    return M


def _legacy_rhs(q, qdot, F, p: PhysicsParams) -> np.ndarray:
    """Return the original point-mass right-hand side."""
    n = q.size - 1
    theta, thd, xd = q[1:], qdot[1:], qdot[0]
    m, ln, c = p.link_masses(n), p.link_lengths(n), p.joint_frictions(n)
    mu = _mu(m)
    r = np.zeros(n + 1)
    r[0] = F - p.b * xd + float(np.sum(mu * ln * np.sin(theta) * thd**2))
    for k in range(n):
        grav = mu[k] * p.g * ln[k] * np.sin(theta[k])
        cor = 0.0
        for j in range(n):
            if j != k:
                cor += (
                    mu[max(j, k)]
                    * ln[k]
                    * ln[j]
                    * np.sin(theta[j] - theta[k])
                    * thd[j] ** 2
                )
        omega_own = thd[k] - (thd[k - 1] if k >= 1 else 0.0)
        fric = -c[k] * omega_own
        if k + 1 < n:
            fric += c[k + 1] * (thd[k + 1] - thd[k])
        r[k + 1] = grav + cor + fric
    return r


def _rigid(n: int, rng: np.random.Generator, **kw) -> PhysicsParams:
    """Random rigid-body links: COM inside the link, inertia of a bar+tip mass."""
    masses = rng.uniform(0.05, 0.3, n)
    lengths = rng.uniform(0.15, 0.35, n)
    com = lengths * rng.uniform(0.3, 0.9, n)
    inertia = masses * lengths**2 * rng.uniform(0.02, 0.12, n)
    return PhysicsParams(
        M=rng.uniform(0.5, 1.5),
        masses=tuple(masses),
        lengths=tuple(lengths),
        com=tuple(com),
        inertia=tuple(inertia),
        **kw,
    )


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_point_mass_defaults_match_legacy_formulas(n: int) -> None:
    """com=None / inertia=0 reproduces the original model exactly."""
    rng = np.random.default_rng(n)
    p = PhysicsParams(masses=tuple(rng.uniform(0.1, 0.4, n)), lengths=0.25, b=0.2)
    for _ in range(20):
        q = rng.uniform(-np.pi, np.pi, n + 1)
        qd = rng.uniform(-4, 4, n + 1)
        F = rng.uniform(-20, 20)
        np.testing.assert_allclose(mass_matrix(q[1:], p), _legacy_mass_matrix(q[1:], p))
        np.testing.assert_allclose(
            rhs(q, qd, F, p), _legacy_rhs(q, qd, F, p), atol=1e-12
        )


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_rigid_mass_matrix_is_positive_definite(n: int) -> None:
    """Rigid-body mass matrices are symmetric positive definite."""
    rng = np.random.default_rng(10 + n)
    p = _rigid(n, rng)
    for _ in range(50):
        M = mass_matrix(rng.uniform(-np.pi, np.pi, n), p)
        np.testing.assert_allclose(M, M.T)
        assert np.linalg.eigvalsh(M).min() > 0


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_rigid_rhs_satisfies_euler_lagrange(n: int) -> None:
    """Rhs equals the Euler-Lagrange forcing computed by finite differences.

    For ``T = 1/2 qd' M(theta) qd`` and ``V = g sum a_k cos(theta_k)``, frictionless,
    ``M qdd = -dM/dt qd + dT/dq - dV/dq + [F, 0, ...]``. The right side is
    evaluated from ``mass_matrix`` and ``potential_energy`` alone, independently
    of the hand-written ``rhs`` terms.
    """
    rng = np.random.default_rng(20 + n)
    p = _rigid(n, rng, b=0.0, joint_friction=0.0)
    h = 1e-6
    for _ in range(10):
        theta = rng.uniform(-np.pi, np.pi, n)
        x, xd = rng.uniform(-0.3, 0.3), rng.uniform(-2, 2)
        thd = rng.uniform(-4, 4, n)
        F = rng.uniform(-10, 10)
        qdot = np.concatenate(([xd], thd))

        def kin(th: np.ndarray, qdot: np.ndarray = qdot) -> float:
            return 0.5 * float(qdot @ mass_matrix(th, p) @ qdot)

        def pot(th: np.ndarray) -> float:
            st = np.zeros(2 + 2 * n)
            st[2::2] = th
            return potential_energy(st, p)

        dT = np.zeros(n + 1)
        dV = np.zeros(n + 1)
        Mdot = np.zeros((n + 1, n + 1))
        for k in range(n):
            e = np.zeros(n)
            e[k] = h
            dT[k + 1] = (kin(theta + e) - kin(theta - e)) / (2 * h)
            dV[k + 1] = (pot(theta + e) - pot(theta - e)) / (2 * h)
            Mdot += (
                (mass_matrix(theta + e, p) - mass_matrix(theta - e, p))
                / (2 * h)
                * thd[k]
            )
        expected = -Mdot @ qdot + dT - dV
        expected[0] += F
        q = np.concatenate(([x], theta))
        np.testing.assert_allclose(rhs(q, qdot, F, p), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_rigid_energy_conservation(n: int) -> None:
    """Frictionless rigid-body links conserve total energy."""
    rng = np.random.default_rng(30 + n)
    p = _rigid(n, rng, b=0.0, joint_friction=0.0)
    # Gentle velocities like test_n_links: from a violent chaotic start the RK45
    # default tolerance, not the model, limits conservation.
    state = np.zeros(2 + 2 * n)
    state[1] = 0.2
    state[2::2] = rng.uniform(-np.pi, np.pi, n)
    state[3::2] = rng.uniform(-0.5, 0.5, n)
    e0 = total_energy(state, p)
    for _ in range(300):
        state = step(state, 0.0, p)
    assert abs(total_energy(state, p) - e0) < 2e-3 * max(1.0, abs(e0))


def test_total_energy_batch_matches_scalar() -> None:
    """The batched energy equals the scalar energy row by row."""
    rng = np.random.default_rng(5)
    p = _rigid(3, rng)
    states = rng.uniform(-2, 2, (7, 8))
    np.testing.assert_allclose(
        total_energy_batch(states, p), [total_energy(s, p) for s in states]
    )
    assert total_energy_batch(states[0], p).shape == ()


def test_legacy_pickled_params_default_to_point_masses() -> None:
    """Checkpoints pickled before com/inertia existed load as point masses."""
    p = PhysicsParams.__new__(PhysicsParams)
    p.__setstate__(
        {
            "M": 1.0,
            "g": 9.81,
            "dt": 0.01,
            "force_max": 20.0,
            "x_lim": 0.5,
            "b": 0.1,
            "masses": (0.2, 0.15),
            "lengths": 0.25,
            "joint_friction": 0.002,
        }
    )
    np.testing.assert_allclose(p.link_coms(2), p.link_lengths(2))
    np.testing.assert_allclose(p.link_inertias(2), 0.0)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_fast_ode_matches_mass_matrix_and_rhs(n: int) -> None:
    """The integrator's precomputed-coefficient path equals solve(M, rhs)."""
    from n_cartpole.env.dynamics import _model, _ode

    rng = np.random.default_rng(40 + n)
    p = _rigid(n, rng, b=0.15, joint_friction=tuple(rng.uniform(0.001, 0.01, n)))
    mdl = _model(p, n)
    for _ in range(20):
        state = np.empty(2 + 2 * n)
        state[:] = rng.uniform(-3, 3, state.size)
        F = rng.uniform(-15, 15)
        q = np.concatenate(([state[0]], state[2::2]))
        qd = np.concatenate(([state[1]], state[3::2]))
        acc = np.linalg.solve(mass_matrix(q[1:], p), rhs(q, qd, F, p))
        out = _ode(0.0, state, F, mdl)
        np.testing.assert_allclose(out[1], acc[0], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(out[3::2], acc[1:], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(out[0::2], state[1::2])
