"""Lagrangian equations of motion for a single pendulum cartpole.

System: cart (mass M) + one pendulum (point mass m1, rod length l1).

Generalized coordinates: q = [x, theta1]
  x      — cart horizontal position (m)
  theta1 — angle of the pendulum from vertical, 0 = upright (rad)

State vector: [x, x_dot, theta1, theta1_dot]
Action: scalar horizontal force F applied to the cart (N)

This is the ``n_links == 1`` special case of the double model in
``dynamics.py`` (drop link 2: m2 = l2 = c2 = 0). It uses the SAME sign and
angle conventions (theta = 0 upright, positive potential m·g·l·cos theta), the
same ``PhysicsParams`` (M, m1, l1, g, dt, force_max, x_lim, b, c1), and the same
RK45 integration, so a policy trained here transfers cleanly to the two-link
task as a warm start.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from n_cartpole.env.dynamics import PhysicsParams


def mass_matrix(theta1: float, p: PhysicsParams) -> np.ndarray:
    """Return the 2x2 positive-definite mass matrix M(theta)."""
    c1 = np.cos(theta1)
    return np.array(
        [
            [p.M + p.m1, p.m1 * p.l1 * c1],
            [p.m1 * p.l1 * c1, p.m1 * p.l1**2],
        ]
    )


def rhs(q: np.ndarray, qdot: np.ndarray, F: float, p: PhysicsParams) -> np.ndarray:
    """Return the right-hand side of M * q_ddot = rhs.

    Collects Coriolis/centrifugal, gravity, external force, and viscous friction.
    """
    theta1 = q[1]
    td1 = qdot[1]
    s1 = np.sin(theta1)

    # Row 0 (x): external force + centrifugal + cart viscous friction.
    r0 = F + p.m1 * p.l1 * s1 * td1**2 - p.b * qdot[0]
    # Row 1 (theta1): gravity + pivot viscous friction.
    r1 = p.m1 * p.g * p.l1 * s1 - p.c1 * td1
    return np.array([r0, r1])


def _ode(t: float, state: np.ndarray, F: float, p: PhysicsParams) -> np.ndarray:
    """ODE function for scipy.integrate.solve_ivp."""
    x, xd, th1, th1d = state
    q = np.array([x, th1])
    qdot = np.array([xd, th1d])

    M = mass_matrix(th1, p)
    b = rhs(q, qdot, F, p)
    q_ddot = np.linalg.solve(M, b)

    return np.array([xd, q_ddot[0], th1d, q_ddot[1]])


def step(state: np.ndarray, action: float, p: PhysicsParams) -> np.ndarray:
    """Integrate one timestep using RK45.

    Args:
        state: [x, x_dot, theta1, theta1_dot]
        action: horizontal force applied to the cart (clipped to force_max)
        p: physics parameters

    Returns:
        Next state array of the same shape.

    """
    F = float(np.clip(action, -p.force_max, p.force_max))
    sol = solve_ivp(
        _ode,
        t_span=(0.0, p.dt),
        y0=state,
        method="RK45",
        max_step=p.dt / 4,
        args=(F, p),
        dense_output=False,
    )
    return sol.y[:, -1]


def kinetic_energy(state: np.ndarray, p: PhysicsParams) -> float:
    """Return total kinetic energy of the system."""
    _x, xd, th1, th1d = state
    qdot = np.array([xd, th1d])
    M = mass_matrix(th1, p)
    return 0.5 * float(qdot @ M @ qdot)


def potential_energy(state: np.ndarray, p: PhysicsParams) -> float:
    """Return total potential energy (measured from cart height)."""
    th1 = state[2]
    return p.m1 * p.g * p.l1 * np.cos(th1)


def total_energy(state: np.ndarray, p: PhysicsParams) -> float:
    """Return total mechanical energy KE + PE."""
    return kinetic_energy(state, p) + potential_energy(state, p)
