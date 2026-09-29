"""Lagrangian equations of motion for a double pendulum cartpole.

System: cart (mass M) + pendulum 1 (point mass m1, rod length l1) +
        pendulum 2 (point mass m2, rod length l2) attached to the tip of pendulum 1.

Generalized coordinates: q = [x, theta1, theta2]
  x      — cart horizontal position (m)
  theta1 — angle of pendulum 1 from vertical, 0 = upright (rad)
  theta2 — angle of pendulum 2 from vertical, 0 = upright (rad)

State vector: [x, x_dot, theta1, theta1_dot, theta2, theta2_dot]
Action: scalar horizontal force F applied to the cart (N)

EOM derived via Lagrangian mechanics:
  M(theta) * q_ddot = rhs(q, q_dot, F)
where M is the 3x3 symmetric positive-definite mass matrix and rhs collects
Coriolis, centrifugal, gravity, and input terms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class PhysicsParams:
    """Physical parameters of the double pendulum cartpole."""

    M: float = 1.0
    m1: float = 0.3
    m2: float = 0.3
    l1: float = 0.5
    l2: float = 0.5
    g: float = 9.81
    dt: float = 0.02
    force_max: float = 20.0
    x_lim: float = 5.0


def mass_matrix(theta1: float, theta2: float, p: PhysicsParams) -> np.ndarray:
    """Return the 3x3 positive-definite mass matrix M(theta)."""
    c1 = np.cos(theta1)
    c2 = np.cos(theta2)
    c12 = np.cos(theta1 - theta2)
    M = np.array(
        [
            [
                p.M + p.m1 + p.m2,
                (p.m1 + p.m2) * p.l1 * c1,
                p.m2 * p.l2 * c2,
            ],
            [
                (p.m1 + p.m2) * p.l1 * c1,
                (p.m1 + p.m2) * p.l1**2,
                p.m2 * p.l1 * p.l2 * c12,
            ],
            [
                p.m2 * p.l2 * c2,
                p.m2 * p.l1 * p.l2 * c12,
                p.m2 * p.l2**2,
            ],
        ]
    )
    return M


def rhs(q: np.ndarray, qdot: np.ndarray, F: float, p: PhysicsParams) -> np.ndarray:
    """Return the right-hand side of M * q_ddot = rhs.

    Collects Coriolis/centrifugal, gravity, and external force terms.
    """
    theta1, theta2 = q[1], q[2]
    td1, td2 = qdot[1], qdot[2]
    s1 = np.sin(theta1)
    s2 = np.sin(theta2)
    s12 = np.sin(theta1 - theta2)

    # Row 0 (x): Coriolis/centrifugal + external force
    r0 = F + (p.m1 + p.m2) * p.l1 * s1 * td1**2 + p.m2 * p.l2 * s2 * td2**2

    # Row 1 (theta1): gravity + Coriolis (theta2 rotating relative to theta1)
    r1 = (p.m1 + p.m2) * p.g * p.l1 * s1 - p.m2 * p.l1 * p.l2 * s12 * td2**2

    # Row 2 (theta2): gravity + Coriolis (theta1 rotating relative to theta2)
    r2 = p.m2 * p.g * p.l2 * s2 + p.m2 * p.l1 * p.l2 * s12 * td1**2

    return np.array([r0, r1, r2])


def _ode(t: float, state: np.ndarray, F: float, p: PhysicsParams) -> np.ndarray:
    """ODE function for scipy.integrate.solve_ivp."""
    x, xd, th1, th1d, th2, th2d = state
    q = np.array([x, th1, th2])
    qdot = np.array([xd, th1d, th2d])

    M = mass_matrix(th1, th2, p)
    b = rhs(q, qdot, F, p)
    q_ddot = np.linalg.solve(M, b)

    return np.array([xd, q_ddot[0], th1d, q_ddot[1], th2d, q_ddot[2]])


def step(state: np.ndarray, action: float, p: PhysicsParams) -> np.ndarray:
    """Integrate one timestep using RK45.

    Args:
        state: [x, x_dot, theta1, theta1_dot, theta2, theta2_dot]
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
    _x, xd, th1, th1d, th2, th2d = state
    qdot = np.array([xd, th1d, th2d])
    M = mass_matrix(th1, th2, p)
    return 0.5 * float(qdot @ M @ qdot)


def potential_energy(state: np.ndarray, p: PhysicsParams) -> float:
    """Return total potential energy (measured from cart height)."""
    th1, th2 = state[2], state[4]
    return (p.m1 + p.m2) * p.g * p.l1 * np.cos(th1) + p.m2 * p.g * p.l2 * np.cos(th2)


def total_energy(state: np.ndarray, p: PhysicsParams) -> float:
    """Return total mechanical energy KE + PE."""
    return kinetic_energy(state, p) + potential_energy(state, p)
