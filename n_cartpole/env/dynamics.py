"""Lagrangian equations of motion for a cart with ``n`` serial pendulum links.

System: cart (mass ``M``) on a horizontal track + ``n`` point-mass pendulums in
series. Link ``i`` (0-indexed) has bob mass ``m_i`` at the tip of a massless rod
of length ``l_i``; its proximal joint (to the cart for ``i = 0``, otherwise to
link ``i - 1``) has viscous friction ``c_i``. All angles are **absolute**,
measured from the upward vertical (``0`` = upright).

Generalized coordinates: ``q = [x, theta_0, ..., theta_{n-1}]``
State vector: ``[x, x_dot, theta_0, theta_0_dot, ..., theta_{n-1}, theta_{n-1}_dot]``
Action: scalar horizontal force ``F`` applied to the cart (N).

EOM in manipulator form ``M(theta) q_ddot = rhs(q, qdot, F)`` with the
``(n+1) x (n+1)`` symmetric positive-definite mass matrix

    M[0, 0]        = M + sum_k m_k
    M[0, k+1]      = mu_k l_k cos(theta_k)
    M[j+1, k+1]    = mu_{max(j,k)} l_j l_k cos(theta_j - theta_k)

where ``mu_k = sum_{i >= k} m_i`` is the mass carried at or beyond link ``k``.
This closed form reduces **exactly** to the hand-derived single- and double-link
models it replaces (verified term-by-term, symbolically for n up to 4, and by
energy conservation in the tests). Using absolute angles is what makes the
velocity coupling purely centrifugal — only ``theta_dot_j**2`` terms appear, no
``theta_dot_i theta_dot_j`` cross terms — which keeps the ``rhs`` compact for
arbitrary ``n``.

No sympy at runtime: the equations above are evaluated directly with numpy. The
symbolic derivation is only used off-line to check this file (see the tests).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

# A per-link parameter: either one value shared by every link, or one per link.
LinkSpec = float | int | Sequence[float]


def _broadcast(spec: LinkSpec, n: int) -> np.ndarray:
    """Resolve a per-link spec to a length-``n`` float array.

    A scalar is broadcast to every link. A sequence of length ``n`` is used as
    is. A shorter sequence is extended by repeating its **last** value (so the
    2-link default ``(0.20, 0.15)`` gives ``(0.20, 0.15, 0.15, ...)`` for more
    links); a longer sequence is truncated to the first ``n`` (so a single-link
    env reads only link 0's value). This is what lets ``--links 3`` / ``4`` work
    out of the box while still allowing a fully explicit per-link build.
    """
    if isinstance(spec, int | float):
        return np.full(n, float(spec))
    vals = np.asarray(spec, dtype=float).ravel()
    if vals.size == 0:
        raise ValueError("per-link spec must not be empty")
    if vals.size == n:
        return vals
    if vals.size == 1:
        return np.full(n, vals[0])
    if vals.size < n:
        return np.concatenate([vals, np.full(n - vals.size, vals[-1])])
    return vals[:n]


@dataclass
class PhysicsParams:
    """Physical parameters of the ``n``-link pendulum cartpole.

    The cart terms (``M``, ``b``) and the global terms (``g``, ``dt``,
    ``force_max``, ``x_lim``) are scalars. The per-link terms are given as
    ``LinkSpec`` — a scalar shared by every link, or one value per link:

      - ``masses``: link bob masses (kg).
      - ``lengths``: rod lengths (m).
      - ``joint_friction``: viscous friction at each link's proximal joint,
        ``N·m/(rad/s)`` (``joint_friction[0]`` is the cart-link0 joint).

    Resolve them for a given link count with :meth:`link_masses`,
    :meth:`link_lengths`, and :meth:`joint_frictions`. Only ``masses = lengths =
    joint_friction = 0`` at the joints (and ``b = 0``) conserves energy.

    Defaults describe a small, buildable bench rig (belt-driven cart on a ~1 m
    rail, aluminium rods with end bobs) at a 100 Hz control loop. The pendulums
    are point masses on massless rods — a bob-on-rod idealization.
    """

    M: float = 1.0  # cart mass (kg)
    g: float = 9.81
    dt: float = 0.01  # 100 Hz control loop
    force_max: float = 20.0  # motor force limit (N)
    x_lim: float = 0.5  # rail half-length (m)
    b: float = 0.10  # cart viscous friction, N/(m/s)
    # Per-link specs (scalar = shared; sequence = per link). Defaults match the
    # historical 2-link rig (m1=0.20, m2=0.15; l=0.25; joint friction 0.002).
    masses: LinkSpec = (0.20, 0.15)
    lengths: LinkSpec = 0.25
    joint_friction: LinkSpec = 0.002

    def link_masses(self, n: int) -> np.ndarray:
        """Return the ``(n,)`` array of link bob masses."""
        return _broadcast(self.masses, n)

    def link_lengths(self, n: int) -> np.ndarray:
        """Return the ``(n,)`` array of rod lengths."""
        return _broadcast(self.lengths, n)

    def joint_frictions(self, n: int) -> np.ndarray:
        """Return the ``(n,)`` array of proximal-joint friction coefficients."""
        return _broadcast(self.joint_friction, n)

    # --- Backward-compatible scalar accessors (links 0 and 1) ----------------
    # Kept so existing code/tests that reach for l1/l2/m1/m2/c1/c2 keep working.
    @property
    def m1(self) -> float:
        """Link-0 bob mass (kg)."""
        return float(_broadcast(self.masses, 2)[0])

    @property
    def m2(self) -> float:
        """Link-1 bob mass (kg)."""
        return float(_broadcast(self.masses, 2)[1])

    @property
    def l1(self) -> float:
        """Link-0 rod length (m)."""
        return float(_broadcast(self.lengths, 2)[0])

    @property
    def l2(self) -> float:
        """Link-1 rod length (m)."""
        return float(_broadcast(self.lengths, 2)[1])

    @property
    def c1(self) -> float:
        """Cart-link0 joint friction."""
        return float(_broadcast(self.joint_friction, 2)[0])

    @property
    def c2(self) -> float:
        """Link0-link1 joint friction."""
        return float(_broadcast(self.joint_friction, 2)[1])

    def __setstate__(self, state: dict) -> None:
        """Unpickle, upgrading legacy checkpoints saved with m1/m2/l1/l2/c1/c2."""
        if "masses" not in state and "m1" in state:
            legacy = dict(state)
            state = {
                "M": legacy.get("M", 1.0),
                "g": legacy.get("g", 9.81),
                "dt": legacy.get("dt", 0.01),
                "force_max": legacy.get("force_max", 20.0),
                "x_lim": legacy.get("x_lim", 0.5),
                "b": legacy.get("b", 0.10),
                "masses": (legacy["m1"], legacy["m2"]),
                "lengths": (legacy["l1"], legacy["l2"]),
                "joint_friction": (legacy["c1"], legacy["c2"]),
            }
        self.__dict__.update(state)


def _mu(masses: np.ndarray) -> np.ndarray:
    """Cumulative mass carried at or beyond each link: ``mu_k = sum_{i>=k} m_i``."""
    return np.cumsum(masses[::-1])[::-1]


def mass_matrix(theta: np.ndarray, p: PhysicsParams) -> np.ndarray:
    """Return the ``(n+1) x (n+1)`` positive-definite mass matrix ``M(theta)``.

    Args:
        theta: link angles ``(n,)`` (absolute, 0 = upright).
        p: physics parameters.

    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    n = theta.size
    m = p.link_masses(n)
    ln = p.link_lengths(n)
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


def rhs(q: np.ndarray, qdot: np.ndarray, F: float, p: PhysicsParams) -> np.ndarray:
    """Return the right-hand side of ``M q_ddot = rhs``.

    Collects gravity, centrifugal, external-force, and viscous-friction terms.

    Args:
        q: generalized coordinates ``[x, theta_0, ..., theta_{n-1}]``.
        qdot: generalized velocities, same layout as ``q``.
        F: horizontal force on the cart (N).
        p: physics parameters.

    """
    n = q.size - 1
    theta = q[1:]
    thd = qdot[1:]
    xd = qdot[0]
    m = p.link_masses(n)
    ln = p.link_lengths(n)
    c = p.joint_frictions(n)
    mu = _mu(m)

    r = np.zeros(n + 1)
    # Cart row: external force, cart friction, and centrifugal reaction of links.
    r[0] = F - p.b * xd + float(np.sum(mu * ln * np.sin(theta) * thd**2))

    for k in range(n):
        # Gravity (positive: upright theta=0 is unstable).
        grav = mu[k] * p.g * ln[k] * np.sin(theta[k])
        # Centrifugal coupling from every other link (no Coriolis cross terms in
        # absolute angles).
        cor = 0.0
        for j in range(n):
            if j == k:
                continue
            cor += (
                mu[max(j, k)]
                * ln[k]
                * ln[j]
                * np.sin(theta[j] - theta[k])
                * thd[j] ** 2
            )
        # Viscous joint friction (dissipative, odd in velocity so mirror symmetry
        # is preserved). Each joint opposes the RELATIVE rate across it; the
        # child joint's reaction feeds back onto this link.
        omega_own = thd[k] - (thd[k - 1] if k >= 1 else 0.0)
        fric = -c[k] * omega_own
        if k + 1 < n:
            fric += c[k + 1] * (thd[k + 1] - thd[k])
        r[k + 1] = grav + cor + fric
    return r


def _ode(t: float, state: np.ndarray, F: float, p: PhysicsParams) -> np.ndarray:
    """ODE function for :func:`scipy.integrate.solve_ivp`."""
    xd = state[1]
    theta = state[2::2]
    thd = state[3::2]
    q = np.concatenate(([state[0]], theta))
    qdot = np.concatenate(([xd], thd))

    M = mass_matrix(theta, p)
    q_ddot = np.linalg.solve(M, rhs(q, qdot, F, p))

    out = np.empty_like(state)
    out[0] = xd
    out[1] = q_ddot[0]
    out[2::2] = thd
    out[3::2] = q_ddot[1:]
    return out


def step(state: np.ndarray, action: float, p: PhysicsParams) -> np.ndarray:
    """Integrate one timestep using RK45.

    Args:
        state: ``[x, x_dot, theta_0, theta_0_dot, ...]`` of length ``2 + 2n``.
        action: horizontal force applied to the cart (clipped to ``force_max``).
        p: physics parameters.

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
    theta = state[2::2]
    qdot = np.concatenate(([state[1]], state[3::2]))
    M = mass_matrix(theta, p)
    return 0.5 * float(qdot @ M @ qdot)


def potential_energy(state: np.ndarray, p: PhysicsParams) -> float:
    """Return total potential energy (measured from cart height)."""
    theta = state[2::2]
    n = theta.size
    mu = _mu(p.link_masses(n))
    ln = p.link_lengths(n)
    return float(p.g * np.sum(mu * ln * np.cos(theta)))


def total_energy(state: np.ndarray, p: PhysicsParams) -> float:
    """Return total mechanical energy KE + PE."""
    return kinetic_energy(state, p) + potential_energy(state, p)
