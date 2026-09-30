"""Lagrangian equations of motion for a cart with ``n`` serial pendulum links.

System: cart (mass ``M``) on a horizontal track + ``n`` rigid links in series.
Link ``i`` (0-indexed) has mass ``m_i``, length ``l_i`` (proximal joint to distal
joint), centre of mass ``c_i`` from its proximal joint, and moment of inertia
``I_i`` about its centre of mass. The defaults ``c_i = l_i``, ``I_i = 0`` make
each link a point-mass bob on a massless rod (the original model); a real bar
has ``c_i`` inside the rod and ``I_i > 0``. Its proximal joint (to the cart for
``i = 0``, otherwise to link ``i - 1``) has viscous friction ``f_i``. All angles
are **absolute**, measured from the upward vertical (``0`` = upright).

Generalized coordinates: ``q = [x, theta_0, ..., theta_{n-1}]``
State vector: ``[x, x_dot, theta_0, theta_0_dot, ..., theta_{n-1}, theta_{n-1}_dot]``
Action: scalar horizontal force ``F`` applied to the cart (N).

EOM in manipulator form ``M(theta) q_ddot = rhs(q, qdot, F)`` with the
``(n+1) x (n+1)`` symmetric positive-definite mass matrix

    M[0, 0]        = M + sum_k m_k
    M[0, k+1]      = a_k cos(theta_k)
    M[k+1, k+1]    = d_k
    M[j+1, k+1]    = l_min(j,k) a_max(j,k) cos(theta_j - theta_k)     (j != k)

with ``a_k = m_k c_k + l_k sum_{i>k} m_i`` (first moment of the mass carried by
link ``k`` about its proximal joint) and ``d_k = I_k + m_k c_k^2 + l_k^2
sum_{i>k} m_i``. For ``c_k = l_k``, ``I_k = 0`` these are ``a_k = mu_k l_k`` and
``d_k = mu_k l_k^2`` with ``mu_k = sum_{i >= k} m_i`` — the closed form of the
point-mass model, which reduces **exactly** to the hand-derived single- and
double-link models it replaced (verified term-by-term, and by energy
conservation in the tests). Using absolute angles is what makes the velocity
coupling purely centrifugal — only ``theta_dot_j**2`` terms appear, no
``theta_dot_i theta_dot_j`` cross terms — which keeps the ``rhs`` compact for
arbitrary ``n``. Gravity is ``g a_k sin(theta_k)`` and the potential energy is
``g sum_k a_k cos(theta_k)``. The general-link terms are checked against a
finite-difference Euler-Lagrange residual in the tests.

No sympy at runtime: the equations above are evaluated directly with numpy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

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

      - ``masses``: link masses (kg) — bob masses for the point-mass model,
        total link mass (rod + attachments) for a rigid-body link.
      - ``lengths``: joint-to-joint link lengths (m).
      - ``com``: distance from each link's proximal joint to its centre of mass
        (m). ``None`` (default) puts it at the tip (``com = lengths``), i.e. a
        point-mass bob.
      - ``inertia``: moment of inertia of each link about its own centre of mass
        (kg·m²); ``0`` (default) for a point mass.
      - ``joint_friction``: viscous friction at each link's proximal joint,
        ``N·m/(rad/s)`` (``joint_friction[0]`` is the cart-link0 joint).

    Resolve them for a given link count with :meth:`link_masses`,
    :meth:`link_lengths`, :meth:`link_coms`, :meth:`link_inertias`, and
    :meth:`joint_frictions`. Only ``masses = lengths =
    joint_friction = 0`` at the joints (and ``b = 0``) conserves energy.

    Defaults describe a small bench rig at a 100 Hz control loop with the
    pendulums idealized as point masses on massless rods. ``config/rig.toml``
    holds a rigid-body reference rig (bar links with inertia) for a real build.
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
    com: LinkSpec | None = None
    inertia: LinkSpec = 0.0

    def link_masses(self, n: int) -> np.ndarray:
        """Return the ``(n,)`` array of link bob masses."""
        return _broadcast(self.masses, n)

    def link_lengths(self, n: int) -> np.ndarray:
        """Return the ``(n,)`` array of rod lengths."""
        return _broadcast(self.lengths, n)

    def link_coms(self, n: int) -> np.ndarray:
        """Return the ``(n,)`` joint-to-centre-of-mass distances (tip if unset)."""
        if self.com is None:
            return self.link_lengths(n)
        return _broadcast(self.com, n)

    def link_inertias(self, n: int) -> np.ndarray:
        """Return the ``(n,)`` link moments of inertia about their own COM."""
        return _broadcast(self.inertia, n)

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


def link_from_parts(
    length: float, rod_mass: float, tip_mass: float
) -> tuple[float, float, float]:
    """Mass, COM distance and COM inertia of a uniform bar plus a tip assembly.

    The bar runs joint-to-joint (``length``) with mass ``rod_mass``; ``tip_mass``
    (a bearing/encoder block, ballast, …) is a point mass at the distal joint.
    Returns ``(mass, com, inertia)`` where ``com`` is measured from the proximal
    joint and ``inertia`` is about the link's own centre of mass (bar ``m L²/12``
    plus parallel-axis terms).
    """
    mass = rod_mass + tip_mass
    com = (rod_mass * length / 2.0 + tip_mass * length) / mass
    inertia = (
        rod_mass * length**2 / 12.0
        + rod_mass * (length / 2.0 - com) ** 2
        + tip_mass * (length - com) ** 2
    )
    return mass, com, inertia


def _mu(masses: np.ndarray) -> np.ndarray:
    """Cumulative mass carried at or beyond each link: ``mu_k = sum_{i>=k} m_i``."""
    return np.cumsum(masses[::-1])[::-1]


def _coeffs(
    p: PhysicsParams, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Link coefficients ``(m, l, a, d, K)`` of the general-link model.

    ``a_k = m_k c_k + l_k sum_{i>k} m_i``, ``d_k = I_k + m_k c_k^2 + l_k^2
    sum_{i>k} m_i`` and the off-diagonal coupling ``K[j, k] = l_min(j,k)
    a_max(j,k)`` (its diagonal is unused; ``d`` replaces it).
    """
    m = p.link_masses(n)
    ln = p.link_lengths(n)
    c = p.link_coms(n)
    inertia = p.link_inertias(n)
    beyond = _mu(m) - m  # mass carried by links further out than k
    a = m * c + ln * beyond
    d = inertia + m * c**2 + ln**2 * beyond
    idx = np.arange(n)
    K = ln[np.minimum.outer(idx, idx)] * a[np.maximum.outer(idx, idx)]
    return m, ln, a, d, K


def mass_matrix_batch(theta: np.ndarray, p: PhysicsParams) -> np.ndarray:
    """Mass matrices for a batch of link angles: ``(B, n) -> (B, n+1, n+1)``."""
    theta = np.asarray(theta, dtype=float)
    n = theta.shape[-1]
    m, _, a, d, K = _coeffs(p, n)
    M = np.zeros((*theta.shape[:-1], n + 1, n + 1))
    M[..., 0, 0] = p.M + m.sum()
    cart = a * np.cos(theta)
    M[..., 0, 1:] = cart
    M[..., 1:, 0] = cart
    block = K * np.cos(theta[..., :, None] - theta[..., None, :])
    idx = np.arange(n)
    block[..., idx, idx] = d
    M[..., 1:, 1:] = block
    return M


def mass_matrix(theta: np.ndarray, p: PhysicsParams) -> np.ndarray:
    """Return the ``(n+1) x (n+1)`` positive-definite mass matrix ``M(theta)``.

    Args:
        theta: link angles ``(n,)`` (absolute, 0 = upright).
        p: physics parameters.

    """
    return mass_matrix_batch(np.atleast_1d(np.asarray(theta, dtype=float)), p)


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
    _, _, a, _, K = _coeffs(p, n)
    c = p.joint_frictions(n)

    r = np.zeros(n + 1)
    # Cart row: external force, cart friction, and centrifugal reaction of links.
    r[0] = F - p.b * xd + float(np.sum(a * np.sin(theta) * thd**2))

    # Gravity (positive: upright theta=0 is unstable) plus the centrifugal
    # coupling from every other link (no Coriolis cross terms in absolute
    # angles): sum_{j != k} K[j, k] sin(theta_j - theta_k) thd_j^2.
    grav = a * p.g * np.sin(theta)
    diff = np.sin(theta[:, None] - theta[None, :])  # [j, k] = sin(theta_j - theta_k)
    # The diagonal's sin(0) factor zeroes the j == k term, so K's diagonal is moot.
    cor = np.sum(K * diff * (thd**2)[:, None], axis=0)

    # Viscous joint friction (dissipative, odd in velocity so mirror symmetry is
    # preserved). Each joint opposes the RELATIVE rate across it; the child
    # joint's reaction feeds back onto this link.
    omega = thd - np.concatenate(([0.0], thd[:-1]))
    fric = -c * omega
    fric[:-1] += c[1:] * omega[1:]
    r[1:] = grav + cor + fric
    return r


class _Model(NamedTuple):
    """Link coefficients precomputed once per integration step (hot path)."""

    m00: float  # M[0, 0]
    a: np.ndarray  # cart coupling / gravity moment per link
    Kd: np.ndarray  # K with d_k on the diagonal (cos(0) = 1 there)
    K: np.ndarray
    fric: np.ndarray  # joint friction per link
    b: float
    g: float


def _model(p: PhysicsParams, n: int) -> _Model:
    m, _, a, d, K = _coeffs(p, n)
    Kd = K.copy()
    Kd[np.arange(n), np.arange(n)] = d
    return _Model(p.M + float(m.sum()), a, Kd, K, p.joint_frictions(n), p.b, p.g)


def _ode(t: float, state: np.ndarray, F: float, mdl: _Model) -> np.ndarray:
    """ODE function for :func:`scipy.integrate.solve_ivp`."""
    n = mdl.a.size
    xd = state[1]
    theta = state[2::2]
    thd = state[3::2]

    diff = theta[:, None] - theta[None, :]  # [j, k] = theta_j - theta_k
    cart = mdl.a * np.cos(theta)
    M = np.empty((n + 1, n + 1))
    M[0, 0] = mdl.m00
    M[0, 1:] = cart
    M[1:, 0] = cart
    M[1:, 1:] = mdl.Kd * np.cos(diff)

    thd2 = thd * thd
    r = np.empty(n + 1)
    r[0] = F - mdl.b * xd + float(np.dot(mdl.a * np.sin(theta), thd2))
    # Gravity + centrifugal coupling (see ``rhs``) + viscous joint friction.
    omega = thd.copy()
    omega[1:] -= thd[:-1]
    fric = -mdl.fric * omega
    fric[:-1] += mdl.fric[1:] * omega[1:]
    r[1:] = mdl.a * mdl.g * np.sin(theta) + (mdl.K * np.sin(diff)).T @ thd2 + fric
    q_ddot = np.linalg.solve(M, r)

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
        args=(F, _model(p, (state.size - 2) // 2)),
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
    _, _, a, _, _ = _coeffs(p, theta.size)
    return float(p.g * np.sum(a * np.cos(theta)))


def total_energy(state: np.ndarray, p: PhysicsParams) -> float:
    """Return total mechanical energy KE + PE."""
    return kinetic_energy(state, p) + potential_energy(state, p)


def total_energy_batch(states: np.ndarray, p: PhysicsParams) -> np.ndarray:
    """Total mechanical energy of a batch of states ``(..., 2+2n) -> (...)``."""
    states = np.asarray(states, dtype=float)
    theta = states[..., 2::2]
    qdot = np.concatenate([states[..., 1:2], states[..., 3::2]], axis=-1)
    M = mass_matrix_batch(theta, p)
    ke = 0.5 * np.einsum("...i,...ij,...j->...", qdot, M, qdot)
    _, _, a, _, _ = _coeffs(p, theta.shape[-1])
    return ke + p.g * np.sum(a * np.cos(theta), axis=-1)


def goal_energy(goal: np.ndarray, p: PhysicsParams) -> np.ndarray:
    """Mechanical energy of the goal pose at rest, ``(..., n) -> (...)``.

    The links sit at absolute angles ``goal`` (0 = up, pi = down) with no motion,
    so only the potential energy ``g sum a_k cos(goal_k)`` counts.
    """
    goal = np.asarray(goal, dtype=float)
    _, _, a, _, _ = _coeffs(p, goal.shape[-1])
    return p.g * np.sum(a * np.cos(goal), axis=-1)


def energy_range(p: PhysicsParams, n: int) -> float:
    """Span between the lowest and highest link potential energy (all-up vs all-down)."""
    _, _, a, _, _ = _coeffs(p, n)
    return float(2.0 * p.g * np.sum(a))
