"""LQR hold controllers around the up/down equilibria of the n-link cartpole.

Every goal (``UU``, ``DU``, ``UD``, ``DD``) is an equilibrium of the nominal model, so each gets a
linearisation and a discrete LQR gain. The design accounts for what the real pipeline does between
the controller and the plant:

* the action delay: the command computed at step ``k`` reaches the plant at step ``k + delay``
  (``HardwareConfig.delay_steps``), so the state is augmented with the commands in flight;
* the actuator's rate limit: the controlled input is the *change* of the commanded force per step
  (``v``), which is also what the env's ``action_mode="delta"`` takes, and ``v`` is penalised so the
  hold is smooth by construction.

Rail Coulomb friction is left out of the linear model (its regularised slope, ``coulomb /
coulomb_vscale``, is a huge fictitious viscous damping at rest); it acts as a disturbance.

The augmented state is ``z = [s - s*, c_{k-1}, ..., c_{k-m}]`` with ``m = max(delay, 1)``; the
cost-to-go ``z' P z`` doubles as the *level* used to decide when a hand-over from the learned
policy is safe.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm, solve_discrete_are

from n_cartpole.env.dynamics import PhysicsParams, _model, _ode
from n_cartpole.env.goals import goal_configs, goal_labels


@dataclass(frozen=True)
class LQRWeights:
    """Cost weights (per control step)."""

    q_x: float = 10.0  # cart position (m)
    q_xd: float = 1.0  # cart speed (m/s)
    q_th: float = 100.0  # link angle error (rad)
    q_thd: float = 1.0  # link angular rate (rad/s)
    q_cmd: float = 0.01  # commanded force in flight (N)
    r_dcmd: float = 1.0  # change of the commanded force per step (N)


def equilibrium(goal: np.ndarray) -> np.ndarray:
    """State ``[0, 0, th_0*, 0, th_1*, 0, ...]`` of a goal (absolute link angles)."""
    s = np.zeros(2 + 2 * len(goal))
    s[2::2] = goal
    return s


def linearize(
    p: PhysicsParams, n: int, goal: np.ndarray, eps: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """Continuous ``(A, B)`` of the nominal model at a goal equilibrium, zero force.

    Central differences of the same ``_ode`` the env integrates, with rail Coulomb friction removed.
    """
    mdl = _model(dataclasses.replace(p, cart_coulomb=0.0), n)
    s0 = equilibrium(goal)

    def f(s: np.ndarray, u: float) -> np.ndarray:
        return _ode(0.0, s, u, mdl)

    dim = s0.size
    a = np.empty((dim, dim))
    for i in range(dim):
        d = np.zeros(dim)
        d[i] = eps
        a[:, i] = (f(s0 + d, 0.0) - f(s0 - d, 0.0)) / (2 * eps)
    b = ((f(s0, eps) - f(s0, -eps)) / (2 * eps)).reshape(-1, 1)
    return a, b


def discretize(
    a: np.ndarray, b: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Zero-order-hold discretisation."""
    ns = a.shape[0]
    m = np.zeros((ns + 1, ns + 1))
    m[:ns, :ns] = a * dt
    m[:ns, ns:] = b * dt
    e = expm(m)
    return e[:ns, :ns], e[:ns, ns:]


def augment_delay(
    ad: np.ndarray, bd: np.ndarray, delay: int
) -> tuple[np.ndarray, np.ndarray]:
    """State ``[s, c_{k-1}, ..., c_{k-m}]``, input ``v`` with ``c_k = c_{k-1} + v``.

    The plant sees ``c_{k-delay}`` during step ``k``; with ``delay = 0`` it sees ``c_k`` itself.
    """
    ns = ad.shape[0]
    m = max(delay, 1)
    aa = np.zeros((ns + m, ns + m))
    ba = np.zeros((ns + m, 1))
    aa[:ns, :ns] = ad
    aa[ns, ns] = 1.0  # c_{k-1} carries into the new c_k = c_{k-1} + v
    ba[ns, 0] = 1.0
    for i in range(1, m):
        aa[ns + i, ns + i - 1] = 1.0  # shift the line
    if delay == 0:
        aa[:ns, ns] = bd[:, 0]  # the plant sees c_k = c_{k-1} + v
        ba[:ns] = bd
    else:
        aa[:ns, ns + delay - 1] = bd[:, 0]
    return aa, ba


@dataclass(frozen=True)
class LQRDesign:
    """Gain and cost-to-go of one goal."""

    label: str
    goal: np.ndarray
    k: np.ndarray  # (1, ns + m): v = -k @ z
    p: np.ndarray  # cost-to-go z' p z
    poles: np.ndarray  # closed-loop eigenvalues (discrete)
    controllability_rank: int
    ns: int
    m: int


def design(
    physics: PhysicsParams,
    n: int,
    goal: np.ndarray,
    label: str = "",
    weights: LQRWeights | None = None,
    delay: int = 1,
) -> LQRDesign:
    """Discrete LQR for one goal on the delay-augmented system."""
    weights = weights or LQRWeights()
    a, b = linearize(physics, n, goal)
    ad, bd = discretize(a, b, physics.dt)
    aa, ba = augment_delay(ad, bd, delay)
    ns, m = ad.shape[0], aa.shape[0] - ad.shape[0]
    qs = [weights.q_x, weights.q_xd]
    for _ in range(n):
        qs += [weights.q_th, weights.q_thd]
    q = np.diag(np.concatenate([qs, np.full(m, weights.q_cmd)]))
    r = np.array([[weights.r_dcmd]])
    pm = solve_discrete_are(aa, ba, q, r)
    k = np.linalg.solve(r + ba.T @ pm @ ba, ba.T @ pm @ aa)
    poles = np.linalg.eigvals(aa - ba @ k)
    dim = aa.shape[0]
    ctrb = np.hstack([np.linalg.matrix_power(aa, i) @ ba for i in range(dim)])
    return LQRDesign(
        label,
        np.asarray(goal, float),
        k,
        pm,
        poles,
        int(np.linalg.matrix_rank(ctrb)),
        ns,
        m,
    )


class LQRBank:
    """One :class:`LQRDesign` per goal of an ``n``-link cartpole."""

    def __init__(
        self,
        physics: PhysicsParams,
        n: int,
        weights: LQRWeights | dict[str, LQRWeights] | None = None,
        delay: int = 1,
    ) -> None:
        """Design every goal; ``weights`` may differ per goal label."""
        weights = weights or LQRWeights()
        self.n = n
        self.delay = delay
        self.physics = physics
        goals = goal_configs(n)
        labels = goal_labels(n)
        self.designs = [
            design(
                physics,
                n,
                g,
                lab,
                weights
                if isinstance(weights, LQRWeights)
                else weights.get(lab, LQRWeights()),
                delay,
            )
            for g, lab in zip(goals, labels, strict=True)
        ]

    def error_state(
        self, goal_idx: int, state: np.ndarray, cmd_hist: np.ndarray
    ) -> np.ndarray:
        """``z`` for an estimated state ``[x, xd, th_0, th_0d, ...]`` and ``[c_{k-1}, ...]``."""
        d = self.designs[goal_idx]
        e = np.asarray(state, float) - equilibrium(d.goal)
        e[2::2] = np.angle(np.exp(1j * e[2::2]))  # wrap the angle errors to (-pi, pi]
        hist = np.zeros(d.m)
        h = np.asarray(cmd_hist, float)[: d.m]
        hist[: h.size] = h
        return np.concatenate([e, hist])

    def rate(self, goal_idx: int, state: np.ndarray, cmd_hist: np.ndarray) -> float:
        """Commanded-force change for this step (N per step), unclipped."""
        d = self.designs[goal_idx]
        return float(-(d.k @ self.error_state(goal_idx, state, cmd_hist))[0])

    def level(self, goal_idx: int, state: np.ndarray, cmd_hist: np.ndarray) -> float:
        """Cost-to-go ``z' P z``: small near the goal, large far from it."""
        z = self.error_state(goal_idx, state, cmd_hist)
        return float(z @ self.designs[goal_idx].p @ z)
