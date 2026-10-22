"""Offline plant identification from device recordings (multiple shooting).

Each recording is cut into short windows (default 0.25 s). Every window is
simulated with the model of :mod:`n_cartpole.env.dynamics` from its own free
initial state, driven by the recorded command through the env's actuator chain
(``delay_steps`` delay -> slew limit -> ``available_force`` at the simulated cart
speed -> force gain), and compared with the position readings. A
:func:`scipy.optimize.least_squares` fit over the window states and the plant
parameters minimizes the scaled reading errors; the integer delay is chosen by
a grid search. Window states start from a non-causal Savitzky-Golay estimate of
the readings (positions and velocities, no lag offline).

Parameters (positive ones fitted in log space):

- ``M``: cart mass (the link masses stay at the config's weighed values),
- ``b``, ``cart_coulomb``: viscous and Coulomb rail friction,
- ``force_gain``: delivered force / command,
- ``links``: per-link first and second mass moments about the proximal joint,
  ``a_k = m_k c_k + l_k sum_{i>k} m_i`` and ``d_k = I_k + m_k c_k^2 + l_k^2
  sum_{i>k} m_i`` (they are what the dynamics depend on); mapped back to
  ``PhysicsParams.com`` / ``inertia`` with the config's masses and lengths,
- ``joint_friction``: viscous friction per joint.

The equations of motion are invariant when every inertia, friction and the force
gain are scaled together, so ``M``, ``force_gain`` and ``links`` cannot all be fitted
at once: by default the gain is fixed at 1, which identifies the plant in command
units — exactly what the policy acts in.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
from loguru import logger
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from scipy.sparse import lil_matrix

from n_cartpole.env.dynamics import PhysicsParams, _coeffs
from n_cartpole.sysid.recording import Recording

FIT_GROUPS = ("M", "b", "cart_coulomb", "force_gain", "links", "joint_friction")
DEFAULT_FIT = ("M", "b", "cart_coulomb", "links", "joint_friction")
# Start values for parameters the config sets to zero (log space needs > 0).
_FLOORS = {"b": 1e-3, "cart_coulomb": 1e-2, "joint_friction": 1e-4}


@dataclass
class FitOptions:
    """Settings of :func:`fit_recordings`."""

    fit: Sequence[str] = DEFAULT_FIT
    delays: Sequence[int] | None = None  # actuator delays to try (default 0..delay_max)
    delay_max: int = 3
    window: float = 0.25  # s per shooting window
    skip: int = 25  # samples skipped at each segment start (actuator history unknown)
    substeps: int = 2  # RK4 steps per control period
    x_scale: float = 1e-3  # residual unit for the cart reading (m)
    angle_scale: float = 1e-2  # residual unit for the angle readings (rad)
    loss: str = "soft_l1"  # robust to hand contact / bumps; "linear" = plain LSQ
    f_scale: float = 3.0  # in residual units
    savgol_window: int = 15
    savgol_order: int = 3
    max_nfev: int = 60
    ftol: float = 1e-8  # least_squares tolerances
    xtol: float = 1e-8


@dataclass
class ParamEstimate:
    """One reported parameter: nominal (config) value, estimate, standard error."""

    name: str
    nominal: float
    value: float
    se: float
    fitted: bool  # directly fitted (else derived from fitted ones / fixed)

    @property
    def rel_change(self) -> float:
        """``value / nominal - 1`` (NaN for a zero nominal)."""
        return self.value / self.nominal - 1.0 if self.nominal else float("nan")


@dataclass
class Windows:
    """Shooting windows cut from the recordings (B windows of W samples)."""

    t: np.ndarray  # (B, W) sample times (s), per recording
    meas: np.ndarray  # (B, W, n + 1) readings
    force: np.ndarray  # (B, W - 1) actuator force before back-EMF and gain (N)
    init: np.ndarray  # (B, 2 + 2n) smoothed initial-state guess
    rec: np.ndarray  # (B,) recording index


@dataclass
class FitResult:
    """Outcome of :func:`fit_recordings`."""

    physics: PhysicsParams  # fitted nominal plant
    nominal: PhysicsParams
    n_links: int
    delay: int
    nominal_delay: int
    force_gain: float
    estimates: list[ParamEstimate]
    rms: dict[str, float]  # per reading channel, fitted model (physical units)
    rms_nominal: dict[str, float]  # same windows, config model (states re-fitted)
    delay_cost: dict[int, float]
    windows: Windows
    sim: np.ndarray  # (B, W, n + 1) fitted-model readings
    sim_nominal: np.ndarray
    log_se: dict[str, float] = field(default_factory=dict)  # SE of log(param)
    warnings: list[str] = field(default_factory=list)

    def estimate(self, name: str) -> ParamEstimate:
        """Look up a reported parameter by name."""
        for e in self.estimates:
            if e.name == name:
                return e
        raise KeyError(name)

    def summary(self) -> str:
        """Fixed-width table of every parameter and the fit RMS."""
        lines = [
            f"{'parameter':<18}{'nominal':>12}{'fitted':>12}{'± se':>11}{'change':>9}"
        ]
        for e in self.estimates:
            se = f"{e.se:.2g}" if np.isfinite(e.se) else "-"
            ch = f"{100 * e.rel_change:+.1f}%" if np.isfinite(e.rel_change) else "-"
            lines.append(
                f"{e.name:<18}{e.nominal:>12.5g}{e.value:>12.5g}{se:>11}{ch:>9}"
            )
        lines.append(f"{'delay_steps':<18}{self.nominal_delay:>12d}{self.delay:>12d}")
        lines.append(
            "reading RMS (nominal -> fitted): "
            + ", ".join(
                f"{k} {self.rms_nominal[k]:.3g} -> {self.rms[k]:.3g}" for k in self.rms
            )
        )
        return "\n".join(lines + self.warnings)


class _Plant(NamedTuple):
    """Coefficients of the batched model (dynamics._model with free a, d)."""

    m00: float
    a: np.ndarray
    Kd: np.ndarray
    K: np.ndarray
    f: np.ndarray
    b: float
    coulomb: float
    gain: float


def _plant(p: PhysicsParams, n: int, v: dict[str, np.ndarray]) -> _Plant:
    """Model coefficients for parameter values ``v`` (lengths and masses from ``p``)."""
    ln = p.link_lengths(n)
    a = v["a"]
    idx = np.arange(n)
    K = ln[np.minimum.outer(idx, idx)] * a[np.maximum.outer(idx, idx)]
    Kd = K.copy()
    Kd[idx, idx] = v["d"]
    return _Plant(
        float(v["M"][0] + p.link_masses(n).sum()),
        a,
        Kd,
        K,
        v["joint_friction"],
        float(v["b"][0]),
        float(v["cart_coulomb"][0]),
        float(v["force_gain"][0]),
    )


def _ode_batch(S: np.ndarray, F: np.ndarray, c: _Plant, p: PhysicsParams) -> np.ndarray:
    """Batched dynamics._ode: ``S`` (B, 2 + 2n), ``F`` (B,) -> d(S)/dt."""
    n = c.a.size
    xd = S[:, 1]
    th = S[:, 2::2]
    thd = S[:, 3::2]
    cos, sin = np.cos(th), np.sin(th)
    diff = th[:, :, None] - th[:, None, :]  # [b, j, k] = th_j - th_k
    M = np.empty((S.shape[0], n + 1, n + 1))
    M[:, 0, 0] = c.m00
    M[:, 0, 1:] = c.a * cos
    M[:, 1:, 0] = c.a * cos
    M[:, 1:, 1:] = c.Kd * np.cos(diff)
    thd2 = thd * thd
    r = np.empty((S.shape[0], n + 1))
    r[:, 0] = F - c.b * xd + np.sum(c.a * sin * thd2, axis=1)
    if c.coulomb > 0.0:
        r[:, 0] -= c.coulomb * np.tanh(xd / p.cart_coulomb_vscale)
    omega = thd.copy()
    omega[:, 1:] -= thd[:, :-1]
    fric = -c.f * omega
    fric[:, :-1] += c.f[1:] * omega[:, 1:]
    cor = np.einsum("bjk,bj->bk", c.K * np.sin(diff), thd2)
    r[:, 1:] = c.a * p.g * sin + cor + fric
    qdd = np.linalg.solve(M, r[..., None])[..., 0]
    out = np.empty_like(S)
    out[:, 0] = xd
    out[:, 1] = qdd[:, 0]
    out[:, 2::2] = thd
    out[:, 3::2] = qdd[:, 1:]
    return out


def _available(F: np.ndarray, xd: np.ndarray, p: PhysicsParams) -> np.ndarray:
    """Vectorized dynamics.available_force."""
    limit = np.full_like(F, p.force_max)
    if p.v_noload is not None:
        along = F * xd > 0.0
        limit = np.where(
            along, p.force_max * np.maximum(0.0, 1.0 - np.abs(xd) / p.v_noload), limit
        )
    return np.clip(F, -limit, limit)


def simulate_windows(
    S0: np.ndarray, force: np.ndarray, c: _Plant, p: PhysicsParams, substeps: int
) -> np.ndarray:
    """Simulate every window from ``S0`` (B, dim); return readings ``(B, W, n + 1)``."""
    B, W1 = force.shape
    n = c.a.size
    out = np.empty((B, W1 + 1, n + 1))
    S = S0.copy()
    h = p.dt / substeps
    pos_idx = [0] + [2 + 2 * k for k in range(n)]
    out[:, 0] = S[:, pos_idx]
    for k in range(W1):
        F = c.gain * _available(force[:, k], S[:, 1], p)
        for _ in range(substeps):
            k1 = _ode_batch(S, F, c, p)
            k2 = _ode_batch(S + 0.5 * h * k1, F, c, p)
            k3 = _ode_batch(S + 0.5 * h * k2, F, c, p)
            k4 = _ode_batch(S + h * k3, F, c, p)
            S = S + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        out[:, k + 1] = S[:, pos_idx]
    return out


def actuator_force(
    cmd: np.ndarray, armed: np.ndarray, delay: int, slew: float | None, dt: float
) -> np.ndarray:
    """Apply the env's actuator chain before back-EMF: delay -> slew (one segment).

    The drive only acts while armed. The delay line and slew state start from the
    first sample's command (the history before the segment is unknown).
    """
    u = np.where(armed, cmd, 0.0)
    if delay > 0:
        u = np.concatenate([np.full(delay, u[0]), u[:-delay]])
    if slew is None:
        return u
    out = np.empty_like(u)
    prev = u[0]
    dF = slew * dt
    for i, v in enumerate(u):
        prev = prev + min(max(v - prev, -dF), dF)
        out[i] = prev
    return out


def make_windows(
    recs: Sequence[Recording],
    p: PhysicsParams,
    slew: float | None,
    delay: int,
    opts: FitOptions,
) -> Windows:
    """Cut every recording's contiguous segments into shooting windows."""
    W = max(3, int(round(opts.window / p.dt)))
    ts, meas, force, init, rec_idx = [], [], [], [], []
    for r_i, rec in enumerate(recs):
        if abs(rec.dt - p.dt) > 1e-9:
            raise ValueError(
                f"{rec.source or 'recording'}: dt {rec.dt} != config dt {p.dt}"
            )
        n = rec.n_links
        for start, stop in rec.segments():
            L = stop - start
            if L < opts.skip + W:
                continue
            pos = rec.pos[start:stop]
            u = actuator_force(
                rec.cmd[start:stop], rec.armed[start:stop], delay, slew, p.dt
            )
            win = opts.savgol_window | 1  # odd; L >= skip + W is longer
            order = min(opts.savgol_order, win - 1)
            smooth = savgol_filter(pos, win, order, axis=0)
            vel = savgol_filter(pos, win, order, deriv=1, delta=p.dt, axis=0)
            for s in range(opts.skip, L - W + 1, W):
                ts.append(rec.t[start + s : start + s + W])
                meas.append(pos[s : s + W])
                force.append(u[s : s + W - 1])
                st = np.empty(2 + 2 * n)
                st[0], st[1] = smooth[s, 0], vel[s, 0]
                st[2::2], st[3::2] = smooth[s, 1:], vel[s, 1:]
                init.append(st)
                rec_idx.append(r_i)
    if not meas:
        raise ValueError("recordings too short for one shooting window")
    return Windows(
        np.array(ts), np.array(meas), np.array(force), np.array(init), np.array(rec_idx)
    )


class _Spec:
    """Which parameters are free, and their mapping to/from the log vector."""

    def __init__(self, p: PhysicsParams, n: int, fit: Sequence[str]) -> None:
        unknown = set(fit) - set(FIT_GROUPS)
        if unknown:
            raise ValueError(
                f"unknown fit parameter(s) {sorted(unknown)}; choose from {FIT_GROUPS}"
            )
        if {"M", "force_gain", "links"} <= set(fit):
            raise ValueError(
                "M, force_gain and links cannot all be fitted: the equations of "
                "motion are unchanged when every inertia, friction and the force "
                "gain are scaled together (fix one, e.g. leave out force_gain)"
            )
        _, _, a, d, _ = _coeffs(p, n)
        self.n = n
        self.base: dict[str, np.ndarray] = {
            "M": np.array([p.M]),
            "b": np.array([p.b]),
            "cart_coulomb": np.array([p.cart_coulomb]),
            "force_gain": np.array([1.0]),
            "a": a.copy(),
            "d": d.copy(),
            "joint_friction": p.joint_frictions(n).copy(),
        }
        groups = [g for g in FIT_GROUPS if g in fit]
        self.keys = [k for g in groups for k in (("a", "d") if g == "links" else (g,))]
        self.names: list[str] = []
        for k in self.keys:
            size = self.base[k].size
            self.names += (
                [k]
                if size == 1 and k != "joint_friction"
                else [f"{k}{i}" for i in range(size)]
            )

    def x0(self) -> np.ndarray:
        """Log of the config values (floored where the config has 0)."""
        parts = [np.maximum(self.base[k], _FLOORS.get(k, 1e-9)) for k in self.keys]
        return np.log(np.concatenate(parts)) if parts else np.zeros(0)

    def values(self, z: np.ndarray) -> dict[str, np.ndarray]:
        """Parameter dict for log vector ``z`` (fixed ones from the config)."""
        v = {k: x.copy() for k, x in self.base.items()}
        i = 0
        for k in self.keys:
            size = v[k].size
            v[k] = np.exp(np.clip(z[i : i + size], -50.0, 50.0))  # trial steps
            i += size
        return v


def _residual_fn(
    w: Windows, p: PhysicsParams, spec: _Spec, opts: FitOptions, n_global: int
) -> Callable[[np.ndarray], np.ndarray]:
    n = spec.n
    dim = 2 + 2 * n
    scale = np.array([opts.x_scale] + [opts.angle_scale] * n)
    B = w.meas.shape[0]
    fixed_z = spec.x0()

    def fun(x: np.ndarray) -> np.ndarray:
        z = x[:n_global] if n_global else fixed_z
        S0 = x[n_global:].reshape(B, dim)
        plant = _plant(p, n, spec.values(z))
        with np.errstate(over="ignore", invalid="ignore"):
            sim = simulate_windows(S0, w.force, plant, p, opts.substeps)
            r = ((sim - w.meas) / scale).ravel()
        # A trial step that blew up a window: a large finite residual steers away.
        return np.clip(np.nan_to_num(r, nan=1e6), -1e6, 1e6)

    return fun


def _sparsity(B: int, rows_per: int, dim: int, n_global: int) -> lil_matrix:
    J = lil_matrix((B * rows_per, n_global + B * dim), dtype=int)
    if n_global:
        J[:, :n_global] = 1
    for b in range(B):
        J[
            b * rows_per : (b + 1) * rows_per,
            n_global + b * dim : n_global + (b + 1) * dim,
        ] = 1
    return J


def _solve(
    w: Windows,
    p: PhysicsParams,
    spec: _Spec,
    opts: FitOptions,
    z0: np.ndarray | None,
    free_params: bool,
):
    """One least-squares fit; ``free_params=False`` fits only the window states."""
    n = spec.n
    dim = 2 + 2 * n
    B, W = w.meas.shape[:2]
    n_global = len(spec.names) if free_params else 0
    x0 = np.concatenate([(spec.x0() if z0 is None else z0)[:n_global], w.init.ravel()])
    fun = _residual_fn(w, p, spec, opts, n_global)
    res = least_squares(
        fun,
        x0,
        jac_sparsity=_sparsity(B, W * (n + 1), dim, n_global),
        loss=opts.loss,
        f_scale=opts.f_scale,
        x_scale="jac",
        max_nfev=opts.max_nfev,
        ftol=opts.ftol,
        xtol=opts.xtol,
    )
    return res, n_global


def _global_cov(res, n_global: int, B: int, dim: int) -> np.ndarray:
    """Covariance of the global parameters (Schur complement over window states)."""
    J = res.jac.tocsc() if hasattr(res.jac, "tocsc") else res.jac
    m = J.shape[0]
    rows = m // B
    Jg = np.asarray(
        J[:, :n_global].todense() if hasattr(J, "todense") else J[:, :n_global]
    )
    info = Jg.T @ Jg
    Jcsr = J.tocsr() if hasattr(J, "tocsr") else J
    for b in range(B):
        r = slice(b * rows, (b + 1) * rows)
        blk = Jcsr[r]
        Js = np.asarray(
            blk[:, n_global + b * dim : n_global + (b + 1) * dim].todense()
            if hasattr(blk, "todense")
            else blk[:, n_global + b * dim : n_global + (b + 1) * dim]
        )
        Jgb = Jg[r]
        C = Js.T @ Js
        Bm = Jgb.T @ Js
        info -= Bm @ np.linalg.pinv(C) @ Bm.T
    dof = max(m - J.shape[1], 1)
    sigma2 = 2.0 * res.cost / dof
    return sigma2 * np.linalg.pinv(info)


def _channel_rms(sim: np.ndarray, meas: np.ndarray, n: int) -> dict[str, float]:
    names = ["x"] + [f"th{k}" for k in range(n)]
    err = sim - meas
    return {nm: float(np.sqrt(np.mean(err[..., i] ** 2))) for i, nm in enumerate(names)}


def _derived(
    spec: _Spec, p: PhysicsParams, v: dict[str, np.ndarray]
) -> dict[str, float]:
    """Physical parameters (PhysicsParams terms) from fitted values ``v``."""
    n = spec.n
    m, ln = p.link_masses(n), p.link_lengths(n)
    beyond = np.cumsum(m[::-1])[::-1] - m
    com = (v["a"] - ln * beyond) / m
    inertia = v["d"] - m * com**2 - ln**2 * beyond
    out = {
        "M": float(v["M"][0]),
        "b": float(v["b"][0]),
        "cart_coulomb": float(v["cart_coulomb"][0]),
        "force_gain": float(v["force_gain"][0]),
    }
    for k in range(n):
        out[f"com{k}"] = float(com[k])
        out[f"inertia{k}"] = float(inertia[k])
        out[f"joint_friction{k}"] = float(v["joint_friction"][k])
    for k in range(n):
        out[f"a{k}"] = float(v["a"][k])
        out[f"d{k}"] = float(v["d"][k])
    return out


def consistency_warnings(
    com: np.ndarray, inertia: np.ndarray, lengths: np.ndarray
) -> list[str]:
    """Flag fitted links that the config's weighed masses cannot explain."""
    out = []
    for k in range(len(com)):
        if not 0.0 < com[k] <= lengths[k] * 1.05:
            out.append(
                f"warning: link {k} centre of mass {com[k]:.4f} m lies outside the "
                f"link (length {lengths[k]:.3f} m): check its weighed mass"
            )
        if inertia[k] < 0.0:
            out.append(
                f"warning: link {k} inertia came out negative ({inertia[k]:.2e}); "
                "clipped to 0 — the weighed masses disagree with the swing"
            )
    return out


def fit_recordings(
    recs: Sequence[Recording],
    physics: PhysicsParams,
    n_links: int,
    force_slew: float | None,
    nominal_delay: int = 0,
    opts: FitOptions | None = None,
) -> FitResult:
    """Fit the plant parameters (and the integer delay) to ``recs``.

    Args:
        recs: recordings from the device (or a simulation of it).
        physics: the config's nominal plant (start values, fixed parameters,
            link masses/lengths, actuator limits).
        n_links: number of links (must match the recordings).
        force_slew: the actuator slew limit (N/s) the firmware models.
        nominal_delay: the config's ``delay_steps``, for the report.
        opts: fit settings.

    """
    opts = opts or FitOptions()
    for r in recs:
        if r.n_links != n_links:
            raise ValueError(f"{r.source}: {r.n_links} links, config has {n_links}")
    spec = _Spec(physics, n_links, opts.fit)
    dim = 2 + 2 * n_links
    delays = (
        list(opts.delays)
        if opts.delays is not None
        else list(range(opts.delay_max + 1))
    )
    # Full fit at the config's delay (or the first candidate) from the config
    # values. Every other delay is scored with those plant parameters held fixed
    # (only the window states re-fitted: cheap); a full fit is redone at the best.
    first = nominal_delay if nominal_delay in delays else delays[0]
    w = make_windows(recs, physics, force_slew, first, opts)
    res, n_global = _solve(w, physics, spec, opts, None, True)
    z_first = res.x[:n_global]
    delay_cost: dict[int, float] = {first: float(np.mean(res.fun**2))}
    fixed = _Spec(physics, n_links, ())
    fixed.base = spec.values(z_first)
    for D in delays:
        if D == first:
            continue
        wd = make_windows(recs, physics, force_slew, D, opts)
        rd, _ = _solve(wd, physics, fixed, opts, None, False)
        delay_cost[D] = float(np.mean(rd.fun**2))
    for D in sorted(delay_cost):
        logger.info(f"sysid: delay {D} -> mean sq. residual {delay_cost[D]:.4g}")
    best_delay = min(delay_cost, key=lambda d: delay_cost[d])
    if best_delay != first:
        w = make_windows(recs, physics, force_slew, best_delay, opts)
        res, n_global = _solve(w, physics, spec, opts, z_first, True)
        delay_cost[best_delay] = float(np.mean(res.fun**2))
    best = (best_delay, w, res, n_global)
    D, w, res, n_global = best
    B = w.meas.shape[0]
    z = res.x[:n_global]
    cov = _global_cov(res, n_global, B, dim)
    log_se = dict(zip(spec.names, np.sqrt(np.maximum(np.diag(cov), 0.0)), strict=True))

    vals = spec.values(z)
    plant = _plant(physics, n_links, vals)
    S0 = res.x[n_global:].reshape(B, dim)
    sim = simulate_windows(S0, w.force, plant, physics, opts.substeps)

    # Derived parameters and their delta-method standard errors.
    derived = _derived(spec, physics, vals)
    nominal_vals = _derived(spec, physics, spec.base)
    names = list(derived)
    G = np.zeros((len(names), len(z)))
    for j in range(len(z)):
        h = 1e-6
        zp, zm = z.copy(), z.copy()
        zp[j] += h
        zm[j] -= h
        dp = _derived(spec, physics, spec.values(zp))
        dm = _derived(spec, physics, spec.values(zm))
        G[:, j] = [(dp[k] - dm[k]) / (2 * h) for k in names]
    dcov = G @ cov @ G.T
    fitted_names = set(spec.names)
    depends = np.any(G != 0.0, axis=1)
    estimates = [
        ParamEstimate(
            k,
            nominal_vals[k],
            derived[k],
            float(np.sqrt(max(dcov[i, i], 0.0))) if depends[i] else float("nan"),
            k in fitted_names,
        )
        for i, k in enumerate(names)
    ]

    ln = physics.link_lengths(n_links)
    com = np.array([derived[f"com{k}"] for k in range(n_links)])
    inertia = np.array([derived[f"inertia{k}"] for k in range(n_links)])
    warnings = consistency_warnings(com, inertia, ln)
    for msg in warnings:
        logger.warning(msg)
    fitted = dataclasses.replace(
        physics,
        M=derived["M"],
        b=derived["b"],
        cart_coulomb=derived["cart_coulomb"],
        masses=tuple(physics.link_masses(n_links)),
        lengths=tuple(ln),
        com=tuple(com),
        inertia=tuple(np.maximum(inertia, 0.0)),
        joint_friction=tuple(derived[f"joint_friction{k}"] for k in range(n_links)),
    )

    # The config's model on the same windows (window states re-fitted).
    w_nom = make_windows(recs, physics, force_slew, nominal_delay, opts)
    nom_spec = _Spec(physics, n_links, ())
    res_nom, _ = _solve(w_nom, physics, nom_spec, opts, None, False)
    sim_nom = simulate_windows(
        res_nom.x.reshape(-1, dim),
        w_nom.force,
        _plant(physics, n_links, nom_spec.base),
        physics,
        opts.substeps,
    )
    return FitResult(
        physics=fitted,
        nominal=physics,
        n_links=n_links,
        delay=D,
        nominal_delay=nominal_delay,
        force_gain=derived["force_gain"],
        estimates=estimates,
        rms=_channel_rms(sim, w.meas, n_links),
        rms_nominal=_channel_rms(sim_nom, w_nom.meas, n_links),
        delay_cost=delay_cost,
        windows=w,
        sim=sim,
        sim_nominal=sim_nom,
        log_se=log_se,
        warnings=warnings,
    )
