"""Screen physical rig designs by how easy they are to control on real hardware.

Stage 1 (seconds per design): a delay-aware discrete LQR at each unstable
equilibrium (UU, UD, DU) on the linearized plant, scored with the rig's encoder
noise model, robustness to ±parameter error, tolerated extra delay, recoverable
perturbation, and swing-up work headroom.

Stage 2 (a minute per design): the best designs are simulated closed-loop in the
FULL environment — encoder noise/quantization with differenced velocities, rail
Coulomb friction, motor back-EMF speed limit, slew limit, action delay, and a
randomized true plant — holding each equilibrium.

This is a screening tool, not a proof: the LQR stands in for "how easy is this plant
to control", and swing-up / transitions still need a learned-policy check on the
shortlist (``scripts/train.py``). Design variables are link length, tip masses, and
the drive (motor + pulley), which sets the effective cart mass, peak force and speed
limit through ``config.drive_from_parts``.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.linalg as la
from loguru import logger

from n_cartpole.config import drive_from_parts, load_tqc_config
from n_cartpole.env.cartpole import EnvConfig
from n_cartpole.env.dynamics import (
    PhysicsParams,
    energy_range,
    link_from_parts,
    step,
)
from n_cartpole.env.factory import make_env
from n_cartpole.env.hardware_config import SensorModel
from n_cartpole.env.randomization import PlantRandomization

N = 6  # state dimension of the double pendulum cartpole
EQUILIBRIA = {"UU": (0.0, 0.0), "UD": (0.0, math.pi), "DU": (math.pi, 0.0)}

# Generic motor classes: (rotor+pulley inertia kg·m², peak torque N·m, no-load rpm).
# These are assumptions to be replaced with your candidate parts' datasheet values.
MOTORS = {
    "bldc-small": (1.5e-5, 0.15, 4000.0),
    "bldc-mid": (3.0e-5, 0.30, 3000.0),
    "stepper-17": (5.4e-5, 0.40, 1200.0),
}
PULLEYS = {"GT2-20T": 0.00635, "GT2-30T": 0.00955, "GT2-40T": 0.01273}


@dataclass(frozen=True)
class Design:
    """One candidate rig: link geometry and drive."""

    length: float
    tip1: float
    tip2: float
    motor: str
    pulley: str
    cart_mass: float = 0.55

    def physics(self, base: PhysicsParams) -> PhysicsParams:
        """Resolve to ``PhysicsParams``: bar links (25x3 mm Al) + tips + derived drive."""
        rod = 0.025 * 0.003 * 2700.0 * self.length
        links = [
            link_from_parts(self.length, rod, self.tip1),
            link_from_parts(self.length, rod, self.tip2),
        ]
        j, tau, rpm = MOTORS[self.motor]
        m_eff, force, v_noload = drive_from_parts(
            self.cart_mass, j, PULLEYS[self.pulley], tau, rpm, derate=0.65
        )
        return dataclasses.replace(
            base,
            M=m_eff,
            force_max=force,
            v_noload=v_noload,
            masses=tuple(p[0] for p in links),
            lengths=(self.length, self.length),
            com=tuple(p[1] for p in links),
            inertia=tuple(p[2] for p in links),
        )

    def label(self) -> str:
        """Compact human-readable name."""
        return (
            f"L={self.length:.2f} tips={self.tip1:.2f}/{self.tip2:.2f} "
            f"{self.motor}/{self.pulley}"
        )


def linearize(
    p: PhysicsParams, eq: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Discrete-time ``(A, B)`` about an equilibrium, by central differences of ``step``."""
    # Sliding friction is left out of the linear model: its regularized slope at zero
    # speed (mu / vscale) is a huge fake viscous damper. Stage 2 simulates it properly.
    free = dataclasses.replace(p, force_max=1e9, v_noload=None, cart_coulomb=0.0)
    x0 = np.array([0.0, 0.0, eq[0], 0.0, eq[1], 0.0])
    eps = 1e-5
    A = np.zeros((N, N))
    for i in range(N):
        d = np.zeros(N)
        d[i] = eps
        A[:, i] = (step(x0 + d, 0.0, free) - step(x0 - d, 0.0, free)) / (2 * eps)
    B = ((step(x0, eps, free) - step(x0, -eps, free)) / (2 * eps)).reshape(N, 1)
    return A, B


def delay_augment(
    A: np.ndarray, B: np.ndarray, delay: int
) -> tuple[np.ndarray, np.ndarray]:
    """Augment with a command shift register: the plant sees the command ``delay`` steps old."""
    m = N + delay
    Aa, Ba = np.zeros((m, m)), np.zeros((m, 1))
    Aa[:N, :N] = A
    Aa[:N, m - 1] = B[:, 0]
    for j in range(delay - 1):
        Aa[N + j + 1, N + j] = 1.0
    Ba[N, 0] = 1.0
    return Aa, Ba


def lqr_gain(A, B, delay=1, qx=5.0, qth=100.0, r=1.0) -> np.ndarray:
    """Delay-aware discrete LQR gain on ``[state, command history]``."""
    Aa, Ba = delay_augment(A, B, delay)
    q = np.zeros(Aa.shape[0])
    q[:N] = [qx, 1, qth, 1, qth, 1]
    P = la.solve_discrete_are(Aa, Ba, np.diag(q), np.array([[r]]))
    return np.linalg.solve(r + Ba.T @ P @ Ba, Ba.T @ P @ Aa)[0]


def _closed_loop_radius(A, B, K, delay) -> float:
    Aa, Ba = delay_augment(A, B, delay)
    return float(np.max(np.abs(np.linalg.eigvals(Aa - Ba @ K[None, :]))))


def measurement_std(s: SensorModel, dt: float) -> np.ndarray:
    """Std of each state channel as the controller sees it (differenced rates)."""
    px = math.sqrt(s.x_noise_std**2 + s.x_resolution**2 / 12.0)
    pa = math.sqrt(s.angle_noise_std**2 + s.angle_resolution**2 / 12.0)
    return np.array(
        [
            px,
            math.sqrt(2) * px / dt,
            pa,
            math.sqrt(2) * pa / dt,
            pa,
            math.sqrt(2) * pa / dt,
        ]
    )


@dataclass
class Stage1:
    """Linear screen results at one equilibrium (``None`` fields mean unbalanceable)."""

    u_rms: float  # control-force RMS driven by sensor noise (N)
    robust: float  # fraction of ±param-error plants stabilized by the nominal gain
    delay_margin: int  # extra delay steps the nominal gain tolerates
    roa_deg: float  # largest recoverable angle error within force / rail limits (deg)


def stage1(
    p: PhysicsParams,
    eq: tuple[float, float],
    sensors: SensorModel,
    rnd: PlantRandomization,
    rng: np.random.Generator,
    samples: int = 30,
) -> Stage1 | None:
    """Score one design at one equilibrium; None if the nominal loop is unstable."""
    A, B = linearize(p, eq)
    K = lqr_gain(A, B)
    Aa, Ba = delay_augment(A, B, 1)
    Acl = Aa - Ba @ K[None, :]
    if np.max(np.abs(np.linalg.eigvals(Acl))) >= 0.9999:
        return None
    sig = measurement_std(sensors, p.dt)
    Nn = np.diag(sig**2)
    G = -Ba @ K[None, :N]
    cov = la.solve_discrete_lyapunov(Acl, G @ Nn @ G.T)
    u_rms = math.sqrt(max(float(K @ cov @ K + K[:N] @ Nn @ K[:N]), 0.0))
    ok = 0
    for _ in range(samples):
        q, _ = rnd.sample(p, 2, rng)
        Aq, Bq = linearize(q, eq)
        ok += _closed_loop_radius(Aq, Bq, K, 1) < 0.9999
    margin = 0
    for extra in (1, 2, 3):
        Ax, Bx = delay_augment(A, B, 1 + extra)
        Kx = np.concatenate([K, np.zeros(extra)])
        if np.max(np.abs(np.linalg.eigvals(Ax - Bx @ Kx[None, :]))) < 0.9999:
            margin = extra
        else:
            break
    z = np.zeros(N + 1)
    z[2] = z[4] = 1.0
    peak_u = peak_x = 1e-12
    for _ in range(300):
        peak_u = max(peak_u, abs(float(-K @ z)))
        peak_x = max(peak_x, abs(z[0]))
        z = Acl @ z
    roa = math.degrees(min(p.force_max / peak_u, p.x_lim / peak_x))
    return Stage1(u_rms, ok / samples, margin, roa)


def score_design(
    d: Design,
    base: PhysicsParams,
    sensors: SensorModel,
    rnd: PlantRandomization,
    seed: int = 0,
) -> dict | None:
    """Stage-1 metrics for every equilibrium plus the constraint quantities."""
    p = d.physics(base)
    rng = np.random.default_rng(seed)
    per = {}
    for name, eq in EQUILIBRIA.items():
        s1 = stage1(p, eq, sensors, rnd, rng)
        if s1 is None:
            return None
        per[name] = s1
    return {
        "design": d,
        "physics": p,
        "per": per,
        "worst_u_rms": max(s.u_rms for s in per.values()),
        "min_robust": min(s.robust for s in per.values()),
        "min_delay": min(s.delay_margin for s in per.values()),
        "min_roa": min(s.roa_deg for s in per.values()),
        "accel": p.force_max / p.M,
        "work_ratio": p.force_max * 2 * p.x_lim / energy_range(p, 2),
        "v_noload": p.v_noload,
    }


def feasible(r: dict, min_accel=10.0, min_vnl=2.0, min_work=5.0) -> bool:
    """Hard requirements for a buildable, swing-up-capable, robust design."""
    return (
        r["accel"] >= min_accel
        and (r["v_noload"] is None or r["v_noload"] >= min_vnl)
        and r["work_ratio"] >= min_work
        and r["min_robust"] >= 0.95
        and r["min_delay"] >= 1
    )


def stage2(
    r: dict,
    env_cfg: EnvConfig,
    seeds: int = 6,
    seconds: float = 10.0,
) -> dict[str, dict]:
    """Closed-loop LQR holds in the full environment, per equilibrium."""
    p = r["physics"]
    cfg = dataclasses.replace(
        env_cfg,
        physics=p,
        goal_conditioned=True,
        goal_hold_steps=None,
        max_steps=int(seconds / p.dt),
        randomize=env_cfg.randomize or PlantRandomization(),
    )
    env = make_env(cfg)
    out = {}
    for name, eq in EQUILIBRIA.items():
        A, B = linearize(p, eq)
        K = lqr_gain(A, B)
        ok, rms, ddf = 0, [], []
        for seed in range(seeds):
            obs, _ = env.reset(seed=seed, options={"start": name, "goal": name})
            u_prev, applied = 0.0, []
            for _ in range(cfg.max_steps):
                x, xd, c1, s1, w1, c2, s2, w2 = obs[:8]
                th = np.array(
                    [
                        (math.atan2(s1, c1) - eq[0] + math.pi) % (2 * math.pi)
                        - math.pi,
                        (math.atan2(s2, c2) - eq[1] + math.pi) % (2 * math.pi)
                        - math.pi,
                    ]
                )
                z = np.array([x, xd, th[0], w1, th[1], w2, u_prev])
                u = float(np.clip(-K @ z, -p.force_max, p.force_max))
                obs, _, term, trunc, info = env.step(np.array([u]))
                u_prev = u
                applied.append(info["applied_force"])
                if "segment" in info:
                    ok += int(info["segment"][2])
                if term or trunc:
                    break
            tail = np.array(applied[len(applied) // 2 :])
            rms.append(float(np.sqrt(np.mean(tail**2))))
            ddf.append(float(np.mean(np.abs(np.diff(tail)))))
        out[name] = {
            "ok": ok,
            "seeds": seeds,
            "rms": float(np.mean(rms)),
            "ddf": float(np.mean(ddf)),
        }
    return out


# --------------------------------------------------------------------------- sensing


class HoldController:
    """LQR state feedback on ``[state, previous command]`` with a pluggable estimator.

    ``estimator='diff'`` uses the observation as given (positions plus differenced,
    optionally filtered, velocities). ``estimator='kalman'`` ignores the velocity
    channels and runs a steady-state Kalman filter on the linear model, fed with
    the three position readings only — what a firmware observer would do.
    """

    def __init__(
        self,
        p: PhysicsParams,
        eq: tuple[float, float],
        sensors: SensorModel,
        estimator: str,
        process_force_std: float = 0.5,
    ) -> None:
        """Design the gain (and observer) for ``p`` at equilibrium ``eq``."""
        self.eq, self.estimator = eq, estimator
        A, B = linearize(p, eq)
        self.K = lqr_gain(A, B)
        self.Aa, self.Ba = delay_augment(A, B, 1)
        self.u_prev = 0.0
        if estimator == "kalman":
            sig = measurement_std(sensors, p.dt)
            self.C = np.zeros((3, N + 1))
            self.C[0, 0] = self.C[1, 2] = self.C[2, 4] = 1.0
            R = np.diag(sig[[0, 2, 4]] ** 2)
            Qw = np.zeros((N + 1, N + 1))
            Qw[:N, :N] = process_force_std**2 * (B @ B.T) + 1e-10 * np.eye(N)
            # P = steady-state PREDICTION covariance; the "current estimator" gain
            # folds the newest measurement in before the command is computed, which
            # avoids a spurious extra step of delay (the predictor form uses y_{k-1}).
            P = la.solve_discrete_are(self.Aa.T, self.C.T, Qw, R)
            self.L = P @ self.C.T @ np.linalg.inv(self.C @ P @ self.C.T + R)
            self.zhat = np.zeros(N + 1)  # prediction of the current state

    def reset(self) -> None:
        """Forget the previous command and the observer state."""
        self.u_prev = 0.0
        if self.estimator == "kalman":
            self.zhat = np.zeros(N + 1)

    def act(self, obs: np.ndarray, force_max: float) -> float:
        """Return the commanded force for the current observation."""
        x, xd, c1, s1, w1, c2, s2, w2 = obs[:8]
        th = [
            (math.atan2(s1, c1) - self.eq[0] + math.pi) % (2 * math.pi) - math.pi,
            (math.atan2(s2, c2) - self.eq[1] + math.pi) % (2 * math.pi) - math.pi,
        ]
        if self.estimator == "kalman":
            y = np.array([x, th[0], th[1]])
            filtered = self.zhat + self.L @ (y - self.C @ self.zhat)
            u = float(np.clip(-self.K @ filtered, -force_max, force_max))
            self.zhat = self.Aa @ filtered + self.Ba[:, 0] * u
        else:
            z = np.array([x, xd, th[0], w1, th[1], w2, self.u_prev])
            u = float(np.clip(-self.K @ z, -force_max, force_max))
        self.u_prev = u
        return u


def sensing_trials(
    p: PhysicsParams,
    env_cfg: EnvConfig,
    sensors: SensorModel,
    estimator: str,
    seeds: int = 5,
    seconds: float = 10.0,
    process_force_std: float = 0.5,
    randomize: bool = True,
) -> dict[str, dict]:
    """Closed-loop holds (UU/UD/DU) in the full env for one sensing/estimator choice.

    ``process_force_std`` (N) is the Kalman observer's unmodelled-force assumption
    (rail friction, model error); larger trusts the measurements more.
    """
    assert env_cfg.hardware is not None
    hw = dataclasses.replace(env_cfg.hardware, sensors=sensors)
    cfg = dataclasses.replace(
        env_cfg,
        physics=p,
        hardware=hw,
        goal_conditioned=True,
        goal_hold_steps=None,
        max_steps=int(seconds / p.dt),
        randomize=(env_cfg.randomize or PlantRandomization()) if randomize else None,
    )
    env = make_env(cfg)
    out = {}
    for name, eq in EQUILIBRIA.items():
        ctrl = HoldController(p, eq, sensors, estimator, process_force_std)
        ok, rms, ddf = 0, [], []
        for seed in range(seeds):
            obs, _ = env.reset(seed=seed, options={"start": name, "goal": name})
            ctrl.reset()
            applied = []
            for _ in range(cfg.max_steps):
                obs, _, term, trunc, info = env.step(
                    np.array([ctrl.act(obs, p.force_max)])
                )
                applied.append(info["applied_force"])
                if "segment" in info:
                    ok += int(info["segment"][2])
                if term or trunc:
                    break
            tail = np.array(applied[len(applied) // 2 :])
            rms.append(float(np.sqrt(np.mean(tail**2))))
            ddf.append(float(np.mean(np.abs(np.diff(tail)))))
        out[name] = {
            "ok": ok,
            "seeds": seeds,
            "rms": float(np.mean(rms)),
            "ddf": float(np.mean(ddf)),
        }
    return out


def sensing_study(env_cfg: EnvConfig, designs: dict[str, Design], seeds: int) -> None:
    """Sweep encoder noise, loop rate and velocity estimator for each design."""
    assert env_cfg.hardware is not None and env_cfg.hardware.sensors is not None
    base_sensors = env_cfg.hardware.sensors
    print(
        "\nsensing study: full-env LQR holds. cell = holds OK, RMS force (N), mean |dF| per step (N)"
    )
    print(
        f"{'design':8s} {'loop':>6s} {'angle noise':>11s} {'estimator':>12s} | UU | UD | DU"
    )
    cases = []
    for noise in (0.15e-3, 0.5e-3, 1.5e-3):
        for est, alpha in (("diff", 0.0), ("diff", 0.8), ("kalman", 0.0)):
            cases.append((100.0, noise, est, alpha))
    cases.append((200.0, 0.5e-3, "diff", 0.8))
    cases.append((200.0, 0.5e-3, "kalman", 0.0))
    for dname, d in designs.items():
        for rate, noise, est, alpha in cases:
            p = dataclasses.replace(d.physics(env_cfg.physics), dt=1.0 / rate)
            sens = dataclasses.replace(
                base_sensors, angle_noise_std=noise, velocity_filter=alpha
            )
            res = sensing_trials(p, env_cfg, sens, est, seeds=seeds)
            label = est if est == "kalman" else f"diff a={alpha:.1f}"
            cells = " | ".join(
                f"{v['ok']}/{v['seeds']} {v['rms']:4.2f} {v['ddf']:4.2f}"
                for v in res.values()
            )
            print(
                f"{dname:8s} {rate:4.0f}Hz {noise * 1e3:8.2f} mrad {label:>12s} | {cells}",
                flush=True,
            )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, default=Path("config/rig.toml"))
    ap.add_argument("--top", type=int, default=8, help="Designs to verify in stage 2")
    ap.add_argument("--stage1-only", action="store_true")
    ap.add_argument(
        "--sensing",
        action="store_true",
        help="Sweep encoder noise / loop rate / velocity estimator on reference designs",
    )
    ap.add_argument(
        "--seeds", type=int, default=6, help="Stage-2 seeds per equilibrium"
    )
    return ap.parse_args()


def main() -> None:
    """Run the screen and print the ranked table."""
    args = parse_args()
    env_cfg = load_tqc_config(args.config).env
    base = env_cfg.physics
    hw = env_cfg.hardware
    if hw is None or hw.sensors is None:
        raise SystemExit(f"{args.config} needs a [hardware.sensors] table")
    rnd = env_cfg.randomize or PlantRandomization()
    if args.sensing:
        designs = {
            "ref": Design(0.25, 0.08, 0.05, "bldc-small", "GT2-20T"),
            "mid": Design(0.20, 0.05, 0.05, "bldc-mid", "GT2-40T"),
        }
        sensing_study(env_cfg, designs, args.seeds)
        return

    grid = [
        Design(L, t1, t2, motor, pulley)
        for L, (t1, t2), motor, pulley in itertools.product(
            (0.15, 0.20, 0.25, 0.30),
            ((0.03, 0.03), (0.05, 0.05), (0.08, 0.08), (0.15, 0.08)),
            MOTORS,
            PULLEYS,
        )
    ]
    ref = Design(
        0.25, 0.08, 0.05, "bldc-small", "GT2-20T"
    )  # config/rig.toml as shipped
    results = [r for d in [ref, *grid] if (r := score_design(d, base, hw.sensors, rnd))]
    logger.info(f"{len(results)}/{len(grid) + 1} designs balanceable; screening…")
    ok = [r for r in results if feasible(r)]
    ok.sort(key=lambda r: r["worst_u_rms"])
    logger.info(
        f"{len(ok)} meet the constraints (accel>=10, v_noload>=2, work/E>=5, robust>=95%, +1 delay)"
    )

    def row(r: dict) -> str:
        per = r["per"]
        return (
            f"{r['design'].label():44s} M={r['physics'].M:4.2f} F={r['physics'].force_max:5.1f} "
            f"vnl={(r['v_noload'] or 0):4.1f} | u_rms UU/UD/DU = "
            + "/".join(f"{per[k].u_rms:4.2f}" for k in EQUILIBRIA)
            + f" N | ROA {r['min_roa']:4.1f}° | delay+{r['min_delay']} | work/E {r['work_ratio']:4.1f}"
        )

    print("\nreference (config/rig.toml as shipped):")
    ref_r = next((r for r in results if r["design"] == ref), None)
    print("  " + (row(ref_r) if ref_r else "unbalanceable"))
    print("\nbest by noise-driven control effort:")
    for r in ok[: args.top]:
        print("  " + row(r))
    if args.stage1_only:
        return
    print(
        "\nstage 2: full-env closed-loop LQR holds (sensors, Coulomb, motor limit, delay, randomized plant)"
    )
    for r in ([ref_r] if ref_r else []) + ok[: args.top]:
        res = stage2(r, env_cfg, seeds=args.seeds)
        cells = "  ".join(
            f"{k}: {v['ok']}/{v['seeds']} {v['rms']:4.2f}N dF{v['ddf']:4.2f}"
            for k, v in res.items()
        )
        print(f"  {r['design'].label():44s} {cells}")


if __name__ == "__main__":
    main()
