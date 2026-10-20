"""Measure how hard each equilibrium is for a classical controller, not a learned policy.

For a set of plants (the idealized site plant, the buildable rig, and the rig with one
factor reverted at a time) this reports (1) the open-loop unstable pole speeds at UU/UD/DU
and (2) how large a perturbation a delay-aware LQR can still recover at UU, simulated
nonlinearly with the plant's force limit, slew limit, action delay and drive limit and
EXACT sensing. If a plant is hard for this controller, no learned policy will find it
easy; if it is easy, the difficulty is in learning.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
from pathlib import Path

import numpy as np
from design_study import EQUILIBRIA, HoldController, linearize

from n_cartpole.config import load_tqc_config
from n_cartpole.env.cartpole import EnvConfig
from n_cartpole.env.dynamics import PhysicsParams
from n_cartpole.env.factory import make_env
from n_cartpole.env.hardware_config import HardwareConfig


def poles(p: PhysicsParams, eq: tuple[float, float]) -> list[float]:
    """Open-loop unstable pole speeds (rad/s), largest first."""
    A, _ = linearize(p, eq)
    lam = np.log(np.abs(np.linalg.eigvals(A))) / p.dt
    return sorted((float(x) for x in lam if x > 1e-3), reverse=True)


def recovers(
    p: PhysicsParams,
    slew: float | None,
    delay: int,
    delta: float,
    direction: float,
    seconds: float = 8.0,
) -> bool:
    """Hold UU for ``seconds`` from a ``delta`` rad tilt (direction in the (th0, th1) plane)."""
    cfg = EnvConfig(
        physics=p,
        goal_conditioned=True,
        goal_hold_steps=None,
        force_slew=slew,
        hardware=HardwareConfig(sysid_context=False, delay_steps=delay),
        max_steps=round(seconds / p.dt),
        init_noise=0.0,
    )
    env = make_env(cfg)
    env.reset(seed=0, options={"start": "UU", "goal": "UU"})
    st = np.zeros(6)
    st[2], st[4] = delta * math.cos(direction), delta * math.sin(direction)
    env._state = st
    env._reset_potentials()
    ctrl = HoldController(p, EQUILIBRIA["UU"], None, "diff")  # type: ignore[arg-type]
    ctrl.reset()
    obs = env._make_obs()
    for _ in range(cfg.max_steps):
        obs, _, term, trunc, _ = env.step(np.array([ctrl.act(obs, p.force_max)]))
        if term:
            return False
        if trunc:
            break
    s = env.get_state()
    return bool(np.all(np.abs(s[2::2]) < 0.2) and abs(s[0]) < p.x_lim)


def main() -> None:
    """Print poles and the LQR recovery table for each plant variant."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rig", type=Path, default=Path("config/rig.toml"))
    ap.add_argument("--directions", type=int, default=8)
    args = ap.parse_args()
    rig_env = load_tqc_config(args.rig).env
    rig = rig_env.physics
    site = PhysicsParams()  # the site plant: point masses, 20 N, +-0.5 m
    rig_delay = 1
    variants: list[tuple[str, PhysicsParams, float | None, int]] = [
        ("site plant (20 N, 800 N/s, point masses, no delay)", site, 800.0, 0),
        (
            "rig as configured (15.4 N, 300 N/s, 5 ms delay)",
            rig,
            rig_env.force_slew,
            rig_delay,
        ),
        (
            "rig, drive speed limit removed",
            dataclasses.replace(rig, v_noload=None),
            rig_env.force_slew,
            rig_delay,
        ),
        (
            "rig with site force (20 N)",
            dataclasses.replace(rig, force_max=20.0),
            rig_env.force_slew,
            rig_delay,
        ),
        ("rig with site slew (800 N/s)", rig, 800.0, rig_delay),
        ("rig with no delay", rig, rig_env.force_slew, 0),
        (
            "rig with point-mass links (site links)",
            dataclasses.replace(
                rig, masses=site.masses, lengths=site.lengths, com=None, inertia=0.0
            ),
            rig_env.force_slew,
            rig_delay,
        ),
    ]
    print("open-loop unstable poles (rad/s):")
    for name, p, _, _ in variants[:2]:
        print(f"  {name}:")
        for eq in EQUILIBRIA:
            print(f"     {eq}: {[round(x, 1) for x in poles(p, EQUILIBRIA[eq])]}")
    deltas = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4)
    dirs = [math.pi * k / args.directions for k in range(2 * args.directions)]
    print(
        f"\ndelay-aware LQR, UU, exact sensing: share of {len(dirs)} tilt directions recovered, by tilt (rad)"
    )
    print(f"{'plant':56s} " + " ".join(f"{d:5.2f}" for d in deltas), flush=True)
    for name, p, slew, delay in variants:
        row = []
        for d in deltas:
            ok = sum(recovers(p, slew, delay, d, th) for th in dirs)
            row.append(ok / len(dirs))
        print(f"{name:56s} " + " ".join(f"{r:5.0%}" for r in row), flush=True)


if __name__ == "__main__":
    main()
