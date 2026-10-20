"""Add rig realism to an idealized-plant policy one factor at a time.

Takes a goal-conditioned checkpoint trained on the idealized plant (the policy behind
the personal-site demo) and re-evaluates DD -> UU while the plant is made more like the
buildable rig, cumulatively: action delay, rig drive limits, rig links, slew limit,
differenced-velocity sensing, encoder noise/quantization, rail friction, plant
randomization. Shows which realism factor each policy tolerates and which one breaks it.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Any

import numpy as np

from n_cartpole.config import load_tqc_config
from n_cartpole.env.factory import make_env
from n_cartpole.env.hardware_config import HardwareConfig, SensorModel
from n_cartpole.env.randomization import PlantRandomization
from n_cartpole.policy.loader import load_policy
from n_cartpole.training.evaluate import _rollout


def main() -> None:
    """Run the cumulative ablation and print one row per step."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--rig", type=Path, default=Path("config/rig.toml"))
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--start", default="DD")
    ap.add_argument("--goal", default="UU")
    args = ap.parse_args()

    bundle = load_policy(args.checkpoint)
    base = dataclasses.replace(
        bundle.env_config,
        goal_hold_steps=None,
        max_steps=round(args.seconds / bundle.physics.dt),
    )
    rig = load_tqc_config(args.rig).env
    dt = bundle.physics.dt  # evaluate at the policy's own control rate
    assert rig.hardware is not None and rig.hardware.sensors is not None
    rig_phys = dataclasses.replace(rig.physics, dt=dt)
    cfg = base
    plan: list[tuple[str, dict[str, Any]]] = [
        (
            "+ 1-step action delay",
            {"hardware": HardwareConfig(sysid_context=False, delay_steps=1)},
        ),
        (
            "+ rig drive: 15.4 N, +-0.40 m rail",
            {
                "physics": dataclasses.replace(
                    cfg.physics, force_max=rig_phys.force_max, x_lim=rig_phys.x_lim
                )
            },
        ),
        (
            "+ drive speed limit (back-EMF)",
            {"physics": None, "v_noload": rig_phys.v_noload},
        ),
        ("+ slew limit 300 N/s", {"force_slew": rig.force_slew}),
        (
            "+ rig links (rod mass, inertia), cart 0.92 kg",
            {
                "physics": None,
                "links": (
                    rig_phys.M,
                    rig_phys.masses,
                    rig_phys.lengths,
                    rig_phys.com,
                    rig_phys.inertia,
                ),
            },
        ),
        ("+ differenced-velocity sensing (no noise)", {"sensors": SensorModel()}),
        ("+ encoder noise + 14-bit quantization", {"sensors": rig.hardware.sensors}),
        (
            "+ rail friction 0.4 N",
            {"physics": None, "cart_coulomb": rig_phys.cart_coulomb},
        ),
        (
            "+ plant randomization (full rig realism)",
            {"randomize": rig.randomize or PlantRandomization()},
        ),
    ]
    rows = [("idealized plant (what the site runs)", base)]
    for label, change in plan:
        kw = dict(change)
        phys = cfg.physics
        if "links" in kw:
            m_cart, masses, lengths, com, inertia = kw.pop("links")
            phys = dataclasses.replace(
                phys, M=m_cart, masses=masses, lengths=lengths, com=com, inertia=inertia
            )
        if "v_noload" in kw:
            phys = dataclasses.replace(phys, v_noload=kw.pop("v_noload"))
        if "cart_coulomb" in kw:
            phys = dataclasses.replace(phys, cart_coulomb=kw.pop("cart_coulomb"))
        phys = kw.pop("physics", None) or phys
        if "sensors" in kw:
            hw = cfg.hardware or HardwareConfig(sysid_context=False, delay_steps=0)
            kw["hardware"] = dataclasses.replace(hw, sensors=kw.pop("sensors"))
        cfg = dataclasses.replace(cfg, physics=phys, **kw)
        rows.append((label, cfg))

    print(
        f"{args.checkpoint}: {args.start} -> {args.goal}, {args.seeds} seeds, {args.seconds:.0f} s, cumulative"
    )
    print(
        f"{'plant':46s} {'settled':>8s} {'median t':>9s} {'off-rail':>9s}", flush=True
    )
    for label, c in rows:
        env = make_env(c)
        ok, off, ts = 0, 0, []
        for seed in range(args.seeds):
            trial, _ = _rollout(
                bundle,
                env,
                env.goal_labels.index(args.start),
                env.goal_labels.index(args.goal),
                seed,
            )
            ok += trial.success
            off += trial.terminated
            if trial.success and not np.isnan(trial.settle_time):
                ts.append(trial.settle_time)
        t = f"{np.median(ts):.1f}s" if ts else "-"
        print(
            f"{label:46s} {ok:4d}/{args.seeds:<3d} {t:>9s} {off:6d}/{args.seeds}",
            flush=True,
        )


if __name__ == "__main__":
    main()
