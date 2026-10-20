"""Sweep checkpoints of one run and report full-hold rate, stay length and hold effort.

For each numbered checkpoint (plain swing-up policy) and each requested realism level,
roll ``--seeds`` episodes and report: the number that stay within 20 degrees of upright
for the whole final ``--hold-seconds`` (a "hold"), episodes with a stay over 3 s, the
longest stay, the upright share of the final window, and — in the held episodes only —
the RMS applied force and its per-step change. Use it to pick the best-holding
checkpoint of a run instead of defaulting to the last one, and (with ``--realism``) to
see how a policy trained on an easy plant degrades as realism is added.

Realism levels per checkpoint: the level scheduled by ``--ramp-start/--ramp-steps``
(from ``--base-realism`` to 1) if given, plus every value passed to ``--realism``.
With neither, the checkpoint's own configured level is used.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np

from n_cartpole.env.factory import make_env
from n_cartpole.policy.loader import load_policy


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--min-step", type=int, default=0)
    ap.add_argument("--max-step", type=int, default=10**9)
    ap.add_argument("--every", type=int, default=1, help="Use every Nth checkpoint")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--hold-seconds", type=float, default=5.0)
    ap.add_argument(
        "--realism",
        type=float,
        nargs="*",
        default=None,
        help="Also evaluate every checkpoint at these fixed realism levels",
    )
    ap.add_argument(
        "--estimator",
        choices=("diff", "ekf"),
        default=None,
        help="Override where the velocities in the observation come from",
    )
    ap.add_argument("--ekf-accel-std", type=float, default=None)
    ap.add_argument("--ekf-force-std", type=float, default=None)
    ap.add_argument("--ramp-start", type=int, default=None)
    ap.add_argument("--ramp-steps", type=int, default=0)
    ap.add_argument("--base-realism", type=float, default=0.0)
    return ap.parse_args()


def evaluate(path: Path, level: float | None, args: argparse.Namespace) -> str:
    """Roll episodes of one checkpoint at one realism level; return a table row."""
    bundle = load_policy(path)
    dt = bundle.physics.dt
    steps = round(args.seconds / dt)
    hold = round(args.hold_seconds / dt)
    cfg = dataclasses.replace(bundle.env_config, max_steps=steps)
    if level is not None:
        cfg = dataclasses.replace(cfg, realism=level)
    overrides = {
        k: v
        for k, v in (
            ("estimator", args.estimator),
            ("ekf_accel_std", args.ekf_accel_std),
            ("ekf_force_std", args.ekf_force_std),
        )
        if v is not None
    }
    if overrides and cfg.hardware is not None and cfg.hardware.sensors is not None:
        sensors = dataclasses.replace(cfg.hardware.sensors, **overrides)
        cfg = dataclasses.replace(
            cfg, hardware=dataclasses.replace(cfg.hardware, sensors=sensors)
        )
    env = make_env(cfg)
    held = off = 0
    stays: list[float] = []
    frac: list[float] = []
    rms: list[float] = []
    ddf: list[float] = []
    for seed in range(args.seeds):
        obs, _ = env.reset(seed=seed)
        states = [env.get_state()]
        forces: list[float] = []
        done = False
        while not done:
            act = bundle.select_action(bundle.normalize(np.asarray(obs)[None]))
            obs, _, term, trunc, info = env.step(act[0])
            done = term or trunc
            states.append(env.get_state())
            forces.append(info["applied_force"])
        st = np.array(states)
        off += len(st) < steps + 1
        wrapped = np.abs((st[:, 2::2] + np.pi) % (2 * np.pi) - np.pi)
        up = (np.degrees(wrapped) < 20).all(axis=1)
        frac.append(float(up[-hold:].mean()))
        longest, i = 0.0, 0
        while i < len(up):
            if up[i]:
                j = i
                while j < len(up) and up[j]:
                    j += 1
                longest = max(longest, (j - i) * dt)
                i = j
            else:
                i += 1
        stays.append(longest)
        if up[-hold:].mean() > 0.95:
            held += 1
            tail = np.array(forces[-hold:])
            rms.append(float(np.sqrt(np.mean(tail**2))))
            ddf.append(float(np.abs(np.diff(tail)).mean()))
    step_n = int(path.stem.split("_")[1])
    n = args.seeds
    return (
        f"{step_n:8d} {cfg.realism:8.2f} {held:3d}/{n:<3d} "
        f"{sum(s > 3 for s in stays):4d}/{n:<3d} {max(stays):7.1f}s {np.mean(frac):8.2f} "
        f"{(np.median(rms) if rms else float('nan')):9.2f} N "
        f"{(np.median(ddf) if ddf else float('nan')):8.3f} N {off:8d}"
    )


def main() -> None:
    """Evaluate every selected checkpoint at every requested realism level."""
    args = parse_args()
    ckpts = sorted(
        (p for p in args.run_dir.glob("tqc_[0-9]*.pt")),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    ckpts = [
        p for p in ckpts if args.min_step <= int(p.stem.split("_")[1]) <= args.max_step
    ][:: args.every]
    print(
        f"{'step':>8s} {'realism':>8s} {'held':>7s} {'stay>3s':>8s} {'longest':>8s} "
        f"{'upright':>8s} {'held RMS F':>11s} {'held |dF|':>10s} {'off-rail':>8s}",
        flush=True,
    )
    for path in ckpts:
        step_n = int(path.stem.split("_")[1])
        levels: list[float | None] = []
        if args.ramp_start is not None and args.ramp_steps > 0:
            f = float(np.clip((step_n - args.ramp_start) / args.ramp_steps, 0.0, 1.0))
            levels.append(args.base_realism + (1.0 - args.base_realism) * f)
        for r in args.realism or []:
            if r not in levels:
                levels.append(r)
        for level in levels or [None]:
            print(evaluate(path, level, args), flush=True)


if __name__ == "__main__":
    main()
