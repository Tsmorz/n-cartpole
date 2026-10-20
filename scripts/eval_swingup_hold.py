"""Evaluate swing-up + hold of a plain (non-goal) checkpoint, time-consistently.

Rolls the deterministic policy for ``--seconds`` from the usual reset distribution
and reports: reached upright, held upright over the last ``--hold-seconds``, time to
upright, off-rail count, and force statistics taken from the environment itself so
absolute and delta action modes compare fairly: the APPLIED force's RMS and mean
per-step change over the hold window, and the COMMANDED force's mean and worst
per-step jump over the whole episode (how smooth the policy's own command is). Windows are in seconds so it is correct at
any control rate. ``--realism`` overrides the checkpoint's own level (e.g. evaluate a
clean-plant policy on the full-realism plant).
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
from loguru import logger

from n_cartpole.env.factory import make_env
from n_cartpole.policy.loader import load_policy


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--hold-seconds", type=float, default=5.0)
    ap.add_argument("--realism", type=float, default=None)
    return ap.parse_args()


def main() -> None:
    """Evaluate every given checkpoint and print one comparison table."""
    args = parse_args()
    print(
        f"{'checkpoint':30s} {'reached':>8s} {'held':>6s} {'t_up':>6s} {'off-rail':>8s} "
        f"{'hold RMS F':>11s} {'applied |dF|':>13s} {'cmd |dF| mean/max':>19s} {'upright':>8s}"
    )
    for path in args.checkpoint:
        bundle = load_policy(path)
        dt = bundle.physics.dt
        cfg = dataclasses.replace(bundle.env_config, max_steps=round(args.seconds / dt))
        if args.realism is not None:
            cfg = dataclasses.replace(cfg, realism=args.realism)
        env = make_env(cfg)
        hold = round(args.hold_seconds / dt)
        reach = held = off = 0
        tup, rms, ddf, mdf, cj, frac = [], [], [], [], [], []
        for seed in range(args.seeds):
            obs, _ = env.reset(seed=seed)
            states, applied, command = [env.get_state()], [], []
            done = False
            while not done:
                act = bundle.select_action(bundle.normalize(np.asarray(obs)[None]))
                obs, _, term, trunc, info = env.step(act[0])
                done = term or trunc
                states.append(env.get_state())
                applied.append(info["applied_force"])  # what the actuator delivered
                command.append(env._prev_cmd)  # the force the policy commanded
            st = np.array(states)
            theta = st[:, 2::2]  # theta = 0 is upright
            up = (np.degrees(np.abs((theta + np.pi) % (2 * np.pi) - np.pi)) < 20).all(
                axis=1
            )
            off += len(st) < cfg.max_steps + 1
            reach += bool(up.any())
            held += bool(up[-hold:].mean() > 0.95)
            frac.append(up[-hold:].mean())
            if up.any():
                tup.append(np.argmax(up) * dt)
            tail = np.array(applied[-hold:])
            rms.append(np.sqrt(np.mean(tail**2)))
            ddf.append(np.abs(np.diff(tail)).mean())
            cmd_jump = np.abs(np.diff(np.array(command)))
            mdf.append(cmd_jump.max())
            cj.append(cmd_jump.mean())
        n = args.seeds
        print(
            f"{path.parent.name + '/' + path.name:30s} {reach:5d}/{n:<2d} {held:3d}/{n:<2d} "
            f"{np.median(tup) if tup else float('nan'):5.1f}s {off:5d}/{n:<2d} "
            f"{np.median(rms):9.2f} N {np.median(ddf):11.3f} N "
            f"{np.median(cj):7.3f}/{np.max(mdf):6.2f} N {np.mean(frac):8.2f}"
        )
    logger.info("done")


if __name__ == "__main__":
    main()
