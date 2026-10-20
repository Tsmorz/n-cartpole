"""Measure how long a swing-up policy takes to catch, and whether more time helps.

Rolls ``--seeds`` long episodes (default 30 s) and records, per episode: when it first
stays within 20 degrees of upright for ``--catch-seconds`` (the catch), whether it
leaves the rail, and whether it is holding upright in the final ``--hold-seconds``.
Prints the cumulative share of episodes caught by each time horizon, so the effect of
giving the policy more time (a retry window) can be read off directly.
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
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--catch-seconds", type=float, default=3.0)
    ap.add_argument("--hold-seconds", type=float, default=5.0)
    ap.add_argument("--realism", type=float, default=None)
    ap.add_argument("--estimator", choices=("diff", "ekf"), default=None)
    ap.add_argument("--ekf-accel-std", type=float, default=None)
    return ap.parse_args()


def main() -> None:
    """Run the episodes and print the catch-time distribution."""
    args = parse_args()
    bundle = load_policy(args.checkpoint)
    dt = bundle.physics.dt
    steps = round(args.seconds / dt)
    hold = round(args.hold_seconds / dt)
    need = round(args.catch_seconds / dt)
    cfg = dataclasses.replace(bundle.env_config, max_steps=steps)
    if args.realism is not None:
        cfg = dataclasses.replace(cfg, realism=args.realism)
    over = {
        k: v
        for k, v in (
            ("estimator", args.estimator),
            ("ekf_accel_std", args.ekf_accel_std),
        )
        if v is not None
    }
    if over and cfg.hardware is not None and cfg.hardware.sensors is not None:
        sensors = dataclasses.replace(cfg.hardware.sensors, **over)
        cfg = dataclasses.replace(
            cfg, hardware=dataclasses.replace(cfg.hardware, sensors=sensors)
        )
    env = make_env(cfg)
    catch_t: list[float | None] = []
    rail_t: list[float | None] = []
    holding = 0
    lost_after_catch = 0
    for seed in range(args.seeds):
        obs, _ = env.reset(seed=seed)
        states, done = [env.get_state()], False
        while not done:
            act = bundle.select_action(bundle.normalize(np.asarray(obs)[None]))
            obs, _, term, trunc, _ = env.step(act[0])
            done = term or trunc
            states.append(env.get_state())
        st = np.array(states)
        left = len(st) < steps + 1
        rail_t.append(len(st) * dt if left else None)
        up = (np.degrees(np.abs((st[:, 2::2] + np.pi) % (2 * np.pi) - np.pi)) < 20).all(
            axis=1
        )
        first, i = None, 0
        while i < len(up):
            if up[i]:
                j = i
                while j < len(up) and up[j]:
                    j += 1
                if j - i >= need and first is None:
                    first = i * dt
                i = j
            else:
                i += 1
        catch_t.append(first)
        if not left and up[-hold:].mean() > 0.95:
            holding += 1
        elif first is not None and not left:
            lost_after_catch += 1
    n = args.seeds
    print(
        f"{args.checkpoint.parent.name}/{args.checkpoint.name}: {n} episodes x {args.seconds:.0f} s"
    )
    print(
        f"cumulative share caught (a sustained >={args.catch_seconds:.0f} s upright stay) "
        "by time t:"
    )
    for t in (3, 5, 8, 10, 15, 20, 25, 30):
        if t > args.seconds:
            break
        caught = sum(c is not None and c <= t for c in catch_t)
        railed = sum(r is not None and r <= t for r in rail_t)
        print(
            f"   by {t:2d} s: caught {caught:2d}/{n} ({caught / n:4.0%}), left the rail {railed:2d}/{n}"
        )
    print(
        f"holding upright in the final {args.hold_seconds:.0f} s: {holding}/{n}; caught but later lost it: {lost_after_catch}"
    )
    caught_times = [c for c in catch_t if c is not None]
    if caught_times:
        print(
            f"catch time: median {np.median(caught_times):.1f} s, 90th pct {np.percentile(caught_times, 90):.1f} s, max {max(caught_times):.1f} s"
        )


if __name__ == "__main__":
    main()
