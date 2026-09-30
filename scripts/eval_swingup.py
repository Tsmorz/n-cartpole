"""CLI: time a swing-up policy from randomized low-energy near-hanging starts."""

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
    parser = argparse.ArgumentParser(
        description="Roll out a swing-up policy from randomized near-hanging, "
        "low-rotational-energy starts and report how long it takes to swing up."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/double/tqc/tqc_latest.pt"),
        help="Swing-up checkpoint; a goal-conditioned one is commanded --goal",
    )
    parser.add_argument("--trials", type=int, default=10, help="Number of episodes")
    parser.add_argument(
        "--seconds", type=float, default=20.0, help="Time budget per episode"
    )
    parser.add_argument(
        "--angle-noise",
        type=float,
        default=0.2,
        help="Max radians each link starts away from straight down (pi)",
    )
    parser.add_argument(
        "--vel-noise",
        type=float,
        default=0.2,
        help="Max rad/s (and m/s for the cart) of the low-energy starting velocity",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument(
        "--goal",
        type=str,
        default=None,
        help="Goal-conditioned checkpoints only: target label (default all-up, "
        "e.g. UU)",
    )
    return parser.parse_args()


def main() -> None:
    """Load the checkpoint, roll out randomized near-hanging starts, report timing."""
    args = parse_args()
    bundle = load_policy(args.checkpoint)
    goal = args.goal or "U" * bundle.n_links
    # Goal-conditioned: one fixed goal for the whole episode, from all-down.
    options = {"start": "D" * bundle.n_links, "goal": goal}

    dt = bundle.physics.dt
    cfg = dataclasses.replace(
        bundle.env_config, max_steps=int(round(args.seconds / dt))
    )
    if bundle.goal_conditioned:
        cfg = dataclasses.replace(cfg, goal_hold_steps=None)
    elif args.goal is not None:
        raise ValueError("--goal needs a goal-conditioned checkpoint")
    env = make_env(cfg)
    rng = np.random.default_rng(args.seed)

    times: list[float] = []
    for trial in range(args.trials):
        env.reset(
            seed=int(rng.integers(2**31)),
            options=options if bundle.goal_conditioned else None,
        )
        state = env.get_state()
        state[0] = 0.0
        state[1] = rng.uniform(-args.vel_noise, args.vel_noise)
        state[2::2] = np.pi + rng.uniform(
            -args.angle_noise, args.angle_noise, bundle.n_links
        )
        state[3::2] = rng.uniform(-args.vel_noise, args.vel_noise, bundle.n_links)
        env._state = state
        env._prev_potential = env._angle_potential(state)
        obs = env._make_obs()

        # `env.settled`/`_at_goal` bookkeeping only runs on the goal-conditioned
        # step() path, so a plain swing-up policy never updates it; track the
        # same "at goal" condition locally instead (at the goal within
        # ``goal_tol_angle``/``goal_tol_vel`` for ``goal_settle_steps`` in a row).
        swing_up_time: float | None = None
        settled_steps = 0
        k = 0
        terminated = truncated = False
        while not (terminated or truncated):
            action = bundle.select_action(bundle.normalize(np.asarray(obs)[None]))
            obs, _, terminated, truncated, _ = env.step(action[0])
            k += 1
            if swing_up_time is None:
                settled_steps = settled_steps + 1 if env._at_goal(env._state) else 0
                if settled_steps >= cfg.goal_settle_steps:
                    swing_up_time = (k - cfg.goal_settle_steps + 1) * dt

        status = f"{swing_up_time:5.2f}s" if swing_up_time is not None else "  never"
        logger.info(f"trial {trial:2d}: swing-up in {status}")
        if swing_up_time is not None:
            times.append(swing_up_time)

    n_ok = len(times)
    logger.info(f"{n_ok}/{args.trials} swung up within {args.seconds:.0f}s")
    if times:
        arr = np.array(times)
        logger.info(
            f"time-to-swing-up: mean={arr.mean():.2f}s  "
            f"median={np.median(arr):.2f}s  min={arr.min():.2f}s  max={arr.max():.2f}s"
        )


if __name__ == "__main__":
    main()
