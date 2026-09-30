"""CLI: visualize a trained cartpole policy as an interactive HTML replay."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
from loguru import logger

from n_cartpole.env.cartpole import EnvConfig, NPendulumCartpole
from n_cartpole.env.factory import make_env
from n_cartpole.env.goals import goal_labels, parse_goal_schedule
from n_cartpole.policy.loader import PolicyBundle, load_policy
from n_cartpole.viz.animate import animate_episode


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize a trained cartpole policy (interactive HTML replay)."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/double/tqc/tqc_latest.pt"),
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--episodes", type=int, default=1, help="Number of episodes to run"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("checkpoints/replay.html"),
        help="Output HTML path for the replay",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0, help="Animation playback speed"
    )
    parser.add_argument(
        "--no-open", action="store_true", help="Do not open the HTML in a browser"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--goals",
        type=str,
        default=None,
        help="Goal-conditioned checkpoints: commanded goals with times in seconds, "
        "e.g. 'UU@0,DU@6,DD@12' (U/D per link, base link first). Default: visit "
        "every configuration, 6 s each.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Goal-conditioned checkpoints: starting configuration, e.g. DD "
        "(default: all links down)",
    )
    parser.add_argument(
        "--hold",
        type=float,
        default=6.0,
        help="Seconds to keep running after the last commanded goal",
    )
    return parser.parse_args()


def run_episode(
    bundle: PolicyBundle,
    env: NPendulumCartpole,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Roll out one deterministic episode.

    Returns (states (T,·), actions (T-1,), rewards (T-1,), total_reward).
    """
    obs, _ = env.reset(seed=seed)
    states = [env.get_state()]
    actions: list[float] = []
    rewards: list[float] = []
    total_reward = 0.0

    terminated = truncated = False
    while not (terminated or truncated):
        obs_norm = bundle.normalize(np.asarray(obs)[None, :])
        action = bundle.select_action(obs_norm)[0]
        obs, reward, terminated, truncated, _ = env.step(action)
        states.append(env.get_state())
        actions.append(float(np.asarray(action).reshape(-1)[0]))
        rewards.append(float(reward))
        total_reward += float(reward)

    return np.array(states), np.array(actions), np.array(rewards), total_reward


def run_goal_episode(
    bundle: PolicyBundle,
    env: NPendulumCartpole,
    schedule: list[tuple[float, int]],
    start: str,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Roll out one deterministic episode that follows a commanded goal schedule.

    Returns (states, actions, rewards, total_reward, goals) where ``goals`` is the
    ``(T, n)`` commanded target angles aligned with ``states``.
    """
    dt = env.cfg.physics.dt
    pending = [(round(t / dt), g) for t, g in schedule]
    obs, _ = env.reset(seed=seed, options={"start": start, "goal": pending[0][1]})
    pending = pending[1:]
    states = [env.get_state()]
    goals = [env.goal_angles]
    actions: list[float] = []
    rewards: list[float] = []
    total_reward = 0.0

    k = 0
    terminated = truncated = False
    while not (terminated or truncated):
        if pending and k >= pending[0][0]:
            if k > 0:
                logger.info(
                    f"t={k * dt:5.2f}s  {env.goal}: "
                    f"{'reached' if env.settled else 'NOT reached'}"
                )
            obs = env.set_goal(pending.pop(0)[1])
            logger.info(f"t={k * dt:5.2f}s  goal → {env.goal}")
        obs_norm = bundle.normalize(np.asarray(obs)[None, :])
        action = bundle.select_action(obs_norm)[0]
        obs, reward, terminated, truncated, info = env.step(action)
        k += 1
        if "segment" in info:
            frm, to, ok = info["segment"]
            logger.info(
                f"t={k * dt:5.2f}s  {env.goal_labels[to]}: "
                f"{'reached' if ok else 'NOT reached'}"
            )
        states.append(env.get_state())
        goals.append(env.goal_angles)
        actions.append(float(np.asarray(action).reshape(-1)[0]))
        rewards.append(float(reward))
        total_reward += float(reward)

    return (
        np.array(states),
        np.array(actions),
        np.array(rewards),
        total_reward,
        np.array(goals),
    )


def main() -> None:
    """Load checkpoint, run episodes, and write interactive replays."""
    args = parse_args()
    bundle = load_policy(args.checkpoint)
    logger.info(
        f"Loaded {bundle.algo.upper()} policy from {args.checkpoint} "
        f"({bundle.n_links}-link)"
    )
    # The checkpoint's own env config keeps the observation layout (goal block,
    # sysID context) identical to training.
    env_cfg: EnvConfig = bundle.env_config
    schedule: list[tuple[float, int]] = []
    start = ""
    if bundle.goal_conditioned:
        labels = goal_labels(bundle.n_links)
        spec = args.goals or ",".join(f"{lab}@{6 * i}" for i, lab in enumerate(labels))
        schedule = parse_goal_schedule(spec, bundle.n_links)
        start = args.start or labels[-1]  # all links down
        duration = schedule[-1][0] + args.hold
        env_cfg = dataclasses.replace(
            env_cfg,
            goal_hold_steps=None,  # goals come from the schedule only
            max_steps=int(round(duration / bundle.physics.dt)),
        )
        logger.info(f"Start {start}; goal schedule: {spec}")
    elif args.goals:
        logger.warning("--goals ignored: checkpoint is not goal-conditioned")
    env = make_env(env_cfg)

    for ep in range(args.episodes):
        seed = args.seed + ep
        goals = None
        if bundle.goal_conditioned:
            states, actions, rewards, total_reward, goals = run_goal_episode(
                bundle, env, schedule, start, seed=seed
            )
        else:
            states, actions, rewards, total_reward = run_episode(bundle, env, seed=seed)
        logger.info(
            f"Episode {ep + 1}: {len(states)} steps, total reward = {total_reward:.2f}"
        )

        out = args.out
        if args.episodes > 1:
            out = out.with_stem(f"{out.stem}_ep{ep + 1}")
        animate_episode(
            states,
            bundle.physics,
            dt=bundle.physics.dt,
            save_path=out,
            speed=args.speed,
            actions=actions,
            rewards=rewards,
            goals=goals,
            title=f"{bundle.algo.upper()} · {bundle.n_links}-link · "
            + (
                f"{start} → "
                + " → ".join(env.goal_labels[g] for _, g in schedule)
                + " · "
                if goals is not None
                else ""
            )
            + f"return {total_reward:.1f}",
            open_browser=not args.no_open,
        )


if __name__ == "__main__":
    main()
