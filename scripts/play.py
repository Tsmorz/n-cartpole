"""CLI: visualize a trained cartpole policy as an interactive HTML replay."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from loguru import logger

from n_cartpole.env.double_cartpole import DoublePendulumCartpole, EnvConfig
from n_cartpole.env.factory import make_env
from n_cartpole.env.single_cartpole import SinglePendulumCartpole
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
        default=Path("checkpoints/latest.pt"),
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
    return parser.parse_args()


def run_episode(
    bundle: PolicyBundle,
    env: DoublePendulumCartpole | SinglePendulumCartpole,
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


def main() -> None:
    """Load checkpoint, run episodes, and write interactive replays."""
    args = parse_args()
    bundle = load_policy(args.checkpoint)
    logger.info(
        f"Loaded {bundle.algo.upper()} policy from {args.checkpoint} "
        f"({bundle.n_links}-link)"
    )
    env = make_env(EnvConfig(physics=bundle.physics, n_links=bundle.n_links))

    for ep in range(args.episodes):
        seed = args.seed + ep
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
            title=f"{bundle.algo.upper()} · {bundle.n_links}-link · "
            f"return {total_reward:.1f}",
            open_browser=not args.no_open,
        )


if __name__ == "__main__":
    main()
