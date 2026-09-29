"""CLI entry point for visualizing a trained double cartpole policy."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from n_cartpole.env.double_cartpole import DoublePendulumCartpole, EnvConfig
from n_cartpole.env.dynamics import PhysicsParams
from n_cartpole.policy.actor_critic import Actor, RunningNorm
from n_cartpole.policy.tqc import SquashedGaussianActor
from n_cartpole.viz.animate import animate_episode


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize a trained double cartpole policy."
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
        "--save",
        type=Path,
        default=None,
        help="Save animation to file (.gif or .mp4) instead of showing",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0, help="Animation playback speed"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def run_episode(
    select_action: Callable[[torch.Tensor], np.ndarray],
    norm: RunningNorm,
    env: DoublePendulumCartpole,
    device: torch.device,
    seed: int | None = None,
) -> tuple[np.ndarray, float]:
    """Roll out one deterministic episode and return states + total reward."""
    obs, _ = env.reset(seed=seed)
    states = [env.get_state()]
    total_reward = 0.0

    with torch.no_grad():
        terminated = truncated = False
        while not (terminated or truncated):
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
            obs_norm = norm.normalize(obs_t)
            action = select_action(obs_norm)
            obs, reward, terminated, truncated, _ = env.step(action)
            states.append(env.get_state())
            total_reward += reward

    return np.array(states), total_reward


def main() -> None:
    """Load checkpoint, run episodes, and visualize."""
    args = parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg")
    hidden = cfg.hidden if cfg is not None else 64
    physics = cfg.env.physics if cfg is not None else PhysicsParams()
    algo = ckpt.get("algo", "ppo")
    device = torch.device("cpu")

    norm = RunningNorm(8)
    norm.load_state_dict(ckpt["norm"])

    if algo == "tqc":
        actor = SquashedGaussianActor(hidden=hidden, force_max=physics.force_max)
        actor.load_state_dict(ckpt["actor"])
        actor.eval()

        def select_action(obs_norm: torch.Tensor) -> np.ndarray:
            return actor.act(obs_norm, deterministic=True).squeeze(0).cpu().numpy()
    else:
        ppo_actor = Actor(hidden=hidden)
        ppo_actor.load_state_dict(ckpt["actor"])
        ppo_actor.eval()

        def select_action(obs_norm: torch.Tensor) -> np.ndarray:
            mean, _ = ppo_actor(obs_norm)  # deterministic: use the mean
            return mean.squeeze(0).cpu().numpy()

    logger.info(f"Loaded {algo.upper()} policy from {args.checkpoint}")
    env = DoublePendulumCartpole(EnvConfig(physics=physics))

    for ep in range(args.episodes):
        seed = args.seed + ep
        states, total_reward = run_episode(select_action, norm, env, device, seed=seed)
        logger.info(
            f"Episode {ep + 1}: {len(states)} steps, total reward = {total_reward:.2f}"
        )

        save_path = None
        if args.save is not None:
            stem = args.save.stem + (f"_ep{ep + 1}" if args.episodes > 1 else "")
            save_path = args.save.with_name(stem + args.save.suffix)

        animate_episode(
            states, physics, dt=physics.dt, save_path=save_path, speed=args.speed
        )


if __name__ == "__main__":
    main()
