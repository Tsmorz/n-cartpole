"""CLI entry point for training the double cartpole PPO policy."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from n_cartpole.env.double_cartpole import EnvConfig
from n_cartpole.env.dynamics import PhysicsParams
from n_cartpole.training.trainer import Trainer, TrainingConfig


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a PPO policy for double pendulum cartpole swing-up."
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="Number of rollout workers"
    )
    parser.add_argument(
        "--steps", type=int, default=2048, help="Steps per worker per iteration"
    )
    parser.add_argument(
        "--iterations", type=int, default=300, help="Number of PPO iterations"
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Adam learning rate")
    parser.add_argument("--hidden", type=int, default=64, help="MLP hidden layer size")
    parser.add_argument(
        "--mini-batch",
        type=int,
        default=None,
        help="PPO minibatch size (default: 512)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Gradient-update device. 'auto' picks CPU, which is fastest for this "
        "small network; use 'mps'/'cuda' only if you scale the network up.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--resume", type=Path, default=None, help="Resume from checkpoint path"
    )
    return parser.parse_args()


def main() -> None:
    """Build TrainingConfig, instantiate Trainer, and run."""
    args = parse_args()

    cfg = TrainingConfig(
        env=EnvConfig(physics=PhysicsParams()),
        device=args.device,
        steps_per_worker=args.steps,
        hidden=args.hidden,
        lr=args.lr,
        n_iterations=args.iterations,
        checkpoint_dir=args.checkpoint_dir,
    )
    if args.workers is not None:
        cfg.n_workers = args.workers
    if args.mini_batch is not None:
        cfg.mini_batch_size = args.mini_batch

    trainer = Trainer(cfg)

    if args.resume is not None:
        trainer.load(args.resume)
        logger.info(f"Resuming from {args.resume}")

    trainer.train()


if __name__ == "__main__":
    main()
