"""CLI entry point for off-policy TQC training of the double cartpole policy.

This is the sample-efficient, hardware-oriented alternative to PPO
(`scripts/train.py`). Both share the same environment and checkpoint format.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from n_cartpole.env.double_cartpole import EnvConfig
from n_cartpole.env.dynamics import PhysicsParams
from n_cartpole.training.off_policy import TQCConfig, TQCTrainer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a TQC (off-policy) policy for double cartpole swing-up."
    )
    parser.add_argument("--steps", type=int, default=200_000, help="Total env steps")
    parser.add_argument("--hidden", type=int, default=256, help="MLP hidden size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Adam learning rate")
    parser.add_argument("--batch", type=int, default=256, help="Minibatch size")
    parser.add_argument("--n-critics", type=int, default=2, help="Number of critics")
    parser.add_argument("--n-quantiles", type=int, default=25, help="Atoms per critic")
    parser.add_argument(
        "--drop", type=int, default=2, help="Top atoms to drop from the pooled set"
    )
    parser.add_argument(
        "--no-symmetry", action="store_true", help="Disable symmetric augmentation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Gradient-update device ('auto' picks CPU for this small network)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Checkpoint dir",
    )
    return parser.parse_args()


def main() -> None:
    """Build TQCConfig, instantiate TQCTrainer, and run."""
    args = parse_args()
    cfg = TQCConfig(
        env=EnvConfig(physics=PhysicsParams()),
        device=args.device,
        hidden=args.hidden,
        lr=args.lr,
        batch_size=args.batch,
        n_critics=args.n_critics,
        n_quantiles=args.n_quantiles,
        top_quantiles_to_drop=args.drop,
        symmetry_augment=not args.no_symmetry,
        total_steps=args.steps,
        checkpoint_dir=args.checkpoint_dir,
    )
    trainer = TQCTrainer(cfg)
    logger.info(
        f"TQC: {cfg.n_critics} critics x {cfg.n_quantiles} atoms, "
        f"drop {cfg.top_quantiles_to_drop}, symmetry={cfg.symmetry_augment}"
    )
    trainer.train()


if __name__ == "__main__":
    main()
