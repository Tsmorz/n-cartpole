"""CLI: success matrix of a goal-conditioned policy over every transition."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from n_cartpole.env.goals import goal_labels
from n_cartpole.policy.loader import load_policy
from n_cartpole.training.evaluate import evaluate_transitions


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate every (start → goal) transition of a goal-conditioned "
        "policy and print the success matrix."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/double/tqc-goal/tqc_latest.pt"),
        help="Goal-conditioned checkpoint (trained with --goals)",
    )
    parser.add_argument(
        "--seconds", type=float, default=10.0, help="Time budget per transition"
    )
    parser.add_argument("--trials", type=int, default=5, help="Episodes per pair")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    return parser.parse_args()


def main() -> None:
    """Load the checkpoint, evaluate all transitions, print the matrix."""
    args = parse_args()
    bundle = load_policy(args.checkpoint)
    stats = evaluate_transitions(bundle, args.seconds, args.trials, args.seed)
    logger.info(
        f"{bundle.algo.upper()} {args.checkpoint}: "
        f"{stats.success_rate():.0%} of transitions reached within {args.seconds}s\n"
        + stats.format_matrix(goal_labels(bundle.n_links))
    )


if __name__ == "__main__":
    main()
