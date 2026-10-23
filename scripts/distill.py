"""CLI: distil a trained TQC actor into a smaller one (for the ESP32 firmware).

See ``n_cartpole/training/distill.py``. The student is an ordinary checkpoint:
evaluate it with ``task eval-transitions`` and export it with
``task export-firmware``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from n_cartpole.training.distill import DistillConfig, distill


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--teacher",
        type=Path,
        default=Path("checkpoints/double/tqc-rail05-goals-ft-weighted/tqc_2000000.pt"),
        help="Checkpoint to imitate (its replay buffer is used if it sits beside it)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Default: <teacher dir>-h<H>b<B>/student.pt",
    )
    parser.add_argument("--hidden", type=int, default=64, help="Student width")
    parser.add_argument("--blocks", type=int, default=2, help="Student residual blocks")
    parser.add_argument("--rounds", type=int, default=6, help="DAgger rounds")
    parser.add_argument(
        "--steps-per-round",
        type=int,
        default=60_000,
        help="Env steps collected per round",
    )
    parser.add_argument(
        "--buffer-obs",
        type=int,
        default=400_000,
        help="Teacher buffer observations (0 = none)",
    )
    parser.add_argument(
        "--grad-steps", type=int, default=15_000, help="Adam steps per round"
    )
    parser.add_argument("--batch", type=int, default=2048, help="Minibatch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--lr-decay", type=float, default=1.0, help="Learning-rate factor per round"
    )
    parser.add_argument(
        "--tag", default="", help="Suffix for the default output folder"
    )
    parser.add_argument(
        "--beta", type=float, default=0.5, help="Teacher-acts probability in round 1"
    )
    parser.add_argument(
        "--realism",
        type=float,
        default=None,
        help="Env realism (default: the teacher's)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    return parser.parse_args()


def main() -> None:
    """Build the config and run the distillation."""
    args = parse_args()
    out = args.out or (
        args.teacher.parent.with_name(
            f"{args.teacher.parent.name}-h{args.hidden}b{args.blocks}{args.tag}"
        )
        / "student.pt"
    )
    distill(
        DistillConfig(
            teacher=args.teacher,
            out=out,
            hidden=args.hidden,
            blocks=args.blocks,
            rounds=args.rounds,
            steps_per_round=args.steps_per_round,
            buffer_obs=args.buffer_obs,
            grad_steps=args.grad_steps,
            batch_size=args.batch,
            lr=args.lr,
            lr_decay=args.lr_decay,
            beta=args.beta,
            realism=args.realism,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
