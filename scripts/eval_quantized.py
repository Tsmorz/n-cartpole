"""CLI: the start-by-goal transition test with the actor the firmware would run.

Runs ``evaluate_transition_metrics`` (the same episodes and seeds as
``task eval-transitions``) with the actor's weights as the device holds them:
``torch`` (the checkpoint itself), ``f32`` (the folded float32 tensors through the
NumPy port of ``ncp_policy_act``) or ``i8`` (int8 fc1/fc2, :mod:`n_cartpole.deploy.quantize`).
For ``i8`` it also reports how far the int8 force is from the float force on the
states the int8 policy actually visits.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from n_cartpole.deploy.actor_export import actor_tensors
from n_cartpole.deploy.quantize import actor_forward, numpy_bundle, quantize_tensors
from n_cartpole.env.goals import goal_labels
from n_cartpole.env.randomization import PlantRandomization
from n_cartpole.policy.loader import load_policy
from n_cartpole.training.evaluate import evaluate_transition_metrics


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--weights", choices=("torch", "f32", "i8"), default="i8")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nominal", action="store_true")
    parser.add_argument("--randomize", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Evaluate the chosen weights and print the matrices."""
    args = parse_args()
    bundle = load_policy(args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tensors = actor_tensors(ckpt)
    force_max = float(bundle.physics.force_max)
    diffs: list[np.ndarray] = []
    if args.weights == "f32":
        bundle = numpy_bundle(bundle, tensors)
    elif args.weights == "i8":
        ref = [(n, np.asarray(a, np.float32)) for n, a in tensors]
        bundle = numpy_bundle(bundle, quantize_tensors(tensors))
        act = bundle.select_action

        def select_action(obs: torch.Tensor) -> np.ndarray:
            out = act(obs)
            diffs.append(np.abs(out[:, 0] - actor_forward(ref, obs.numpy(), force_max)))
            return out

        bundle.select_action = select_action
    stats, metrics = evaluate_transition_metrics(
        bundle,
        args.seconds,
        args.trials,
        args.seed,
        randomize=PlantRandomization() if args.randomize else None,
        nominal=args.nominal,
    )
    labels = goal_labels(bundle.n_links)
    report = (
        f"{args.weights} {args.checkpoint}: {stats.success_rate():.0%} of transitions "
        f"reached within {args.seconds}s ({args.trials} trials, seed {args.seed})\n"
        + stats.format_matrix(labels)
        + "\n\n"
        + metrics.format(labels)
    )
    if diffs:
        d = np.concatenate(diffs)
        report += (
            f"\n\n|F_i8 - F_f32| on the visited states ({d.size:,} steps): "
            f"mean {d.mean():.3f} N, p99 {np.percentile(d, 99):.3f} N, "
            f"max {d.max():.3f} N (force_max {force_max:.2f} N)"
        )
    logger.info(report)


if __name__ == "__main__":
    main()
