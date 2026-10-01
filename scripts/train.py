"""CLI entry point for off-policy TQC training of the cartpole policy."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import torch
from loguru import logger

from n_cartpole.config import load_tqc_config, rig_summary
from n_cartpole.env.cartpole import EnvConfig, RewardShape
from n_cartpole.env.factory import checkpoint_subdir, links_name
from n_cartpole.env.randomization import PlantRandomization
from n_cartpole.training.off_policy import TQCConfig, TQCTrainer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a TQC (off-policy) policy for double cartpole swing-up."
    )
    parser.add_argument(
        "--links",
        type=int,
        default=2,
        help="Number of pendulum links (>=1): 1 (single, warm-up), 2 (double), "
        "3 (triple), 4 (quadruple), ...",
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
        "--goals",
        action="store_true",
        help="Goal-conditioned training: learn transitions between every "
        "up/down configuration (UU, DU, UD, DD for 2 links) with one network. "
        "Checkpoints go to checkpoints/<links>/tqc-goal/.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Episode length in steps (default: 1000, or 3000 with --goals so "
        "each episode covers several transitions)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Checkpoint directory (default: checkpoints/<single|double|triple|…>/tqc, "
        "chosen from --links)",
    )
    parser.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="With --goals: warm-start from a plain swing-up TQC checkpoint "
        "(network size and physics come from it; the goal inputs start at zero "
        "weight, so training begins from the swing-up policy).",
    )
    parser.add_argument(
        "--seed-steps",
        type=int,
        default=None,
        help="Fill the replay buffer with this many steps of the warm-started "
        "policy before training (default: 100000 with --init-from, else 0)",
    )
    parser.add_argument(
        "--target-entropy",
        type=float,
        default=None,
        help="SAC entropy target in force units (default: log(force_max) - 1, "
        "the -1 convention on the [-1, 1]-scaled action)",
    )
    parser.add_argument(
        "--hold-prob",
        type=float,
        default=0.0,
        help="With --goals: probability a reset starts near an equilibrium and "
        "must hold it (perturbation grows as holding succeeds)",
    )
    parser.add_argument(
        "--hold-phase-steps",
        type=int,
        default=0,
        help="With --goals: first N env steps are hold-only episodes "
        "(stabilize every configuration before learning transitions)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="TOML file supplying the environment: physics (incl. rigid-body links), "
        "hardware pipeline, plant randomization, actuator slew (e.g. "
        "config/rig.toml). Training hyperparameters stay CLI flags. Checkpoints "
        "default to a '-<config name>' folder so they never overwrite the "
        "point-mass ones.",
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Randomize the TRUE plant each episode (masses, lengths, friction, "
        "actuator gain) so the policy tolerates model error. Uses the config's "
        "[randomize] ranges, or the defaults without a config.",
    )
    parser.add_argument(
        "--energy-shaping",
        type=float,
        default=None,
        help="Weight of the energy-gap shaping potential (removes the reward dead "
        "zone at hanging-down; overrides the config's [reward] value; 0 = off)",
    )
    parser.add_argument(
        "--wall-weight",
        type=float,
        default=None,
        help="Weight of the rail-margin reward factor (overrides [reward]; 0 = off)",
    )
    parser.add_argument(
        "--hold-effort",
        type=float,
        default=None,
        help="Weight of the hold-gated force penalty (overrides [reward]; 0 = off)",
    )
    parser.add_argument(
        "--smooth-temporal",
        type=float,
        default=0.0,
        help="CAPS temporal smoothness weight on the actor's action change between "
        "consecutive states (0 = off); suppresses high-frequency force dither",
    )
    parser.add_argument(
        "--smooth-spatial",
        type=float,
        default=0.0,
        help="CAPS spatial smoothness weight: action change under a small "
        "observation perturbation, i.e. a cap on the actor's local gain (0 = off)",
    )
    parser.add_argument(
        "--smooth-gate",
        type=float,
        default=0.0,
        help="Gate the smoothness penalty by closeness to the goal, alignment**N "
        "(0 = ungated). Quiet only when settled; swing-up stays aggressive.",
    )
    parser.add_argument(
        "--smooth-start",
        type=int,
        default=0,
        help="Env step at which the smoothness penalty starts ramping in",
    )
    parser.add_argument(
        "--smooth-ramp",
        type=int,
        default=0,
        help="Steps over which the smoothness penalty ramps from 0 to full weight "
        "(0 = immediate once started)",
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=0.05,
        help="Std of the observation perturbation (normalized units) for "
        "--smooth-spatial",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from a TQC checkpoint (architecture/env come from the "
        "checkpoint; --steps is the new ABSOLUTE total). Pass tqc_latest.pt to "
        "also restore the replay buffer.",
    )
    return parser.parse_args()


def main() -> None:
    """Build TQCConfig, instantiate TQCTrainer, and run."""
    args = parse_args()
    if args.links < 1:
        raise SystemExit(f"--links must be >= 1, got {args.links}")
    if args.init_from is not None and (not args.goals or args.resume is not None):
        raise SystemExit(
            "--init-from needs --goals and cannot be combined with --resume"
        )

    subdir = checkpoint_subdir("tqc", args.goals)
    base_env = EnvConfig()
    if args.config is not None:
        base_env = load_tqc_config(args.config).env
        subdir += f"-{args.config.stem}"
    overrides = {
        "energy_weight": args.energy_shaping,
        "wall_weight": args.wall_weight,
        "effort_weight": args.hold_effort,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}
    if overrides:
        shape = dataclasses.replace(base_env.reward_shape or RewardShape(), **overrides)
        base_env = dataclasses.replace(base_env, reward_shape=shape)
    if args.randomize and base_env.randomize is None:
        base_env = dataclasses.replace(base_env, randomize=PlantRandomization())
    checkpoint_dir = (
        args.checkpoint_dir or Path("checkpoints") / links_name(args.links) / subdir
    )

    cfg = TQCConfig(
        env=dataclasses.replace(
            base_env,
            n_links=args.links,
            goal_conditioned=args.goals,
            max_steps=args.max_steps or (3000 if args.goals else 1000),
            hold_prob=args.hold_prob,
        ),
        device=args.device,
        hidden=args.hidden,
        lr=args.lr,
        batch_size=args.batch,
        n_critics=args.n_critics,
        n_quantiles=args.n_quantiles,
        top_quantiles_to_drop=args.drop,
        symmetry_augment=not args.no_symmetry,
        target_entropy=args.target_entropy,
        hold_phase_steps=args.hold_phase_steps,
        smooth_temporal=args.smooth_temporal,
        smooth_spatial=args.smooth_spatial,
        smooth_sigma=args.smooth_sigma,
        smooth_gate_power=args.smooth_gate,
        smooth_start_step=args.smooth_start,
        smooth_ramp_steps=args.smooth_ramp,
        total_steps=args.steps,
        checkpoint_dir=checkpoint_dir,
    )
    if args.config is not None:
        logger.info(
            f"Rig from {args.config}:\n{rig_summary(cfg.env.physics, args.links)}"
        )
    if cfg.env.randomize is not None:
        logger.info(f"Plant randomization: {cfg.env.randomize}")
    if cfg.env.reward_shape is not None:
        logger.info(f"Reward shaping: {cfg.env.reward_shape}")
    if args.init_from is not None:
        # The warm-started network must have the source's shape and physics.
        src = torch.load(args.init_from, map_location="cpu", weights_only=False)["cfg"]
        cfg = dataclasses.replace(
            cfg,
            hidden=src.hidden,
            n_critics=src.n_critics,
            n_quantiles=src.n_quantiles,
            env=dataclasses.replace(
                cfg.env, physics=src.env.physics, hardware=src.env.hardware
            ),
        )
        logger.info(
            f"Network/physics from {args.init_from}: hidden={src.hidden}, "
            f"{src.n_critics} critics x {src.n_quantiles} atoms"
        )
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if ckpt.get("algo") != "tqc":
            raise SystemExit(f"{args.resume} is not a TQC checkpoint")
        cfg = dataclasses.replace(
            ckpt["cfg"],
            device=args.device,
            total_steps=args.steps,
            checkpoint_dir=args.checkpoint_dir or args.resume.parent,
        )
    trainer = TQCTrainer(cfg)
    if args.resume is not None:
        trainer.load(args.resume)
    if args.init_from is not None:
        trainer.warm_start(args.init_from)
    seed_steps = args.seed_steps
    if seed_steps is None:
        seed_steps = 100_000 if args.init_from is not None else 0
    if seed_steps > 0:
        trainer.seed_buffer(seed_steps)
    logger.info(
        f"TQC: {cfg.n_critics} critics x {cfg.n_quantiles} atoms, "
        f"drop {cfg.top_quantiles_to_drop}, symmetry={cfg.symmetry_augment}"
    )
    trainer.train()


if __name__ == "__main__":
    main()
