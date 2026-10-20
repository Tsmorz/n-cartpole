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
        default=None,
        help="CAPS temporal smoothness weight on the actor's action change between "
        "consecutive states (0 = off); suppresses high-frequency force dither",
    )
    parser.add_argument(
        "--smooth-spatial",
        type=float,
        default=None,
        help="CAPS spatial smoothness weight: action change under a small "
        "observation perturbation, i.e. a cap on the actor's local gain (0 = off)",
    )
    parser.add_argument(
        "--smooth-gate",
        type=float,
        default=None,
        help="Gate the smoothness penalty by closeness to the goal, alignment**N "
        "(0 = ungated). Quiet only when settled; swing-up stays aggressive.",
    )
    parser.add_argument(
        "--smooth-start",
        type=int,
        default=None,
        help="Env step at which the smoothness penalty starts ramping in",
    )
    parser.add_argument(
        "--smooth-ramp",
        type=int,
        default=None,
        help="Steps over which the smoothness penalty ramps from 0 to full weight "
        "(0 = immediate once started)",
    )
    parser.add_argument(
        "--action-history",
        type=int,
        default=0,
        help="Append the last N commanded forces to the observation (the command "
        "in flight through the actuator delay/slew is not visible in the state)",
    )
    parser.add_argument(
        "--obs-history",
        type=int,
        default=0,
        help="Append the last N past measurement frames to the observation so the "
        "policy can learn its own velocity filtering (no hand-set lag)",
    )
    parser.add_argument(
        "--estimator",
        choices=("diff", "ekf"),
        default=None,
        help="Where the velocities in the observation come from: finite-differenced "
        "encoder readings (diff) or a model-based EKF (ekf; no filter lag). Overrides "
        "the config's [hardware.sensors]; with --resume it changes the resumed run.",
    )
    parser.add_argument(
        "--ekf-accel-std",
        type=float,
        default=None,
        help="EKF trust in its model: unmodelled link acceleration (rad/s^2). Smaller "
        "= smoother estimates, more reliance on the model (0.5 tested best zero-shot)",
    )
    parser.add_argument(
        "--ekf-force-std",
        type=float,
        default=None,
        help="EKF unmodelled cart force (N): rail friction, force-gain error",
    )
    parser.add_argument(
        "--resume-lr",
        type=float,
        default=None,
        help="With --resume, set the actor/critic learning rate (fine-tune gently)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25_000,
        help="Save a numbered checkpoint every N env steps (each is ~45 MB)",
    )
    parser.add_argument(
        "--actor-warmup-steps",
        type=int,
        default=0,
        help="Critic warm-up: freeze the actor (and temperature) for this many env "
        "steps after a start/resume while the critic fits the new data",
    )
    parser.add_argument(
        "--reset-buffer",
        action="store_true",
        help="With --resume, start with an empty replay buffer (its old transitions "
        "came from a different observation pipeline)",
    )
    parser.add_argument(
        "--force-slew",
        type=float,
        default=None,
        help="Override the actuator slew limit (N/s). With --resume it changes the "
        "limit of the resumed run (a diagnostic: the old replay data stays).",
    )
    parser.add_argument(
        "--realism",
        type=float,
        default=None,
        help="Start level in [0, 1] for rail friction, encoder noise/quantization and "
        "plant randomization (0 = clean plant, 1 = as configured). Overrides the "
        "config; with --resume it changes the level of the resumed run.",
    )
    parser.add_argument(
        "--realism-ramp-start",
        type=int,
        default=0,
        help="Env step at which realism starts ramping up to 1",
    )
    parser.add_argument(
        "--realism-ramp-steps",
        type=int,
        default=0,
        help="Steps over which realism ramps linearly to 1 (0 = no ramp)",
    )
    parser.add_argument(
        "--action-mode",
        choices=("absolute", "delta"),
        default=None,
        help="absolute: the action is the commanded force. delta: the action is the "
        "force CHANGE within the actuator rate limit (smooth by construction; "
        "needs --action-history >= 1)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Discount factor (default 0.99 per 10 ms, rescaled for the control "
        "period so the horizon in seconds is the same at 200 Hz)",
    )
    parser.add_argument(
        "--explore-noise",
        choices=("white", "pink"),
        default="white",
        help="Exploration noise color: white (independent per step) or pink "
        "(1/f, temporally correlated within an episode)",
    )
    parser.add_argument(
        "--explore-beta",
        type=float,
        default=1.0,
        help="Spectral exponent of --explore-noise pink (1 = pink, 2 = red)",
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


def _with_sensor_overrides(env: EnvConfig, args: argparse.Namespace) -> EnvConfig:
    """Apply --estimator / --ekf-* to ``env.hardware.sensors`` (no-op if none given)."""
    fields = {
        k: v
        for k, v in (
            ("estimator", args.estimator),
            ("ekf_accel_std", args.ekf_accel_std),
            ("ekf_force_std", args.ekf_force_std),
        )
        if v is not None
    }
    if not fields:
        return env
    if env.hardware is None or env.hardware.sensors is None:
        raise SystemExit(
            "--estimator/--ekf-* need a [hardware.sensors] model (use --config)"
        )
    sensors = dataclasses.replace(env.hardware.sensors, **fields)
    return dataclasses.replace(
        env, hardware=dataclasses.replace(env.hardware, sensors=sensors)
    )


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
    if args.resume is None:  # on resume the overrides apply to the checkpoint's env
        base_env = _with_sensor_overrides(base_env, args)
    if args.realism is not None:
        base_env = dataclasses.replace(base_env, realism=args.realism)
    if args.force_slew is not None:
        base_env = dataclasses.replace(base_env, force_slew=args.force_slew)
    if args.action_mode is not None:
        base_env = dataclasses.replace(base_env, action_mode=args.action_mode)
    if args.action_history or args.obs_history:
        base_env = dataclasses.replace(
            base_env,
            action_history=args.action_history,
            obs_history=args.obs_history,
        )
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
            # 10 s episodes (30 s with goals), whatever the control rate
            max_steps=args.max_steps
            or round((30.0 if args.goals else 10.0) / base_env.physics.dt),
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
        smooth_temporal=args.smooth_temporal or 0.0,
        smooth_spatial=args.smooth_spatial or 0.0,
        smooth_sigma=args.smooth_sigma,
        realism_ramp_start=args.realism_ramp_start,
        realism_ramp_steps=args.realism_ramp_steps,
        actor_warmup_steps=args.actor_warmup_steps,
        checkpoint_every=args.checkpoint_every,
        gamma=args.gamma
        if args.gamma is not None
        else 0.99 ** (base_env.physics.dt / 0.01),
        explore_noise=args.explore_noise,
        explore_beta=args.explore_beta,
        smooth_gate_power=args.smooth_gate or 0.0,
        smooth_start_step=args.smooth_start or 0,
        smooth_ramp_steps=args.smooth_ramp or 0,
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
        if args.realism is not None:  # fine-tune the same policy at a new level
            cfg = dataclasses.replace(
                cfg, env=dataclasses.replace(cfg.env, realism=args.realism)
            )
        if args.force_slew is not None:
            cfg = dataclasses.replace(
                cfg, env=dataclasses.replace(cfg.env, force_slew=args.force_slew)
            )
        cfg = dataclasses.replace(cfg, env=_with_sensor_overrides(cfg.env, args))
        overrides = {
            k: v
            for k, v in (
                ("smooth_temporal", args.smooth_temporal),
                ("smooth_spatial", args.smooth_spatial),
                ("smooth_gate_power", args.smooth_gate),
                ("smooth_start_step", args.smooth_start),
                ("smooth_ramp_steps", args.smooth_ramp),
            )
            if v is not None
        }
        if overrides:  # e.g. --smooth-temporal 0 --smooth-spatial 0 turns it off
            cfg = dataclasses.replace(cfg, **overrides)
        if args.actor_warmup_steps:
            cfg = dataclasses.replace(cfg, actor_warmup_steps=args.actor_warmup_steps)
        if args.realism_ramp_steps:
            cfg = dataclasses.replace(
                cfg,
                realism_ramp_start=args.realism_ramp_start,
                realism_ramp_steps=args.realism_ramp_steps,
            )
    trainer = TQCTrainer(cfg)
    if args.resume is not None:
        trainer.load(args.resume)
        if args.reset_buffer:
            trainer.buffer.clear()
            logger.info("Replay buffer cleared; learning resumes once it refills")
        if args.resume_lr is not None:
            for opt in (trainer.actor_opt, trainer.critic_opt):
                for group in opt.param_groups:
                    group["lr"] = args.resume_lr
            logger.info(f"Learning rate set to {args.resume_lr:g}")
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
