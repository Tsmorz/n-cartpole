"""Load training configuration from a TOML file.

``config/default.toml`` is the single source of truth for all physical
parameters, environment settings, and training hyperparameters.  The functions
here read that file and construct the typed dataclasses consumed by the
trainers and environment.

Usage::

    from n_cartpole.config import load_tqc_config

    tqc_cfg = load_tqc_config("config/default.toml")
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from n_cartpole.training.off_policy import TQCConfig

from n_cartpole.env.cartpole import EnvConfig
from n_cartpole.env.dynamics import PhysicsParams
from n_cartpole.env.hardware_config import HardwareConfig


def _physics_from(d: dict) -> PhysicsParams:
    return PhysicsParams(
        M=d.get("M", 1.0),
        g=d.get("g", 9.81),
        dt=d.get("dt", 0.01),
        force_max=d.get("force_max", 20.0),
        x_lim=d.get("x_lim", 0.5),
        b=d.get("b", 0.10),
        masses=d.get("masses", (0.20, 0.15)),
        lengths=d.get("lengths", 0.25),
        joint_friction=d.get("joint_friction", 0.002),
    )


def _env_from(data: dict) -> EnvConfig:
    phys = _physics_from(data.get("physics", {}))
    env_d = data.get("env", {})

    hw: HardwareConfig | None = None
    if "hardware" in data:
        hw_d = data["hardware"]
        hw_phys = _physics_from(hw_d.get("physics", data.get("physics", {})))
        noise = hw_d.get("noise", {})
        pipe = hw_d.get("pipeline", {})
        hw = HardwareConfig(
            physics=hw_phys,
            M_noise=noise.get("M_noise", 0.05),
            b_noise=noise.get("b_noise", 0.03),
            masses_noise=noise.get("masses_noise", 0.005),
            lengths_noise=noise.get("lengths_noise", 0.005),
            joint_friction_noise=noise.get("joint_friction_noise", 0.0005),
            delay_steps=pipe.get("delay_steps", 1),
            sensor_noise_std=pipe.get("sensor_noise_std", 0.0),
        )

    hold = env_d.get("goal_hold_steps", [400, 800])
    return EnvConfig(
        physics=phys,
        n_links=env_d.get("n_links", 2),
        max_steps=env_d.get("max_steps", 1000),
        init_noise=env_d.get("init_noise", 0.05),
        init_random_prob=env_d.get("init_random_prob", 0.3),
        init_vel_noise=env_d.get("init_vel_noise", 2.0),
        hardware=hw,
        goal_conditioned=env_d.get("goal_conditioned", False),
        goal_hold_steps=tuple(hold) if hold else None,
        goal_curriculum=env_d.get("goal_curriculum", True),
    )


def load_tqc_config(path: str | Path) -> TQCConfig:
    """Construct a :class:`~n_cartpole.training.off_policy.TQCConfig` from TOML."""
    from n_cartpole.training.off_policy import TQCConfig

    with open(path, "rb") as f:
        data = tomllib.load(f)

    env = _env_from(data)
    tqc = data.get("tqc", {})

    return TQCConfig(
        env=env,
        device=tqc.get("device", "auto"),
        hidden=tqc.get("hidden", 256),
        n_critics=tqc.get("n_critics", 2),
        n_quantiles=tqc.get("n_quantiles", 25),
        top_quantiles_to_drop=tqc.get("top_quantiles_to_drop", 2),
        gamma=tqc.get("gamma", 0.99),
        tau=tqc.get("tau", 0.005),
        lr=tqc.get("lr", 3e-4),
        batch_size=tqc.get("batch_size", 256),
        buffer_size=tqc.get("buffer_size", 1_000_000),
        start_steps=tqc.get("start_steps", 5_000),
        updates_per_step=tqc.get("updates_per_step", 1),
        symmetry_augment=tqc.get("symmetry_augment", True),
        total_steps=tqc.get("total_steps", 200_000),
        checkpoint_every=tqc.get("checkpoint_every", 25_000),
        log_every=tqc.get("log_every", 5_000),
    )
