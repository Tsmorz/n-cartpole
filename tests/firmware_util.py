"""Shared helpers for the firmware tests: a tiny rig-config checkpoint to export."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import torch

from n_cartpole.config import load_tqc_config
from n_cartpole.env.cartpole import EnvConfig
from n_cartpole.training.off_policy import TQCConfig, TQCTrainer

REPO = Path(__file__).resolve().parents[1]
RIG_TOML = REPO / "config" / "rig.toml"


def make_checkpoint(
    path: Path, env: EnvConfig | None = None, seed: int = 0, step: int = 1234
) -> Path:
    """Save an untrained hidden-32 TQC checkpoint with a non-trivial normalizer.

    Default env: :func:`rig_env` — ``config/rig.toml`` (rigid-body links, rail
    Coulomb friction, back-EMF limit, 1-step delay, encoder model) with the EKF,
    goal conditioning, delta actions and 2 past commands. The output head is scaled up so the untrained actor
    commands forces across the whole range instead of ~0.
    """
    if env is None:
        env = rig_env()
    cfg = TQCConfig(
        env=env,
        device="cpu",
        hidden=32,
        batch_size=32,
        n_quantiles=8,
        buffer_size=1000,
        checkpoint_dir=path.parent,
    )
    torch.manual_seed(seed)
    trainer = TQCTrainer(cfg)
    dim = trainer.norm.mean.shape[0]
    with torch.no_grad():
        trainer.norm.mean.copy_(0.3 * torch.randn(dim))
        trainer.norm.var.copy_(0.2 + 2.0 * torch.rand(dim))
        trainer.actor.net.out_proj.weight.mul_(100.0)
    trainer.step = step
    trainer.save(path)
    return path


def rig_env(**overrides: Any) -> EnvConfig:
    """Return the rig env as the deployed policy trains on it, plus ``overrides``.

    Goal-conditioned, delta actions, 2 past commands, and the EKF estimator
    (``ekf_accel_std`` 0.5, as ``--estimator ekf --ekf-accel-std 0.5``).
    """
    rig = load_tqc_config(RIG_TOML).env
    hw = rig.hardware
    assert hw is not None and hw.sensors is not None
    sensors = dataclasses.replace(hw.sensors, estimator="ekf", ekf_accel_std=0.5)
    base = dataclasses.replace(
        rig,
        goal_conditioned=True,
        action_mode="delta",
        action_history=2,
        hardware=dataclasses.replace(hw, sensors=sensors),
    )
    return dataclasses.replace(base, **overrides)
