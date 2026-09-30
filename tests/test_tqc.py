"""Tests for the TQC networks, losses, and off-policy update."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from n_cartpole.policy.tqc import (
    QuantileCritic,
    SquashedGaussianActor,
    quantile_huber_loss,
)
from n_cartpole.training.off_policy import ReplayBuffer, TQCConfig, TQCTrainer


def test_actor_action_bounded_and_logprob_shape() -> None:
    """Actor actions must respect force_max and log_prob must be per-sample."""
    actor = SquashedGaussianActor(hidden=32, force_max=20.0)
    obs = torch.randn(16, SquashedGaussianActor.OBS_DIM)
    action, log_prob = actor.sample(obs)
    assert action.shape == (16, 1)
    assert log_prob.shape == (16,)
    assert action.abs().max().item() <= 20.0 + 1e-4


def test_critic_atom_shape() -> None:
    """Critic must return (batch, n_critics, n_quantiles) atoms."""
    critic = QuantileCritic(hidden=32, n_critics=3, n_quantiles=10)
    obs = torch.randn(8, QuantileCritic.OBS_DIM)
    act = torch.randn(8, QuantileCritic.ACT_DIM)
    atoms = critic(obs, act)
    assert atoms.shape == (8, 3, 10)


def test_quantile_huber_loss_nonnegative_and_zero_at_match() -> None:
    """Loss is >= 0 and ~0 when predictions equal a single shared target."""
    taus = (torch.arange(5) + 0.5) / 5
    atoms = torch.full((4, 5), 2.0)
    target = torch.full((4, 1), 2.0)
    loss = quantile_huber_loss(atoms, target, taus)
    assert loss.item() >= 0.0
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_replay_buffer_add_and_sample() -> None:
    """Buffer stores transitions and samples correctly shaped batches."""
    buf = ReplayBuffer(capacity=100, obs_dim=8)
    for _ in range(50):
        buf.add(np.zeros(8), np.zeros(1), 0.5, np.ones(8), False)
    assert buf.size == 50
    obs, act, rew, next_obs, done = buf.sample(16)
    assert obs.shape == (16, 8)
    assert act.shape == (16, 1)
    assert rew.shape == (16, 1)


def test_tqc_update_returns_finite_metrics() -> None:
    """A single TQC update on a filled buffer yields finite losses."""
    cfg = TQCConfig(device="cpu", hidden=32, batch_size=32, n_quantiles=8)
    trainer = TQCTrainer(cfg)
    for _ in range(64):
        trainer.buffer.add(
            np.random.randn(8).astype(np.float32),
            np.random.randn(1).astype(np.float32),
            float(np.random.rand()),
            np.random.randn(8).astype(np.float32),
            False,
        )
    metrics = trainer._update()
    for k, v in metrics.items():
        assert math.isfinite(v), f"{k} is not finite: {v}"
