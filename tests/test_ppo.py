"""Tests for the Actor-Critic networks and PPO update logic."""

from __future__ import annotations

import pytest
import torch

from n_cartpole.policy.actor_critic import Actor, Critic, RunningNorm
from n_cartpole.policy.ppo import Batch, compute_gae, ppo_update

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def actor() -> Actor:
    """Small Actor for fast tests."""
    return Actor(hidden=32)


@pytest.fixture()
def critic() -> Critic:
    """Small Critic for fast tests."""
    return Critic(hidden=32)


@pytest.fixture()
def batch_size() -> int:
    """Return the number of transitions in a synthetic batch."""
    return 64


@pytest.fixture()
def obs_batch(batch_size: int) -> torch.Tensor:
    """Random normalized observations."""
    torch.manual_seed(0)
    return torch.randn(batch_size, Actor.OBS_DIM)


@pytest.fixture()
def synthetic_batch(
    actor: Actor, critic: Critic, obs_batch: torch.Tensor, batch_size: int
) -> Batch:
    """Return a complete synthetic PPO batch with GAE-computed advantages."""
    with torch.no_grad():
        actions, log_probs = actor.sample_action(obs_batch, force_max=20.0)
        values = critic(obs_batch)

    rewards = torch.randn(batch_size) * 0.5 + 1.0
    dones = torch.zeros(batch_size)
    dones[-1] = 1.0

    advantages, returns = compute_gae(rewards, values.detach(), 0.0, dones)

    return Batch(
        obs=obs_batch,
        actions=actions.detach(),
        log_probs=log_probs.detach(),
        advantages=advantages,
        returns=returns,
        values=values.detach(),
    )


# ---------------------------------------------------------------------------
# Actor tests
# ---------------------------------------------------------------------------


def test_actor_output_shapes(actor: Actor, obs_batch: torch.Tensor) -> None:
    """Actor must output (batch, 1) mean and (batch, 1) log_std."""
    mean, log_std = actor(obs_batch)
    assert mean.shape == (len(obs_batch), 1)
    assert log_std.shape == (len(obs_batch), 1)


def test_actor_sample_action_shapes(actor: Actor, obs_batch: torch.Tensor) -> None:
    """sample_action must return clipped action and scalar log_prob per sample."""
    with torch.no_grad():
        action, log_prob = actor.sample_action(obs_batch, force_max=20.0)
    assert action.shape == (len(obs_batch), 1)
    assert log_prob.shape == (len(obs_batch),)


def test_actor_action_clipped(actor: Actor, obs_batch: torch.Tensor) -> None:
    """Sampled actions must be within [-force_max, force_max]."""
    force_max = 10.0
    with torch.no_grad():
        action, _ = actor.sample_action(obs_batch, force_max=force_max)
    assert action.abs().max().item() <= force_max + 1e-5


def test_critic_output_shape(critic: Critic, obs_batch: torch.Tensor) -> None:
    """Critic must return a scalar value per observation."""
    with torch.no_grad():
        values = critic(obs_batch)
    assert values.shape == (len(obs_batch),)


# ---------------------------------------------------------------------------
# RunningNorm tests
# ---------------------------------------------------------------------------


def test_running_norm_update_changes_stats() -> None:
    """Running mean/var must change after an update."""
    norm = RunningNorm(8)
    x = torch.randn(100, 8)
    norm.update(x)
    assert norm.count.item() == 100
    # Mean should approximate x.mean(0)
    torch.testing.assert_close(norm.mean, x.mean(0), atol=0.2, rtol=0.0)


def test_running_norm_normalize_zero_mean() -> None:
    """After updating with data, normalizing that data should yield ~0 mean."""
    norm = RunningNorm(4)
    x = torch.randn(500, 4) * 3 + 2
    norm.update(x)
    x_norm = norm.normalize(x)
    assert x_norm.mean(0).abs().max().item() < 0.1


# ---------------------------------------------------------------------------
# GAE tests
# ---------------------------------------------------------------------------


def test_gae_shapes() -> None:
    """compute_gae must return tensors of length T."""
    T = 16
    rewards = torch.ones(T)
    values = torch.zeros(T)
    dones = torch.zeros(T)
    dones[-1] = 1.0
    advantages, returns = compute_gae(rewards, values, 0.0, dones)
    assert advantages.shape == (T,)
    assert returns.shape == (T,)


def test_gae_no_discount_no_bootstrap() -> None:
    """With gamma=1, lam=1, no dones, advantages should equal sum of future rewards - value."""
    T = 5
    rewards = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
    values = torch.zeros(T)
    dones = torch.zeros(T)
    # last_value = 0
    advantages, returns = compute_gae(rewards, values, 0.0, dones, gamma=1.0, lam=1.0)
    # advantage[0] = 5, advantage[1] = 4, ...
    expected_advs = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    torch.testing.assert_close(advantages, expected_advs, atol=1e-5, rtol=0.0)


def test_gae_done_cuts_trajectory() -> None:
    """A 'done=1' at step t must prevent bootstrapping V(s_{t+1}) into step t's delta.

    Convention: dones[t]=1 means the episode ended after step t.
    GAE formula: delta_t = r_t + gamma*V_{t+1}*(1-done_t) - V_t
                 A_t     = delta_t + gamma*lam*(1-done_t)*A_{t+1}

    With gamma=lam=1, values=0, dones=[0,1,0,0], last_value=100:
      t=3: delta=1+100*1=101,  A[3]=101
      t=2: delta=1+0*1=1,      A[2]=1+1*101=102
      t=1: delta=1+0*(1-1)=1,  A[1]=1+1*(1-1)*102=1  ← done cuts recursion
      t=0: delta=1+0*1=1,      A[0]=1+1*1*1=2
    """
    T = 4
    rewards = torch.ones(T)
    values = torch.zeros(T)
    dones = torch.zeros(T)
    dones[1] = 1.0  # episode ends at t=1

    advantages, _ = compute_gae(rewards, values, 100.0, dones, gamma=1.0, lam=1.0)

    expected = torch.tensor([2.0, 1.0, 102.0, 101.0])
    torch.testing.assert_close(advantages, expected, atol=1e-5, rtol=0.0)


# ---------------------------------------------------------------------------
# PPO update tests
# ---------------------------------------------------------------------------


def test_ppo_update_returns_finite_losses(
    actor: Actor, critic: Critic, synthetic_batch: Batch
) -> None:
    """ppo_update must return finite scalar losses."""
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=1e-3
    )
    metrics = ppo_update(
        actor, critic, optimizer, synthetic_batch, n_epochs=2, mini_batch_size=32
    )
    for k, v in metrics.items():
        assert not (v != v), f"{k} is NaN"  # noqa: PLR0124
        assert abs(v) < 1e6, f"{k}={v} is unexpectedly large"  # type: ignore[arg-type]


def test_ppo_advantage_normalization(
    actor: Actor, critic: Critic, synthetic_batch: Batch
) -> None:
    """Advantages should be normalized to ~mean=0, std=1 inside ppo_update."""
    # This is internal, but we can verify by patching and checking the output
    # Indirectly: just verify losses stay sane after normalization
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=3e-4
    )
    metrics = ppo_update(
        actor, critic, optimizer, synthetic_batch, n_epochs=5, mini_batch_size=32
    )
    assert metrics["policy_loss"] < 5.0


def test_ppo_value_loss_decreases(
    actor: Actor, critic: Critic, obs_batch: torch.Tensor
) -> None:
    """Value loss should decrease over repeated updates on a fixed batch."""
    # Build a batch with fixed targets so the critic can fit
    targets = torch.ones(len(obs_batch)) * 2.0  # constant V = 2

    with torch.no_grad():
        _, log_probs = actor.sample_action(obs_batch, force_max=20.0)
        values_init = critic(obs_batch)

    batch = Batch(
        obs=obs_batch,
        actions=torch.zeros(len(obs_batch), 1),
        log_probs=log_probs.detach(),
        advantages=torch.zeros(len(obs_batch)),
        returns=targets,
        values=values_init.detach(),
    )

    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=1e-2
    )
    metrics_before = ppo_update(
        actor, critic, optimizer, batch, n_epochs=1, mini_batch_size=64
    )
    metrics_after = ppo_update(
        actor, critic, optimizer, batch, n_epochs=10, mini_batch_size=64
    )
    assert metrics_after["value_loss"] < metrics_before["value_loss"]
