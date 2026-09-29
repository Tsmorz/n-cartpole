"""PPO-clip update logic with GAE advantage estimation."""

from __future__ import annotations

from typing import TypedDict

import torch
from torch import nn

from n_cartpole.policy.actor_critic import Actor, Critic


class Batch(TypedDict):
    """One rollout batch assembled from parallel workers."""

    obs: torch.Tensor  # (N, 8)
    actions: torch.Tensor  # (N, 1)
    log_probs: torch.Tensor  # (N,)  — old policy
    advantages: torch.Tensor  # (N,)
    returns: torch.Tensor  # (N,)
    values: torch.Tensor  # (N,)  — old critic predictions


class PPOMetrics(TypedDict):
    """Scalar training metrics returned by ppo_update."""

    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    last_value: float,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation and discounted returns.

    Args:
        rewards:    (T,) reward at each step
        values:     (T,) critic value estimates V(s_t)
        last_value: V(s_{T+1}), the bootstrap value at episode end
        dones:      (T,) bool; True when episode terminated at this step
        gamma:      discount factor
        lam:        GAE smoothing parameter

    Returns:
        advantages: (T,) GAE estimates
        returns:    (T,) discounted returns for value regression target

    """
    T = len(rewards)
    advantages = torch.zeros(T, device=values.device)
    last_gae = 0.0

    for t in reversed(range(T)):
        next_value = values[t + 1].item() if t < T - 1 else last_value
        not_done = 1.0 - float(dones[t])

        delta = rewards[t].item() + gamma * next_value * not_done - values[t].item()
        last_gae = delta + gamma * lam * not_done * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def ppo_update(
    actor: Actor,
    critic: Critic,
    optimizer: torch.optim.Optimizer,
    batch: Batch,
    *,
    clip_eps: float = 0.2,
    value_clip_eps: float = 0.2,
    entropy_coeff: float = 0.01,
    n_epochs: int = 10,
    mini_batch_size: int = 256,
    max_grad_norm: float = 0.5,
) -> PPOMetrics:
    """Run K epochs of PPO-clip mini-batch updates.

    Both actor and critic share the same optimizer so parameter groups
    can be managed in one place (e.g., for LR scheduling).

    Returns aggregate scalar metrics averaged over all mini-batch steps.
    """
    obs = batch["obs"]
    actions = batch["actions"]
    old_log_probs = batch["log_probs"].detach()
    advantages = batch["advantages"].detach()
    returns = batch["returns"].detach()
    old_values = batch["values"].detach()

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    N = obs.shape[0]

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_kl = 0.0
    total_clip_frac = 0.0
    updates = 0

    for _ in range(n_epochs):
        perm = torch.randperm(N, device=obs.device)
        for start in range(0, N, mini_batch_size):
            idx = perm[start : start + mini_batch_size]
            mb_obs = obs[idx]
            mb_actions = actions[idx]
            mb_old_log_probs = old_log_probs[idx]
            mb_advantages = advantages[idx]
            mb_returns = returns[idx]
            mb_old_values = old_values[idx]

            # Actor forward
            new_log_probs, entropy = actor.evaluate_actions(mb_obs, mb_actions)

            # Policy loss (PPO-clip)
            ratio = (new_log_probs - mb_old_log_probs).exp()
            pg_unclipped = ratio * mb_advantages
            pg_clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * mb_advantages
            policy_loss = -torch.min(pg_unclipped, pg_clipped).mean()

            # Value loss (clipped)
            new_values = critic(mb_obs)
            v_clipped = mb_old_values + (new_values - mb_old_values).clamp(
                -value_clip_eps, value_clip_eps
            )
            value_loss = (
                0.5
                * torch.max(
                    (new_values - mb_returns).pow(2),
                    (v_clipped - mb_returns).pow(2),
                ).mean()
            )

            # Combined loss
            loss = policy_loss + value_loss - entropy_coeff * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(actor.parameters()) + list(critic.parameters()), max_grad_norm
            )
            optimizer.step()

            # Diagnostics
            with torch.no_grad():
                approx_kl = (mb_old_log_probs - new_log_probs).mean().item()
                clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean().item()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
            total_kl += approx_kl
            total_clip_frac += clip_frac
            updates += 1

    denom = max(1, updates)
    return PPOMetrics(
        policy_loss=total_policy_loss / denom,
        value_loss=total_value_loss / denom,
        entropy=total_entropy / denom,
        approx_kl=total_kl / denom,
        clip_fraction=total_clip_frac / denom,
    )
