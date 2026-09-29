"""Actor-Critic neural networks and observation normalization for PPO."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal

from n_cartpole.policy.simba import SimbaNet


class RunningNorm(nn.Module):
    """Online running mean/variance normalizer using Welford's algorithm."""

    mean: torch.Tensor
    var: torch.Tensor
    count: torch.Tensor

    def __init__(self, shape: int) -> None:
        """Initialize with zero mean and unit variance."""
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Update running statistics from a batch of observations."""
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_n = x.shape[0]

        total = self.count + batch_n
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * (batch_n / total)
        self.var = (
            self.var * self.count
            + batch_var * batch_n
            + delta**2 * self.count * batch_n / total
        ) / total
        self.count = total

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize x using running statistics."""
        return (x - self.mean) / (self.var.sqrt() + 1e-8)


def _mlp(in_dim: int, hidden: int, out_dim: int, n_blocks: int = 2) -> SimbaNet:
    return SimbaNet(in_dim, hidden, out_dim, n_blocks=n_blocks)


class Actor(nn.Module):
    """Gaussian policy network.

    Input:  normalized 8D observation
    Output: action mean (unbounded); log_std is a learnable scalar parameter
    """

    OBS_DIM = 8
    ACT_DIM = 1

    def __init__(
        self,
        hidden: int = 256,
        obs_dim: int | None = None,
        log_std_init: float = 0.0,
        blocks: int = 2,
    ) -> None:
        """Initialize Actor with given hidden size, obs width, and initial log_std.

        ``log_std_init`` sets the initial exploration scale in *force units*. It
        must be matched to ``force_max``: the default std of 1 N explores a tiny
        band next to a 20 N limit, far too little force to pump a pendulum up, so
        swing-up training needs a larger initial std (e.g. log_std_init≈1.6 → 5 N).
        Entropy annealing then narrows it as the policy converges.
        """
        super().__init__()
        self.obs_dim = obs_dim if obs_dim is not None else self.OBS_DIM
        self.net = _mlp(self.obs_dim, hidden, self.ACT_DIM, n_blocks=blocks)
        self.log_std = nn.Parameter(torch.full((self.ACT_DIM,), log_std_init))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, log_std) for the Gaussian policy."""
        mean = self.net(obs)
        log_std = self.log_std.expand_as(mean)
        return mean, log_std

    def get_dist(self, obs: torch.Tensor) -> Normal:
        """Return the Normal distribution over actions."""
        mean, log_std = self(obs)
        return Normal(mean, log_std.exp())

    def sample_action(
        self, obs: torch.Tensor, force_max: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample an action and its log probability.

        Returns:
            action: clipped to [-force_max, force_max]
            log_prob: log probability of the raw (unclipped) Gaussian sample

        """
        dist = self.get_dist(obs)
        raw = dist.rsample()
        log_prob = dist.log_prob(raw).sum(dim=-1)
        action = raw.clamp(-force_max, force_max)
        return action, log_prob

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute log_prob and entropy for given obs-action pairs.

        Used during the PPO update step.

        Returns:
            log_prob: (batch,) log prob of each action under current policy
            entropy:  (batch,) differential entropy of the distribution

        """
        dist = self.get_dist(obs)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class Critic(nn.Module):
    """State-value network.

    Input:  normalized 8D observation
    Output: scalar value estimate V(s)
    """

    OBS_DIM = 8

    def __init__(
        self, hidden: int = 256, obs_dim: int | None = None, blocks: int = 2
    ) -> None:
        """Initialize Critic with given hidden layer size and observation width."""
        super().__init__()
        self.obs_dim = obs_dim if obs_dim is not None else self.OBS_DIM
        self.net = _mlp(self.obs_dim, hidden, 1, n_blocks=blocks)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return value estimates with shape (batch,)."""
        return self.net(obs).squeeze(-1)
