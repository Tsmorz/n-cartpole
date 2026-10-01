"""Truncated Quantile Critics (TQC): off-policy distributional actor-critic.

Used by Lee et al. for real multi-pendulum swing-up (see docs/).

TQC = SAC with a distributional critic. Each critic outputs a set of quantile
"atoms" of the return distribution instead of a scalar Q; the target pools all
atoms across critics and drops the largest few (truncation) to counter the
overestimation bias that plagues value learning in high-dimensional control.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal

from n_cartpole.policy.simba import SimbaNet

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def _mlp(in_dim: int, hidden: int, out_dim: int, n_blocks: int = 2) -> SimbaNet:
    return SimbaNet(in_dim, hidden, out_dim, n_blocks=n_blocks)


class SquashedGaussianActor(nn.Module):
    """SAC-style stochastic policy: a tanh-squashed Gaussian bounded to force_max."""

    OBS_DIM = 8
    ACT_DIM = 1

    def __init__(
        self,
        hidden: int = 256,
        force_max: float = 20.0,
        blocks: int = 2,
        obs_dim: int | None = None,
    ) -> None:
        """Build the policy network for the given action bound."""
        super().__init__()
        self.obs_dim = obs_dim if obs_dim is not None else self.OBS_DIM
        self.net = _mlp(self.obs_dim, hidden, 2 * self.ACT_DIM, n_blocks=blocks)
        self.force_max = float(force_max)

    def _mean_logstd(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.net(obs).chunk(2, dim=-1)
        return mean, log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (action, log_prob) with the tanh change-of-variables correction."""
        mean, log_std = self._mean_logstd(obs)
        dist = Normal(mean, log_std.exp())
        u = dist.rsample()
        tanh_u = torch.tanh(u)
        action = tanh_u * self.force_max
        # log_prob(action) = log_prob(u) - sum log|d(action)/du|
        log_prob = dist.log_prob(u).sum(-1)
        log_prob = log_prob - torch.log(
            self.force_max * (1.0 - tanh_u.pow(2)) + 1e-6
        ).sum(-1)
        return action, log_prob

    def mean_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action scaled to [-1, 1], with gradients (for smoothness)."""
        mean, _ = self._mean_logstd(obs)
        return torch.tanh(mean)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, *, deterministic: bool = False) -> torch.Tensor:
        """Return an action for interaction (mean action when deterministic)."""
        mean, log_std = self._mean_logstd(obs)
        if deterministic:
            return torch.tanh(mean) * self.force_max
        u = Normal(mean, log_std.exp()).sample()
        return torch.tanh(u) * self.force_max


class QuantileCritic(nn.Module):
    """N independent critics, each mapping (obs, action) to n_quantiles atoms."""

    OBS_DIM = 8
    ACT_DIM = 1

    def __init__(
        self,
        hidden: int = 256,
        n_critics: int = 2,
        n_quantiles: int = 25,
        blocks: int = 2,
        obs_dim: int | None = None,
    ) -> None:
        """Build ``n_critics`` quantile heads."""
        super().__init__()
        self.obs_dim = obs_dim if obs_dim is not None else self.OBS_DIM
        self.n_critics = n_critics
        self.n_quantiles = n_quantiles
        self.nets = nn.ModuleList(
            _mlp(self.obs_dim + self.ACT_DIM, hidden, n_quantiles, n_blocks=blocks)
            for _ in range(n_critics)
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return atoms of shape (batch, n_critics, n_quantiles)."""
        x = torch.cat([obs, action], dim=-1)
        return torch.stack([net(x) for net in self.nets], dim=1)


def quantile_huber_loss(
    atoms: torch.Tensor, target: torch.Tensor, taus: torch.Tensor
) -> torch.Tensor:
    """Quantile Huber (κ=1) regression loss.

    Args:
        atoms:  (B, n) predicted quantiles at fractions ``taus``.
        target: (B, m) target atoms (treated as samples of the target dist).
        taus:   (n,) quantile fractions for the predicted atoms.

    Returns:
        Scalar mean loss over the (n, m) pairwise pinball terms.

    """
    # pairwise TD errors: (B, n, m)
    delta = target.unsqueeze(1) - atoms.unsqueeze(2)
    abs_delta = delta.abs()
    huber = torch.where(abs_delta <= 1.0, 0.5 * delta.pow(2), abs_delta - 0.5)
    taus = taus.view(1, -1, 1)
    loss = (taus - (delta < 0).float()).abs() * huber
    return loss.mean()
