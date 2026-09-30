"""SimBa-style residual MLP backbone.

Ha et al., "SimBa: Simplicity Bias for Scaling Up Parameters in Deep
Reinforcement Learning" (arXiv:2410.09754). A pre-LayerNorm residual MLP that
lets actor/critic networks scale up in width/depth without the optimization
instability plain MLPs show at larger sizes. Drop-in replacement for a plain
``Linear -> activation -> Linear`` stack; used by both PPO (actor_critic.py)
and TQC (tqc.py).
"""

from __future__ import annotations

import torch
from torch import nn


class _ResidualBlock(nn.Module):
    """Pre-LayerNorm residual block: LN -> Linear(h, 4h) -> ReLU -> Linear(4h, h)."""

    def __init__(self, hidden: int, expand: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, hidden * expand)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden * expand, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.act(self.fc1(h))
        h = self.fc2(h)
        return x + h


class SimbaNet(nn.Module):
    """Input projection + N residual blocks + final LayerNorm + output head.

    Output head is initialized with a small gain so the network starts close
    to zero output (matches the paper and keeps early-training behavior
    similar to a freshly-initialized plain MLP).
    """

    def __init__(
        self, in_dim: int, hidden: int, out_dim: int, n_blocks: int = 2
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([_ResidualBlock(hidden) for _ in range(n_blocks)])
        self.final_norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, out_dim)
        nn.init.orthogonal_(self.out_proj.weight, gain=0.01)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the network output for input `x`."""
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h)
        h = self.final_norm(h)
        return self.out_proj(h)
