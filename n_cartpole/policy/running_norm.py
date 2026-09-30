"""Online running mean/variance observation normalizer."""

from __future__ import annotations

import torch
from torch import nn


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
