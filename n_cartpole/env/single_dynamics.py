"""Single-link views onto the general ``n``-link dynamics (back-compat shim).

The single-link cartpole is just ``n = 1`` of :mod:`n_cartpole.env.dynamics`.
This module re-exports that dynamics and adds a scalar-``theta`` ``mass_matrix``
so older call sites (``mass_matrix(theta1, p)`` → 2x2) keep working. ``step`` and
the energy helpers already accept a 4D single-link state directly.
"""

from __future__ import annotations

import numpy as np

from n_cartpole.env.dynamics import (
    PhysicsParams,
    kinetic_energy,
    potential_energy,
    step,
    total_energy,
)
from n_cartpole.env.dynamics import mass_matrix as _mass_matrix

__all__ = [
    "PhysicsParams",
    "kinetic_energy",
    "mass_matrix",
    "potential_energy",
    "step",
    "total_energy",
]


def mass_matrix(theta1: float, p: PhysicsParams) -> np.ndarray:
    """Return the 2x2 positive-definite mass matrix for the single-link cartpole."""
    return _mass_matrix(np.array([theta1], dtype=float), p)
