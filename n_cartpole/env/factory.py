"""Factory helpers to build the single- or double-link cartpole from config.

Keeps the training/eval code link-count agnostic: it asks the factory for an
env, its observation width, and its mirror-symmetry sign vector, all derived
from ``EnvConfig.n_links``.
"""

from __future__ import annotations

import numpy as np

from n_cartpole.env.double_cartpole import OBS_MIRROR_SIGN as DOUBLE_MIRROR
from n_cartpole.env.double_cartpole import DoublePendulumCartpole, EnvConfig
from n_cartpole.env.single_cartpole import OBS_MIRROR_SIGN as SINGLE_MIRROR
from n_cartpole.env.single_cartpole import SinglePendulumCartpole


def make_env(
    config: EnvConfig,
) -> DoublePendulumCartpole | SinglePendulumCartpole:
    """Return the env matching ``config.n_links`` (1 → single, else double)."""
    if config.n_links == 1:
        return SinglePendulumCartpole(config)
    return DoublePendulumCartpole(config)


def env_spec(n_links: int) -> tuple[int, np.ndarray]:
    """Return ``(obs_dim, mirror_sign)`` for the given link count."""
    if n_links == 1:
        return SinglePendulumCartpole.OBS_DIM, SINGLE_MIRROR
    return DoublePendulumCartpole.OBS_DIM, DOUBLE_MIRROR
