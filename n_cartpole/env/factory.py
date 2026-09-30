"""Factory helpers to build the ``n``-link cartpole from config.

Keeps the training/eval code link-count agnostic: it asks the factory for an
env, its observation width, and its mirror-symmetry sign vector, all derived
from ``EnvConfig``.
"""

from __future__ import annotations

import numpy as np

from n_cartpole.env.cartpole import (
    EnvConfig,
    NPendulumCartpole,
    obs_dim,
    obs_mirror_sign,
    sysid_dim,
)

# Human-readable directory/label names per link count; falls back to "<n>link".
_LINK_NAMES = {1: "single", 2: "double", 3: "triple", 4: "quadruple"}


def make_env(config: EnvConfig) -> NPendulumCartpole:
    """Return an ``n``-link environment matching ``config.n_links``."""
    return NPendulumCartpole(config)


def env_spec(config: EnvConfig) -> tuple[int, np.ndarray]:
    """Return ``(obs_dim, mirror_sign)`` for the given config.

    Includes the sysID context dimensions when ``config.hardware`` is set.
    """
    n = config.n_links
    with_sysid = config.hardware is not None
    dim = obs_dim(n) + (sysid_dim(n) if with_sysid else 0)
    sign = obs_mirror_sign(n, with_sysid=with_sysid)
    return dim, sign


def links_name(n_links: int) -> str:
    """Return the checkpoint-folder name for a link count (``single``/``double``/…)."""
    return _LINK_NAMES.get(n_links, f"{n_links}link")
