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
    goal_dim,
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

    Includes the goal dimensions when ``config.goal_conditioned`` and the sysID
    context dimensions when ``config.hardware`` is set.
    """
    n = config.n_links
    with_sysid = config.hardware is not None and config.hardware.sysid_context
    # getattr: configs pickled in checkpoints before goal conditioning existed.
    with_goal = bool(getattr(config, "goal_conditioned", False))
    dim = (
        obs_dim(n)
        + (goal_dim(n) if with_goal else 0)
        + (sysid_dim(n) if with_sysid else 0)
    )
    sign = obs_mirror_sign(n, with_sysid=with_sysid, with_goal=with_goal)
    return dim, sign


def checkpoint_subdir(algo: str, goal_conditioned: bool) -> str:
    """Algorithm folder under ``checkpoints/<links>/``: ``tqc`` or ``tqc-goal``."""
    return f"{algo}-goal" if goal_conditioned else algo


def links_name(n_links: int) -> str:
    """Return the checkpoint-folder name for a link count (``single``/``double``/…)."""
    return _LINK_NAMES.get(n_links, f"{n_links}link")
