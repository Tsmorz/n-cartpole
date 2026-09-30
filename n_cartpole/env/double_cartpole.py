"""Double pendulum cartpole (``n_links = 2``).

Thin wrapper over the general ``n``-link environment in
:mod:`n_cartpole.env.cartpole`. Kept as a named class, module constant, and
re-export of ``EnvConfig`` for back-compat with existing imports and
checkpoints — the behavior lives in ``NPendulumCartpole``.
"""

from __future__ import annotations

import numpy as np

from n_cartpole.env.cartpole import (
    EnvConfig,
    NPendulumCartpole,
    encode_obs,
    obs_dim,
    obs_mirror_sign,
)

__all__ = ["OBS_MIRROR_SIGN", "DoublePendulumCartpole", "EnvConfig"]

# Left-right mirror of the 8D observation (see ``cartpole.obs_mirror_sign``):
#   [x, ẋ, cosθ1, sinθ1, θ̇1, cosθ2, sinθ2, θ̇2]
OBS_MIRROR_SIGN = obs_mirror_sign(2)


def _encode_obs(state: np.ndarray) -> np.ndarray:
    """Encode a raw 6D state to the 8D cos/sin observation (back-compat alias)."""
    return encode_obs(state)


class DoublePendulumCartpole(NPendulumCartpole):
    """Double pendulum cartpole (``n_links = 2``); see :class:`NPendulumCartpole`."""

    OBS_DIM = obs_dim(2)
    N_LINKS = 2

    def __init__(self, config: EnvConfig | None = None) -> None:
        """Initialize with a 2-link config by default."""
        super().__init__(config or EnvConfig(n_links=2))
