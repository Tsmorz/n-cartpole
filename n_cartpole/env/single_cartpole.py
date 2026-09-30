"""Single pendulum cartpole (``n_links = 1``) — the swing-up warm-up.

Thin wrapper over the general ``n``-link environment in
:mod:`n_cartpole.env.cartpole`. Kept as a named class and module constant for
back-compat; the behavior lives in ``NPendulumCartpole``.
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

__all__ = ["OBS_MIRROR_SIGN", "SinglePendulumCartpole"]

# Left-right mirror of the 5D observation (see ``cartpole.obs_mirror_sign``):
#   [x, ẋ, cosθ1, sinθ1, θ̇1]
OBS_MIRROR_SIGN = obs_mirror_sign(1)


def _encode_obs(state: np.ndarray) -> np.ndarray:
    """Encode a raw 4D state to the 5D cos/sin observation (back-compat alias)."""
    return encode_obs(state)


class SinglePendulumCartpole(NPendulumCartpole):
    """Single pendulum cartpole (``n_links = 1``); see :class:`NPendulumCartpole`."""

    OBS_DIM = obs_dim(1)
    N_LINKS = 1

    def __init__(self, config: EnvConfig | None = None) -> None:
        """Initialize with a 1-link config by default."""
        super().__init__(config or EnvConfig(n_links=1))
