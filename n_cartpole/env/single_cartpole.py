"""Gymnasium environment for the single pendulum cartpole swing-up task.

This is the ``n_links == 1`` warm-up for the two-link task in
``double_cartpole.py``. It shares ``EnvConfig``, the bounded multiplicative
reward style, the diverse initial-state distribution, and the mirror-symmetry
convention, so training code can switch between one and two links by config
alone.
"""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from n_cartpole.env.double_cartpole import EnvConfig
from n_cartpole.env.single_dynamics import step

# Left-right mirror of the 5D observation:
#   [x, ẋ, cosθ1, sinθ1, θ̇1]
#   → negate x, ẋ, sinθ (odd), θ̇; keep cosθ (even). Force also negates.
OBS_MIRROR_SIGN = np.array([-1, -1, 1, -1, -1], dtype=np.float32)


def _encode_obs(state: np.ndarray) -> np.ndarray:
    """Encode raw 4D state to 5D cos/sin observation.

    Input state: [x, x_dot, theta1, theta1_dot]
    Output obs:  [x, x_dot, cos(theta1), sin(theta1), theta1_dot]
    """
    x, xd, th1, th1d = state
    return np.array([x, xd, np.cos(th1), np.sin(th1), th1d], dtype=np.float32)


class SinglePendulumCartpole(gym.Env):
    """Single pendulum cartpole swing-up environment.

    Observation space (5D, float32):
        [x, x_dot, cos(theta1), sin(theta1), theta1_dot]

    Action space (1D, float32):
        Horizontal force on cart, clipped to [-force_max, force_max].

    Reward:
        Bounded, multiplicative shaping in (0, 1] (same style as the two-link
        env): r = r_angle * r_pos * r_vel * r_act, each factor in [~0.5, 1] or
        [0, 1]; the product peaks at 1.0 only when the pole is upright, the cart
        is centered, angular velocity is low, and little force is used. On top
        of that, a potential-based shaping term (Ng, Harada & Russell, ICML
        1999) rewards progress toward upright every step — see double_cartpole.py
        for the full rationale; same Φ = r_angle, same γ and weight.

    Episode ends when:
        - terminated: |x| > x_lim (cart out of bounds)
        - truncated:  step count exceeds max_steps
    """

    metadata: ClassVar[dict] = {"render_modes": ["rgb_array"]}  # type: ignore[misc]

    OBS_DIM: ClassVar[int] = 5
    N_LINKS: ClassVar[int] = 1

    SHAPING_GAMMA: ClassVar[float] = 0.99
    SHAPING_WEIGHT: ClassVar[float] = 0.2

    def __init__(self, config: EnvConfig | None = None) -> None:
        """Initialize the environment."""
        super().__init__()
        self.cfg = config or EnvConfig(n_links=1)
        p = self.cfg.physics

        obs_high = np.array(
            [p.x_lim * 2, np.inf, 1.0, 1.0, np.inf],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-p.force_max], dtype=np.float32),
            high=np.array([p.force_max], dtype=np.float32),
            dtype=np.float32,
        )

        self._state = np.zeros(4)
        self._step_count = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset to hanging-down (swing-up) or, with ``init_random_prob``, anywhere."""
        super().reset(seed=seed)
        cfg = self.cfg
        if self.np_random.random() < cfg.init_random_prob:
            v = cfg.init_vel_noise
            xl = cfg.physics.x_lim
            self._state = np.array(
                [
                    self.np_random.uniform(-0.5 * xl, 0.5 * xl),
                    self.np_random.uniform(-v, v),
                    self.np_random.uniform(-np.pi, np.pi),
                    self.np_random.uniform(-v, v),
                ],
                dtype=np.float64,
            )
        else:
            noise = cfg.init_noise
            self._state = np.array(
                [0.0, 0.0, np.pi, 0.0], dtype=np.float64
            ) + self.np_random.uniform(-noise, noise, 4)
        self._step_count = 0
        self._prev_potential = self._angle_potential(self._state)
        return _encode_obs(self._state), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply action and advance physics by one timestep."""
        p = self.cfg.physics
        F = float(np.clip(action[0], -p.force_max, p.force_max))

        self._state = step(self._state, F, p)
        self._step_count += 1

        obs = _encode_obs(self._state)
        reward = self._compute_reward(self._state, F)

        terminated = bool(abs(self._state[0]) > p.x_lim)
        truncated = self._step_count >= self.cfg.max_steps

        return obs, reward, terminated, truncated, {}

    def _angle_potential(self, state: np.ndarray) -> float:
        """Upright-alignment term, in [0, 1]; used as the shaping potential Φ(s)
        as well as the base reward's r_angle factor.
        """
        th1 = state[2]
        return float(0.5 + 0.5 * np.cos(th1))

    def _compute_reward(self, state: np.ndarray, F: float) -> float:
        """Bounded multiplicative reward in (0, 1] (mirror-invariant), plus
        potential-based shaping (see class docstring / double_cartpole.py).
        """
        x, th1d = state[0], state[3]
        p = self.cfg.physics

        r_angle = self._angle_potential(state)
        r_pos = 0.5 + 0.5 * np.exp(-0.7 * (x / p.x_lim) ** 2)
        # Velocity penalty GATED by upright alignment: spinning is free during
        # swing-up (align≈0) and penalized only near the top (align≈1), so the
        # reward rewards *balancing* without fighting the energy pumping needed
        # to get there. Bounded to [0.5, 1].
        vel_pen = np.exp(-0.1 * th1d**2)
        r_vel = 1.0 - 0.5 * r_angle * (1.0 - vel_pen)
        a = F / p.force_max
        r_act = 0.8 + 0.2 * max(1.0 - a**2, 0.0)

        base = r_angle * r_pos * r_vel * r_act

        shaping = self.SHAPING_GAMMA * r_angle - self._prev_potential
        self._prev_potential = r_angle

        return float(base + self.SHAPING_WEIGHT * shaping)

    def get_state(self) -> np.ndarray:
        """Return the current raw physics state (4D)."""
        return self._state.copy()
