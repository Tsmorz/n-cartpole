"""Gymnasium environment for the double pendulum cartpole swing-up task."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from n_cartpole.env.dynamics import PhysicsParams, step


@dataclass
class EnvConfig:
    """Configuration for the double cartpole environment."""

    physics: PhysicsParams = field(default_factory=PhysicsParams)
    max_steps: int = 1000
    init_noise: float = 0.05


def _encode_obs(state: np.ndarray) -> np.ndarray:
    """Encode raw state to 8D cos/sin observation.

    Input state: [x, x_dot, theta1, theta1_dot, theta2, theta2_dot]
    Output obs:  [x, x_dot, cos(theta1), sin(theta1), theta1_dot,
                             cos(theta2), sin(theta2), theta2_dot]
    """
    x, xd, th1, th1d, th2, th2d = state
    return np.array(
        [x, xd, np.cos(th1), np.sin(th1), th1d, np.cos(th2), np.sin(th2), th2d],
        dtype=np.float32,
    )


class DoublePendulumCartpole(gym.Env):
    """Double pendulum cartpole swing-up environment.

    Observation space (8D, float32):
        [x, x_dot, cos(theta1), sin(theta1), theta1_dot,
                   cos(theta2), sin(theta2), theta2_dot]

    Action space (1D, float32):
        Horizontal force on cart, clipped to [-force_max, force_max].

    Reward:
        cos(theta1) + cos(theta2) - 0.1*|x| - 0.001*F^2
        Range approximately [-2.0 - penalties, 2.0]
        Both poles upright maximises reward.

    Episode ends when:
        - terminated: |x| > x_lim (cart out of bounds)
        - truncated:  step count exceeds max_steps
    """

    metadata: ClassVar[dict] = {"render_modes": ["rgb_array"]}  # type: ignore[misc]

    def __init__(self, config: EnvConfig | None = None) -> None:
        """Initialize the environment."""
        super().__init__()
        self.cfg = config or EnvConfig()
        p = self.cfg.physics

        obs_high = np.array(
            [p.x_lim * 2, np.inf, 1.0, 1.0, np.inf, 1.0, 1.0, np.inf],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-p.force_max], dtype=np.float32),
            high=np.array([p.force_max], dtype=np.float32),
            dtype=np.float32,
        )

        self._state = np.zeros(6)
        self._step_count = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset to a small random perturbation from the hanging-down equilibrium."""
        super().reset(seed=seed)
        noise = self.cfg.init_noise
        # Hanging-down equilibrium: both poles pointing straight down (theta = pi)
        self._state = np.array(
            [0.0, 0.0, np.pi, 0.0, np.pi, 0.0], dtype=np.float64
        ) + self.np_random.uniform(-noise, noise, 6)
        self._step_count = 0
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

    def _compute_reward(self, state: np.ndarray, F: float) -> float:
        """Dense swing-up reward encouraging both poles upright near center."""
        th1, th2 = state[2], state[4]
        x = state[0]
        return float(np.cos(th1) + np.cos(th2) - 0.1 * abs(x) - 0.001 * F**2)

    def get_state(self) -> np.ndarray:
        """Return the current raw physics state (6D)."""
        return self._state.copy()
