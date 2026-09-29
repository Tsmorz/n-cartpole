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
    # Small perturbation around the hanging-down start (the swing-up task).
    init_noise: float = 0.05
    # "Recovery characteristics" (Lee et al.): with this probability, reset to a
    # fully random state instead of hanging-down, so the policy learns to reach
    # and hold upright from anywhere and to recover from disturbances. Requires
    # a simulator (a real rig can only be reset to hanging-down), which is exactly
    # why this diverse exploration is done in sim.
    init_random_prob: float = 0.3
    init_vel_noise: float = 2.0  # rad/s (and m/s for the cart) for random resets


# Left-right mirror of the 8D observation. Mirroring the whole rig flips
# x, ẋ, θ, θ̇ and the applied force. In the cos/sin encoding that means:
#   [x, ẋ, cosθ1, sinθ1, θ̇1, cosθ2, sinθ2, θ̇2]
#   → negate x, ẋ, sinθ (odd), θ̇; keep cosθ (even). Force also negates.
OBS_MIRROR_SIGN = np.array([-1, -1, 1, -1, -1, 1, -1, -1], dtype=np.float32)


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
        Bounded, multiplicative shaping in (0, 1] (Lee et al.; see docs/):
            r = r_angle * r_pos * r_vel * r_act
        Each factor lies in [~0.5, 1] or [0, 1]; the product peaks at 1.0 only
        when both poles are upright, the cart is centered, velocities are low,
        and little force is used. The angle factor is a PRODUCT over both links,
        so partial credit for a single upright pole is suppressed; the velocity
        factor rewards actually balancing rather than spinning through upright.
        Always positive → well-scaled returns (no value normalization needed).

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
        """Reset to hanging-down (swing-up) or, with ``init_random_prob``, anywhere.

        Mixing in fully random starts teaches the policy to reach upright from any
        configuration and to recover from disturbances, not just to swing up from
        rest.
        """
        super().reset(seed=seed)
        cfg = self.cfg
        if self.np_random.random() < cfg.init_random_prob:
            # Fully random state: angles anywhere, modest random velocities.
            v = cfg.init_vel_noise
            xl = cfg.physics.x_lim
            self._state = np.array(
                [
                    self.np_random.uniform(-0.5 * xl, 0.5 * xl),
                    self.np_random.uniform(-v, v),
                    self.np_random.uniform(-np.pi, np.pi),
                    self.np_random.uniform(-v, v),
                    self.np_random.uniform(-np.pi, np.pi),
                    self.np_random.uniform(-v, v),
                ],
                dtype=np.float64,
            )
        else:
            # Hanging-down equilibrium (theta = pi) with a small perturbation.
            noise = cfg.init_noise
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
        """Bounded multiplicative reward in (0, 1].

        Mirror-invariant by construction (every factor depends on x², cos θ, θ̇²,
        or a²), which keeps the symmetry augmentation used in training valid.
        """
        x, th1, th1d, th2, th2d = state[0], state[2], state[3], state[4], state[5]
        p = self.cfg.physics

        # Upright alignment — product over links (both must be up to score high).
        r_angle = (0.5 + 0.5 * np.cos(th1)) * (0.5 + 0.5 * np.cos(th2))
        # Cart centered on the rail (x normalized by the rail half-length).
        r_pos = 0.5 + 0.5 * np.exp(-0.7 * (x / p.x_lim) ** 2)
        # Low angular velocity — encourages balancing, not spinning through upright.
        r_vel = (0.5 + 0.5 * np.exp(-0.1 * th1d**2)) * (
            0.5 + 0.5 * np.exp(-0.1 * th2d**2)
        )
        # Mild energy/effort term on the normalized force.
        a = F / p.force_max
        r_act = 0.8 + 0.2 * max(1.0 - a**2, 0.0)

        return float(r_angle * r_pos * r_vel * r_act)

    def get_state(self) -> np.ndarray:
        """Return the current raw physics state (6D)."""
        return self._state.copy()
