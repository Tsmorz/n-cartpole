"""Gymnasium environment for the ``n``-link pendulum cartpole swing-up task.

One parameterized environment covers every link count. ``EnvConfig.n_links``
selects the number of pendulums; the observation width, mirror-symmetry sign
vector, reward, and reset distribution all derive from it, so single/double/
triple/… differ by config alone. The historical ``SinglePendulumCartpole`` and
``DoublePendulumCartpole`` are thin subclasses kept for back-compat (see
``single_cartpole.py`` / ``double_cartpole.py``).

Angle convention: ``theta = 0`` is upright, ``theta = pi`` hangs straight down.
Swing-up starts from a small perturbation of the all-down state.

SysID context (optional):
    When ``EnvConfig.hardware`` is set, a ``sysid_dim(n_links)``-wide context
    vector ``[M, b, m_0, l_0, c_0, ...]`` is appended to every observation.
    The context is sampled once per episode from the hardware nominal params
    plus Gaussian measurement noise, so the policy learns to be robust to the
    gap between the bench measurement and the true hardware values.  Action
    latency and sensor noise are also modelled when hardware is configured.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from n_cartpole.env.dynamics import PhysicsParams, step

if TYPE_CHECKING:
    from n_cartpole.env.hardware_config import HardwareConfig


@dataclass
class EnvConfig:
    """Configuration for the ``n``-link cartpole environment."""

    physics: PhysicsParams = field(default_factory=PhysicsParams)
    # Number of pendulum links (1 = single, 2 = double, 3+ = higher order).
    # Selects the env/dynamics via ``n_cartpole.env.factory`` and the obs width.
    n_links: int = 2
    max_steps: int = 1000
    # Small perturbation around the hanging-down start (the swing-up task).
    init_noise: float = 0.05
    # "Recovery characteristics" (Lee et al.): with this probability, reset to a
    # fully random state instead of hanging-down, so the policy learns to reach
    # and hold upright from anywhere and to recover from disturbances. Requires
    # a simulator (a real rig can only be reset to hanging-down), which is why
    # this diverse exploration is done in sim.
    init_random_prob: float = 0.3
    init_vel_noise: float = 2.0  # rad/s (and m/s for the cart) for random resets

    # Actuator slew limit (N/s): the applied force can change by at most
    # ``force_slew * dt`` per step, like a real motor/drive. None = unlimited.
    # The default lets a full-scale reversal (2*force_max) take ~50 ms.
    force_slew: float | None = 800.0

    # Optional hardware configuration.  When set, the env appends a sysID
    # context vector to every observation and models action latency / sensor
    # noise.  Leave as None for pure simulation training without sysID.
    hardware: HardwareConfig | None = None


def obs_dim(n_links: int) -> int:
    """Return base kinematic observation width: ``[x, ẋ]`` + ``[cos, sin, θ̇]`` per link."""
    return 2 + 3 * n_links


def sysid_dim(n_links: int) -> int:
    """Width of the sysID context appended to the observation when hardware is set.

    Layout: ``[M, b, m_0, l_0, c_0, m_1, l_1, c_1, ...]``
    """
    return 2 + 3 * n_links


def obs_mirror_sign(n_links: int, with_sysid: bool = False) -> np.ndarray:
    """Left-right mirror sign vector for the observation of ``n_links``.

    Mirroring the rig flips ``x, ẋ, θ, θ̇`` and the applied force. In the cos/sin
    encoding that negates ``x, ẋ, sin θ`` (odd) and ``θ̇``, and keeps ``cos θ``
    (even) — per link.  SysID params (masses, lengths, frictions) are symmetric
    under mirroring, so their sign is +1.
    """
    base = np.array([-1, -1] + [1, -1, -1] * n_links, dtype=np.float32)
    if with_sysid:
        return np.concatenate([base, np.ones(sysid_dim(n_links), dtype=np.float32)])
    return base


def encode_obs(state: np.ndarray) -> np.ndarray:
    """Encode a raw ``2 + 2n`` state to the ``2 + 3n`` cos/sin observation.

    Input state: ``[x, x_dot, theta_0, theta_0_dot, ...]``
    Output obs:  ``[x, x_dot, cos θ_0, sin θ_0, θ̇_0, ...]``
    """
    state = np.asarray(state, dtype=float)
    cols: list[float] = [state[0], state[1]]
    theta = state[2::2]
    thd = state[3::2]
    for th, td in zip(theta, thd, strict=True):
        cols += [np.cos(th), np.sin(th), td]
    return np.array(cols, dtype=np.float32)


def _build_sysid_context(
    physics: PhysicsParams,
    n_links: int,
    hw: HardwareConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a noisy sysID context vector from hardware nominal params.

    Layout matches ``sysid_dim``: ``[M, b, m_0, l_0, c_0, ...]``.
    """
    masses = hw.physics.link_masses(n_links)
    lengths = hw.physics.link_lengths(n_links)
    frictions = hw.physics.joint_frictions(n_links)

    ctx: list[float] = [
        hw.physics.M + rng.normal(0.0, hw.M_noise),
        hw.physics.b + rng.normal(0.0, hw.b_noise),
    ]
    for i in range(n_links):
        ctx.append(float(masses[i]) + rng.normal(0.0, hw.masses_noise))
        ctx.append(float(lengths[i]) + rng.normal(0.0, hw.lengths_noise))
        ctx.append(float(frictions[i]) + rng.normal(0.0, hw.joint_friction_noise))
    return np.array(ctx, dtype=np.float32)


class NPendulumCartpole(gym.Env):
    """``n``-link pendulum cartpole swing-up environment.

    Observation space (``2 + 3n`` float32, or ``2(2 + 3n)`` with sysID):
        ``[x, x_dot, cos θ_0, sin θ_0, θ̇_0, ...,
           [M, b, m_0, l_0, c_0, ...]]``    ← sysID appended when hardware set

    Action space (1D float32):
        Horizontal force on the cart, clipped to ``[-force_max, force_max]``.

    Reward:
        Bounded, multiplicative shaping in (0, 1] (Lee et al.; see docs/):
            ``r = r_angle * r_pos * r_vel * r_act * r_rate``
        Each factor lies in [~0.5, 1] or [0, 1]; the product peaks at 1.0 only
        when every pole is upright, the cart is centered, velocities are low, and
        little force is used and the command changes smoothly (``r_rate``; the
        applied force is also slew-limited, see ``EnvConfig.force_slew``). The
        angle factor is a PRODUCT over all links, so partial credit for some-but-not-all links upright is suppressed; the
        velocity factor rewards actually balancing rather than spinning through
        upright. Nearly always positive and tightly bounded → well-scaled returns
        without value normalization.

        On top of that, a potential-based shaping term rewards *progress* toward
        upright every step, using ``r_angle`` as the potential Φ(s) (Ng, Harada &
        Russell, ICML 1999): ``F(s,s') = g*P(s') - P(s)``. This is provably
        policy-invariant but makes swing-up happen sooner during training. It can
        push the total slightly negative (to ``-SHAPING_WEIGHT``) when alignment
        regresses; the reward stays in a small fixed range regardless.

    Episode ends when:
        - terminated: ``|x| > x_lim`` (cart out of bounds)
        - truncated:  step count exceeds ``max_steps``
    """

    metadata: ClassVar[dict] = {"render_modes": ["rgb_array"]}  # type: ignore[misc]

    # Discount for the potential-based shaping term (Ng et al. 1999); matches the
    # PPO/TQC training discount so the telescoping shaping sum is consistent with
    # how the value function discounts.
    SHAPING_GAMMA: ClassVar[float] = 0.99
    # Weight on the shaping term relative to the base [0,1]-bounded reward; small
    # enough that the base multiplicative reward still dominates return scale.
    SHAPING_WEIGHT: ClassVar[float] = 0.2
    # Max fractional reward lost to command chatter (see ``r_rate``).
    RATE_PENALTY: ClassVar[float] = 0.2

    def __init__(self, config: EnvConfig | None = None) -> None:
        """Initialize the environment."""
        super().__init__()
        self.cfg = config or EnvConfig()
        n = self.cfg.n_links
        if n < 1:
            raise ValueError(f"n_links must be >= 1, got {n}")
        self.n_links = n
        self.N_LINKS = n
        hw = self.cfg.hardware

        # Observation width: base kinematic dims + sysID context when configured.
        self.OBS_DIM = obs_dim(n) + (sysid_dim(n) if hw is not None else 0)
        self.state_dim = 2 + 2 * n
        p = self.cfg.physics

        highs = [p.x_lim * 2, np.inf]
        for _ in range(n):
            highs += [1.0, 1.0, np.inf]
        if hw is not None:
            highs += [np.inf] * sysid_dim(n)
        obs_high = np.array(highs, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-p.force_max], dtype=np.float32),
            high=np.array([p.force_max], dtype=np.float32),
            dtype=np.float32,
        )

        self._state = np.zeros(self.state_dim)
        self._step_count = 0
        self._sysid_context: np.ndarray | None = None
        self._prev_cmd = 0.0  # last commanded force (for the rate penalty)
        self._prev_applied = 0.0  # last force applied to the plant (slew limit)

        # Action delay buffer: holds the last ``delay_steps`` commands; the oldest
        # is applied to the plant each step.  None when no delay is configured.
        if hw is not None and hw.delay_steps > 0:
            self._action_buf: collections.deque[float] | None = collections.deque(
                [0.0] * hw.delay_steps, maxlen=hw.delay_steps
            )
        else:
            self._action_buf = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset to hanging-down (swing-up) or, with ``init_random_prob``, anywhere.

        When hardware is configured, the sysID context is resampled from the
        hardware nominal params plus measurement noise once per episode.
        """
        super().reset(seed=seed)
        cfg = self.cfg
        n = self.n_links
        if self.np_random.random() < cfg.init_random_prob:
            # Fully random state: angles anywhere, modest random velocities.
            v = cfg.init_vel_noise
            xl = cfg.physics.x_lim
            state = np.empty(self.state_dim, dtype=np.float64)
            state[0] = self.np_random.uniform(-0.5 * xl, 0.5 * xl)
            state[1] = self.np_random.uniform(-v, v)
            state[2::2] = self.np_random.uniform(-np.pi, np.pi, n)
            state[3::2] = self.np_random.uniform(-v, v, n)
            self._state = state
        else:
            # Hanging-down equilibrium (theta = pi per link) with perturbation.
            base = np.zeros(self.state_dim, dtype=np.float64)
            base[2::2] = np.pi
            self._state = base + self.np_random.uniform(
                -cfg.init_noise, cfg.init_noise, self.state_dim
            )
        self._step_count = 0
        self._prev_cmd = 0.0
        self._prev_applied = 0.0
        self._prev_potential = self._angle_potential(self._state)

        hw = self.cfg.hardware
        if hw is not None:
            # Resample sysID context for this episode.
            self._sysid_context = _build_sysid_context(
                self.cfg.physics, n, hw, self.np_random
            )
            # Reset action delay buffer.
            if hw.delay_steps > 0:
                self._action_buf = collections.deque(
                    [0.0] * hw.delay_steps, maxlen=hw.delay_steps
                )

        return self._make_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply action and advance physics by one timestep.

        When hardware is configured, the commanded action enters the delay
        buffer and the oldest buffered command is applied to the plant.
        """
        p = self.cfg.physics
        raw_F = float(np.clip(action[0], -p.force_max, p.force_max))

        if self._action_buf is not None:
            applied_F = float(self._action_buf[0])
            self._action_buf.append(raw_F)
        else:
            applied_F = raw_F

        # Actuator slew limit: the plant can't follow an arbitrarily fast command.
        slew = self.cfg.force_slew
        if slew is not None:
            dF = slew * p.dt
            applied_F = self._prev_applied + float(
                np.clip(applied_F - self._prev_applied, -dF, dF)
            )
        self._prev_applied = applied_F

        self._state = step(self._state, applied_F, p)
        self._step_count += 1

        obs = self._make_obs()
        # The rate penalty acts on the *commanded* change so the policy gets a
        # gradient even where the slew limit would hide the jump from the plant.
        reward = self._compute_reward(self._state, applied_F, raw_F - self._prev_cmd)
        self._prev_cmd = raw_F

        terminated = bool(abs(self._state[0]) > p.x_lim)
        truncated = self._step_count >= self.cfg.max_steps

        return obs, reward, terminated, truncated, {}

    def _make_obs(self) -> np.ndarray:
        """Build the full observation: kinematic encoding + optional sysID context."""
        obs = encode_obs(self._state)
        hw = self.cfg.hardware
        if hw is not None and self._sysid_context is not None:
            obs = np.concatenate([obs, self._sysid_context])
        if hw is not None and hw.sensor_noise_std > 0.0:
            obs = obs + self.np_random.normal(
                0.0, hw.sensor_noise_std, obs.shape
            ).astype(np.float32)
        return obs

    def _angle_potential(self, state: np.ndarray) -> float:
        """Upright-alignment product over links, in [0, 1].

        Used as the shaping potential and the base reward's ``r_angle`` factor.
        """
        theta = np.asarray(state)[2::2]
        return float(np.prod(0.5 + 0.5 * np.cos(theta)))

    def _compute_reward(
        self, state: np.ndarray, F: float, dF_cmd: float = 0.0
    ) -> float:
        """Bounded multiplicative reward in (0, 1], plus potential-based shaping.

        Mirror-invariant by construction (every factor depends on x², cos θ, θ̇²,
        or a²), which keeps the symmetry augmentation used in training valid.
        """
        state = np.asarray(state)
        x = state[0]
        thd = state[3::2]
        p = self.cfg.physics

        # Upright alignment — product over links (all must be up to score high).
        r_angle = self._angle_potential(state)
        # Cart centered on the rail (x normalized by the rail half-length). Gated
        # by upright alignment like the velocity term: free to travel during
        # swing-up, but once balanced the cart is pulled back toward x=0. Bounded
        # to [0.5, 1]; the sharper falloff (k=4) penalizes even modest offsets.
        pos_pen = np.exp(-4.0 * (x / p.x_lim) ** 2)
        r_pos = 1.0 - 0.5 * r_angle * (1.0 - pos_pen)
        # Velocity penalty GATED by upright alignment: spinning is free during
        # swing-up (r_angle≈0) and penalized only near the top (r_angle≈1), so the
        # reward rewards *balancing* without fighting the energy pumping needed to
        # get every link up. Bounded to [0.5, 1].
        vel_pen = np.exp(-0.1 * float(np.sum(thd**2)))
        r_vel = 1.0 - 0.5 * r_angle * (1.0 - vel_pen)
        # Mild energy/effort term on the normalized force.
        a = F / p.force_max
        r_act = 0.8 + 0.2 * max(1.0 - a**2, 0.0)

        # Command-rate term: penalize changing the commanded force between steps
        # (normalized; a full reversal saturates it). Bounded to [0.8, 1], even.
        da = dF_cmd / p.force_max
        r_rate = 1.0 - self.RATE_PENALTY * min(da * da, 1.0)

        base = r_angle * r_pos * r_vel * r_act * r_rate

        # Potential-based shaping (Ng et al. 1999): F(s,s') = g*P(s') - P(s), with
        # Φ = r_angle. Rewards progress toward upright every step instead of only
        # at the top, without changing the optimal policy.
        shaping = self.SHAPING_GAMMA * r_angle - self._prev_potential
        self._prev_potential = r_angle

        return float(base + self.SHAPING_WEIGHT * shaping)

    def get_state(self) -> np.ndarray:
        """Return the current raw physics state (``2 + 2n``)."""
        return self._state.copy()
