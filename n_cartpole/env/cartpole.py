"""Gymnasium environment for the ``n``-link pendulum cartpole swing-up task.

One parameterized environment covers every link count. ``EnvConfig.n_links``
selects the number of pendulums; the observation width, mirror-symmetry sign
vector, reward, and reset distribution all derive from it, so single/double/
triple/… differ by config alone. The historical ``SinglePendulumCartpole`` and
``DoublePendulumCartpole`` are thin subclasses kept for back-compat (see
``single_cartpole.py`` / ``double_cartpole.py``).

Angle convention: ``theta = 0`` is upright, ``theta = pi`` hangs straight down.
Swing-up starts from a small perturbation of the all-down state.

Goal-conditioned mode (``EnvConfig.goal_conditioned``):
    Instead of always driving every link upright, the policy is told a target
    configuration (each link up or down, see ``goals.py``) through ``n`` extra
    observation dims (``cos`` of each link's target angle, i.e. ±1). The reward
    measures alignment to that target, episodes start at any equilibrium, and the
    goal is re-drawn every ``goal_hold_steps`` steps, so one network learns every
    transition (UU↔DU↔UD↔DD for the double link) and to switch on command.
    Use :meth:`NPendulumCartpole.set_goal` to command a goal at any time.

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

from n_cartpole.env.dynamics import (
    PhysicsParams,
    available_force,
    energy_range,
    goal_energy,
    step,
    total_energy_batch,
)
from n_cartpole.env.goals import (
    FROM_OTHER,
    TransitionStats,
    goal_configs,
    goal_index,
    goal_labels,
)

if TYPE_CHECKING:
    from n_cartpole.env.hardware_config import HardwareConfig
    from n_cartpole.env.randomization import PlantRandomization


@dataclass
class RewardShape:
    """Optional reward terms for fast, steady, in-bounds goal transitions.

    ``EnvConfig.reward_shape = None`` (the default) is the original reward exactly;
    every term here is off at its zero weight. All terms are even in ``x`` and the
    velocities, so left-right mirror symmetry is preserved.

    - ``energy_weight``: potential-based shaping on the gap between the rig's
      mechanical energy and the goal pose's energy, faded out by ``1 - r_angle``.
      The angle potential is flat (zero value *and* slope) at hanging-down for any
      goal with an upright link; the energy gap is not, so pumping energy pays from
      the first swing. Being potential-based it does not change the optimal policy.
    - ``wall_weight`` / ``wall_start``: a factor on the reward that falls off
      quadratically once ``|x|`` passes ``wall_start`` of the rail half-length, so
      transitions stay clear of the end-stops (termination alone is too late).
    - ``effort_weight`` / ``effort_scale``: a hold-gated force penalty
      ``1 - w r_angle (1 - exp(-(F/scale)^2))`` — free while transitioning, costly
      once at the goal, so holds are quiet.
    - ``vel_k``: coefficient of the hold-gated joint-speed penalty
      ``exp(-vel_k sum thd^2)`` (the original reward uses 0.1).
    """

    energy_weight: float = 0.0
    wall_weight: float = 0.0
    wall_start: float = 0.7
    effort_weight: float = 0.0
    effort_scale: float = 2.0
    vel_k: float = 0.1


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
    # Optional per-episode randomization of the TRUE plant (masses, lengths,
    # friction, actuator gain) around ``physics``; None = always nominal. Rewards,
    # shaping and replay relabeling keep using the nominal ``physics``.
    randomize: PlantRandomization | None = None
    # Optional extra reward terms (see ``RewardShape``); None = original reward.
    reward_shape: RewardShape | None = None

    # Goal-conditioned transitions between equilibria (see module docstring).
    # False keeps the original all-upright swing-up task and observation.
    goal_conditioned: bool = False
    # Each commanded goal is held for a uniform random number of steps in
    # [lo, hi] before a new one is drawn; None = never switch automatically (the
    # caller commands goals via ``set_goal``). 400-800 steps = 4-8 s at 100 Hz.
    goal_hold_steps: tuple[int, int] | None = (400, 800)
    # Oversample (from, to) transitions with a low running success rate.
    goal_curriculum: bool = True
    # "At goal": every link within this angle (rad) of its target and every
    # joint slower than this (rad/s), continuously for ``goal_settle_steps``.
    goal_tol_angle: float = 0.3
    goal_tol_vel: float = 1.5
    goal_settle_steps: int = 50
    # Hold curriculum: with this probability a reset starts *near* a random
    # equilibrium and commands that same configuration, so the policy learns to
    # stabilize every pose, not just move between them. 0 = off.
    hold_prob: float = 0.0
    # (min, max) angle perturbation (rad) of a hold start; joint velocities get
    # up to twice that in rad/s. Each equilibrium's scale starts at min, grows
    # after a held segment and shrinks after a failed one (a self-paced reverse
    # curriculum: the catch region is widened only as fast as it is mastered).
    hold_noise: tuple[float, float] = (0.05, 1.0)


# Reward constants (shared by the env and the vectorized ``goal_reward`` used for
# replay-buffer goal relabeling). See ``NPendulumCartpole`` for their meaning.
SHAPING_GAMMA = 0.99
SHAPING_WEIGHT = 0.2
RATE_PENALTY = 0.2


def goal_dim(n_links: int) -> int:
    """Width of the goal block in goal-conditioned mode: ``cos`` of each target."""
    return n_links


def obs_dim(n_links: int) -> int:
    """Return base kinematic observation width: ``[x, ẋ]`` + ``[cos, sin, θ̇]`` per link."""
    return 2 + 3 * n_links


def sysid_dim(n_links: int) -> int:
    """Width of the sysID context appended to the observation when hardware is set.

    Layout: ``[M, b, m_0, l_0, c_0, m_1, l_1, c_1, ...]``
    """
    return 2 + 3 * n_links


def obs_mirror_sign(
    n_links: int, with_sysid: bool = False, with_goal: bool = False
) -> np.ndarray:
    """Left-right mirror sign vector for the observation of ``n_links``.

    Mirroring the rig flips ``x, ẋ, θ, θ̇`` and the applied force. In the cos/sin
    encoding that negates ``x, ẋ, sin θ`` (odd) and ``θ̇``, and keeps ``cos θ``
    (even) — per link.  Goal targets are 0 or π, which mirroring leaves
    unchanged, and SysID params (masses, lengths, frictions) are symmetric under
    mirroring, so both blocks have sign +1.

    Layout: ``[kinematic, goal (if with_goal), sysID (if with_sysid)]``.
    """
    parts = [np.array([-1, -1] + [1, -1, -1] * n_links, dtype=np.float32)]
    if with_goal:
        parts.append(np.ones(goal_dim(n_links), dtype=np.float32))
    if with_sysid:
        parts.append(np.ones(sysid_dim(n_links), dtype=np.float32))
    return np.concatenate(parts)


def angle_potential(theta: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """Goal-alignment product over links in [0, 1] (vectorized over leading dims).

    ``prod_i (0.5 + 0.5 cos(θ_i - θ*_i))``: 1 only when every link sits at its
    target. The difference inside ``cos`` makes it indifferent to angle wrapping.
    """
    theta = np.asarray(theta, dtype=float)
    return np.prod(0.5 + 0.5 * np.cos(theta - goal), axis=-1)


def base_reward(
    state: np.ndarray,
    F: np.ndarray | float,
    dF_cmd: np.ndarray | float,
    goal: np.ndarray,
    physics: PhysicsParams,
    shape: RewardShape | None = None,
) -> np.ndarray:
    """Bounded multiplicative reward in (0, 1] toward ``goal`` (vectorized).

    ``state`` is ``(..., 2 + 2n)``; ``F``/``dF_cmd`` broadcast against the
    leading dims; ``goal`` is ``(n,)`` or ``(..., n)`` target angles. ``shape``
    adds the optional terms of :class:`RewardShape` (``None`` = original reward).
    """
    state = np.asarray(state, dtype=float)
    x = state[..., 0]
    thd = state[..., 3::2]

    # Goal alignment — product over links (all must match to score high).
    r_angle = angle_potential(state[..., 2::2], goal)
    # Cart centered on the rail (x normalized by the rail half-length). Gated
    # by goal alignment like the velocity term: free to travel during a
    # transition, but once at the goal the cart is pulled back toward x=0.
    # Bounded to [0.5, 1]; the sharper falloff (k=4) penalizes modest offsets.
    pos_pen = np.exp(-4.0 * (x / physics.x_lim) ** 2)
    r_pos = 1.0 - 0.5 * r_angle * (1.0 - pos_pen)
    # Velocity penalty GATED by goal alignment: spinning is free mid-transition
    # (r_angle≈0) and penalized only near the goal (r_angle≈1), so the reward
    # rewards *holding* the configuration — including bringing a hanging pose to
    # rest — without fighting the energy pumping a transition needs.
    vel_k = 0.1 if shape is None else shape.vel_k
    vel_pen = np.exp(-vel_k * np.sum(thd**2, axis=-1))
    r_vel = 1.0 - 0.5 * r_angle * (1.0 - vel_pen)
    # Mild energy/effort term on the normalized force.
    a = np.asarray(F, dtype=float) / physics.force_max
    r_act = 0.8 + 0.2 * np.maximum(1.0 - a**2, 0.0)
    # Command-rate term: penalize changing the commanded force between steps
    # (normalized; a full reversal saturates it). Bounded to [0.8, 1], even.
    da = np.asarray(dF_cmd, dtype=float) / physics.force_max
    r_rate = 1.0 - RATE_PENALTY * np.minimum(da * da, 1.0)

    reward = r_angle * r_pos * r_vel * r_act * r_rate
    if shape is not None:
        if shape.effort_weight > 0.0:
            f = np.asarray(F, dtype=float)
            quiet = np.exp(-((f / shape.effort_scale) ** 2))
            reward = reward * (1.0 - shape.effort_weight * r_angle * (1.0 - quiet))
        if shape.wall_weight > 0.0:
            edge = np.clip(
                (np.abs(x) / physics.x_lim - shape.wall_start)
                / (1.0 - shape.wall_start),
                0.0,
                1.0,
            )
            reward = reward * (1.0 - shape.wall_weight * edge**2)
    return reward


def energy_potential(
    state: np.ndarray,
    goal: np.ndarray,
    physics: PhysicsParams,
    shape: RewardShape | None,
) -> np.ndarray:
    """Energy-gap shaping potential ``-w (1 - r_angle) min(|E - E*|/range, 1)``.

    ``E*`` is the energy of the goal pose at rest. Zero when the energy shaping is
    off. Vectorized over leading dims of ``state`` (and ``goal``).
    """
    state = np.asarray(state, dtype=float)
    if shape is None or shape.energy_weight <= 0.0:
        return np.zeros(state.shape[:-1])
    n = (state.shape[-1] - 2) // 2
    gap = np.abs(total_energy_batch(state, physics) - goal_energy(goal, physics))
    fade = 1.0 - angle_potential(state[..., 2::2], goal)
    return -shape.energy_weight * fade * np.minimum(gap / energy_range(physics, n), 1.0)


def goal_reward(
    prev_state: np.ndarray,
    state: np.ndarray,
    F: np.ndarray | float,
    dF_cmd: np.ndarray | float,
    goal: np.ndarray,
    physics: PhysicsParams,
    shape: RewardShape | None = None,
) -> np.ndarray:
    """Full per-step reward (base + potential shaping) for an arbitrary goal.

    Matches what the env returns for the transition ``prev_state → state`` under
    ``goal`` (the env recomputes its stored potential when the goal changes, so
    the shaping term is always taken under the current goal). Used to relabel
    replay-buffer transitions with a different goal.
    """
    shaping = SHAPING_GAMMA * angle_potential(
        np.asarray(state)[..., 2::2], goal
    ) - angle_potential(np.asarray(prev_state)[..., 2::2], goal)
    reward = (
        base_reward(state, F, dF_cmd, goal, physics, shape) + SHAPING_WEIGHT * shaping
    )
    if shape is not None and shape.energy_weight > 0.0:
        reward = reward + (
            SHAPING_GAMMA * energy_potential(state, goal, physics, shape)
            - energy_potential(prev_state, goal, physics, shape)
        )
    return reward


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

    Observation space (``2 + 3n`` float32, plus ``n`` goal dims when goal-
    conditioned, plus ``2 + 3n`` sysID dims when hardware is set):
        ``[x, x_dot, cos θ_0, sin θ_0, θ̇_0, ...,
           [cos θ*_0, ..., cos θ*_{n-1}],   ← goal (±1 per link) when conditioned
           [M, b, m_0, l_0, c_0, ...]]``    ← sysID appended when hardware set

    Action space (1D float32):
        Horizontal force on the cart, clipped to ``[-force_max, force_max]``.

    Reward:
        Bounded, multiplicative shaping in (0, 1] (Lee et al.; see docs/):
            ``r = r_angle * r_pos * r_vel * r_act * r_rate``
        Each factor lies in [~0.5, 1] or [0, 1]; the product peaks at 1.0 only
        when every pole is at its target (upright, or the commanded goal in
        goal-conditioned mode), the cart is centered, velocities are low, and
        little force is used and the command changes smoothly (``r_rate``; the
        applied force is also slew-limited, see ``EnvConfig.force_slew``). The
        angle factor is a PRODUCT over all links, so partial credit for
        some-but-not-all links at their target is suppressed; the
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
    # TQC training discount so the telescoping shaping sum is consistent with
    # how the value function discounts.
    SHAPING_GAMMA: ClassVar[float] = SHAPING_GAMMA
    # Weight on the shaping term relative to the base [0,1]-bounded reward; small
    # enough that the base multiplicative reward still dominates return scale.
    SHAPING_WEIGHT: ClassVar[float] = SHAPING_WEIGHT
    # Max fractional reward lost to command chatter (see ``r_rate``).
    RATE_PENALTY: ClassVar[float] = RATE_PENALTY
    # Hold-curriculum scale factors: x1.05 per held segment, /1.1 per failure,
    # which settles where ~2/3 of hold segments succeed.
    HOLD_GROW: ClassVar[float] = 1.05
    HOLD_SHRINK: ClassVar[float] = 1.1

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
        self._uses_sysid = hw is not None and hw.sysid_context

        # Observation width: kinematic dims + goal + sysID context when configured.
        self.goal_conditioned = bool(self.cfg.goal_conditioned)
        self.OBS_DIM = (
            obs_dim(n)
            + (goal_dim(n) if self.goal_conditioned else 0)
            + (sysid_dim(n) if self._uses_sysid else 0)
        )
        self.state_dim = 2 + 2 * n
        p = self.cfg.physics

        highs = [p.x_lim * 2, np.inf]
        for _ in range(n):
            highs += [1.0, 1.0, np.inf]
        if self.goal_conditioned:
            highs += [1.0] * goal_dim(n)
        if self._uses_sysid:
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
        # The plant the dynamics integrate: nominal, or a per-episode draw.
        self._plant = self.cfg.physics
        self._force_gain = 1.0
        self._prev_energy_potential = 0.0  # energy-shaping potential (see RewardShape)
        # Encoder-style sensing state (see ``SensorModel``): last position reading
        # and the differenced (optionally filtered) velocity estimate.
        self._meas_prev: np.ndarray | None = None
        self._meas_vel = np.zeros(n + 1)

        # Goal state. Non-conditioned mode keeps the all-upright target forever.
        self.goal_configs = goal_configs(n)
        self.goal_labels = goal_labels(n)
        self.transition_stats = TransitionStats(len(self.goal_configs))
        self._goal_idx = 0
        self._goal = self.goal_configs[0].copy()
        self._seg_from = FROM_OTHER  # where the current goal segment started
        self._seg_steps = 0  # steps since the current goal was commanded
        self._seg_len = 0  # steps until the next automatic goal switch
        self._settled = 0  # consecutive steps spent at the goal
        # Hold curriculum: per-equilibrium start perturbation (rad), and whether
        # the current segment is a hold started by ``_reset_hold``. With
        # ``hold_only`` (set by the trainer for a stabilize-first phase) every
        # unpinned reset is a hold and the episode ends with that segment.
        self.hold_scale = np.full(len(self.goal_configs), self.cfg.hold_noise[0])
        self.hold_only = False
        self._hold_seg = False

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

        In goal-conditioned mode the non-random start is any equilibrium and the
        first goal is drawn (curriculum-weighted) for it; with ``hold_prob`` (or
        ``hold_only``) it is instead a perturbed equilibrium commanded to hold.
        ``options`` may pin either: ``{"start": "DD", "goal": "UU"}`` (labels or
        indices); pinning disables the hold start.

        When hardware is configured, the sysID context is resampled from the
        hardware nominal params plus measurement noise once per episode.
        """
        super().reset(seed=seed)
        cfg = self.cfg
        n = self.n_links
        options = options or {}
        if self.goal_conditioned:
            self._reset_goal_conditioned(options)
        elif self.np_random.random() < cfg.init_random_prob:
            self._state = self._random_state()
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
        self._reset_potentials()
        self._settled = 0
        self._meas_prev = None
        self._meas_vel = np.zeros(n + 1)

        if cfg.randomize is not None:
            self._plant, self._force_gain = cfg.randomize.sample(
                cfg.physics, n, self.np_random
            )
        hw = self.cfg.hardware
        if hw is not None:
            if self._uses_sysid:
                # Resample sysID context for this episode.
                self._sysid_context = _build_sysid_context(
                    self.cfg.physics, n, hw, self.np_random
                )
            # Reset action delay buffer.
            if hw.delay_steps > 0:
                self._action_buf = collections.deque(
                    [0.0] * hw.delay_steps, maxlen=hw.delay_steps
                )

        return self._make_obs(), self._goal_info()

    def _random_state(self) -> np.ndarray:
        """Fully random state: angles anywhere, modest random velocities."""
        v = self.cfg.init_vel_noise
        xl = self.cfg.physics.x_lim
        n = self.n_links
        state = np.empty(self.state_dim, dtype=np.float64)
        state[0] = self.np_random.uniform(-0.5 * xl, 0.5 * xl)
        state[1] = self.np_random.uniform(-v, v)
        state[2::2] = self.np_random.uniform(-np.pi, np.pi, n)
        state[3::2] = self.np_random.uniform(-v, v, n)
        return state

    def _reset_goal_conditioned(self, options: dict[str, Any]) -> None:
        """Pick a start (an equilibrium or random) and a first goal."""
        cfg = self.cfg
        n_goals = len(self.goal_configs)
        start = options.get("start")
        goal = options.get("goal")
        if (
            start is None
            and goal is None
            and (
                self.hold_only
                or (cfg.hold_prob > 0 and self.np_random.random() < cfg.hold_prob)
            )
        ):
            self._reset_hold()
            return
        if start is None and self.np_random.random() < cfg.init_random_prob:
            self._state = self._random_state()
            frm = FROM_OTHER
        else:
            frm = (
                goal_index(start, self.n_links)
                if start is not None
                else int(self.np_random.integers(n_goals))
            )
            base = np.zeros(self.state_dim, dtype=np.float64)
            base[2::2] = self.goal_configs[frm]
            self._state = base + self.np_random.uniform(
                -cfg.init_noise, cfg.init_noise, self.state_dim
            )
        gi = goal_index(goal, self.n_links) if goal is not None else None
        self._begin_segment(self._sample_goal(frm) if gi is None else gi, frm)

    def _reset_hold(self) -> None:
        """Start near a random equilibrium, perturbed at its scale, and hold it.

        The segment counts as ``FROM_OTHER`` (it did not start settled); its
        outcome adapts ``hold_scale`` for that equilibrium.
        """
        cfg = self.cfg
        gi = int(self.np_random.integers(len(self.goal_configs)))
        s = self.hold_scale[gi]
        state = np.zeros(self.state_dim, dtype=np.float64)
        state[2::2] = self.goal_configs[gi]
        state += self.np_random.uniform(-cfg.init_noise, cfg.init_noise, self.state_dim)
        state[2::2] += self.np_random.uniform(-s, s, self.n_links)
        state[3::2] += self.np_random.uniform(-2.0 * s, 2.0 * s, self.n_links)
        self._state = state
        self._begin_segment(gi, FROM_OTHER)
        self._hold_seg = True

    def _update_hold_scale(self, ok: bool) -> None:
        """Widen the current equilibrium's hold perturbation on success, else narrow."""
        lo, hi = self.cfg.hold_noise
        g = self._goal_idx
        s = self.hold_scale[g] * (self.HOLD_GROW if ok else 1.0 / self.HOLD_SHRINK)
        self.hold_scale[g] = float(np.clip(s, lo, hi))

    def _sample_goal(self, frm: int) -> int:
        """Draw the next goal, oversampling transitions that still fail."""
        n_goals = len(self.goal_configs)
        if self.cfg.goal_curriculum:
            p = self.transition_stats.sample_weights(frm)
            return int(self.np_random.choice(n_goals, p=p))
        return int(self.np_random.integers(n_goals))

    def _begin_segment(self, goal_idx: int, frm: int) -> None:
        """Command ``goal_idx`` and restart the segment counters.

        The shaping potential is recomputed under the new goal so the switch
        itself produces no reward spike (see ``goal_reward``).
        """
        self._goal_idx = goal_idx
        self._goal = self.goal_configs[goal_idx].copy()
        self._seg_from = frm
        self._seg_steps = 0
        self._settled = 0
        self._hold_seg = False
        hold = self.cfg.goal_hold_steps
        self._seg_len = (
            int(self.np_random.integers(hold[0], hold[1] + 1))
            if hold is not None
            else 0
        )
        self._reset_potentials()

    def set_goal(self, goal: int | str) -> np.ndarray:
        """Command a new target configuration now; return the updated observation.

        ``goal`` is an index into ``goal_configs`` or a ``U``/``D`` label such as
        ``"DU"`` (base link first). Requires ``goal_conditioned``.
        """
        if not self.goal_conditioned:
            raise RuntimeError("set_goal requires EnvConfig(goal_conditioned=True)")
        ok = self._is_settled()
        if self._seg_steps > 0:  # close the interrupted segment
            self.transition_stats.record(self._seg_from, self._goal_idx, ok)
        frm = self._goal_idx if ok else FROM_OTHER
        self._begin_segment(goal_index(goal, self.n_links), frm)
        return self._make_obs()

    @property
    def settled(self) -> bool:
        """Whether the rig has been held at the current goal long enough."""
        return self._is_settled()

    @property
    def goal(self) -> str:
        """Label of the current target configuration (e.g. ``"UU"``)."""
        return self.goal_labels[self._goal_idx]

    @property
    def goal_angles(self) -> np.ndarray:
        """Target angle per link of the current goal (0 = up, π = down)."""
        return self._goal.copy()

    def _at_goal(self, state: np.ndarray) -> bool:
        """Every link near its target angle and nearly still."""
        cfg = self.cfg
        dth = np.angle(np.exp(1j * (state[2::2] - self._goal)))  # wrap to (-π, π]
        return bool(
            np.all(np.abs(dth) < cfg.goal_tol_angle)
            and np.all(np.abs(state[3::2]) < cfg.goal_tol_vel)
        )

    def _is_settled(self) -> bool:
        return self._settled >= self.cfg.goal_settle_steps

    def _goal_info(self) -> dict[str, Any]:
        return {"goal": self.goal} if self.goal_conditioned else {}

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
        # Drive speed limit (back-EMF): what the motor can deliver at this speed.
        applied_F = available_force(applied_F, float(self._state[1]), p)

        self._state = step(self._state, applied_F * self._force_gain, self._plant)
        self._step_count += 1

        obs = self._make_obs()
        # The rate penalty acts on the *commanded* change so the policy gets a
        # gradient even where the slew limit would hide the jump from the plant.
        dF_cmd = raw_F - self._prev_cmd
        reward = self._compute_reward(self._state, applied_F, dF_cmd)
        self._prev_cmd = raw_F

        terminated = bool(abs(self._state[0]) > p.x_lim)
        truncated = self._step_count >= self.cfg.max_steps

        info: dict[str, Any] = {"applied_force": applied_F, "dF_cmd": dF_cmd}
        if self.goal_conditioned:
            info.update(self._goal_info())
            if self._advance_segment(terminated, truncated, info):
                obs = self._make_obs()  # the next action should see the new goal
            # A hold-only episode is exactly one hold segment.
            truncated = truncated or (self.hold_only and "segment" in info)
        return obs, reward, terminated, truncated, info

    def _advance_segment(
        self, terminated: bool, truncated: bool, info: dict[str, Any]
    ) -> bool:
        """Track goal progress; close the segment and maybe switch goals.

        A closed segment is reported as ``info["segment"] = (from, to, success)``
        and recorded in ``transition_stats``; a closed hold segment also adapts
        ``hold_scale`` and is reported as ``info["hold"] = (goal, success)``.
        Returns True if the goal changed.
        """
        self._seg_steps += 1
        self._settled = self._settled + 1 if self._at_goal(self._state) else 0
        hold = self.cfg.goal_hold_steps
        switch = hold is not None and self._seg_steps >= self._seg_len
        # A truncation shortly after a switch says nothing about that transition.
        long_enough = hold is None or self._seg_steps >= hold[0]
        if not (terminated or switch or (truncated and long_enough)):
            return False
        ok = not terminated and self._is_settled()
        seg = (self._seg_from, self._goal_idx, ok)
        info["segment"] = seg
        self.transition_stats.record(*seg)
        if self._hold_seg:
            self._update_hold_scale(ok)
            info["hold"] = (self._goal_idx, ok)
            self._hold_seg = False
        if not switch or terminated or truncated or self.hold_only:
            return False
        frm = self._goal_idx if ok else FROM_OTHER
        self._begin_segment(self._sample_goal(frm), frm)
        info["goal"] = self.goal
        return True

    def _make_obs(self) -> np.ndarray:
        """Build the observation: kinematics [+ goal] [+ sysID context].

        Sensor noise applies to measured quantities only, never to the goal
        (a command, not a measurement).
        """
        hw = self.cfg.hardware
        sensors = hw.sensors if hw is not None else None
        obs = encode_obs(
            self._state if sensors is None else self._measured_state(sensors)
        )
        if hw is not None and self._sysid_context is not None:
            obs = np.concatenate([obs, self._sysid_context])
        if hw is not None and hw.sensor_noise_std > 0.0:
            obs = obs + self.np_random.normal(
                0.0, hw.sensor_noise_std, obs.shape
            ).astype(np.float32)
        if self.goal_conditioned:
            k = obs_dim(self.n_links)
            goal_obs = np.cos(self._goal).astype(np.float32)
            obs = np.concatenate([obs[:k], goal_obs, obs[k:]])
        return obs

    def _read(self, true_pos: np.ndarray, noise: np.ndarray, res: np.ndarray):
        """Noisy, quantized encoder readings of the positions ``true_pos``."""
        reading = true_pos + self.np_random.normal(0.0, 1.0, true_pos.shape) * noise
        return np.where(
            res > 0.0, np.round(reading / np.where(res > 0, res, 1.0)) * res, reading
        )

    def _measured_state(self, sensors) -> np.ndarray:
        """State as the controller sees it: encoder positions, differenced rates.

        Call once per control step (it advances the velocity estimator).
        """
        dt = self.cfg.physics.dt
        n = self.n_links
        noise = np.array([sensors.x_noise_std] + [sensors.angle_noise_std] * n)
        res = np.array([sensors.x_resolution] + [sensors.angle_resolution] * n)
        pos = np.concatenate(([self._state[0]], self._state[2::2]))
        meas = self._read(pos, noise, res)
        if self._meas_prev is None:  # first reading after reset: rates unknown -> 0
            vel = np.zeros(n + 1)
        else:
            diff = (meas - self._meas_prev) / dt
            a = sensors.velocity_filter
            vel = a * self._meas_vel + (1.0 - a) * diff
        self._meas_prev, self._meas_vel = meas, vel
        out = np.empty(self.state_dim)
        out[0], out[1] = meas[0], vel[0]
        out[2::2], out[3::2] = meas[1:], vel[1:]
        return out

    def _angle_potential(self, state: np.ndarray) -> float:
        """Goal-alignment product over links, in [0, 1].

        Used as the shaping potential and the base reward's ``r_angle`` factor.
        Without goal conditioning the goal is all-upright.
        """
        return float(angle_potential(np.asarray(state)[2::2], self._goal))

    def _energy_potential(self, state: np.ndarray) -> float:
        """Energy-gap shaping potential under the current goal (0 when off)."""
        return float(
            energy_potential(state, self._goal, self.cfg.physics, self.cfg.reward_shape)
        )

    def _reset_potentials(self) -> None:
        """Re-anchor both shaping potentials to the current state and goal.

        Called on reset and on every goal switch so the switch itself produces no
        shaping spike.
        """
        self._prev_potential = self._angle_potential(self._state)
        self._prev_energy_potential = self._energy_potential(self._state)

    def _compute_reward(
        self, state: np.ndarray, F: float, dF_cmd: float = 0.0
    ) -> float:
        """Bounded multiplicative reward in (0, 1], plus potential-based shaping.

        Mirror-invariant by construction (every factor depends on x², cos θ, θ̇²,
        or a²; goal targets are 0/π), which keeps the symmetry augmentation used
        in training valid. See ``base_reward`` for the individual factors.
        """
        shape = self.cfg.reward_shape
        base = float(base_reward(state, F, dF_cmd, self._goal, self.cfg.physics, shape))
        r_angle = self._angle_potential(state)

        # Potential-based shaping (Ng et al. 1999): F(s,s') = g*P(s') - P(s), with
        # Φ = r_angle. Rewards progress toward the goal every step instead of
        # only on arrival, without changing the optimal policy.
        shaping = self.SHAPING_GAMMA * r_angle - self._prev_potential
        self._prev_potential = r_angle

        reward = base + self.SHAPING_WEIGHT * shaping
        if shape is not None and shape.energy_weight > 0.0:
            energy = self._energy_potential(state)
            reward += self.SHAPING_GAMMA * energy - self._prev_energy_potential
            self._prev_energy_potential = energy
        return float(reward)

    def get_state(self) -> np.ndarray:
        """Return the current raw physics state (``2 + 2n``)."""
        return self._state.copy()
