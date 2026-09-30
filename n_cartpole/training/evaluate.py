"""Evaluate a goal-conditioned policy on every (start, goal) transition.

Rolls the deterministic policy from each equilibrium to each target and reports
the success matrix (a transition succeeds if the rig is settled at the target —
see ``EnvConfig.goal_tol_*`` — when the time budget runs out). The metrics path
also records how *fast* the rig settles and how *steady* and cheap the hold is:
time-to-settle, post-settle RMS force and deviation, and the peak force / cart
speed the transition demanded (to compare against a real drive).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from n_cartpole.env.cartpole import NPendulumCartpole
from n_cartpole.env.factory import make_env
from n_cartpole.env.goals import TransitionStats
from n_cartpole.env.randomization import PlantRandomization
from n_cartpole.policy.loader import PolicyBundle


@dataclass
class TrialMetrics:
    """Outcome of one (start, goal) episode."""

    success: bool
    terminated: bool  # left the rail
    settle_time: float  # s until the rig first settled at the goal; NaN if never
    hold_rms_force: float  # RMS applied force after settling (N); NaN if never
    hold_max_x: float  # max |x| after settling (m); NaN if never
    hold_max_dev: float  # max |theta - theta*| after settling (rad); NaN if never
    peak_force: float  # max |applied force| over the episode (N)
    peak_xdot: float  # max |cart speed| over the episode (m/s)


class TransitionMetrics:
    """Per-(start, goal) trial metrics with summary matrices."""

    def __init__(self, n_goals: int) -> None:
        """Start empty for ``n_goals`` goals."""
        self.n_goals = n_goals
        self.trials: list[list[list[TrialMetrics]]] = [
            [[] for _ in range(n_goals)] for _ in range(n_goals)
        ]

    def record(self, start: int, goal: int, trial: TrialMetrics) -> None:
        """Add one trial."""
        self.trials[start][goal].append(trial)

    def _matrix(self, fn) -> np.ndarray:
        out = np.full((self.n_goals, self.n_goals), np.nan)
        for s in range(self.n_goals):
            for g in range(self.n_goals):
                ts = self.trials[s][g]
                if ts:
                    out[s, g] = fn(ts)
        return out

    @staticmethod
    def _nanstat(values: Sequence[float], stat) -> float:
        arr = np.asarray(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        return float(stat(arr)) if arr.size else float("nan")

    def median_settle_time(self) -> np.ndarray:
        """Median time-to-settle (s); NaN where no trial settled."""
        return self._matrix(
            lambda ts: self._nanstat([t.settle_time for t in ts], np.median)
        )

    def hold_rms_force(self) -> np.ndarray:
        """Mean over settled trials of the post-settle RMS force (N)."""
        return self._matrix(
            lambda ts: self._nanstat([t.hold_rms_force for t in ts], np.mean)
        )

    def hold_max_dev(self) -> np.ndarray:
        """Worst post-settle angular deviation from the target (rad)."""
        return self._matrix(
            lambda ts: self._nanstat([t.hold_max_dev for t in ts], np.max)
        )

    def peak_force(self) -> np.ndarray:
        """Largest applied |force| over the trials (N)."""
        return self._matrix(lambda ts: max(t.peak_force for t in ts))

    def peak_xdot(self) -> np.ndarray:
        """Largest cart speed over the trials (m/s)."""
        return self._matrix(lambda ts: max(t.peak_xdot for t in ts))

    def off_rail(self) -> np.ndarray:
        """Count the trials that left the rail."""
        return self._matrix(lambda ts: sum(t.terminated for t in ts))

    def format(self, labels: Sequence[str]) -> str:
        """Render every summary matrix as fixed-width tables."""
        tables = [
            ("median time-to-settle (s)", self.median_settle_time(), "{:.1f}"),
            ("hold RMS force (N)", self.hold_rms_force(), "{:.2f}"),
            ("hold max |theta - theta*| (rad)", self.hold_max_dev(), "{:.2f}"),
            ("peak |force| (N)", self.peak_force(), "{:.1f}"),
            ("peak |cart speed| (m/s)", self.peak_xdot(), "{:.2f}"),
            ("off-rail trials", self.off_rail(), "{:.0f}"),
        ]
        w = max(6, *(len(s) for s in labels))
        blocks = []
        for title, mat, fmt in tables:
            lines = [
                title,
                "from\\to".ljust(8) + "".join(s.rjust(w + 2) for s in labels),
            ]
            for r, name in enumerate(labels):
                cells = [
                    ("-" if np.isnan(v) else fmt.format(v)).rjust(w + 2) for v in mat[r]
                ]
                lines.append(name.ljust(8) + "".join(cells))
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)


def _rollout(
    bundle: PolicyBundle, env: NPendulumCartpole, start: int, goal: int, seed: int
) -> tuple[TrialMetrics, tuple[int, int, bool] | None]:
    """Run one deterministic episode; return its metrics and closed segment."""
    dt = bundle.physics.dt
    settle_steps = env.cfg.goal_settle_steps
    obs, _ = env.reset(seed=seed, options={"start": start, "goal": goal})
    target = env.goal_angles
    forces: list[float] = []
    xs: list[float] = []
    devs: list[float] = []
    speeds: list[float] = []
    settled_at: int | None = None  # index into the per-step lists
    segment: tuple[int, int, bool] | None = None
    terminated = truncated = False
    while not (terminated or truncated):
        act = bundle.select_action(bundle.normalize(np.asarray(obs)[None]))
        obs, _, terminated, truncated, info = env.step(act[0])
        state = env.get_state()
        forces.append(float(info["applied_force"]))
        xs.append(float(state[0]))
        speeds.append(abs(float(state[1])))
        devs.append(
            float(np.max(np.abs(np.angle(np.exp(1j * (state[2::2] - target))))))
        )
        if settled_at is None and env._settled >= settle_steps:
            settled_at = len(forces) - settle_steps  # first step inside tolerance
        if "segment" in info:
            segment = info["segment"]

    if settled_at is None:
        settle_time = hold_rms = hold_x = hold_dev = float("nan")
    else:
        settle_time = settled_at * dt
        window = slice(settled_at + settle_steps, None)
        hold_f = np.asarray(forces[window])
        hold_rms = float(np.sqrt(np.mean(hold_f**2))) if hold_f.size else float("nan")
        hold_x = float(np.max(np.abs(xs[window]))) if hold_f.size else float("nan")
        hold_dev = float(np.max(devs[window])) if hold_f.size else float("nan")
    trial = TrialMetrics(
        success=bool(segment and segment[2]),
        terminated=bool(terminated),
        settle_time=settle_time,
        hold_rms_force=hold_rms,
        hold_max_x=hold_x,
        hold_max_dev=hold_dev,
        peak_force=float(np.max(np.abs(forces))),
        peak_xdot=float(np.max(speeds)),
    )
    return trial, segment


def evaluate_transition_metrics(
    bundle: PolicyBundle,
    seconds: float = 8.0,
    trials: int = 3,
    seed: int = 0,
    randomize: PlantRandomization | None = None,
) -> tuple[TransitionStats, TransitionMetrics]:
    """Run ``trials`` episodes per (start, goal); return success stats + metrics.

    ``randomize`` evaluates on a per-episode perturbed TRUE plant (the policy only
    knows the nominal one), to check robustness to model error on a real rig.
    """
    if not bundle.goal_conditioned:
        raise ValueError("checkpoint is not goal-conditioned (train with --goals)")
    cfg = dataclasses.replace(
        bundle.env_config,
        goal_hold_steps=None,  # one goal per episode, judged at truncation
        max_steps=int(round(seconds / bundle.physics.dt)),
        randomize=randomize if randomize is not None else bundle.env_config.randomize,
    )
    env = make_env(cfg)
    n_goals = len(env.goal_configs)
    stats = TransitionStats(n_goals)
    metrics = TransitionMetrics(n_goals)
    for start in range(n_goals):
        for goal in range(n_goals):
            for trial in range(trials):
                result, segment = _rollout(bundle, env, start, goal, seed + trial)
                metrics.record(start, goal, result)
                if segment is not None:
                    stats.record(*segment)
    return stats, metrics


def evaluate_transitions(
    bundle: PolicyBundle,
    seconds: float = 8.0,
    trials: int = 3,
    seed: int = 0,
) -> TransitionStats:
    """Run ``trials`` episodes per (start, goal) pair; return the success stats."""
    return evaluate_transition_metrics(bundle, seconds, trials, seed)[0]
