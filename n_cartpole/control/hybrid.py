"""Env-facing controllers: LQR alone, or the learned policy with an LQR hold.

These objects quack like a :class:`~n_cartpole.policy.loader.PolicyBundle` for
``training.evaluate._rollout`` (``physics``, ``normalize``, ``select_action``) and read the env's
goal and last command to build the LQR state, so they are for evaluation and studies, not for the
training loop. The firmware would carry the same logic with its own estimator state.
"""

from __future__ import annotations

import numpy as np

from n_cartpole.control.lqr import LQRBank
from n_cartpole.env.cartpole import NPendulumCartpole
from n_cartpole.policy.loader import PolicyBundle


def estimate_from_obs(obs: np.ndarray, n: int) -> np.ndarray:
    """``[x, xd, th_0, th_0d, ...]`` from the kinematic block of an env observation."""
    s = np.empty(2 + 2 * n)
    s[0], s[1] = obs[0], obs[1]
    for k in range(n):
        s[2 + 2 * k] = np.arctan2(obs[3 + 3 * k], obs[2 + 3 * k])
        s[3 + 2 * k] = obs[4 + 3 * k]
    return s


class HoldController:
    """LQR hold, optionally handing over from (and back to) a learned policy.

    With ``policy=None`` the LQR acts at every step (a pure hold test from a start near the
    goal). With a policy, the policy drives until the LQR cost-to-go of the estimated state drops
    below ``engage[goal]``, the LQR then drives until it rises above ``release[goal]``
    (hysteresis, so the controller does not chatter at the boundary).
    """

    def __init__(
        self,
        env: NPendulumCartpole,
        bank: LQRBank,
        policy: PolicyBundle | None = None,
        engage: list[float] | None = None,
        release: list[float] | None = None,
    ) -> None:
        """Bind to an env; ``engage``/``release`` are per-goal cost-to-go thresholds."""
        self.env = env
        self.bank = bank
        self.policy = policy
        self.n = env.n_links
        self.physics = env.cfg.physics
        n_goals = len(bank.designs)
        self.engage = engage if engage is not None else [np.inf] * n_goals
        self.release = release if release is not None else [np.inf] * n_goals
        self._hist: list[float] = []
        self.lqr_active = False
        self.lqr_steps = 0
        self.steps = 0

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """Pass the raw observation through (the policy normalises it itself)."""
        return obs

    def _reset(self) -> None:
        self._hist = []
        self.lqr_active = self.policy is None
        self.lqr_steps = 0
        self.steps = 0

    def _rate_action(self, v: float) -> float:
        """Env action for a commanded-force change ``v`` (N per step)."""
        cfg = self.env.cfg
        fmax = self.physics.force_max
        if cfg.action_mode == "delta":
            assert cfg.force_slew is not None
            return float(
                np.clip(v / (cfg.force_slew * self.physics.dt), -1.0, 1.0) * fmax
            )
        return float(np.clip(self.env._prev_cmd + v, -fmax, fmax))

    def select_action(self, obs_norm: np.ndarray) -> np.ndarray:
        """Return the action (shape ``(1, 1)``, like a policy) for the current observation."""
        env = self.env
        if env._step_count == 0:
            self._reset()
        o = np.asarray(obs_norm)[0]
        self._hist.insert(0, float(env._prev_cmd))  # c_{k-1}, newest first
        del self._hist[self.bank.designs[0].m :]
        gi = env._goal_idx
        state = estimate_from_obs(o, self.n)
        if self.policy is not None:
            level = self.bank.level(gi, state, np.array(self._hist))
            if not self.lqr_active and level < self.engage[gi]:
                self.lqr_active = True
            elif self.lqr_active and level > self.release[gi]:
                self.lqr_active = False
        self.steps += 1
        if self.lqr_active:
            self.lqr_steps += 1
            v = self.bank.rate(gi, state, np.array(self._hist))
            return np.array([[self._rate_action(v)]])
        assert self.policy is not None
        return self.policy.select_action(self.policy.normalize(o[None]))
