"""Evaluate a goal-conditioned policy on every (start, goal) transition.

Rolls the deterministic policy from each equilibrium to each target and reports
the success matrix (a transition succeeds if the rig is settled at the target —
see ``EnvConfig.goal_tol_*`` — when the time budget runs out).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from n_cartpole.env.factory import make_env
from n_cartpole.env.goals import TransitionStats
from n_cartpole.policy.loader import PolicyBundle


def evaluate_transitions(
    bundle: PolicyBundle,
    seconds: float = 8.0,
    trials: int = 3,
    seed: int = 0,
) -> TransitionStats:
    """Run ``trials`` episodes per (start, goal) pair; return the success stats."""
    if not bundle.goal_conditioned:
        raise ValueError("checkpoint is not goal-conditioned (train with --goals)")
    cfg = dataclasses.replace(
        bundle.env_config,
        goal_hold_steps=None,  # one goal per episode, judged at truncation
        max_steps=int(round(seconds / bundle.physics.dt)),
    )
    env = make_env(cfg)
    n_goals = len(env.goal_configs)
    stats = TransitionStats(n_goals)
    for start in range(n_goals):
        for goal in range(n_goals):
            for trial in range(trials):
                obs, _ = env.reset(
                    seed=seed + trial, options={"start": start, "goal": goal}
                )
                done = False
                while not done:
                    act = bundle.select_action(bundle.normalize(np.asarray(obs)[None]))
                    obs, _, terminated, truncated, info = env.step(act[0])
                    done = terminated or truncated
                    if "segment" in info:
                        stats.record(*info["segment"])
    return stats
