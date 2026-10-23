"""LQR hold design: linearisation, delay augmentation, and closed loops (model and env)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from n_cartpole.control.hybrid import HoldController, estimate_from_obs
from n_cartpole.control.lqr import (
    LQRBank,
    LQRWeights,
    augment_delay,
    discretize,
    equilibrium,
    linearize,
)
from n_cartpole.env.cartpole import encode_obs
from n_cartpole.env.dynamics import _model, _ode, step
from n_cartpole.env.factory import make_env
from n_cartpole.env.goals import goal_configs
from tests.firmware_util import rig_env

N = 2


@pytest.fixture(scope="module")
def physics():
    """Return the rig plant parameters."""
    return rig_env().physics


@pytest.mark.parametrize("goal", goal_configs(N).tolist())
def test_goals_are_equilibria_and_controllable(physics, goal) -> None:
    """Each up/down configuration is an equilibrium with a controllable linearisation."""
    g = np.asarray(goal)
    mdl = _model(dataclasses.replace(physics, cart_coulomb=0.0), N)
    assert np.abs(_ode(0.0, equilibrium(g), 0.0, mdl)).max() < 1e-9
    a, b = linearize(physics, N, g)
    ctrb = np.hstack([np.linalg.matrix_power(a, i) @ b for i in range(a.shape[0])])
    assert np.linalg.matrix_rank(ctrb) == a.shape[0]


def test_linearization_matches_the_nonlinear_model(physics) -> None:
    """A small step from the equilibrium follows ``A``, ``B`` to second order."""
    g = goal_configs(N)[0]
    a, b = linearize(physics, N, g)
    mdl = _model(dataclasses.replace(physics, cart_coulomb=0.0), N)
    rng = np.random.default_rng(0)
    e = 1e-4 * rng.normal(size=a.shape[0])
    u = 1e-3
    exact = _ode(0.0, equilibrium(g) + e, u, mdl)
    np.testing.assert_allclose(exact, a @ e + b[:, 0] * u, atol=1e-6)


@pytest.mark.parametrize("delay", [0, 1, 2])
def test_closed_loop_is_stable_for_every_goal(physics, delay: int) -> None:
    """The Riccati gain stabilises the delay-augmented model of every goal."""
    bank = LQRBank(physics, N, LQRWeights(), delay=delay)
    for d in bank.designs:
        assert d.controllability_rank == d.ns + d.m
        assert np.abs(d.poles).max() < 1.0


@pytest.mark.parametrize("delay", [0, 1, 2])
def test_delay_augmentation_matches_a_delayed_simulation(physics, delay: int) -> None:
    """``augment_delay`` reproduces plant + command line, step by step."""
    g = goal_configs(N)[0]
    a, b = linearize(physics, N, g)
    ad, bd = discretize(a, b, physics.dt)
    aa, ba = augment_delay(ad, bd, delay)
    m = max(delay, 1)
    rng = np.random.default_rng(1)
    s = 1e-3 * rng.normal(size=ad.shape[0])
    c = list(rng.normal(size=m))  # c_{k-1}, c_{k-2}, ...
    z = np.concatenate([s, c])
    for _ in range(6):
        v = float(rng.normal())
        z = aa @ z + ba[:, 0] * v
        new_c = c[0] + v  # c_k
        line = [new_c, *c]  # c_k, c_{k-1}, ...
        s = ad @ s + bd[:, 0] * line[delay]  # the plant sees c_{k-delay}
        c = line[:m]
        np.testing.assert_allclose(z, np.concatenate([s, c]), atol=1e-12)


def test_lqr_recovers_the_nominal_plant_through_the_delay(physics) -> None:
    """Stabilise UU on the nonlinear model with a one-step command delay."""
    bank = LQRBank(physics, N, LQRWeights(), delay=1)
    g = goal_configs(N)[0]
    s = equilibrium(g) + np.array([0.02, 0.0, 0.03, 0.0, -0.02, 0.0])
    hist = [0.0]  # c_{k-1}
    plant_cmd = 0.0  # in flight
    for _ in range(600):
        v = bank.rate(0, s, np.array(hist))
        c = float(np.clip(hist[0] + v, -physics.force_max, physics.force_max))
        s = step(s, plant_cmd, physics)
        plant_cmd, hist = hist[0], [c]
    assert abs(s[2]) < 1e-2 and abs(s[4]) < 1e-2 and abs(s[0]) < 0.05


def test_estimate_from_obs_inverts_encode_obs() -> None:
    """The estimate read from an observation is the state that was encoded."""
    s = np.array([0.1, -0.2, 3.0, 0.5, -0.4, 1.5])
    np.testing.assert_allclose(
        estimate_from_obs(encode_obs(s), N), [0.1, -0.2, 3.0, 0.5, -0.4, 1.5], atol=1e-6
    )


def test_hold_controller_balances_the_env() -> None:
    """In the rig env (delay, EKF cold start, nominal plant) the LQR holds UU from a tiny perturbation."""
    cfg = dataclasses.replace(
        rig_env(), goal_hold_steps=None, max_steps=600, randomize=None, init_noise=0.005
    )
    env = make_env(cfg)
    bank = LQRBank(cfg.physics, N, LQRWeights(), delay=1)
    ctrl = HoldController(env, bank)
    obs, _ = env.reset(seed=0, options={"start": "UU", "goal": "UU"})
    term = trunc = False
    while not (term or trunc):
        obs, _, term, trunc, _ = env.step(ctrl.select_action(obs[None])[0])
    assert not term
    assert env._is_settled()


def test_hybrid_hands_over_only_inside_the_engage_level() -> None:
    """With a policy and an engage level of -1 the LQR never acts; with +inf it always does."""

    class Zero:
        physics = None

        def normalize(self, obs):
            return obs

        def select_action(self, obs):
            return np.zeros((1, 1))

    cfg = dataclasses.replace(
        rig_env(), goal_hold_steps=None, max_steps=50, randomize=None
    )
    env = make_env(cfg)
    bank = LQRBank(cfg.physics, N, LQRWeights(), delay=1)
    for engage, expect in ((-1.0, 0), (np.inf, 49)):
        ctrl = HoldController(env, bank, Zero(), [engage] * 4, [np.inf] * 4)  # type: ignore[arg-type]
        obs, _ = env.reset(seed=0, options={"start": "UU", "goal": "UU"})
        done = False
        while not done:
            obs, _, term, trunc, _ = env.step(ctrl.select_action(obs[None])[0])
            done = term or trunc
        assert (ctrl.lqr_steps > expect) if expect else (ctrl.lqr_steps == 0)
