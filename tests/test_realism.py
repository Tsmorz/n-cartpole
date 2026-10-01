"""Cart Coulomb friction, drive speed limit, drive derivation, and encoder sensing."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from n_cartpole.config import drive_from_parts, load_tqc_config
from n_cartpole.env.cartpole import EnvConfig, NPendulumCartpole, obs_dim
from n_cartpole.env.dynamics import (
    PhysicsParams,
    _model,
    _ode,
    available_force,
    mass_matrix,
    rhs,
    step,
    total_energy,
)
from n_cartpole.env.hardware_config import HardwareConfig, SensorModel
from n_cartpole.env.randomization import PlantRandomization

RIG = Path(__file__).parent.parent / "config" / "rig.toml"


def test_coulomb_off_by_default_matches_original_model() -> None:
    """Zero Coulomb friction leaves the cart row untouched."""
    p = PhysicsParams()
    q, qd = np.array([0.1, 0.5, -1.0]), np.array([0.8, 1.0, 2.0])
    on = dataclasses.replace(p, cart_coulomb=0.5)
    r0, r1 = rhs(q, qd, 3.0, p), rhs(q, qd, 3.0, on)
    assert r1[0] == pytest.approx(r0[0] - 0.5 * math.tanh(0.8 / 0.005))
    np.testing.assert_allclose(r1[1:], r0[1:])


def test_coulomb_is_odd_dissipative_and_matches_fast_path() -> None:
    """Friction opposes the motion, mirrors correctly, and ``_ode`` agrees with rhs."""
    p = dataclasses.replace(
        PhysicsParams(), cart_coulomb=0.4, b=0.0, joint_friction=0.0
    )
    s = np.array([0.0, 1.0, 0.3, 0.0, -0.2, 0.0])
    e0 = total_energy(s, p)
    cur = s
    for _ in range(100):
        cur = step(cur, 0.0, p)
    assert total_energy(cur, p) < e0  # sliding friction removes energy
    # odd in cart velocity
    q, qd = np.array([0.0, 0.2, 0.1]), np.array([0.7, 0.0, 0.0])
    r_pos, r_neg = rhs(q, qd, 0.0, p)[0], rhs(q, -qd, 0.0, p)[0]
    assert r_pos < r_neg  # drag against +v vs against -v
    # fast ODE path == solve(M, rhs)
    state = np.array([0.1, 0.9, 0.3, 0.5, -0.4, 0.2])
    acc = np.linalg.solve(
        mass_matrix(state[2::2], p),
        rhs(
            np.array([state[0], state[2], state[4]]),
            np.array([state[1], state[3], state[5]]),
            2.0,
            p,
        ),
    )
    out = _ode(0.0, state, 2.0, _model(p, 2))
    assert out[1] == pytest.approx(acc[0]) and out[3] == pytest.approx(acc[1])


def test_available_force_back_emf_limit() -> None:
    """Pushing along the motion is limited by back-EMF; braking is not."""
    p = dataclasses.replace(PhysicsParams(), force_max=15.0, v_noload=2.0)
    assert available_force(15.0, 0.0, p) == pytest.approx(15.0)
    assert available_force(15.0, 1.0, p) == pytest.approx(7.5)  # half the speed range
    assert available_force(15.0, 2.0, p) == pytest.approx(0.0)  # at no-load speed
    assert available_force(-15.0, 1.0, p) == pytest.approx(-15.0)  # braking is free
    assert available_force(-15.0, -1.0, p) == pytest.approx(-7.5)  # symmetric
    assert available_force(40.0, 0.0, p) == pytest.approx(15.0)  # always clipped
    plain = PhysicsParams()
    assert available_force(plain.force_max, 5.0, plain) == plain.force_max


def test_drive_from_parts_reflected_inertia() -> None:
    """A NEMA17-class rotor on a small pulley dominates the moving mass."""
    m, f, v = drive_from_parts(0.55, 5.4e-5, 0.00635, 0.4, 600.0)
    assert m == pytest.approx(0.55 + 5.4e-5 / 0.00635**2)
    assert m > 1.8  # 1.34 kg of rotor inertia reflected to the cart
    assert f == pytest.approx(0.4 / 0.00635)
    assert v == pytest.approx(600 * 2 * math.pi / 60 * 0.00635)


def test_env_uses_drive_speed_limit() -> None:
    """At speed the delivered force falls toward zero (and is what the reward sees)."""
    p = dataclasses.replace(PhysicsParams(), force_max=15.0, v_noload=1.0)
    env = NPendulumCartpole(EnvConfig(physics=p, force_slew=None))
    env.reset(seed=0)
    env._state[1] = 0.8  # fast to the right
    _, _, _, _, info = env.step(np.array([15.0]))
    assert info["applied_force"] == pytest.approx(15.0 * (1 - 0.8), rel=1e-6)


def test_exact_observation_without_sensor_model() -> None:
    """No SensorModel -> the observation is the exact encoding of the state."""
    env = NPendulumCartpole(EnvConfig(hardware=HardwareConfig(sysid_context=False)))
    obs, _ = env.reset(seed=0)
    np.testing.assert_allclose(obs[:2], env.get_state()[:2], atol=1e-6)


def test_sensor_model_noise_quantization_and_differenced_velocity() -> None:
    """Positions are noisy+quantized; velocities are finite differences of them."""
    sensors = SensorModel(
        x_noise_std=1e-4,
        angle_noise_std=5e-4,
        x_resolution=1e-4,
        angle_resolution=2 * math.pi / 16384,
    )
    cfg = EnvConfig(
        hardware=HardwareConfig(sysid_context=False, sensors=sensors),
        max_steps=2000,
        init_random_prob=0.0,
    )
    env = NPendulumCartpole(cfg)
    obs, _ = env.reset(seed=1)
    assert obs.shape == (obs_dim(2),)
    # first reading after reset: rates unknown -> 0
    assert obs[1] == 0.0 and obs[4] == 0.0 and obs[7] == 0.0
    vel_err, ang_q = [], []
    for _ in range(600):
        obs, *_ = env.step(np.array([0.0]))
        s = env.get_state()
        vel_err.append(obs[7] - s[5])  # link-1 rate error at the hanging pose
        ang_q.append(obs[4])
    vel_err = np.array(vel_err)
    # differenced noise: sqrt(2) * sigma / dt (plus the small quantization term)
    expected = math.sqrt(2) * 5e-4 / 0.01
    assert 0.6 * expected < vel_err.std() < 1.6 * expected
    assert abs(vel_err.mean()) < 0.05
    # the controller's x reading sits on the encoder grid
    x_read = obs[0]
    assert abs(x_read / 1e-4 - round(x_read / 1e-4)) < 1e-3


def test_velocity_filter_smooths_differenced_rates() -> None:
    """The EMA filter reduces velocity noise (modestly: differenced noise is anti-correlated)."""
    base = dict(angle_noise_std=5e-4)
    stds = []
    for alpha in (0.0, 0.8):
        cfg = EnvConfig(
            hardware=HardwareConfig(
                sysid_context=False, sensors=SensorModel(velocity_filter=alpha, **base)
            ),
            max_steps=2000,
            init_random_prob=0.0,
        )
        env = NPendulumCartpole(cfg)
        env.reset(seed=2)
        errs = []
        for _ in range(500):
            obs, *_ = env.step(np.array([0.0]))
            errs.append(obs[7] - env.get_state()[5])
        stds.append(np.std(errs))
    assert stds[1] < 0.95 * stds[0]


def test_reward_uses_true_state_not_measurement() -> None:
    """Sensor noise perturbs observations only; rewards are computed from truth."""
    exact = EnvConfig(hardware=HardwareConfig(sysid_context=False))  # same delay
    noisy = dataclasses.replace(
        exact,
        hardware=HardwareConfig(
            sysid_context=False, sensors=SensorModel(angle_noise_std=0.05)
        ),
    )
    a, b = NPendulumCartpole(exact), NPendulumCartpole(noisy)
    a.reset(seed=3)
    b.reset(seed=3)
    b._state = a._state.copy()
    for _ in range(20):
        _, ra, *_ = a.step(np.array([2.0]))
        _, rb, *_ = b.step(np.array([2.0]))
    assert ra == pytest.approx(rb)  # same true trajectory, same reward


def test_rig_config_has_realism_blocks() -> None:
    """rig.toml derives its drive from parts and sets sensors + Coulomb friction."""
    env = load_tqc_config(RIG).env
    p = env.physics
    assert p.cart_coulomb > 0.0 and p.v_noload is not None and p.v_noload > 1.0
    assert env.hardware is not None and env.hardware.sensors is not None
    s = env.hardware.sensors
    assert s.angle_resolution == pytest.approx(2 * math.pi / 16384, rel=0.01)
    assert s.angle_noise_std > 0.0 and s.x_noise_std > 0.0
    assert not env.hardware.sysid_context
    # force and effective mass came from the drive parts, not stale constants
    assert 5.0 <= p.force_max / p.M <= 40.0


def test_coulomb_is_randomized_when_present() -> None:
    """Plant randomization scales the rail friction; zero stays zero."""
    p = dataclasses.replace(PhysicsParams(), cart_coulomb=0.4)
    rng = np.random.default_rng(0)
    vals = [PlantRandomization().sample(p, 2, rng)[0].cart_coulomb for _ in range(100)]
    assert 0.2 - 1e-9 <= min(vals) and max(vals) <= 0.8 + 1e-9 and np.std(vals) > 0.05
    assert PlantRandomization().sample(PhysicsParams(), 2, rng)[0].cart_coulomb == 0.0
