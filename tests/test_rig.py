"""Reference rig config, link derivation, and true-plant randomization."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from n_cartpole.config import load_tqc_config, rig_summary
from n_cartpole.env.cartpole import EnvConfig, NPendulumCartpole, obs_dim
from n_cartpole.env.dynamics import link_from_parts, mass_matrix
from n_cartpole.env.factory import env_spec
from n_cartpole.env.hardware_config import HardwareConfig
from n_cartpole.env.randomization import PlantRandomization

RIG = Path(__file__).parent.parent / "config" / "rig.toml"


def test_link_from_parts_limits() -> None:
    """A bare bar is centred with mL²/12; a bare tip mass sits at the joint."""
    m, com, inertia = link_from_parts(0.3, rod_mass=0.1, tip_mass=1e-12)
    assert m == pytest.approx(0.1)
    assert com == pytest.approx(0.15)
    assert inertia == pytest.approx(0.1 * 0.3**2 / 12)
    m, com, inertia = link_from_parts(0.3, rod_mass=1e-12, tip_mass=0.2)
    assert (m, com) == (pytest.approx(0.2), pytest.approx(0.3))
    assert inertia == pytest.approx(0.0, abs=1e-12)


def test_link_from_parts_matches_point_mass_decomposition() -> None:
    """The parallel-axis inertia equals the bar + tip inertia about the joint."""
    length, rod, tip = 0.25, 0.05, 0.08
    m, com, inertia = link_from_parts(length, rod, tip)
    about_joint = rod * length**2 / 3 + tip * length**2  # bar about end + tip
    assert inertia + m * com**2 == pytest.approx(about_joint)


def test_rig_config_loads_and_is_physically_plausible() -> None:
    """The reference rig parses and its numbers are buildable."""
    env = load_tqc_config(RIG).env
    p, n = env.physics, env.n_links
    assert n == 2
    m, ln, c, inertia = (
        p.link_masses(n),
        p.link_lengths(n),
        p.link_coms(n),
        p.link_inertias(n),
    )
    assert np.all(m > 0.02) and np.all(m < 0.5)  # real bars, not ballast or dust
    assert np.all((c > 0.3 * ln) & (c < ln))  # COM inside the link, toward the tip
    assert np.all(inertia > 0)
    assert np.all(inertia < m * ln**2)  # below a point mass at the tip
    assert 0.5 <= p.M <= 3.0
    assert 5.0 <= p.force_max / p.M <= 40.0  # cart acceleration of a small belt drive
    assert p.x_lim <= 0.5  # leaves room for carriage + bumpers on a 1 m rail
    assert m.sum() < p.M  # light pendulums on a heavier cart
    assert np.all(np.linalg.eigvalsh(mass_matrix(np.array([0.3, -1.0]), p)) > 0)
    assert "link 1" in rig_summary(p, n)


def test_rig_hardware_has_delay_but_no_sysid_block() -> None:
    """The rig models action delay without adding sysID inputs to the obs."""
    env = load_tqc_config(RIG).env
    assert env.hardware is not None
    assert env.hardware.delay_steps == 1 and not env.hardware.sysid_context
    # Observation layout is purely kinematic (+ goal when conditioned).
    assert env_spec(env)[0] == obs_dim(2)
    goal = dataclasses.replace(env, goal_conditioned=True)
    assert env_spec(goal)[0] == obs_dim(2) + 2
    assert NPendulumCartpole(env).OBS_DIM == obs_dim(2)


def test_sysid_context_default_is_unchanged() -> None:
    """Plain hardware configs still append the sysID context."""
    cfg = EnvConfig(hardware=HardwareConfig())
    assert cfg.hardware is not None and cfg.hardware.sysid_context
    assert env_spec(cfg)[0] > obs_dim(2)


def test_randomization_stays_within_ranges() -> None:
    """Sampled plants respect every configured range and actually vary."""
    nominal = load_tqc_config(RIG).env.physics
    rnd = PlantRandomization()
    rng = np.random.default_rng(0)
    seen_mass = set()
    for _ in range(200):
        plant, gain = rnd.sample(nominal, 2, rng)
        assert 0.9 <= gain <= 1.1
        assert plant.force_max == pytest.approx(nominal.force_max * gain)
        assert 0.9 * nominal.M <= plant.M <= 1.1 * nominal.M
        assert 0.5 * nominal.b <= plant.b <= 2.0 * nominal.b
        m, ln, c = plant.link_masses(2), plant.link_lengths(2), plant.link_coms(2)
        assert np.all(m >= 0.9 * nominal.link_masses(2) - 1e-12)
        assert np.all(m <= 1.1 * nominal.link_masses(2) + 1e-12)
        assert np.all(c <= ln + 1e-12)
        assert np.all(plant.link_inertias(2) > 0)
        f = plant.joint_frictions(2) / nominal.joint_frictions(2)
        assert np.all((f >= 0.5 - 1e-12) & (f <= 2.0 + 1e-12))
        seen_mass.add(round(float(m[0]), 6))
    assert len(seen_mass) > 100  # actually varies
    # The nominal params are never mutated.
    assert nominal.force_max == load_tqc_config(RIG).env.physics.force_max


def test_env_integrates_a_randomized_plant_but_rewards_use_nominal() -> None:
    """The env integrates a per-episode plant; nominal params stay the reference."""
    cfg = dataclasses.replace(
        load_tqc_config(RIG).env, randomize=PlantRandomization(), max_steps=50
    )
    env = NPendulumCartpole(cfg)
    env.reset(seed=1)
    plant_a, gain_a = env._plant, env._force_gain
    env.reset(seed=2)
    assert env._plant != plant_a and env._force_gain != gain_a
    assert env.cfg.physics == cfg.physics  # nominal stays the reference
    for _ in range(10):
        obs, reward, *_ = env.step(np.array([5.0]))
    assert obs.shape == (obs_dim(2),) and np.isfinite(reward)


def test_non_randomized_env_uses_nominal_plant() -> None:
    """Without randomization the env integrates the nominal physics."""
    env = NPendulumCartpole(EnvConfig())
    env.reset(seed=0)
    assert env._plant is env.cfg.physics and env._force_gain == 1.0


def test_randomized_plant_is_mirror_symmetric() -> None:
    """Mirroring the rig (x, theta -> -x, -theta; F -> -F) mirrors the trajectory."""
    cfg = dataclasses.replace(
        load_tqc_config(RIG).env, randomize=PlantRandomization(), hardware=None
    )
    env_a, env_b = NPendulumCartpole(cfg), NPendulumCartpole(cfg)
    env_a.reset(seed=3)
    env_b.reset(seed=3)
    env_b._state = -env_a._state.copy()
    env_b._state[2::2] += 2 * np.pi * np.array([1, 1])  # pi -> -pi, equivalent mod 2 pi
    for _ in range(20):
        env_a.step(np.array([4.0]))
        env_b.step(np.array([-4.0]))
    np.testing.assert_allclose(env_b._state[:2], -env_a._state[:2], atol=1e-6)
    np.testing.assert_allclose(
        np.cos(env_b._state[2::2]), np.cos(env_a._state[2::2]), atol=1e-6
    )
