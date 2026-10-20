"""Realism dial, delta action mode, and time-consistent behavior at 200 Hz."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from n_cartpole.config import load_tqc_config
from n_cartpole.env.cartpole import (
    SHAPING_GAMMA,
    EnvConfig,
    NPendulumCartpole,
    RewardShape,
    goal_reward,
    obs_dim,
    shaping_gamma,
)
from n_cartpole.env.dynamics import PhysicsParams, step, total_energy
from n_cartpole.env.hardware_config import HardwareConfig
from n_cartpole.env.randomization import PlantRandomization
from n_cartpole.training.off_policy import TQCConfig, TQCTrainer

RIG = Path(__file__).parent.parent / "config" / "rig.toml"
KIN = obs_dim(2)


def _rig(**kw) -> EnvConfig:
    cfg = load_tqc_config(RIG).env
    return dataclasses.replace(cfg, **kw)


# --------------------------------------------------------------------- scaled ranges


def test_randomization_scaled_spans_and_identity() -> None:
    """scaled(1) is identical, scaled(0) is nominal, and spans shrink monotonically."""
    base = PlantRandomization()
    assert base.scaled(1.0) == base
    zero = base.scaled(0.0)
    nominal = PhysicsParams(cart_coulomb=0.4)
    plant, gain = zero.sample(nominal, 2, np.random.default_rng(0))
    assert gain == 1.0 and plant.M == nominal.M and plant.cart_coulomb == 0.4
    np.testing.assert_allclose(plant.link_masses(2), nominal.link_masses(2))

    def spread(r: float) -> float:
        rng = np.random.default_rng(1)
        return float(
            np.std([base.scaled(r).sample(nominal, 2, rng)[0].M for _ in range(300)])
        )

    assert spread(0.25) < spread(0.5) < spread(1.0)
    assert base.scaled(0.5).cart_friction_scale == pytest.approx((0.5**0.5, 2.0**0.5))


# ------------------------------------------------------------------- realism in env


def test_realism_zero_is_a_clean_plant() -> None:
    """No rail friction, no randomization, and exact (un-noised) sensors."""
    env = NPendulumCartpole(_rig(realism=0.0, randomize=PlantRandomization()))
    env.reset(seed=3)
    assert env._plant.cart_coulomb == 0.0 and env._force_gain == 1.0
    assert env._plant.M == env.cfg.physics.M
    assert env._sensors_eff is not None
    assert (
        env._sensors_eff.angle_noise_std == 0.0 and env._sensors_eff.x_resolution == 0.0
    )
    # delay, slew and the drive limit are NOT part of the dial
    assert env.cfg.hardware is not None
    assert env.cfg.hardware.delay_steps == 1 and env.cfg.force_slew == 300.0
    assert env.cfg.physics.v_noload is not None


def test_realism_one_is_unchanged() -> None:
    """At full realism the plant and sensors are exactly the configured ones."""
    cfg = _rig(randomize=None)
    env = NPendulumCartpole(cfg)
    env.reset(seed=0)
    assert env._plant is cfg.physics
    assert cfg.hardware is not None
    assert env._sensors_eff == cfg.hardware.sensors


def test_realism_scales_friction_and_sensor_noise_and_applies_on_reset() -> None:
    """Half realism halves friction and noise; the sensors change immediately."""
    env = NPendulumCartpole(_rig(randomize=None))
    env.reset(seed=0)
    assert env._sensors_eff is not None
    full_noise = env._sensors_eff.angle_noise_std
    env.set_realism(0.5)
    assert env._sensors_eff is not None
    assert env._sensors_eff.angle_noise_std == pytest.approx(0.5 * full_noise)
    assert env._plant.cart_coulomb == env.cfg.physics.cart_coulomb  # plant: next reset
    env.reset(seed=1)
    assert env._plant.cart_coulomb == pytest.approx(0.5 * env.cfg.physics.cart_coulomb)


def test_clean_sensors_give_exact_differenced_velocity() -> None:
    """With realism 0 the velocity channel is the exact backward difference."""
    cfg = _rig(realism=0.0, randomize=None, init_random_prob=0.0)
    env = NPendulumCartpole(cfg)
    env.reset(seed=0)
    dt = cfg.physics.dt
    prev = env.get_state()[2]
    for _ in range(30):
        obs, *_ = env.step(np.array([5.0]))
        th = env.get_state()[2]
        assert obs[4] == pytest.approx((th - prev) / dt, abs=1e-3)
        prev = th


# ----------------------------------------------------------------------- delta mode


def _delta_env(**kw) -> NPendulumCartpole:
    cfg = EnvConfig(
        action_mode="delta",
        action_history=1,
        force_slew=300.0,
        hardware=HardwareConfig(sysid_context=False, delay_steps=1),
        physics=PhysicsParams(dt=0.005, force_max=15.0),
        **kw,
    )
    return NPendulumCartpole(cfg)


def test_delta_mode_integrates_a_rate_limited_command() -> None:
    """The command moves by at most slew*dt per step and saturates at force_max."""
    env = _delta_env()
    env.reset(seed=0)
    step_n = 300.0 * 0.005  # 1.5 N per step
    cmd = []
    for _ in range(14):
        obs, _, _, _, info = env.step(np.array([15.0]))  # full positive rate
        cmd.append(obs[KIN] * 15.0)  # newest commanded force (obs extras)
    np.testing.assert_allclose(np.diff([0.0, *cmd])[:9], step_n, atol=1e-6)
    assert max(cmd) == pytest.approx(15.0)  # saturates at the force limit
    # half rate -> half step; reversal ramps back down (no jumps)
    env.reset(seed=1)
    obs, *_ = env.step(np.array([7.5]))
    assert obs[KIN] * 15.0 == pytest.approx(0.5 * step_n)
    for _ in range(3):
        obs, *_ = env.step(np.array([-15.0]))
    assert obs[KIN] * 15.0 == pytest.approx(0.5 * step_n - 3 * step_n)


def test_delta_mode_applied_force_never_exceeds_the_rate_limit() -> None:
    """Whatever the policy asks, the applied force changes by <= slew*dt per step."""
    env = _delta_env()
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    last = 0.0
    for _ in range(400):
        _, _, term, trunc, info = env.step(rng.uniform(-15, 15, 1))
        assert abs(info["applied_force"] - last) <= 300.0 * 0.005 + 1e-9
        last = info["applied_force"]
        if term or trunc:
            env.reset()
            last = 0.0


def test_delta_mode_validation() -> None:
    """Delta mode needs a slew limit and the command in the observation."""
    with pytest.raises(ValueError, match="delta"):
        NPendulumCartpole(EnvConfig(action_mode="delta", action_history=0))
    with pytest.raises(ValueError, match="delta"):
        NPendulumCartpole(
            EnvConfig(action_mode="delta", action_history=1, force_slew=None)
        )
    with pytest.raises(ValueError, match="action_mode"):
        NPendulumCartpole(EnvConfig(action_mode="jerk"))


# --------------------------------------------------------------- time consistency


def test_shaping_gamma_keeps_the_horizon_in_seconds() -> None:
    """Gamma per step rescales with dt: 0.99 at 100 Hz, ~0.995 at 200 Hz."""
    assert shaping_gamma(PhysicsParams(dt=0.01)) == pytest.approx(SHAPING_GAMMA)
    g = shaping_gamma(PhysicsParams(dt=0.005))
    assert g == pytest.approx(0.99**0.5) and 0.9949 < g < 0.9951
    assert g**200 == pytest.approx(SHAPING_GAMMA**100)  # same 1 s horizon


def test_goal_reward_matches_env_reward_at_200hz() -> None:
    """Relabeling reward == env reward with the rescaled shaping gamma and shaping on."""
    cfg = _rig(
        goal_conditioned=True,
        goal_hold_steps=(40, 60),
        hardware=None,
        randomize=None,
        reward_shape=RewardShape(energy_weight=0.1),
        init_random_prob=0.5,
    )
    env = NPendulumCartpole(cfg)
    env.reset(seed=2)
    rng = np.random.default_rng(2)
    for _ in range(200):
        prev, goal = env.get_state(), env.goal_angles
        _, r, term, trunc, info = env.step(rng.uniform(-15, 15, 1).astype(np.float32))
        expected = goal_reward(
            prev,
            env.get_state(),
            info["applied_force"],
            info["dF_cmd"],
            goal,
            cfg.physics,
            cfg.reward_shape,
        )
        assert r == pytest.approx(float(expected), abs=1e-9)
        if term or trunc:
            env.reset()


def test_rig_is_200hz_with_time_consistent_windows() -> None:
    """rig.toml runs at 200 Hz with windows doubled to keep their durations."""
    cfg = load_tqc_config(RIG).env
    assert cfg.hardware is not None
    assert cfg.physics.dt == 0.005
    assert cfg.goal_settle_steps * cfg.physics.dt == pytest.approx(0.5)
    assert cfg.goal_hold_steps is not None
    assert tuple(h * cfg.physics.dt for h in cfg.goal_hold_steps) == pytest.approx(
        (4.0, 8.0)
    )
    assert cfg.hardware.delay_steps * cfg.physics.dt == pytest.approx(0.005)


def test_dynamics_two_half_steps_equal_one_full_step() -> None:
    """The integrator is consistent across control periods, and energy is conserved at 200 Hz."""
    p10 = PhysicsParams(dt=0.01, b=0.0, joint_friction=0.0)
    p5 = dataclasses.replace(p10, dt=0.005)
    s = np.array([0.0, 0.2, 2.5, 0.3, 2.9, -0.2])
    full = step(s, 3.0, p10)
    half = step(step(s, 3.0, p5), 3.0, p5)
    np.testing.assert_allclose(full, half, atol=1e-6)
    e0 = total_energy(s, p5)
    cur = s
    for _ in range(600):
        cur = step(cur, 0.0, p5)
    assert abs(total_energy(cur, p5) - e0) < 2e-3 * max(1.0, abs(e0))


# ------------------------------------------------------------------------ trainer


def test_trainer_realism_schedule_and_ramp(tmp_path: Path) -> None:
    """realism_at ramps linearly from the env level to 1, and the env follows it."""
    cfg = TQCConfig(
        env=EnvConfig(realism=0.2, max_steps=40),
        device="cpu",
        hidden=16,
        batch_size=16,
        n_quantiles=4,
        start_steps=20,
        total_steps=120,
        buffer_size=300,
        log_every=1000,
        checkpoint_every=10_000,
        checkpoint_dir=tmp_path,
        realism_ramp_start=20,
        realism_ramp_steps=80,
    )
    trainer = TQCTrainer(cfg)
    assert trainer.realism_at(0) == pytest.approx(0.2)
    assert trainer.realism_at(60) == pytest.approx(0.2 + 0.8 * 0.5)
    assert trainer.realism_at(500) == pytest.approx(1.0)
    trainer.train()
    assert trainer.env._realism > 0.8  # followed the ramp to the end
    fixed = TQCTrainer(dataclasses.replace(cfg, realism_ramp_steps=0))
    assert fixed.realism_at(10_000) == pytest.approx(0.2)


def test_goal_conditioned_delta_ekf_pipeline_end_to_end(tmp_path: Path) -> None:
    """The rig recipe works for goals: train, save, load, and run the transition evaluator.

    Goal-conditioned obs + delta action mode + EKF velocities + a force history in one
    run, through relabeling, checkpointing, loading, and every (start, goal) pair.
    """
    from n_cartpole.policy.loader import load_policy
    from n_cartpole.training.evaluate import evaluate_transition_metrics

    env_cfg = dataclasses.replace(
        load_tqc_config(RIG).env,
        goal_conditioned=True,
        goal_hold_steps=(40, 60),
        max_steps=300,
        action_mode="delta",
        action_history=2,
        realism=0.0,
    )
    hw = env_cfg.hardware
    assert hw is not None and hw.sensors is not None
    sensors = dataclasses.replace(hw.sensors, estimator="ekf")
    env_cfg = dataclasses.replace(
        env_cfg, hardware=dataclasses.replace(hw, sensors=sensors)
    )
    cfg = TQCConfig(
        env=env_cfg,
        device="cpu",
        hidden=16,
        batch_size=16,
        n_quantiles=4,
        start_steps=40,
        total_steps=150,
        buffer_size=400,
        log_every=1000,
        checkpoint_every=10_000,
        checkpoint_dir=tmp_path,
        hold_phase_steps=50,
    )
    trainer = TQCTrainer(cfg)
    trainer.train()
    assert trainer.buffer.size > 0
    bundle = load_policy(tmp_path / "tqc_latest.pt")
    assert bundle.goal_conditioned
    assert bundle.env_config.action_mode == "delta"
    hw = bundle.env_config.hardware
    assert hw is not None and hw.sensors is not None
    assert hw.sensors.estimator == "ekf"
    stats, metrics = evaluate_transition_metrics(bundle, seconds=1.0, trials=1)
    assert stats.count[1:].sum() == 16  # every (start, goal) pair once
    assert np.all(metrics.peak_force() <= cfg.env.physics.force_max + 1e-6)
