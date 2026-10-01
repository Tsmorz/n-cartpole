"""CAPS actor smoothness regularization and the rig's actuator slew."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from n_cartpole.config import load_tqc_config
from n_cartpole.env.cartpole import EnvConfig, NPendulumCartpole
from n_cartpole.training.off_policy import TQCConfig, TQCTrainer

RIG = Path(__file__).parent.parent / "config" / "rig.toml"


def _trainer(tmp_path: Path, **kw) -> TQCTrainer:
    cfg = TQCConfig(
        env=EnvConfig(goal_conditioned=True, goal_hold_steps=(20, 40)),
        device="cpu",
        hidden=32,
        batch_size=32,
        n_quantiles=8,
        start_steps=64,
        total_steps=160,
        buffer_size=1000,
        checkpoint_dir=tmp_path,
        **kw,
    )
    trainer = TQCTrainer(cfg)
    env, obs = trainer.env, trainer.env.reset(seed=0)[0]
    rng = np.random.default_rng(0)
    for _ in range(200):
        s = env.get_state()
        act = rng.uniform(-15, 15, 1).astype(np.float32)
        nobs, r, term, trunc, info = env.step(act)
        raw = (s, env.get_state(), info["applied_force"], info["dF_cmd"])
        trainer.buffer.add(obs, act, r, nobs, term, raw)
        obs = nobs if not (term or trunc) else env.reset()[0]
    return trainer


def test_smoothness_off_by_default(tmp_path: Path) -> None:
    """With zero weights there is no smoothness term at all."""
    trainer = _trainer(tmp_path)
    obs, _, _, nxt, _ = trainer._sample_batch()
    assert trainer._smoothness_loss(obs, nxt) is None
    assert trainer._update()["smooth_loss"] == 0.0


def test_smoothness_penalty_is_positive_and_trains_down(tmp_path: Path) -> None:
    """The penalty is active, differentiable, and minimizing it flattens the actor."""
    trainer = _trainer(tmp_path, smooth_temporal=5.0, smooth_spatial=5.0)
    obs, _, _, nxt, _ = trainer._sample_batch()
    before = trainer._smoothness_loss(obs, nxt)
    assert before is not None and float(before.detach()) > 0.0
    opt = torch.optim.Adam(trainer.actor.parameters(), lr=1e-2)
    torch.manual_seed(0)
    for _ in range(150):
        loss = trainer._smoothness_loss(obs, nxt)
        assert loss is not None
        opt.zero_grad()
        loss.backward()
        opt.step()
    after = trainer._smoothness_loss(obs, nxt)
    assert after is not None and float(after) < 0.2 * float(before.detach())
    assert trainer._update()["smooth_loss"] >= 0.0  # the full update still runs


def test_smooth_actor_has_lower_local_gain(tmp_path: Path) -> None:
    """After the spatial penalty, a small observation change moves the action less."""
    trainer = _trainer(tmp_path, smooth_spatial=20.0, smooth_sigma=0.1)
    obs, *_ = trainer._sample_batch()
    torch.manual_seed(1)
    noise = 0.1 * torch.randn_like(obs)

    def gain() -> float:
        with torch.no_grad():
            d = trainer.actor.mean_action(obs + noise) - trainer.actor.mean_action(obs)
        return float(d.abs().mean())

    before = gain()
    opt = torch.optim.Adam(trainer.actor.parameters(), lr=1e-2)
    for _ in range(100):
        loss = trainer._smoothness_loss(obs, obs)
        assert loss is not None
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert gain() < 0.5 * before


def test_rig_actuator_slew_is_realistic() -> None:
    """The rig cannot reverse its full force within one control period."""
    cfg = load_tqc_config(RIG).env
    p = cfg.physics
    assert cfg.force_slew is not None
    per_step = cfg.force_slew * p.dt
    assert per_step <= 0.25 * p.force_max  # at most a quarter of range per step
    env = NPendulumCartpole(cfg)
    env.reset(seed=0)
    last = 0.0
    for k in range(40):
        _, _, _, _, info = env.step(np.array([p.force_max if k < 20 else -p.force_max]))
        assert abs(info["applied_force"] - last) <= per_step + 1e-9
        last = info["applied_force"]
    # A full-scale reversal needs 2 * force_max / slew seconds.
    assert 2 * p.force_max / cfg.force_slew >= 0.08


def test_legacy_tqc_config_has_no_smoothness() -> None:
    """Configs pickled before the smoothness fields existed load with them off."""
    cfg = TQCConfig.__new__(TQCConfig)
    cfg.__dict__.update({"hidden": 32})
    assert cfg.smooth_temporal == 0.0 and cfg.smooth_spatial == 0.0
    assert pytest.approx(0.05) == cfg.smooth_sigma
