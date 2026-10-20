"""Critic warm-up: the actor and temperature stay frozen while the critic fits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from n_cartpole.env.cartpole import EnvConfig
from n_cartpole.training.off_policy import TQCConfig, TQCTrainer


def _trainer(tmp_path: Path, **kw) -> TQCTrainer:
    cfg = TQCConfig(
        env=EnvConfig(max_steps=60),
        device="cpu",
        hidden=16,
        batch_size=16,
        n_quantiles=4,
        start_steps=20,
        total_steps=80,
        buffer_size=400,
        log_every=1000,
        checkpoint_every=10_000,
        checkpoint_dir=tmp_path,
        **kw,
    )
    trainer = TQCTrainer(cfg)
    env, obs = trainer.env, trainer.env.reset(seed=0)[0]
    rng = np.random.default_rng(0)
    for _ in range(100):
        act = rng.uniform(-15, 15, 1).astype(np.float32)
        nobs, r, term, trunc, _ = env.step(act)
        trainer.buffer.add(obs, act, r, nobs, term)
        obs = nobs if not (term or trunc) else env.reset()[0]
    return trainer


def _snapshot(module: torch.nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in module.parameters()]


def _changed(before: list[torch.Tensor], module: torch.nn.Module) -> bool:
    return any(
        not torch.equal(b, p) for b, p in zip(before, module.parameters(), strict=True)
    )


def test_frozen_actor_still_trains_the_critic(tmp_path: Path) -> None:
    """During warm-up the critic moves; the actor and alpha do not."""
    torch.manual_seed(0)
    trainer = _trainer(tmp_path)
    trainer.step, trainer._actor_freeze_until = 10, 50
    actor0, critic0, alpha0 = (
        _snapshot(trainer.actor),
        _snapshot(trainer.critic),
        float(trainer.alpha.item()),
    )
    for _ in range(5):
        metrics = trainer._update()
    assert not _changed(actor0, trainer.actor)
    assert float(trainer.alpha.item()) == alpha0
    assert _changed(critic0, trainer.critic)
    assert metrics["actor_loss"] == 0.0


def test_actor_trains_once_the_warmup_ends(tmp_path: Path) -> None:
    """After the freeze step the actor and temperature update normally."""
    torch.manual_seed(0)
    trainer = _trainer(tmp_path)
    trainer.step, trainer._actor_freeze_until = 60, 50
    actor0, alpha0 = _snapshot(trainer.actor), float(trainer.alpha.item())
    for _ in range(5):
        trainer._update()
    assert _changed(actor0, trainer.actor)
    assert float(trainer.alpha.item()) != alpha0


def test_train_loop_applies_the_warmup_window(tmp_path: Path) -> None:
    """The freeze window starts at the current step and the run completes."""
    trainer = _trainer(tmp_path, actor_warmup_steps=30)
    trainer.step = 0
    actor0 = _snapshot(trainer.actor)
    trainer.cfg.total_steps = 25  # entirely inside the warm-up
    trainer.train()
    assert trainer._actor_freeze_until == 30
    assert not _changed(actor0, trainer.actor)
