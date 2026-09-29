"""Smoke test: verify the full training loop runs without error."""

from __future__ import annotations

import pytest

from n_cartpole.training.trainer import Trainer, TrainingConfig


@pytest.fixture()
def tiny_cfg(tmp_path):
    """Minimal config that runs 2 iterations in under a few seconds."""
    return TrainingConfig(
        n_workers=1,
        steps_per_worker=64,
        n_epochs=2,
        n_iterations=2,
        checkpoint_every=1,
        log_every=1,
        hidden=16,
        checkpoint_dir=tmp_path / "checkpoints",
    )


def test_trainer_runs_without_error(tiny_cfg: TrainingConfig) -> None:
    """Full training loop must complete two iterations and save a checkpoint."""
    trainer = Trainer(tiny_cfg)
    trainer.train()
    assert (tiny_cfg.checkpoint_dir / "latest.pt").exists()


def test_trainer_checkpoint_loadable(tiny_cfg: TrainingConfig) -> None:
    """A checkpoint saved during training must be loadable into a fresh Trainer."""
    trainer = Trainer(tiny_cfg)
    trainer.train()

    fresh = Trainer(tiny_cfg)
    fresh.load(tiny_cfg.checkpoint_dir / "latest.pt")
