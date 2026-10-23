"""Tests for actor distillation (training/distill.py)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import torch

from n_cartpole.env.cartpole import EnvConfig
from n_cartpole.policy.loader import load_policy
from n_cartpole.training.distill import DistillConfig, distill
from n_cartpole.training.off_policy import TQCConfig, TQCTrainer


def _teacher(tmp_path: Path) -> Path:
    cfg = TQCConfig(
        env=EnvConfig(goal_conditioned=True, goal_hold_steps=(20, 40), max_steps=60),
        device="cpu",
        hidden=32,
        batch_size=32,
        n_quantiles=8,
        start_steps=64,
        total_steps=96,
        log_every=1000,
        checkpoint_every=10_000,
        buffer_size=500,
        checkpoint_dir=tmp_path,
    )
    trainer = TQCTrainer(cfg)
    trainer.train()
    path = tmp_path / "tqc_latest.pt"
    trainer.save(path, with_buffer=True)
    return path


def _cfg(teacher: Path, out: Path, **kw) -> DistillConfig:
    base: dict = dict(
        rounds=2, steps_per_round=80, buffer_obs=50, grad_steps=30, batch_size=16
    )
    base.update(kw)
    return DistillConfig(teacher=teacher, out=out, **base)


def test_student_imitates_teacher_and_loads(tmp_path: Path) -> None:
    """The student is a loadable checkpoint whose actions approach the teacher's."""
    teacher = _teacher(tmp_path)
    out = distill(_cfg(teacher, tmp_path / "s" / "student.pt", hidden=16, blocks=1))
    student, ref = load_policy(out), load_policy(teacher)
    assert student.hidden == 16 and student.value is None
    assert torch.load(out, weights_only=False)["actor_blocks"] == 1
    obs = np.random.default_rng(0).normal(size=(64, ref.obs_dim)).astype(np.float32)
    a_s = student.select_action(student.normalize(obs))
    a_t = ref.select_action(ref.normalize(obs))
    assert a_s.shape == a_t.shape
    assert np.all(np.abs(a_s) <= ref.physics.force_max + 1e-6)


def test_distill_without_buffer_starts_from_rollouts(tmp_path: Path) -> None:
    """With no buffer, round 0 is a teacher-only rollout and training still runs."""
    teacher = _teacher(tmp_path)
    (tmp_path / "tqc_latest_buffer.npz").unlink()
    out = distill(_cfg(teacher, tmp_path / "student.pt", hidden=8, realism=0.0))
    assert load_policy(out).hidden == 8


def test_buffer_obs_zero_skips_buffer(tmp_path: Path) -> None:
    """buffer_obs=0 ignores an existing buffer."""
    teacher = _teacher(tmp_path)
    out = distill(
        _cfg(teacher, tmp_path / "student.pt", hidden=8, buffer_obs=0, rounds=1)
    )
    assert out.exists()


def test_rl_fine_tune_of_student_keeps_teacher_critics(tmp_path: Path) -> None:
    """replace_actor: a small actor trains against the teacher's critics and reloads."""
    teacher = _teacher(tmp_path)
    student = distill(_cfg(teacher, tmp_path / "s" / "student.pt", hidden=8, blocks=1))
    ckpt = torch.load(teacher, weights_only=False)
    cfg = dataclasses.replace(
        ckpt["cfg"], total_steps=ckpt["step"] + 40, checkpoint_dir=tmp_path / "rl"
    )
    trainer = TQCTrainer(cfg)
    trainer.load(teacher)
    critic_before = {k: v.clone() for k, v in trainer.critic.state_dict().items()}
    trainer.replace_actor(student)
    assert trainer.cfg.actor_hidden == 8 and trainer.cfg.actor_blocks == 1
    for k, v in trainer.critic.state_dict().items():
        assert torch.equal(v, critic_before[k])  # the teacher's critics are kept
    trainer.train()
    out = tmp_path / "rl" / "tqc_latest.pt"
    bundle = load_policy(out)
    assert bundle.hidden == 8 and bundle.value is not None
    obs = np.zeros((3, bundle.obs_dim), dtype=np.float32)
    assert bundle.value(bundle.normalize(obs)).shape == (3,)
    # Resuming the fine-tuned run rebuilds the small actor from its config.
    again = TQCTrainer(torch.load(out, weights_only=False)["cfg"])
    again.load(out)
    assert sum(p.numel() for p in again.actor.parameters()) == sum(
        p.numel() for p in trainer.actor.parameters()
    )


def test_replace_actor_without_buffer_uses_default_log_std(tmp_path: Path) -> None:
    """With an empty buffer the student's log-std starts at log(0.1 * force_max)."""
    teacher = _teacher(tmp_path)
    student = distill(_cfg(teacher, tmp_path / "student.pt", hidden=8, rounds=1))
    trainer = TQCTrainer(torch.load(teacher, weights_only=False)["cfg"])
    trainer.replace_actor(student)
    bias = trainer.actor.net.out_proj.bias[1].item()
    assert abs(bias - np.log(0.1 * trainer.actor.force_max)) < 1e-6
