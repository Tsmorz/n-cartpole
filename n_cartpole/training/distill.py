"""Distil a trained TQC actor into a smaller one that fits a microcontroller.

The 2M-step goal-conditioned policy has a 256-wide SimBa actor (~1.06M
parameters, 4.2 MB as float32): far more than an ESP32-S3 can run in a 5 ms
control period. The behaviour it learned does not need that capacity, so a
small *student* actor is trained by supervised imitation of the *teacher*:

- **Data.** Observations from the teacher's replay buffer (states the RL run
  visited at full realism) and, each DAgger round (Ross et al. 2011), fresh
  rollouts on the training env in which the student itself acts with
  probability ``1 - beta`` — so the student also learns what to do in the states
  *its own* mistakes lead to, which a teacher-only dataset never contains.
- **Labels.** The teacher's deterministic action ``tanh(mean)`` (in [-1, 1]) on
  every observation, regardless of who acted.
- **Loss.** Mean squared error between the student's and the teacher's
  ``tanh(mean)``.

The student reads the same normalized observation as the teacher (the
teacher's ``RunningNorm`` is reused), so the result is an ordinary TQC
checkpoint without critics: ``load_policy``, the evaluators and the firmware
exporter work on it unchanged. ``actor_blocks`` in the checkpoint records a
student with fewer residual blocks than the default two.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from n_cartpole.env.factory import make_env
from n_cartpole.policy.loader import load_policy
from n_cartpole.policy.tqc import SquashedGaussianActor


@dataclass
class DistillConfig:
    """Settings for :func:`distill`."""

    teacher: Path
    out: Path
    hidden: int = 64
    blocks: int = 2
    rounds: int = 6
    # Env steps collected per DAgger round (after round 0).
    steps_per_round: int = 60_000
    # Observations taken from the teacher's replay buffer for round 0 (0 = none).
    buffer_obs: int = 400_000
    grad_steps: int = 15_000  # per round
    batch_size: int = 2048
    lr: float = 1e-3
    # Learning-rate factor per round (1 = constant); the last round always
    # anneals (cosine) to 5 % of its starting rate.
    lr_decay: float = 1.0
    # Probability that the teacher acts during collection in round 1; halves
    # every round after (round 0 without a buffer is teacher-only).
    beta: float = 0.5
    realism: float | None = None  # None: the teacher's own training level
    seed: int = 0
    device: str = "cpu"


def _student_out(student: SquashedGaussianActor, obs: torch.Tensor) -> torch.Tensor:
    """Return the student's deterministic action in [-1, 1], with gradients."""
    return student.mean_action(obs)[:, 0]


class _Teacher:
    """The teacher's deterministic action in [-1, 1] on raw observations."""

    def __init__(self, path: Path) -> None:
        self.bundle = load_policy(path)
        self.fmax = float(self.bundle.physics.force_max)

    def label(self, obs: np.ndarray, chunk: int = 8192) -> np.ndarray:
        """Teacher actions / force_max for raw observations ``(B, obs_dim)``."""
        out = np.empty(len(obs), dtype=np.float32)
        for i in range(0, len(obs), chunk):
            o = self.bundle.normalize(obs[i : i + chunk])
            out[i : i + chunk] = self.bundle.select_action(o)[:, 0] / self.fmax
        return out


def _buffer_observations(teacher: Path, n: int, rng: np.random.Generator) -> np.ndarray:
    """Up to ``n`` observations from the teacher's ``*_latest_buffer.npz``."""
    if n <= 0:
        return np.zeros((0, 0), dtype=np.float32)
    candidates = [
        teacher.with_name(teacher.stem + "_buffer.npz"),
        teacher.with_name("tqc_latest_buffer.npz"),
    ]
    for path in candidates:
        if path.exists():
            with np.load(path) as z:
                obs = z["obs"]
                idx = rng.choice(len(obs), size=min(n, len(obs)), replace=False)
                logger.info(f"{len(idx)} buffer observations ← {path}")
                return np.asarray(obs[np.sort(idx)], dtype=np.float32)
    logger.warning("No teacher replay buffer found; starting from rollouts only")
    return np.zeros((0, 0), dtype=np.float32)


def _collect(
    env,
    teacher: _Teacher,
    student: SquashedGaussianActor,
    steps: int,
    beta: float,
    rng: np.random.Generator,
    seed: int,
) -> np.ndarray:
    """Roll out the beta-mixture of teacher and student; return the observations."""
    obs_list: list[np.ndarray] = []
    obs, _ = env.reset(seed=seed)
    episode = 0
    for _ in range(steps):
        obs_list.append(np.asarray(obs, dtype=np.float32))
        if rng.random() < beta:
            act = teacher.label(np.asarray(obs, dtype=np.float32)[None])[0]
        else:
            with torch.no_grad():
                o = teacher.bundle.normalize(np.asarray(obs, dtype=np.float32)[None])
                act = float(student.mean_action(o)[0, 0])
        obs, _, terminated, truncated, _ = env.step(np.array([act * teacher.fmax]))
        if terminated or truncated:
            episode += 1
            obs, _ = env.reset(seed=seed + episode)
    return np.stack(obs_list)


def _train(
    student: SquashedGaussianActor,
    opt: torch.optim.Optimizer,
    teacher: _Teacher,
    obs: np.ndarray,
    labels: np.ndarray,
    cfg: DistillConfig,
    rng: np.random.Generator,
    lr: float,
    anneal: bool,
) -> float:
    """``cfg.grad_steps`` Adam steps on random minibatches; return the last-100 loss.

    With ``anneal`` the rate follows a cosine from ``lr`` down to ``0.05 * lr``.
    """
    norm_obs = teacher.bundle.normalize(obs)
    target = torch.from_numpy(labels)
    losses: list[float] = []
    for step in range(cfg.grad_steps):
        frac = step / max(cfg.grad_steps - 1, 1)
        cos = 0.5 * (1.0 + math.cos(math.pi * frac))
        for group in opt.param_groups:
            group["lr"] = lr * (0.05 + 0.95 * cos) if anneal else lr
        idx = torch.from_numpy(rng.integers(0, len(obs), cfg.batch_size))
        loss = torch.mean((_student_out(student, norm_obs[idx]) - target[idx]) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return float(np.mean(losses[-100:]))


def distill(cfg: DistillConfig) -> Path:
    """Run the DAgger rounds and save the student checkpoint to ``cfg.out``."""
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    teacher = _Teacher(cfg.teacher)
    ckpt = torch.load(cfg.teacher, map_location="cpu", weights_only=False)
    env_cfg = teacher.bundle.env_config
    if cfg.realism is not None:
        env_cfg = dataclasses.replace(env_cfg, realism=cfg.realism)
    env = make_env(env_cfg)
    obs_dim = teacher.bundle.obs_dim

    student = SquashedGaussianActor(
        hidden=cfg.hidden, force_max=teacher.fmax, blocks=cfg.blocks, obs_dim=obs_dim
    )
    opt = torch.optim.Adam(student.parameters(), lr=cfg.lr)
    n_params = sum(p.numel() for p in student.parameters())
    logger.info(
        f"Student: hidden {cfg.hidden} x {cfg.blocks} blocks, {n_params} parameters "
        f"(teacher: hidden {teacher.bundle.hidden})"
    )

    data = _buffer_observations(cfg.teacher, cfg.buffer_obs, rng)
    labels = teacher.label(data) if len(data) else np.zeros(0, dtype=np.float32)
    for r in range(cfg.rounds):
        if r > 0 or not len(data):
            beta = 1.0 if not len(data) else cfg.beta * 0.5 ** max(r - 1, 0)
            new = _collect(
                env,
                teacher,
                student,
                cfg.steps_per_round,
                beta,
                rng,
                cfg.seed + 1000 * r,
            )
            data = new if not len(data) else np.concatenate([data, new])
            labels = np.concatenate([labels, teacher.label(new)])
        else:
            beta = math.nan
        lr = cfg.lr * cfg.lr_decay**r
        last = r == cfg.rounds - 1
        loss = _train(student, opt, teacher, data, labels, cfg, rng, lr, anneal=last)
        logger.info(
            f"round {r}: beta {beta:.3f}, {len(data)} samples, loss {loss:.2e} "
            f"(rms action error {math.sqrt(loss) * teacher.fmax:.2f} N)"
        )

    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor": student.state_dict(),
            "norm": ckpt["norm"],
            "cfg": dataclasses.replace(ckpt["cfg"], hidden=cfg.hidden),
            "algo": "tqc",
            "step": int(ckpt.get("step", 0)),
            "actor_blocks": cfg.blocks,
            "distilled_from": str(cfg.teacher),
        },
        cfg.out,
    )
    logger.info(f"Saved student → {cfg.out}")
    return cfg.out
