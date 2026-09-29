"""Trainer: orchestrates parallel rollout workers and PPO gradient updates."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.multiprocessing as mp
from loguru import logger
from tqdm import tqdm

from n_cartpole.env.double_cartpole import EnvConfig
from n_cartpole.policy.actor_critic import Actor, Critic, RunningNorm
from n_cartpole.policy.ppo import Batch, compute_gae, ppo_update
from n_cartpole.training.rollout import RolloutResult, rollout_worker


def _resolve_device(pref: str) -> torch.device:
    """Resolve the training device from a preference string.

    Accepts "cpu", "mps", "cuda", or "auto". Anything unavailable falls back
    to CPU.

    Why "auto" picks CPU here: the actor/critic are tiny (8→64→64 MLPs) and the
    per-iteration batch is small, so GPU per-op dispatch overhead dominates the
    actual math. Measured on Apple Silicon, the PPO update is ~3x slower on MPS
    than CPU, and compute_gae — which reads scalars back with .item() each step —
    is ~250x slower on MPS because every read forces a device sync. CPU is the
    fast choice for this workload; MPS/CUDA remain available via an explicit
    preference if the network or batch is scaled up substantially.
    """
    pref = pref.lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        logger.warning("MPS requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    if pref == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        logger.warning("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    # "auto" (and any unknown value): CPU is fastest for this small workload.
    return torch.device("cpu")


@dataclass
class TrainingConfig:
    """All hyperparameters and infrastructure settings for a training run."""

    # Environment
    env: EnvConfig = field(default_factory=EnvConfig)

    # Compute
    # "auto" resolves to CPU: for this tiny net + small batches CPU beats MPS/CUDA.
    device: str = "auto"

    # Rollout collection
    # Leave ~2 cores free for the main process and the OS so the laptop stays
    # responsive during training; more workers past this point mostly oversubscribe.
    n_workers: int = field(
        default_factory=lambda: max(1, min(8, (os.cpu_count() or 2) - 2))
    )
    steps_per_worker: int = 2048
    hidden: int = 64

    # PPO
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    value_clip_eps: float = 0.2
    entropy_coeff_start: float = 0.01
    entropy_coeff_end: float = 0.001
    n_epochs: int = 10
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    # Fixed minibatch size for the PPO update. 512 gives many gradient steps per
    # rollout (good sample efficiency) while staying cheap on CPU.
    mini_batch_size: int = 512

    # Training loop
    n_iterations: int = 300
    checkpoint_dir: Path = Path("checkpoints")
    checkpoint_every: int = 50
    log_every: int = 10


def _merge_rollouts(
    results: list[RolloutResult],
    gamma: float,
    lam: float,
    device: torch.device,
) -> Batch:
    """Compute GAE for each worker's trajectory and assemble a single Batch."""
    all_obs, all_acts, all_lps, all_advs, all_rets, all_vals = [], [], [], [], [], []

    for r in results:
        obs_t = torch.from_numpy(r.obs).to(device)
        act_t = torch.from_numpy(r.actions).to(device)
        lp_t = torch.from_numpy(r.log_probs).to(device)
        rew_t = torch.from_numpy(r.rewards).to(device)
        val_t = torch.from_numpy(r.values).to(device)
        done_t = torch.from_numpy(r.dones.astype("float32")).to(device)

        advs, rets = compute_gae(rew_t, val_t, r.last_value, done_t, gamma, lam)

        all_obs.append(obs_t)
        all_acts.append(act_t)
        all_lps.append(lp_t)
        all_advs.append(advs.to(device))
        all_rets.append(rets.to(device))
        all_vals.append(val_t)

    return Batch(
        obs=torch.cat(all_obs),
        actions=torch.cat(all_acts),
        log_probs=torch.cat(all_lps),
        advantages=torch.cat(all_advs),
        returns=torch.cat(all_rets),
        values=torch.cat(all_vals),
    )


class Trainer:
    """Trains a PPO policy for the double cartpole using parallel CPU workers.

    Architecture:
      - N worker processes collect rollouts using the current policy on CPU.
      - The main process aggregates trajectories, computes GAE, and performs
        the PPO update on the selected device (MPS/CUDA/CPU).
      - Updated weights are broadcast to workers at the start of each iteration.
    """

    def __init__(self, cfg: TrainingConfig | None = None) -> None:
        """Set up networks, optimizer, and device."""
        self.cfg = cfg or TrainingConfig()
        self.device = _resolve_device(self.cfg.device)
        logger.info(f"Training device: {self.device} (requested: {self.cfg.device})")

        self.actor = Actor(hidden=self.cfg.hidden).to(self.device)
        self.critic = Critic(hidden=self.cfg.hidden).to(self.device)
        self.norm = RunningNorm(Actor.OBS_DIM).to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.cfg.lr,
        )

    def _entropy_coeff(self, iteration: int) -> float:
        frac = min(1.0, iteration / self.cfg.n_iterations)
        return self.cfg.entropy_coeff_start + frac * (
            self.cfg.entropy_coeff_end - self.cfg.entropy_coeff_start
        )

    def _cpu_state_dicts(self) -> tuple[dict, dict]:
        actor_sd = {k: v.cpu() for k, v in self.actor.state_dict().items()}
        critic_sd = {k: v.cpu() for k, v in self.critic.state_dict().items()}
        return actor_sd, critic_sd

    def save(self, path: Path) -> None:
        """Save checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "norm": self.norm.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "cfg": self.cfg,
            },
            path,
        )
        logger.info(f"Saved checkpoint → {path}")

    def load(self, path: Path) -> None:
        """Load checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.norm.load_state_dict(ckpt["norm"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        logger.info(f"Loaded checkpoint ← {path}")

    def train(self) -> None:
        """Run the full training loop with parallel rollout workers."""
        cfg = self.cfg
        ctx = mp.get_context("spawn")
        command_queues: list[mp.Queue] = [ctx.Queue() for _ in range(cfg.n_workers)]
        result_queue: mp.Queue = ctx.Queue()

        procs = []
        for i in range(cfg.n_workers):
            p = ctx.Process(
                target=rollout_worker,
                args=(
                    i,
                    command_queues[i],
                    result_queue,
                    cfg.env,
                    cfg.steps_per_worker,
                    cfg.hidden,
                ),
                daemon=True,
            )
            p.start()
            procs.append(p)

        logger.info(
            f"Started {cfg.n_workers} rollout workers, "
            f"{cfg.steps_per_worker} steps each "
            f"({cfg.n_workers * cfg.steps_per_worker} total per iteration)"
        )

        interrupted = False
        try:
            pbar = tqdm(range(1, cfg.n_iterations + 1), desc="Training", unit="iter")
            for iteration in pbar:
                # Broadcast current weights to all workers
                actor_sd, critic_sd = self._cpu_state_dicts()
                for q in command_queues:
                    q.put((actor_sd, critic_sd))

                # Collect results
                results: list[RolloutResult] = []
                for _ in range(cfg.n_workers):
                    msg = result_queue.get()
                    if isinstance(msg, str):
                        raise RuntimeError(msg)
                    results.append(msg)

                # Update observation normalizer
                all_obs = torch.from_numpy(
                    results[0].obs  # use first worker for norm update
                ).to(self.device)
                self.norm.update(all_obs)

                # Normalize observations across all results
                # (norm params are frozen during update — Welford already updated above)
                for r in results:
                    r.obs = (
                        self.norm.normalize(torch.from_numpy(r.obs).to(self.device))
                        .cpu()
                        .numpy()
                    )

                batch = _merge_rollouts(results, cfg.gamma, cfg.lam, self.device)
                n_total = cfg.n_workers * cfg.steps_per_worker
                n_mini = max(1, min(cfg.mini_batch_size, n_total))
                metrics = ppo_update(
                    self.actor,
                    self.critic,
                    self.optimizer,
                    batch,
                    clip_eps=cfg.clip_eps,
                    value_clip_eps=cfg.value_clip_eps,
                    entropy_coeff=self._entropy_coeff(iteration),
                    n_epochs=cfg.n_epochs,
                    mini_batch_size=n_mini,
                    max_grad_norm=cfg.max_grad_norm,
                )

                mean_return = sum(r.episode_return for r in results) / len(results)
                pbar.set_postfix(
                    ret=f"{mean_return:.1f}",
                    π=f"{metrics['policy_loss']:.4f}",
                    V=f"{metrics['value_loss']:.4f}",
                    ent=f"{metrics['entropy']:.3f}",
                )

                if iteration % cfg.log_every == 0:
                    logger.info(
                        f"iter {iteration:4d} | "
                        f"return={mean_return:7.2f} | "
                        f"π_loss={metrics['policy_loss']:6.4f} | "
                        f"v_loss={metrics['value_loss']:6.4f} | "
                        f"ent={metrics['entropy']:5.3f} | "
                        f"kl={metrics['approx_kl']:6.4f}"
                    )

                if iteration % cfg.checkpoint_every == 0:
                    self.save(cfg.checkpoint_dir / f"iter_{iteration:05d}.pt")

        except KeyboardInterrupt:
            interrupted = True
            logger.info("\nTraining interrupted.")
        finally:
            for q in command_queues:
                q.put("stop")
            for p in procs:
                p.join(timeout=5)
            self.save(cfg.checkpoint_dir / "latest.pt")

        if interrupted:
            logger.info("Resume with: task train -- --resume checkpoints/latest.pt")
        else:
            logger.info("Training complete.")
