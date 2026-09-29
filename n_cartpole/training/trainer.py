"""Trainer: orchestrates parallel rollout workers and PPO gradient updates."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from loguru import logger
from tqdm import tqdm

from n_cartpole.env.double_cartpole import EnvConfig
from n_cartpole.env.factory import env_spec
from n_cartpole.policy.actor_critic import Actor, Critic, RunningNorm
from n_cartpole.policy.ppo import Batch, compute_gae, ppo_update
from n_cartpole.training.rollout import RolloutResult, rollout_worker


def _augment_symmetry(
    batch: Batch, device: torch.device, mirror_sign: np.ndarray
) -> Batch:
    """Double the batch with left-right mirrored transitions.

    The cart-pole is mirror-symmetric: mirroring the observation and negating the
    action gives a physically valid transition with the SAME reward, and therefore
    the same advantage, return, value and (for a symmetric policy) log-prob. We
    reuse those quantities for the mirrored half; training on both halves teaches
    the policy/value nets to respect the symmetry and roughly doubles sample count.
    """
    sign = torch.as_tensor(mirror_sign, device=device)
    return Batch(
        obs=torch.cat([batch["obs"], batch["obs"] * sign]),
        actions=torch.cat([batch["actions"], -batch["actions"]]),
        log_probs=torch.cat([batch["log_probs"], batch["log_probs"]]),
        advantages=torch.cat([batch["advantages"], batch["advantages"]]),
        returns=torch.cat([batch["returns"], batch["returns"]]),
        values=torch.cat([batch["values"], batch["values"]]),
    )


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
    hidden: int = 256

    # PPO
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    value_clip_eps: float = 0.2
    # No entropy bonus: exploration comes from a large INITIAL policy std
    # (log_std_init) that then anneals naturally under the policy gradient. A
    # persistent entropy bonus instead pinned the std high (~9 N), keeping the
    # policy too noisy to fine-tune balance at the top and biasing its gradient
    # against the force-clamp. Set >0 only if a run collapses to no exploration.
    entropy_coeff_start: float = 0.0
    entropy_coeff_end: float = 0.0
    # Initial policy std in FORCE units (N). Must be a sizeable fraction of
    # force_max or exploration is too small to discover the forces that swing the
    # pendulum up. log(5) ≈ 1.6 → ~5 N initial std against the 20 N limit.
    log_std_init: float = 1.6
    n_epochs: int = 10
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    # Fixed minibatch size for the PPO update. 512 gives many gradient steps per
    # rollout (good sample efficiency) while staying cheap on CPU.
    mini_batch_size: int = 512
    # Scale the value loss/clip by the running return std so value_clip_eps stays
    # meaningful when returns are large. When True, target_kl also applies.
    normalize_returns: bool = True
    # Augment each PPO batch with left-right mirrored transitions (the cart-pole
    # is mirror-symmetric). ~Free 2x data and enforces a symmetric policy — the
    # "virtual experience replay" idea from Lee et al., adapted to on-policy PPO.
    symmetry_augment: bool = True
    # Early-stop the PPO epoch loop once mean KL exceeds this, bounding how far
    # the policy moves per rollout (prevents over-updating now that the value
    # loss is properly scaled and no longer suppresses actor gradients).
    target_kl: float = 0.03

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

        self.obs_dim, self.mirror_sign = env_spec(self.cfg.env.n_links)
        logger.info(f"Links: {self.cfg.env.n_links} | observation dim: {self.obs_dim}")
        # Per-iteration mean episode return, for the training curve.
        self.return_history: list[float] = []
        # Full per-iteration metrics (return + PPO diagnostics), for the dashboard.
        self.metrics_history: list[dict[str, float]] = []

        self.actor = Actor(
            hidden=self.cfg.hidden,
            obs_dim=self.obs_dim,
            log_std_init=self.cfg.log_std_init,
        ).to(self.device)
        self.critic = Critic(hidden=self.cfg.hidden, obs_dim=self.obs_dim).to(
            self.device
        )
        self.norm = RunningNorm(self.obs_dim).to(self.device)
        # Running mean/std of returns; used to scale the value loss & clip so a
        # fixed value_clip_eps stays meaningful when returns are large.
        self.ret_norm = RunningNorm(1).to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.cfg.lr,
        )

    def _entropy_coeff(self, iteration: int) -> float:
        frac = min(1.0, iteration / self.cfg.n_iterations)
        return self.cfg.entropy_coeff_start + frac * (
            self.cfg.entropy_coeff_end - self.cfg.entropy_coeff_start
        )

    def _cpu_state_dicts(self) -> tuple[dict, dict, dict]:
        actor_sd = {k: v.cpu() for k, v in self.actor.state_dict().items()}
        critic_sd = {k: v.cpu() for k, v in self.critic.state_dict().items()}
        norm_sd = {k: v.cpu() for k, v in self.norm.state_dict().items()}
        return actor_sd, critic_sd, norm_sd

    def save(self, path: Path) -> None:
        """Save checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "norm": self.norm.state_dict(),
                "ret_norm": self.ret_norm.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "cfg": self.cfg,
            },
            path,
        )
        logger.info(f"Saved checkpoint → {path}")

    def save_return_history(self, path: Path) -> None:
        """Write per-iteration mean episode return to a CSV (for plotting)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "mean_return"])
            for i, ret in enumerate(self.return_history, start=1):
                writer.writerow([i, ret])
        logger.info(f"Saved return history → {path}")

    # Column order for metrics.csv (matches the training dashboard's panels).
    _METRIC_COLS = (
        "iteration",
        "mean_return",
        "value_loss",
        "policy_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
    )

    def save_metrics_history(self, path: Path) -> None:
        """Write per-iteration return + PPO diagnostics to a CSV (for the dashboard)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self._METRIC_COLS)
            for i, m in enumerate(self.metrics_history, start=1):
                writer.writerow([i] + [m.get(c, "") for c in self._METRIC_COLS[1:]])
        logger.info(f"Saved metrics history → {path}")

    def load(self, path: Path) -> None:
        """Load checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.norm.load_state_dict(ckpt["norm"])
        if "ret_norm" in ckpt:
            self.ret_norm.load_state_dict(ckpt["ret_norm"])
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
                    self.obs_dim,
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
                # Broadcast current weights + observation-normalizer stats to all
                # workers, so rollout-time action selection uses the same
                # normalized observations the PPO update evaluates log-probs on.
                actor_sd, critic_sd, norm_sd = self._cpu_state_dicts()
                for q in command_queues:
                    q.put((actor_sd, critic_sd, norm_sd))

                # Collect results
                results: list[RolloutResult] = []
                for _ in range(cfg.n_workers):
                    msg = result_queue.get()
                    if isinstance(msg, str):
                        raise RuntimeError(msg)
                    results.append(msg)

                # Normalize observations using the SAME stats the rollout workers
                # were broadcast at the start of this iteration (before any update),
                # so old_log_probs (computed by workers) and the log_probs
                # recomputed during the PPO update agree on input scale — otherwise
                # the importance ratio reflects a normalization shift rather than
                # an actual policy change.
                raw_obs_for_norm_update = torch.from_numpy(
                    results[0].obs  # use first worker for norm update
                ).to(self.device)
                for r in results:
                    r.obs = (
                        self.norm.normalize(torch.from_numpy(r.obs).to(self.device))
                        .cpu()
                        .numpy()
                    )

                batch = _merge_rollouts(results, cfg.gamma, cfg.lam, self.device)

                # Update the running normalizer AFTER building this iteration's
                # batch; the new stats take effect starting with next iteration's
                # broadcast, not retroactively on the batch just collected.
                self.norm.update(raw_obs_for_norm_update)

                # Update the running return normalizer and derive (mean, std) so
                # the value loss/clip are computed in scale-free units.
                if cfg.normalize_returns:
                    self.ret_norm.update(batch["returns"].reshape(-1, 1))
                    value_norm = (
                        self.ret_norm.mean.item(),
                        self.ret_norm.var.sqrt().item(),
                    )
                    target_kl = cfg.target_kl
                else:
                    value_norm = (0.0, 1.0)
                    target_kl = None

                if cfg.symmetry_augment:
                    batch = _augment_symmetry(batch, self.device, self.mirror_sign)

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
                    value_norm=value_norm,
                    target_kl=target_kl,
                )

                mean_return = sum(r.episode_return for r in results) / len(results)
                self.return_history.append(mean_return)
                self.metrics_history.append(
                    {
                        "mean_return": mean_return,
                        "policy_loss": metrics["policy_loss"],
                        "value_loss": metrics["value_loss"],
                        "entropy": metrics["entropy"],
                        "approx_kl": metrics["approx_kl"],
                        "clip_fraction": metrics["clip_fraction"],
                    }
                )
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
            self.save_return_history(cfg.checkpoint_dir / "returns.csv")
            self.save_metrics_history(cfg.checkpoint_dir / "metrics.csv")

        if interrupted:
            logger.info("Resume with: task train -- --resume checkpoints/latest.pt")
        else:
            logger.info("Training complete.")
