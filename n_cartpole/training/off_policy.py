"""Off-policy TQC training loop with a replay buffer and symmetric augmentation.

This is the sample-efficient alternative to PPO used by Lee et al. for real
hardware. It keeps the PPO code path untouched — pick the algorithm at the CLI.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from n_cartpole.env.double_cartpole import EnvConfig
from n_cartpole.env.factory import env_spec, make_env
from n_cartpole.policy.actor_critic import RunningNorm
from n_cartpole.policy.tqc import (
    QuantileCritic,
    SquashedGaussianActor,
    quantile_huber_loss,
)
from n_cartpole.training.trainer import _resolve_device

ACT_DIM = 1


@dataclass
class TQCConfig:
    """Hyperparameters for off-policy TQC training."""

    env: EnvConfig = field(default_factory=EnvConfig)
    device: str = "auto"
    hidden: int = 256
    n_critics: int = 2
    n_quantiles: int = 25
    top_quantiles_to_drop: int = 2  # dropped from the pooled n_critics*n_quantiles

    gamma: float = 0.99
    tau: float = 0.005  # soft target update rate
    lr: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 1_000_000
    start_steps: int = 5_000  # random actions before learning starts
    updates_per_step: int = 1
    symmetry_augment: bool = True  # mirror sampled minibatches (VER)

    total_steps: int = 200_000
    checkpoint_dir: Path = Path("checkpoints")
    checkpoint_every: int = 25_000
    log_every: int = 5_000


class ReplayBuffer:
    """Fixed-capacity circular replay buffer of transitions."""

    def __init__(self, capacity: int, obs_dim: int) -> None:
        """Preallocate storage for ``capacity`` transitions."""
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, ACT_DIM), dtype=np.float32)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store one transition, overwriting the oldest when full."""
        i = self.ptr
        self.obs[i] = obs
        self.act[i] = act
        self.rew[i] = rew
        self.next_obs[i] = next_obs
        self.done[i] = float(done)
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def save(self, path: Path) -> None:
        """Write the filled part of the buffer to an uncompressed ``.npz``."""
        n = self.size
        np.savez(
            path,
            obs=self.obs[:n],
            act=self.act[:n],
            rew=self.rew[:n],
            next_obs=self.next_obs[:n],
            done=self.done[:n],
            ptr=self.ptr,
        )

    def load(self, path: Path) -> None:
        """Restore transitions saved by :meth:`save` (truncated to capacity)."""
        d = np.load(path)
        n = min(len(d["obs"]), self.capacity)
        self.obs[:n] = d["obs"][:n]
        self.act[:n] = d["act"][:n]
        self.rew[:n] = d["rew"][:n]
        self.next_obs[:n] = d["next_obs"][:n]
        self.done[:n] = d["done"][:n]
        self.size = n
        self.ptr = int(d["ptr"]) % self.capacity if n == self.capacity else n

    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        """Return a uniform random minibatch of transitions."""
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.obs[idx],
            self.act[idx],
            self.rew[idx],
            self.next_obs[idx],
            self.done[idx],
        )


class TQCTrainer:
    """Trains a TQC policy for the double cartpole (single-env, replay-based)."""

    def __init__(self, cfg: TQCConfig | None = None) -> None:
        """Build networks, optimizers, buffer, and environment."""
        self.cfg = cfg or TQCConfig()
        self.device = _resolve_device(self.cfg.device)
        logger.info(
            f"TQC training device: {self.device} (requested: {self.cfg.device})"
        )

        obs_dim, mirror_sign = env_spec(self.cfg.env)
        fmax = self.cfg.env.physics.force_max
        self.actor = SquashedGaussianActor(self.cfg.hidden, fmax, obs_dim=obs_dim).to(
            self.device
        )
        self.critic = QuantileCritic(
            self.cfg.hidden, self.cfg.n_critics, self.cfg.n_quantiles, obs_dim=obs_dim
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)
        self.norm = RunningNorm(obs_dim).to(self.device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr)

        # Auto-tuned entropy temperature (SAC): target entropy = -action_dim.
        self.target_entropy = -float(ACT_DIM)
        self.log_alpha = torch.zeros(1, device=self.device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.cfg.lr)

        self.buffer = ReplayBuffer(self.cfg.buffer_size, obs_dim)
        self.env = make_env(self.cfg.env)
        self._mirror = torch.as_tensor(mirror_sign, device=self.device)
        # Env steps already taken (non-zero after ``load``); ``resumed`` skips the
        # random-action warm-up because the actor is already trained.
        self.step = 0
        self.resumed = False

    @property
    def alpha(self) -> torch.Tensor:
        """Current entropy temperature."""
        return self.log_alpha.exp()

    def _normalize(self, obs_np: np.ndarray) -> torch.Tensor:
        return self.norm.normalize(torch.as_tensor(obs_np, device=self.device))

    def _sample_batch(self) -> tuple[torch.Tensor, ...]:
        obs, act, rew, next_obs, done = self.buffer.sample(self.cfg.batch_size)
        t = lambda a: torch.as_tensor(a, device=self.device)
        obs_t, act_t, rew_t, next_t, done_t = (
            t(obs),
            t(act),
            t(rew),
            t(next_obs),
            t(done),
        )
        if self.cfg.symmetry_augment:
            obs_t = torch.cat([obs_t, obs_t * self._mirror])
            next_t = torch.cat([next_t, next_t * self._mirror])
            act_t = torch.cat([act_t, -act_t])
            rew_t = torch.cat([rew_t, rew_t])
            done_t = torch.cat([done_t, done_t])
        # Normalize observations with running stats (frozen during the update).
        return (
            self.norm.normalize(obs_t),
            act_t,
            rew_t,
            self.norm.normalize(next_t),
            done_t,
        )

    def _update(self) -> dict[str, float]:
        obs, act, rew, next_obs, done = self._sample_batch()
        cfg = self.cfg
        n_atoms = cfg.n_critics * cfg.n_quantiles
        keep = n_atoms - cfg.top_quantiles_to_drop

        with torch.no_grad():
            next_a, next_logp = self.actor.sample(next_obs)
            next_atoms = self.critic_target(next_obs, next_a)  # (B, N, M)
            pooled, _ = next_atoms.reshape(next_atoms.shape[0], -1).sort(dim=1)
            z = pooled[:, :keep]  # truncate the largest atoms
            target = rew + cfg.gamma * (1.0 - done) * (
                z - self.alpha * next_logp.unsqueeze(1)
            )  # (B, keep)

        atoms = self.critic(obs, act)  # (B, N, M)
        taus = (
            torch.arange(cfg.n_quantiles, device=self.device) + 0.5
        ) / cfg.n_quantiles
        critic_loss = torch.stack(
            [
                quantile_huber_loss(atoms[:, n, :], target, taus)
                for n in range(cfg.n_critics)
            ]
        ).sum()
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Actor: maximize mean atom value minus entropy-weighted log-prob.
        a_new, logp = self.actor.sample(obs)
        q = self.critic(obs, a_new).mean(dim=(1, 2))  # (B,)
        actor_loss = (self.alpha.detach() * logp - q).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Temperature: drive entropy toward the target.
        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # Soft target update.
        with torch.no_grad():
            for p, tp in zip(
                self.critic.parameters(), self.critic_target.parameters(), strict=True
            ):
                tp.mul_(1.0 - cfg.tau).add_(cfg.tau * p)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.alpha.item()),
            "entropy": float(-logp.mean().item()),
        }

    @staticmethod
    def buffer_path(path: Path) -> Path:
        """Replay-buffer file stored beside checkpoint ``path``."""
        return path.with_name(path.stem + "_buffer.npz")

    def save(self, path: Path, with_buffer: bool = False) -> None:
        """Save a checkpoint compatible with scripts/play.py's actor loading.

        Holds the full learner state (target critic, entropy temperature,
        optimizers, step) so ``load`` can resume exactly. The replay buffer is
        large, so it is written only when ``with_buffer`` is set (``*_latest``).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "actor_opt": self.actor_opt.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
                "alpha_opt": self.alpha_opt.state_dict(),
                "norm": self.norm.state_dict(),
                "step": self.step,
                "cfg": self.cfg,
                "algo": "tqc",
            },
            path,
        )
        if with_buffer:
            self.buffer.save(self.buffer_path(path))
        logger.info(f"Saved TQC checkpoint → {path}")

    def load(self, path: Path) -> None:
        """Resume from a checkpoint written by :meth:`save`.

        Older checkpoints (actor/critic/norm only) warm-start: the target critic
        is copied from the critic and alpha/optimizers start fresh. The replay buffer
        is restored if ``<ckpt>_buffer.npz`` exists; otherwise learning pauses
        until ``start_steps`` new transitions have been collected.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.norm.load_state_dict(ckpt["norm"])
        if "critic_target" in ckpt:
            self.critic_target.load_state_dict(ckpt["critic_target"])
            with torch.no_grad():
                self.log_alpha.copy_(ckpt["log_alpha"].to(self.device))
            self.actor_opt.load_state_dict(ckpt["actor_opt"])
            self.critic_opt.load_state_dict(ckpt["critic_opt"])
            self.alpha_opt.load_state_dict(ckpt["alpha_opt"])
            self.step = int(ckpt["step"])
        else:
            self.critic_target.load_state_dict(ckpt["critic"])
            logger.warning(
                "Old-format checkpoint: warm start (fresh alpha/optimizers)."
            )
            m = re.search(r"tqc_(\d+)$", path.stem)
            self.step = int(m.group(1)) if m else 0
        buf = self.buffer_path(path)
        if buf.exists():
            self.buffer.load(buf)
            logger.info(f"Restored {self.buffer.size} transitions ← {buf}")
        else:
            logger.warning("No replay buffer found; refilling before updates.")
        self.resumed = True
        logger.info(f"Loaded checkpoint ← {path} (step {self.step})")

    def train(self) -> None:
        """Run the off-policy training loop."""
        cfg = self.cfg
        obs, _ = self.env.reset()
        ep_return = 0.0
        ep_len = 0
        recent_returns: list[float] = []
        metrics: dict[str, float] = {}

        pbar = tqdm(
            range(self.step + 1, cfg.total_steps + 1),
            desc="TQC",
            unit="step",
            initial=self.step,
            total=cfg.total_steps,
        )
        try:
            for step in pbar:
                self.step = step
                self.norm.update(torch.as_tensor(obs, device=self.device).unsqueeze(0))
                if not self.resumed and step < cfg.start_steps:
                    action = self.env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = (
                            self.actor.act(self._normalize(obs).unsqueeze(0))
                            .squeeze(0)
                            .cpu()
                            .numpy()
                        )

                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                # Only a genuine terminal (out of bounds) bootstraps as done; a
                # time-limit truncation does not.
                self.buffer.add(obs, action, reward, next_obs, terminated)
                obs = next_obs
                ep_return += reward
                ep_len += 1

                if terminated or truncated:
                    recent_returns.append(ep_return)
                    recent_returns = recent_returns[-20:]
                    obs, _ = self.env.reset()
                    ep_return = 0.0
                    ep_len = 0

                if self.buffer.size >= max(cfg.start_steps, cfg.batch_size):
                    for _ in range(cfg.updates_per_step):
                        metrics = self._update()

                if recent_returns:
                    pbar.set_postfix(
                        ret=f"{np.mean(recent_returns):.1f}",
                        α=f"{metrics.get('alpha', 0.0):.3f}",
                    )

                if step % cfg.log_every == 0 and recent_returns:
                    logger.info(
                        f"step {step:7d} | return={np.mean(recent_returns):6.1f} | "
                        f"critic={metrics.get('critic_loss', 0):.3f} | "
                        f"actor={metrics.get('actor_loss', 0):.3f} | "
                        f"alpha={metrics.get('alpha', 0):.3f}"
                    )
                if step % cfg.checkpoint_every == 0:
                    self.save(cfg.checkpoint_dir / f"tqc_{step:07d}.pt")
                    self.save(cfg.checkpoint_dir / "tqc_latest.pt", with_buffer=True)
        except KeyboardInterrupt:
            logger.info("\nTQC training interrupted.")
        finally:
            self.save(cfg.checkpoint_dir / "tqc_latest.pt", with_buffer=True)
        logger.info("TQC training complete.")
