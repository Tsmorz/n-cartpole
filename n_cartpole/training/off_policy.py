"""Off-policy TQC training loop with a replay buffer and symmetric augmentation.

This is the sample-efficient alternative to PPO used by Lee et al. for real
hardware. It keeps the PPO code path untouched — pick the algorithm at the CLI.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from n_cartpole.env.double_cartpole import (
    OBS_MIRROR_SIGN,
    DoublePendulumCartpole,
    EnvConfig,
)
from n_cartpole.policy.actor_critic import RunningNorm
from n_cartpole.policy.tqc import (
    QuantileCritic,
    SquashedGaussianActor,
    quantile_huber_loss,
)
from n_cartpole.training.trainer import _resolve_device

OBS_DIM = 8
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

    def __init__(self, capacity: int) -> None:
        """Preallocate storage for ``capacity`` transitions."""
        self.capacity = capacity
        self.obs = np.zeros((capacity, OBS_DIM), dtype=np.float32)
        self.act = np.zeros((capacity, ACT_DIM), dtype=np.float32)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, OBS_DIM), dtype=np.float32)
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

        fmax = self.cfg.env.physics.force_max
        self.actor = SquashedGaussianActor(self.cfg.hidden, fmax).to(self.device)
        self.critic = QuantileCritic(
            self.cfg.hidden, self.cfg.n_critics, self.cfg.n_quantiles
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)
        self.norm = RunningNorm(OBS_DIM).to(self.device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr)

        # Auto-tuned entropy temperature (SAC): target entropy = -action_dim.
        self.target_entropy = -float(ACT_DIM)
        self.log_alpha = torch.zeros(1, device=self.device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.cfg.lr)

        self.buffer = ReplayBuffer(self.cfg.buffer_size)
        self.env = DoublePendulumCartpole(self.cfg.env)
        self._mirror = torch.as_tensor(OBS_MIRROR_SIGN, device=self.device)

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

    def save(self, path: Path) -> None:
        """Save a checkpoint compatible with scripts/play.py's actor loading."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "norm": self.norm.state_dict(),
                "cfg": self.cfg,
                "algo": "tqc",
            },
            path,
        )
        logger.info(f"Saved TQC checkpoint → {path}")

    def train(self) -> None:
        """Run the off-policy training loop."""
        cfg = self.cfg
        obs, _ = self.env.reset()
        ep_return = 0.0
        ep_len = 0
        recent_returns: list[float] = []
        metrics: dict[str, float] = {}

        pbar = tqdm(range(1, cfg.total_steps + 1), desc="TQC", unit="step")
        try:
            for step in pbar:
                self.norm.update(torch.as_tensor(obs, device=self.device).unsqueeze(0))
                if step < cfg.start_steps:
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

                if step >= cfg.start_steps and self.buffer.size >= cfg.batch_size:
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
        except KeyboardInterrupt:
            logger.info("\nTQC training interrupted.")
        finally:
            self.save(cfg.checkpoint_dir / "tqc_latest.pt")
        logger.info("TQC training complete.")
