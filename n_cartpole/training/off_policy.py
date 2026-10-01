"""Off-policy TQC training loop with a replay buffer and symmetric augmentation.

TQC is the sample-efficient, off-policy algorithm Lee et al. use for real
hardware; it is this project's only learner.
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from n_cartpole.env.cartpole import NPendulumCartpole, goal_dim, goal_reward
from n_cartpole.env.cartpole import obs_dim as kin_obs_dim
from n_cartpole.env.double_cartpole import EnvConfig
from n_cartpole.env.factory import env_spec, make_env
from n_cartpole.env.goals import TransitionStats, goal_configs
from n_cartpole.policy.running_norm import RunningNorm
from n_cartpole.policy.tqc import (
    QuantileCritic,
    SquashedGaussianActor,
    quantile_huber_loss,
)

ACT_DIM = 1


def _resolve_device(pref: str) -> torch.device:
    """Resolve the training device from a preference string.

    Accepts "cpu", "mps", "cuda", or "auto". Anything unavailable falls back
    to CPU.

    Why "auto" picks CPU here: the actor/critic are small and minibatches are
    small, so GPU per-op dispatch overhead dominates the actual math (measured
    ~3x slower on Apple Silicon MPS than CPU for this workload). MPS/CUDA remain
    available via an explicit preference if the network or batch is scaled up
    substantially.
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
    # Goal-conditioned only: fraction of each minibatch relabeled with a random
    # goal (reward recomputed from the stored raw transition). The dynamics don't
    # depend on the goal, so every transition is valid training data for every
    # goal — this multiplies the data per goal by the number of goals.
    relabel_frac: float = 0.5
    # Entropy target (nats) of the *force* distribution for the auto-tuned
    # temperature. None = SAC's ``-action_dim`` measured on the action scaled to
    # [-1, 1], i.e. ``action_dim * (log(force_max) - 1)`` in force units (~2.0
    # for 20 N, exploration std ~1.8 N). The previous behavior, ``-1`` in force
    # units, allowed only ~0.09 N of exploration noise.
    target_entropy: float | None = None
    # Goal-conditioned only: for the first this many env steps every episode is
    # one hold segment (start near an equilibrium, stabilize it; see
    # ``EnvConfig.hold_noise``) before regular transitions take over.
    hold_phase_steps: int = 0
    # Actor smoothness regularization (CAPS, Mysore et al. 2021) on the
    # deterministic action scaled to [-1, 1]. A policy with a 1-step action delay
    # and high local gain falls into a fast limit cycle at equilibrium (a 10 Hz+
    # force dither the actuator could never follow). The temporal term penalizes
    # the action change between consecutive states; the spatial term penalizes
    # the change under a small perturbation of the normalized observation (a cap
    # on the actor's local gain). Both weights 0 = off (the original update).
    smooth_temporal: float = 0.0
    smooth_spatial: float = 0.0
    smooth_sigma: float = 0.05  # perturbation std in normalized-observation units

    total_steps: int = 200_000
    checkpoint_dir: Path = Path("checkpoints")
    checkpoint_every: int = 25_000
    log_every: int = 5_000


class ReplayBuffer:
    """Fixed-capacity circular replay buffer of transitions."""

    # Raw-transition arrays kept only when ``state_dim`` is given (goal relabeling).
    _RAW = ("state", "next_state", "force", "dforce")

    def __init__(self, capacity: int, obs_dim: int, state_dim: int = 0) -> None:
        """Preallocate storage for ``capacity`` transitions.

        With ``state_dim > 0`` the raw physics states, applied force and command
        change are stored too, so the reward can be recomputed for another goal.
        """
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, ACT_DIM), dtype=np.float32)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.has_raw = state_dim > 0
        if self.has_raw:
            self.state = np.zeros((capacity, state_dim), dtype=np.float64)
            self.next_state = np.zeros((capacity, state_dim), dtype=np.float64)
            self.force = np.zeros(capacity, dtype=np.float32)
            self.dforce = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
        raw: tuple[np.ndarray, np.ndarray, float, float] | None = None,
    ) -> None:
        """Store one transition, overwriting the oldest when full.

        ``raw`` is ``(state, next_state, applied_force, dF_cmd)`` and is required
        when the buffer was built with a ``state_dim``.
        """
        i = self.ptr
        self.obs[i] = obs
        self.act[i] = act
        self.rew[i] = rew
        self.next_obs[i] = next_obs
        self.done[i] = float(done)
        if self.has_raw:
            if raw is None:
                raise ValueError("this buffer stores raw transitions; pass raw=")
            self.state[i], self.next_state[i], self.force[i], self.dforce[i] = raw
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
            **{k: getattr(self, k)[:n] for k in self._RAW if self.has_raw},
        )

    def load(self, path: Path) -> None:
        """Restore transitions saved by :meth:`save` (truncated to capacity)."""
        d = np.load(path)
        if self.has_raw and not all(k in d for k in self._RAW):
            raise ValueError(f"{path} lacks raw transitions for goal relabeling")
        n = min(len(d["obs"]), self.capacity)
        self.obs[:n] = d["obs"][:n]
        self.act[:n] = d["act"][:n]
        self.rew[:n] = d["rew"][:n]
        self.next_obs[:n] = d["next_obs"][:n]
        self.done[:n] = d["done"][:n]
        if self.has_raw:
            for k in self._RAW:
                getattr(self, k)[:n] = d[k][:n]
        self.size = n
        self.ptr = int(d["ptr"]) % self.capacity if n == self.capacity else n

    def sample_indices(self, batch_size: int) -> np.ndarray:
        """Uniform random row indices into the filled part of the buffer."""
        return np.random.randint(0, self.size, size=batch_size)

    def sample(
        self, batch_size: int, idx: np.ndarray | None = None
    ) -> tuple[np.ndarray, ...]:
        """Return a uniform random minibatch (copies) of transitions."""
        if idx is None:
            idx = self.sample_indices(batch_size)
        return (
            self.obs[idx],
            self.act[idx],
            self.rew[idx],
            self.next_obs[idx],
            self.done[idx],
        )


def _insert_input_dims(
    state: dict[str, torch.Tensor], at: int, k: int
) -> dict[str, torch.Tensor]:
    """Widen every SimBa input projection by ``k`` zero-weight columns at ``at``.

    A network loaded from the result computes exactly what the original did,
    whatever values the new inputs take.
    """
    out = dict(state)
    for name, w in state.items():
        if name.endswith("in_proj.weight"):
            zeros = w.new_zeros(w.shape[0], k)
            out[name] = torch.cat([w[:, :at], zeros, w[:, at:]], dim=1)
    return out


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

        # Auto-tuned entropy temperature (SAC). ``logp`` is a density over force
        # in newtons, so the default target converts SAC's -action_dim (on the
        # [-1, 1] action) to force units.
        te = self.cfg.target_entropy
        self.target_entropy = (
            ACT_DIM * (math.log(fmax) - 1.0) if te is None else float(te)
        )
        self.log_alpha = torch.zeros(1, device=self.device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.cfg.lr)

        self.env = make_env(self.cfg.env)
        self.goal_conditioned = bool(self.cfg.env.goal_conditioned)
        self.buffer = ReplayBuffer(
            self.cfg.buffer_size,
            obs_dim,
            state_dim=self.env.state_dim if self.goal_conditioned else 0,
        )
        n = self.cfg.env.n_links
        self._goals = goal_configs(n)
        self._goal_slice = slice(kin_obs_dim(n), kin_obs_dim(n) + n)
        # Per-(from, to) transition success, reset at each log line.
        self.transition_stats = TransitionStats(len(self._goals))
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

    def _relabel(
        self,
        idx: np.ndarray,
        obs: np.ndarray,
        rew: np.ndarray,
        next_obs: np.ndarray,
    ) -> None:
        """Swap a random subset of rows to a random goal (in place).

        Both ``obs`` and ``next_obs`` get the new goal and the reward is
        recomputed with ``goal_reward`` from the stored raw transition, exactly
        as the env would have rewarded it under that goal.
        """
        rows = np.flatnonzero(np.random.random(len(idx)) < self.cfg.relabel_frac)
        if rows.size == 0:
            return
        goals = self._goals[np.random.randint(len(self._goals), size=rows.size)]
        j = idx[rows]
        b = self.buffer
        goal_obs = np.cos(goals).astype(np.float32)
        obs[rows, self._goal_slice] = goal_obs
        next_obs[rows, self._goal_slice] = goal_obs
        rew[rows, 0] = goal_reward(
            b.state[j],
            b.next_state[j],
            b.force[j],
            b.dforce[j],
            goals,
            self.cfg.env.physics,
            self.cfg.env.reward_shape,
        )

    def _sample_batch(self) -> tuple[torch.Tensor, ...]:
        idx = self.buffer.sample_indices(self.cfg.batch_size)
        obs, act, rew, next_obs, done = self.buffer.sample(self.cfg.batch_size, idx)
        if self.goal_conditioned and self.cfg.relabel_frac > 0:
            self._relabel(idx, obs, rew, next_obs)
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
        smooth = self._smoothness_loss(obs, next_obs)
        if smooth is not None:
            actor_loss = actor_loss + smooth
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
            "smooth_loss": 0.0 if smooth is None else float(smooth.item()),
        }

    def _smoothness_loss(
        self, obs: torch.Tensor, next_obs: torch.Tensor
    ) -> torch.Tensor | None:
        """Weighted CAPS penalties on the actor's deterministic action, or None."""
        cfg = self.cfg
        if cfg.smooth_temporal <= 0.0 and cfg.smooth_spatial <= 0.0:
            return None
        mu = self.actor.mean_action(obs)
        loss = obs.new_zeros(())
        if cfg.smooth_temporal > 0.0:
            mu_next = self.actor.mean_action(next_obs)
            loss = loss + cfg.smooth_temporal * (mu - mu_next).pow(2).mean()
        if cfg.smooth_spatial > 0.0:
            noisy = obs + cfg.smooth_sigma * torch.randn_like(obs)
            mu_noisy = self.actor.mean_action(noisy)
            loss = loss + cfg.smooth_spatial * (mu - mu_noisy).pow(2).mean()
        return loss

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

    def warm_start(self, path: Path) -> None:
        """Initialize this goal-conditioned run from a plain (non-goal) checkpoint.

        Actor, critics, observation normalizer, and entropy temperature are
        copied. The goal inputs are inserted with zero weights (and identity
        normalization), so the network starts out as the source policy — which
        ignores the goal — and learns to use it from there. Optimizers, step,
        and replay buffer start fresh; see :meth:`seed_buffer`.
        """
        if not self.goal_conditioned:
            raise ValueError("warm_start initializes a goal-conditioned run")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if ckpt.get("algo") != "tqc":
            raise ValueError(f"{path} is not a TQC checkpoint")
        src = ckpt["cfg"]
        if src.env.goal_conditioned:
            raise ValueError(f"{path} is already goal-conditioned; use --resume")
        cfg = self.cfg
        mismatch = [
            name
            for name, a, b in [
                ("n_links", src.env.n_links, cfg.env.n_links),
                ("hardware", src.env.hardware is None, cfg.env.hardware is None),
                ("hidden", src.hidden, cfg.hidden),
                ("n_critics", src.n_critics, cfg.n_critics),
                ("n_quantiles", src.n_quantiles, cfg.n_quantiles),
            ]
            if a != b
        ]
        if mismatch:
            raise ValueError(f"{path} does not match this run's {', '.join(mismatch)}")

        n = cfg.env.n_links
        at, k = kin_obs_dim(n), goal_dim(n)
        self.actor.load_state_dict(_insert_input_dims(ckpt["actor"], at, k))
        self.critic.load_state_dict(_insert_input_dims(ckpt["critic"], at, k))
        self.critic_target.load_state_dict(
            _insert_input_dims(ckpt.get("critic_target", ckpt["critic"]), at, k)
        )

        def insert(t: torch.Tensor, value: float) -> torch.Tensor:
            return torch.cat([t[:at], t.new_full((k,), value), t[at:]])

        norm = ckpt["norm"]
        self.norm.load_state_dict(
            {
                "mean": insert(norm["mean"], 0.0),
                "var": insert(norm["var"], 1.0),
                "count": norm["count"],
            }
        )
        if "log_alpha" in ckpt:
            with torch.no_grad():
                self.log_alpha.copy_(ckpt["log_alpha"].to(self.device))
        self.resumed = True  # the actor is trained: skip random-action warm-up
        logger.info(f"Warm-started goal-conditioned run ← {path}")

    def seed_buffer(self, n_steps: int) -> None:
        """Fill the replay buffer with ``n_steps`` of the current actor's rollouts.

        Meant for right after :meth:`warm_start`: the source policy's swing-ups
        are stored with their raw states, so goal relabeling turns them into
        data for every goal and the run starts with transitions that actually
        reach and hold upright. Uses a separate env so the curriculum statistics
        of the training env aren't skewed; the normalizer updates as in training.
        """
        env = make_env(self.cfg.env)
        obs, _ = env.reset()
        for _ in tqdm(range(n_steps), desc="seed buffer", unit="step"):
            self.norm.update(torch.as_tensor(obs, device=self.device).unsqueeze(0))
            next_obs, _, terminated, truncated, _ = self._store_step(
                env, obs, self._policy_action(obs)
            )
            obs = env.reset()[0] if terminated or truncated else next_obs
        logger.info(f"Seeded replay buffer with {n_steps} policy steps")

    def _policy_action(self, obs: np.ndarray) -> np.ndarray:
        """Sample an exploration action from the actor for one observation."""
        with torch.no_grad():
            return (
                self.actor.act(self._normalize(obs).unsqueeze(0))
                .squeeze(0)
                .cpu()
                .numpy()
            )

    def _store_step(
        self, env: NPendulumCartpole, obs: np.ndarray, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Step ``env`` and add the transition to the buffer; return the step."""
        state = env.get_state()
        next_obs, reward, terminated, truncated, info = env.step(action)
        raw = None
        stored_next = next_obs
        if self.goal_conditioned:
            raw = (state, env.get_state(), info["applied_force"], info["dF_cmd"])
            # On a goal switch ``next_obs`` already shows the new goal, but the
            # reward was for the old one: bootstrap under the old goal (like a
            # time-limit truncation), matching relabeled rows' fixed goal.
            stored_next = next_obs.copy()
            stored_next[self._goal_slice] = obs[self._goal_slice]
        # Only a genuine terminal (out of bounds) bootstraps as done; a
        # time-limit truncation does not.
        self.buffer.add(obs, action, reward, stored_next, terminated, raw)
        return next_obs, reward, terminated, truncated, info

    def train(self) -> None:
        """Run the off-policy training loop."""
        cfg = self.cfg

        def in_hold_phase(step: int) -> bool:
            return self.goal_conditioned and step < cfg.hold_phase_steps

        self.env.hold_only = in_hold_phase(self.step)
        if self.env.hold_only:
            logger.info(f"Hold-only episodes until step {cfg.hold_phase_steps}")
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
                    action = self._policy_action(obs)

                next_obs, reward, terminated, truncated, info = self._store_step(
                    self.env, obs, action
                )
                if "segment" in info:
                    self.transition_stats.record(*info["segment"])
                obs = next_obs
                ep_return += reward
                ep_len += 1

                if terminated or truncated:
                    recent_returns.append(ep_return)
                    recent_returns = recent_returns[-20:]
                    if self.env.hold_only and not in_hold_phase(step):
                        logger.info(f"Hold phase over at step {step}")
                    self.env.hold_only = in_hold_phase(step)
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
                    if self.goal_conditioned:
                        logger.info(
                            f"transition success (last {cfg.log_every} steps, "
                            f"{self.transition_stats.success_rate():.0%} overall):\n"
                            + self.transition_stats.format_matrix(self.env.goal_labels)
                        )
                        self.transition_stats = TransitionStats(len(self._goals))
                        if self.env.hold_only or cfg.env.hold_prob > 0:
                            scales = "  ".join(
                                f"{lab} {s:.2f}"
                                for lab, s in zip(
                                    self.env.goal_labels,
                                    self.env.hold_scale,
                                    strict=True,
                                )
                            )
                            logger.info(f"hold start perturbation (rad): {scales}")
                if step % cfg.checkpoint_every == 0:
                    self.save(cfg.checkpoint_dir / f"tqc_{step:07d}.pt")
                    self.save(cfg.checkpoint_dir / "tqc_latest.pt", with_buffer=True)
        except KeyboardInterrupt:
            logger.info("\nTQC training interrupted.")
        finally:
            self.save(cfg.checkpoint_dir / "tqc_latest.pt", with_buffer=True)
        logger.info("TQC training complete.")
