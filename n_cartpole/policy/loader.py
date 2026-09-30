"""Load a trained checkpoint (PPO or TQC) into a uniform, callable bundle.

Both ``scripts/play.py`` (episode replay) and ``scripts/policy_map.py`` (the
input→output map) need the same thing: given a batch of *normalized* observations,
return the policy's deterministic action and — where available — the critic's
value estimate. This hides the PPO/TQC differences behind one small interface.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from n_cartpole.env.dynamics import PhysicsParams
from n_cartpole.env.factory import env_spec
from n_cartpole.policy.actor_critic import Actor, Critic, RunningNorm
from n_cartpole.policy.tqc import QuantileCritic, SquashedGaussianActor

ActionFn = Callable[[torch.Tensor], np.ndarray]
ValueFn = Callable[[torch.Tensor], np.ndarray]


@dataclass
class PolicyBundle:
    """A loaded policy with a uniform action/value interface.

    ``select_action`` and ``value`` both take a batch of *normalized* observations
    (a ``(B, obs_dim)`` tensor) and return numpy arrays. ``value`` is ``None`` when
    the checkpoint carries no critic.
    """

    algo: str
    n_links: int
    obs_dim: int
    hidden: int
    physics: PhysicsParams
    norm: RunningNorm
    select_action: ActionFn
    value: ValueFn | None

    def encode(self, raw_states: np.ndarray) -> np.ndarray:
        """Encode raw physics states ``(B, 2n+2)`` to the cos/sin observation."""
        s = np.atleast_2d(raw_states)
        x, xd = s[:, 0], s[:, 1]
        cols = [x, xd]
        for link in range(self.n_links):
            th, thd = s[:, 2 + 2 * link], s[:, 3 + 2 * link]
            cols += [np.cos(th), np.sin(th), thd]
        return np.stack(cols, axis=1).astype(np.float32)

    def normalize(self, obs: np.ndarray) -> torch.Tensor:
        """Encode is separate; this only applies the running normalizer."""
        return self.norm.normalize(torch.from_numpy(np.asarray(obs, np.float32)))


def load_policy(checkpoint: str | Path, device: str = "cpu") -> PolicyBundle:
    """Load a PPO or TQC checkpoint into a :class:`PolicyBundle`."""
    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    dev = torch.device(device)
    ckpt = torch.load(path, map_location=dev, weights_only=False)
    cfg = ckpt.get("cfg")
    algo = ckpt.get("algo", "ppo")
    hidden = cfg.hidden if cfg is not None else 128
    physics = cfg.env.physics if cfg is not None else PhysicsParams()
    n_links = getattr(cfg.env, "n_links", 2) if cfg is not None else 2
    obs_dim, _ = env_spec(n_links)

    norm = RunningNorm(obs_dim)
    norm.load_state_dict(ckpt["norm"])

    value: ValueFn | None = None
    if algo == "tqc":
        tqc_actor = SquashedGaussianActor(
            hidden=hidden, force_max=physics.force_max, obs_dim=obs_dim
        )
        tqc_actor.load_state_dict(ckpt["actor"])
        tqc_actor.eval()

        def select_action(obs_norm: torch.Tensor) -> np.ndarray:
            with torch.no_grad():
                return tqc_actor.act(obs_norm, deterministic=True).cpu().numpy()

        if "critic" in ckpt:
            tqc_critic = QuantileCritic(hidden=hidden, obs_dim=obs_dim)
            tqc_critic.load_state_dict(ckpt["critic"])
            tqc_critic.eval()

            def value(obs_norm: torch.Tensor) -> np.ndarray:
                with torch.no_grad():
                    act = tqc_actor.act(obs_norm, deterministic=True)
                    atoms = tqc_critic(obs_norm, act)  # (B, n_critics, n_quantiles)
                    return atoms.mean(dim=(1, 2)).cpu().numpy()

    else:
        ppo_actor = Actor(hidden=hidden, obs_dim=obs_dim)
        ppo_actor.load_state_dict(ckpt["actor"])
        ppo_actor.eval()

        def select_action(obs_norm: torch.Tensor) -> np.ndarray:
            with torch.no_grad():
                mean, _ = ppo_actor(obs_norm)  # deterministic: the distribution mean
                return mean.cpu().numpy()

        if "critic" in ckpt:
            ppo_critic = Critic(hidden=hidden, obs_dim=obs_dim)
            ppo_critic.load_state_dict(ckpt["critic"])
            ppo_critic.eval()

            def value(obs_norm: torch.Tensor) -> np.ndarray:
                with torch.no_grad():
                    return ppo_critic(obs_norm).reshape(-1).cpu().numpy()

    return PolicyBundle(
        algo=algo,
        n_links=n_links,
        obs_dim=obs_dim,
        hidden=hidden,
        physics=physics,
        norm=norm,
        select_action=select_action,
        value=value,
    )
