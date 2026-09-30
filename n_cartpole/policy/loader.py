"""Load a trained TQC checkpoint into a uniform, callable bundle.

Both ``scripts/play.py`` (episode replay) and ``scripts/policy_map.py`` (the
input→output map) need the same thing: given a batch of *normalized* observations,
return the policy's deterministic action and — where available — the critic's
value estimate. This wraps the TQC actor/critic behind one small interface.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from n_cartpole.env.cartpole import EnvConfig
from n_cartpole.env.dynamics import PhysicsParams
from n_cartpole.env.factory import env_spec
from n_cartpole.env.goals import goal_configs, goal_index
from n_cartpole.policy.running_norm import RunningNorm
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
    # The env config the policy was trained with (obs layout, goal mode, sysID).
    env_config: EnvConfig = field(default_factory=EnvConfig)

    @property
    def goal_conditioned(self) -> bool:
        """Whether the policy takes a target configuration as input."""
        return bool(getattr(self.env_config, "goal_conditioned", False))

    def encode(
        self, raw_states: np.ndarray, goal: int | str | None = None
    ) -> np.ndarray:
        """Encode raw physics states ``(B, 2n+2)`` to the cos/sin observation.

        For goal-conditioned policies ``goal`` (index or ``U``/``D`` label) is
        required and appended as the ``cos`` of each link's target angle.
        """
        s = np.atleast_2d(raw_states)
        x, xd = s[:, 0], s[:, 1]
        cols = [x, xd]
        for link in range(self.n_links):
            th, thd = s[:, 2 + 2 * link], s[:, 3 + 2 * link]
            cols += [np.cos(th), np.sin(th), thd]
        if self.goal_conditioned:
            if goal is None:
                raise ValueError("goal-conditioned policy: encode() needs a goal")
            target = goal_configs(self.n_links)[goal_index(goal, self.n_links)]
            cols += [np.full(len(s), np.cos(t)) for t in target]
        return np.stack(cols, axis=1).astype(np.float32)

    def normalize(self, obs: np.ndarray) -> torch.Tensor:
        """Encode is separate; this only applies the running normalizer."""
        return self.norm.normalize(torch.from_numpy(np.asarray(obs, np.float32)))


def load_policy(checkpoint: str | Path, device: str = "cpu") -> PolicyBundle:
    """Load a TQC checkpoint into a :class:`PolicyBundle`."""
    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    dev = torch.device(device)
    try:
        ckpt = torch.load(path, map_location=dev, weights_only=False)
    except ModuleNotFoundError as e:
        # PPO checkpoints pickle a TrainingConfig from the removed trainer module.
        raise ValueError(
            f"{path} is not a TQC checkpoint (PPO support was removed)"
        ) from e
    algo = ckpt.get("algo", "ppo")
    if algo != "tqc":
        raise ValueError(
            f"{path} is a {algo.upper()} checkpoint; only TQC is supported"
        )
    cfg = ckpt.get("cfg")
    hidden = cfg.hidden if cfg is not None else 128
    physics = cfg.env.physics if cfg is not None else PhysicsParams()
    n_links = getattr(cfg.env, "n_links", 2) if cfg is not None else 2
    env_config = cfg.env if cfg is not None else EnvConfig(n_links=n_links)
    obs_dim, _ = env_spec(env_config)

    norm = RunningNorm(obs_dim)
    norm.load_state_dict(ckpt["norm"])

    value: ValueFn | None = None
    tqc_actor = SquashedGaussianActor(
        hidden=hidden, force_max=physics.force_max, obs_dim=obs_dim
    )
    tqc_actor.load_state_dict(ckpt["actor"])
    tqc_actor.eval()

    def select_action(obs_norm: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            return tqc_actor.act(obs_norm, deterministic=True).cpu().numpy()

    if "critic" in ckpt:
        tqc_critic = QuantileCritic(
            hidden=hidden,
            n_critics=getattr(cfg, "n_critics", 2),
            n_quantiles=getattr(cfg, "n_quantiles", 25),
            obs_dim=obs_dim,
        )
        tqc_critic.load_state_dict(ckpt["critic"])
        tqc_critic.eval()

        def value(obs_norm: torch.Tensor) -> np.ndarray:
            with torch.no_grad():
                act = tqc_actor.act(obs_norm, deterministic=True)
                atoms = tqc_critic(obs_norm, act)  # (B, n_critics, n_quantiles)
                return atoms.mean(dim=(1, 2)).cpu().numpy()

    return PolicyBundle(
        algo=algo,
        n_links=n_links,
        obs_dim=obs_dim,
        hidden=hidden,
        physics=physics,
        norm=norm,
        select_action=select_action,
        value=value,
        env_config=env_config,
    )
