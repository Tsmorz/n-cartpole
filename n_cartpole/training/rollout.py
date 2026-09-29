"""Parallel rollout worker for collecting trajectories on CPU.

Each worker runs inside a separate process (torch.multiprocessing spawn context).
Workers always use CPU — only the main process moves tensors to MPS/CUDA for updates.

Communication protocol:
  command_queue  <-- main sends actor+critic state_dicts or 'stop'
  result_queue   --> worker sends RolloutResult or exception string
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass

import numpy as np
import torch
import torch.multiprocessing as mp

from n_cartpole.env.double_cartpole import DoublePendulumCartpole, EnvConfig
from n_cartpole.policy.actor_critic import Actor, Critic


@dataclass
class RolloutResult:
    """Collected trajectory from one worker."""

    obs: np.ndarray  # (T, 8)
    actions: np.ndarray  # (T, 1)
    log_probs: np.ndarray  # (T,)
    rewards: np.ndarray  # (T,)
    values: np.ndarray  # (T,)
    dones: np.ndarray  # (T,) bool
    last_value: float  # bootstrap V(s_{T+1})
    episode_return: float  # sum of rewards for the most recently completed episode


def rollout_worker(
    worker_id: int,
    command_queue: mp.Queue,
    result_queue: mp.Queue,
    env_config: EnvConfig,
    n_steps: int,
    hidden: int,
) -> None:
    """Worker process: collect rollouts and send results back.

    Runs on CPU only. Actor and Critic are reconstructed from state dicts
    broadcast by the main process each iteration.
    """
    device = torch.device("cpu")
    env = DoublePendulumCartpole(env_config)
    actor = Actor(hidden=hidden).to(device)
    critic = Critic(hidden=hidden).to(device)

    obs_arr = np.zeros((n_steps, 8), dtype=np.float32)
    act_arr = np.zeros((n_steps, 1), dtype=np.float32)
    lp_arr = np.zeros(n_steps, dtype=np.float32)
    rew_arr = np.zeros(n_steps, dtype=np.float32)
    val_arr = np.zeros(n_steps, dtype=np.float32)
    done_arr = np.zeros(n_steps, dtype=bool)

    obs, _ = env.reset()
    episode_return = 0.0
    completed_return = 0.0

    try:
        while True:
            msg = command_queue.get()
            if msg == "stop":
                break

            actor_sd, critic_sd = msg
            actor.load_state_dict(actor_sd)
            critic.load_state_dict(critic_sd)
            actor.eval()
            critic.eval()

            force_max = env_config.physics.force_max

            with torch.no_grad():
                for t in range(n_steps):
                    obs_t = torch.from_numpy(obs).unsqueeze(0)
                    action, log_prob = actor.sample_action(obs_t, force_max)
                    value = critic(obs_t)

                    act_np = action.squeeze(0).numpy()
                    obs_arr[t] = obs
                    act_arr[t] = act_np
                    lp_arr[t] = log_prob.item()
                    val_arr[t] = value.item()

                    next_obs, reward, terminated, truncated, _ = env.step(act_np)
                    rew_arr[t] = reward
                    done = terminated or truncated
                    done_arr[t] = done

                    episode_return += reward
                    obs = next_obs

                    if done:
                        completed_return = episode_return
                        episode_return = 0.0
                        obs, _ = env.reset()

                # Bootstrap value for last state
                obs_t = torch.from_numpy(obs).unsqueeze(0)
                last_value = critic(obs_t).item()

            result_queue.put(
                RolloutResult(
                    obs=obs_arr.copy(),
                    actions=act_arr.copy(),
                    log_probs=lp_arr.copy(),
                    rewards=rew_arr.copy(),
                    values=val_arr.copy(),
                    dones=done_arr.copy(),
                    last_value=last_value,
                    episode_return=completed_return,
                )
            )

    except Exception:
        result_queue.put(f"Worker {worker_id} error:\n{traceback.format_exc()}")
