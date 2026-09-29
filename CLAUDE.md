# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A reinforcement learning project that trains a PPO policy to swing up and balance a double pendulum cartpole. The system consists of a cart (mass M) on a frictionless track with two pendulums in series — the goal is to apply horizontal forces to the cart to swing both poles from the hanging-down position to upright and hold them there.

The package name is `n_cartpole`. Physics are simulated with a Lagrangian-derived analytical model (no sympy at runtime — equations are hard-coded). Policy learning uses PPO with GAE and parallel rollout collection on multiple CPU workers, with gradient updates on MPS (Apple Silicon), CUDA, or CPU depending on hardware.

## Commands

Uses uv + a Taskfile (go-task). Python 3.13 required. Runtime deps in `[project.dependencies]`; dev tooling (pytest, ruff, mypy, pre-commit) in `[dependency-groups.dev]`.

```bash
task init                                # uv sync
task train                               # train with default settings
task train -- --workers 6 --steps 500   # train with custom args
task play -- --checkpoint checkpoints/latest.pt  # visualize a trained policy
task format                              # ruff format + ruff check --fix + mypy
task test                                # pytest with coverage over n_cartpole/
task ci                                  # format + test (local CI mirror)
task clean                               # remove .venv, caches, checkpoints
```

Run a single test:
```bash
uv run pytest tests/test_dynamics.py::test_energy_conservation -v
```

## Architecture

```
n_cartpole/
  env/
    dynamics.py         — mass_matrix(), coriolis_and_gravity(), step() — pure numpy
    double_cartpole.py  — gymnasium.Env wrapping dynamics; obs encoding, reward, done
  policy/
    actor_critic.py     — Actor, Critic MLPs + RunningNorm observation normalizer
    ppo.py              — compute_gae(), ppo_update() — stateless functions
  training/
    rollout.py          — rollout_worker() for torch.multiprocessing spawn workers
    trainer.py          — Trainer class: broadcasts weights, aggregates trajectories, PPO update
  viz/
    animate.py          — animate_episode() using matplotlib FuncAnimation
scripts/
  train.py              — CLI entry point: argparse → Trainer.train()
  play.py               — CLI entry point: load checkpoint → animate_episode()
tests/
  test_dynamics.py      — energy conservation, equilibrium, mass matrix PD
  test_env.py           — gym API contract, obs shape, reward bounds, termination
  test_ppo.py           — GAE formula, PPO losses finite, advantage normalization
```

**State vs. observation**: The internal physics state is `[x, ẋ, θ₁, θ̇₁, θ₂, θ̇₂]` (6D, angles in radians, 0 = upright). The observation fed to the neural net is `[x, ẋ, cos θ₁, sin θ₁, θ̇₁, cos θ₂, sin θ₂, θ̇₂]` (8D) to remove the angle discontinuity at ±π.

**Angle convention**: θ = 0 means upright; θ = π means hanging straight down. Starting condition for training is a small random perturbation from `[0, 0, π, 0, π, 0]` (both poles down). Reward `cos θ₁ + cos θ₂` peaks at 2.0 (both upright) and bottoms at -2.0 (both down).

**PPO hyperparameters**: γ=0.99, λ=0.95, ε=0.2, LR=3e-4, 10 epochs/rollout, 2048 steps/worker, grad clip 0.5, entropy coeff 0.01 → 0.001.

## Things that will bite you

**MPS and multiprocessing**: MPS (Apple Silicon GPU) cannot be used inside spawned worker processes — the workers always run on CPU. Only the main process (PPO update) uses MPS. Don't try to move tensors to MPS inside `rollout_worker`.

**Angle wrapping**: The internal state angles are not wrapped — they can accumulate beyond ±π during long episodes with rapid spinning. This is intentional: the RK45 integrator handles it correctly and the cos/sin observation encoding is already periodic. Do not wrap angles in dynamics.py or you'll break energy conservation.

**Mass matrix near singular configurations**: The mass matrix M(θ) is guaranteed positive definite for any θ (provable from the kinetic energy). However, numerical conditioning degrades when θ₁ ≈ θ₂ (both poles aligned). `np.linalg.solve` handles this fine for the default parameters; if you change params significantly, verify `np.linalg.cond(M)` stays below ~1e6.

**Multiprocessing on Mac**: `scripts/train.py` uses `if __name__ == "__main__":` guard — required for torch.multiprocessing spawn context on macOS. If you move the `Trainer.train()` call outside this guard, workers will spawn infinitely.

**Torch version and MPS**: MPS support requires PyTorch ≥ 2.1. Not all ops are MPS-supported; the code uses `.to(device)` before ops and falls back to CPU automatically when `torch.backends.mps.is_available()` returns False.
