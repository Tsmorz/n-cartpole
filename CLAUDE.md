# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A reinforcement learning project that trains a PPO policy to swing up and balance a double pendulum cartpole. The system consists of a cart (mass M) on a frictionless track with two pendulums in series — the goal is to apply horizontal forces to the cart to swing both poles from the hanging-down position to upright and hold them there.

The package name is `n_cartpole`. Physics are simulated with a Lagrangian-derived analytical model (no sympy at runtime — equations are hard-coded). Policy learning uses PPO with GAE and parallel rollout collection on multiple CPU workers. The gradient update runs on CPU by default because the networks are tiny; MPS/CUDA are available via `--device` but are slower for this workload (see "Device selection" below).

## Commands

Uses uv + a Taskfile (go-task). Python 3.13 required. Runtime deps in `[project.dependencies]`; dev tooling (pytest, ruff, mypy, pre-commit) in `[dependency-groups.dev]`.

```bash
task init                                # uv sync
task train                               # PPO (on-policy), default settings
task train -- --workers 6 --steps 500   # PPO with custom args
task train-tqc                           # TQC (off-policy, distributional)
task train-tqc -- --steps 300000         # TQC with custom args
task play -- --checkpoint checkpoints/latest.pt  # interactive HTML replay (auto PPO/TQC)
task plot                                # interactive training dashboard (metrics.csv)
task policy-map                          # input→output force/value control-surface map
task format                              # ruff format + ruff check --fix + mypy
task test                                # pytest with coverage over n_cartpole/
task ci                                  # format + test (local CI mirror)
task clean                               # remove .venv, caches, checkpoints
```

Two learners share the environment: **PPO** (`scripts/train.py`, `training/trainer.py`, `policy/ppo.py`) and **TQC** (`scripts/train_tqc.py`, `training/off_policy.py`, `policy/tqc.py`). TQC is off-policy and sample-efficient (Lee et al.'s algorithm); PPO is the on-policy baseline. Neither replaces the other — the PPO path is kept intact.

Run a single test:
```bash
uv run pytest tests/test_dynamics.py::test_energy_conservation -v
```

## Architecture

```
n_cartpole/
  env/
    dynamics.py         — mass_matrix(), rhs() with friction, step() — pure numpy + scipy
    double_cartpole.py  — gymnasium.Env; obs encoding, bounded reward, diverse reset, OBS_MIRROR_SIGN
  policy/
    actor_critic.py     — PPO Actor, Critic MLPs + RunningNorm observation normalizer
    ppo.py              — compute_gae(), ppo_update() (return-norm + target_kl) — stateless
    tqc.py              — SquashedGaussianActor, QuantileCritic, quantile_huber_loss (TQC)
  training/
    rollout.py          — rollout_worker() for torch.multiprocessing spawn workers (PPO)
    trainer.py          — PPO Trainer + _resolve_device() + _augment_symmetry()
    off_policy.py       — TQC ReplayBuffer + TQCTrainer (off-policy loop, symmetric VER)
  policy/
    loader.py           — load_policy(): PPO/TQC checkpoint → uniform action/value bundle
  viz/
    style.py            — shared Plotly palette (validated, CVD-safe) + layout template
    animate.py          — animate_episode(): interactive HTML replay + telemetry sidebar
    plots.py            — plot_training_curves(): interactive training dashboard from metrics.csv
    policy_map.py       — plot_policy_map(): input→output force/value contour phase-portrait
scripts/
  train.py              — PPO CLI: argparse → Trainer.train()
  train_tqc.py          — TQC CLI: argparse → TQCTrainer.train()
  play.py               — load checkpoint (PPO or TQC) → interactive HTML replay
  plot_returns.py       — metrics.csv/returns.csv → interactive training dashboard (task plot)
  policy_map.py         — checkpoint → input→output control-surface map (task policy-map)
tests/
  test_dynamics.py      — energy conservation (frictionless), friction dissipation, equilibria, mass matrix PD
  test_env.py           — gym API contract, obs shape, bounded reward, termination
  test_ppo.py           — GAE formula, PPO losses finite, advantage normalization
  test_training.py      — full PPO loop smoke test
```

**State vs. observation**: The internal physics state is `[x, ẋ, θ₁, θ̇₁, θ₂, θ̇₂]` (6D, angles in radians, 0 = upright). The observation fed to the neural net is `[x, ẋ, cos θ₁, sin θ₁, θ̇₁, cos θ₂, sin θ₂, θ̇₂]` (8D) to remove the angle discontinuity at ±π.

**Angle convention**: θ = 0 means upright; θ = π means hanging straight down. Starting condition for training is a small random perturbation from `[0, 0, π, 0, π, 0]` (both poles down). Reward `cos θ₁ + cos θ₂` peaks at 2.0 (both upright) and bottoms at -2.0 (both down).

**PPO hyperparameters**: γ=0.99, λ=0.95, ε=0.2, LR=3e-4, 10 epochs/rollout, 2048 steps/worker, minibatch 512, grad clip 0.5, entropy coeff 0.01 → 0.001, hidden 128. Defaults live in `TrainingConfig` and are overridable from `scripts/train.py` (`--workers`, `--steps`, `--iterations`, `--lr`, `--hidden`, `--mini-batch`, `--device`).

**Return normalization** (`normalize_returns`, default on): the value loss and its clip are computed in units of the running return std (`Trainer.ret_norm`), so `value_clip_eps=0.2` means "0.2 return-stds" rather than "0.2 raw reward". Without it, the fixed 0.2 clip throttles the critic because raw returns are O(100s). `target_kl=0.03` early-stops the epoch loop so the policy doesn't over-update once the critic is no longer suppressing actor gradients through the shared grad-norm clip.

**Laptop defaults / speed**: The gradient update runs on CPU (`device="auto"` → CPU) and worker count defaults to `min(8, cpu_count-2)` to leave the machine responsive. On Apple Silicon a full run is a few minutes on CPU; forcing `--device mps` is much slower for this tiny network (see "Device selection" below).

## Reward scale (the deep one)

The reward `cos θ₁ + cos θ₂ − 0.1|x| − 0.001F²` is **additive and unbounded below**, so returns are O(±100s–1000s). That poor conditioning is the root cause of the value-clip/critic issues that `normalize_returns` patches around. Lee et al.'s multi-pendulum RL papers (in `docs/`) instead use a **bounded, multiplicative** reward — a product of terms each in ~[0,1] covering upright alignment (product over links), a cart-centering term, an angular-velocity term, and a mild action term — giving per-step reward in (0,1] and always-positive, well-scaled returns. Adopting that style would fix value scaling at the source and add the currently-missing velocity penalty (needed to *balance* rather than spin through upright). See the two PDFs in `docs/` for the exact forms.

## Things that will bite you

**Device selection**: `_resolve_device()` maps `device="auto"` to **CPU**, not MPS. This is deliberate: the actor/critic are tiny (8→64→64) and batches are small, so GPU per-op dispatch overhead outweighs the math. Measured on Apple Silicon, the PPO update is ~3x slower on MPS than CPU, and `compute_gae` is ~250x slower on MPS because its Python loop reads scalars back with `.item()` every step, forcing a device sync each time. Keep `auto`/`cpu` unless you substantially enlarge the network, in which case `--device mps`/`--device cuda` may pay off (and consider vectorizing `compute_gae` first). Rollout workers always run on CPU regardless.

**MPS and multiprocessing**: MPS (Apple Silicon GPU) cannot be used inside spawned worker processes — the workers always run on CPU. Only the main process (PPO update) could use MPS, and only if explicitly requested via `--device mps`. Don't try to move tensors to MPS inside `rollout_worker`.

**Angle wrapping**: The internal state angles are not wrapped — they can accumulate beyond ±π during long episodes with rapid spinning. This is intentional: the RK45 integrator handles it correctly and the cos/sin observation encoding is already periodic. Do not wrap angles in dynamics.py or you'll break energy conservation.

**Mass matrix near singular configurations**: The mass matrix M(θ) is guaranteed positive definite for any θ (provable from the kinetic energy). However, numerical conditioning degrades when θ₁ ≈ θ₂ (both poles aligned). `np.linalg.solve` handles this fine for the default parameters; if you change params significantly, verify `np.linalg.cond(M)` stays below ~1e6.

**Multiprocessing on Mac**: `scripts/train.py` uses `if __name__ == "__main__":` guard — required for torch.multiprocessing spawn context on macOS. If you move the `Trainer.train()` call outside this guard, workers will spawn infinitely.

**Torch version and MPS**: MPS support requires PyTorch ≥ 2.1. Not all ops are MPS-supported; the code uses `.to(device)` before ops and falls back to CPU automatically when `torch.backends.mps.is_available()` returns False.
