# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A reinforcement learning project that trains a TQC policy to swing up and balance a double pendulum cartpole. The system consists of a cart (mass M) on a frictionless track with two pendulums in series — the goal is to apply horizontal forces to the cart to swing both poles from the hanging-down position to upright and hold them there.

The package name is `n_cartpole`. Physics are simulated with a Lagrangian-derived analytical model (no sympy at runtime — equations are hard-coded). Policy learning uses TQC (Truncated Quantile Critics — off-policy, distributional; Lee et al.'s algorithm) with a replay buffer. The gradient update runs on CPU by default because the networks are small; MPS/CUDA are available via `--device` but are slower for this workload (see "Device selection" below).

## Commands

Uses uv + a Taskfile (go-task). Python 3.13 required. Runtime deps in `[project.dependencies]`; dev tooling (pytest, ruff, mypy, pre-commit) in `[dependency-groups.dev]`.

```bash
task init                                       # uv sync
task train                                      # TQC, default settings (--links 2)
task train -- --links 1 --steps 300000          # single-link warm-up, custom args
task train -- --goals --steps 2000000           # goal-conditioned: all UU/DU/UD/DD transitions, one net
task eval-transitions -- --checkpoint checkpoints/double/tqc-goal/tqc_latest.pt  # start×goal success matrix
task play -- --checkpoint checkpoints/double/tqc-goal/tqc_latest.pt --start DD --goals "UU@0,DU@6,DD@12"
task play -- --checkpoint checkpoints/double/tqc/tqc_latest.pt   # interactive HTML replay (auto link count)
task plot -- --csv checkpoints/double/tqc/metrics.csv            # interactive training dashboard
task policy-map -- --checkpoint checkpoints/double/tqc/tqc_latest.pt  # input→output force/value control-surface map
task format                                     # ruff format + ruff check --fix + mypy
task test                                       # pytest with coverage over n_cartpole/
task ci                                         # format + test (local CI mirror)
task clean                                      # remove .venv, caches, checkpoints
```

The only learner is **TQC** (`scripts/train.py`, `training/off_policy.py`, `policy/tqc.py`). PPO was removed; `load_policy()` rejects old PPO checkpoints with a clear error. `--links N` (`n_cartpole/env/factory.py` builds the env and its obs dim/mirror-sign) defaults to 2 (double).

**Checkpoint naming and layout**: checkpoints live under `checkpoints/<single|double>/<tqc|tqc-goal>/`, chosen automatically from `--links`/`--goals` (override with `--checkpoint-dir`). Filenames are `tqc_latest.pt` and `tqc_NNNNNNN.pt`. Every checkpoint embeds its full `TQCConfig` (including `env.n_links`) plus an `"algo": "tqc"` key, so `policy/loader.py::load_policy()` and therefore `scripts/play.py` / `scripts/policy_map.py` reconstruct the right network shape (`n_cartpole/env/factory.py::env_spec()`) from the checkpoint alone — replay and the policy map work for any checkpoint regardless of which folder it's opened from. `policy/tqc.py`'s `SquashedGaussianActor`/`QuantileCritic` take an `obs_dim` override for this (default 8, for backward compatibility with pre-existing double-link checkpoints and `tests/test_tqc.py`).

**Publishing checkpoints**: `checkpoints/` is gitignored — never commit `.pt` files, and nothing trains in CI (no GitHub Actions workflow does model training; `ci.yml` only runs lint/tests). Pretrained models are distributed via GitHub Releases, built entirely locally via Taskfile targets: `task package-models` zips whatever exists under `checkpoints/<single|double>/<tqc|tqc-goal>/` into `dist/{single,double}-{tqc,tqc-goal}.zip` (skipping combinations that haven't been trained — see the guard in `Taskfile.yml`), and `task release-models -- <tag>` runs that, then `git tag`/`git push`, then `gh release create <tag> dist/*.zip --generate-notes` (requires the `gh` CLI, authenticated). `task download-models [-- <tag>]` reverses it: `gh release download` + unzip back into `checkpoints/`. If you change the checkpoint directory convention (`scripts/train.py`'s `--checkpoint-dir` default), update `package-models`'s `for links in single double; for algo in tqc tqc-goal` loop in `Taskfile.yml` to match.

Run a single test:
```bash
uv run pytest tests/test_dynamics.py::test_energy_conservation -v
```

## Architecture

```
n_cartpole/
  env/
    dynamics.py         — GENERAL n-link mass_matrix(theta), rhs(), step() with friction — pure numpy + scipy; PhysicsParams (per-link masses/lengths/joint_friction as scalar-or-sequence LinkSpec)
    cartpole.py         — NPendulumCartpole gymnasium.Env + EnvConfig + obs_dim()/obs_mirror_sign()/encode_obs(); one env parameterized by n_links
    double_cartpole.py  — thin back-compat shim: DoublePendulumCartpole(NPendulumCartpole), OBS_MIRROR_SIGN, re-exports EnvConfig
    single_cartpole.py  — thin back-compat shim: SinglePendulumCartpole(NPendulumCartpole), OBS_MIRROR_SIGN
    single_dynamics.py  — thin shim: re-exports dynamics.py + scalar-theta mass_matrix(theta1)
    factory.py           — make_env()/env_spec()/links_name()/checkpoint_subdir(): build the n-link env + spec from EnvConfig
    goals.py             — goal_configs()/goal_labels()/goal_index()/parse_goal_schedule() + TransitionStats (curriculum + success matrix)
  policy/
    running_norm.py     — RunningNorm (Welford) observation normalizer
    simba.py            — SimbaNet residual MLP backbone used by the TQC actor/critics
    tqc.py              — SquashedGaussianActor, QuantileCritic, quantile_huber_loss (TQC)
  training/
    off_policy.py       — TQC ReplayBuffer + TQCTrainer + _resolve_device() (off-policy loop, symmetric VER, goal relabeling)
    evaluate.py         — evaluate_transitions(): every (start, goal) pair → TransitionStats
  policy/
    loader.py           — load_policy(): TQC checkpoint → uniform action/value bundle
  viz/
    style.py            — shared Plotly palette (validated, CVD-safe) + layout template
    animate.py          — animate_episode(): interactive HTML replay + telemetry sidebar
    plots.py            — plot_training_curves(): interactive training dashboard from metrics.csv
    policy_map.py       — plot_policy_map(): input→output force/value contour phase-portrait
scripts/
  train.py              — TQC CLI: argparse → TQCTrainer.train()
  play.py               — load checkpoint → interactive HTML replay
  plot_returns.py       — metrics.csv/returns.csv → interactive training dashboard (task plot)
  policy_map.py         — checkpoint → input→output control-surface map (task policy-map)
  eval_transitions.py   — goal-conditioned checkpoint → start×goal success matrix (task eval-transitions)
tests/
  test_dynamics.py      — energy conservation (frictionless), friction dissipation, equilibria, mass matrix PD (2-link)
  test_env.py           — gym API contract, obs shape, bounded reward, termination (double)
  test_single.py        — single-link env/dynamics/factory
  test_n_links.py       — n = 3, 4: energy conservation, PD mass matrix, gym contract, reward bounds; PhysicsParams broadcasting + legacy-checkpoint upgrade
  test_tqc.py           — actor/critic shapes, quantile loss, TQC update + save/resume
  test_goals.py         — goal conditioning: obs layout, reward/relabel parity, segments, curriculum, trainers, play
```

**n links**: The env, dynamics, obs encoding, mirror-symmetry, and reward are all generalized to an arbitrary number of pendulum links, selected by `EnvConfig.n_links` (`--links 1|2|3|4|…`; anything ≥1 works). `dynamics.py` builds the `(n+1)×(n+1)` mass matrix and `rhs` from closed forms (absolute angles, so the velocity coupling is purely centrifugal), verified term-by-term against the old hand-derived single/double models and symbolically for n up to 4. Checkpoints go under `checkpoints/<single|double|triple|quadruple|Nlink>/<tqc|tqc-goal>/` via `factory.links_name()`. Per-link physics (`PhysicsParams.masses/lengths/joint_friction`) accept a scalar (shared across links) or a per-link sequence; a shorter sequence is extended by repeating its last value, so `--links 3` just works with the 2-link defaults.

**State vs. observation**: The internal physics state is `[x, ẋ, θ₁, θ̇₁, …, θₙ, θ̇ₙ]` (`2 + 2n`-D, angles in radians, 0 = upright). The observation fed to the net is `[x, ẋ, cos θ₁, sin θ₁, θ̇₁, …]` (`2 + 3n`-D) to remove the angle discontinuity at ±π. So the double link is 6D state / 8D obs, single is 4D/5D, triple is 8D/11D.

**Angle convention**: θ = 0 means upright; θ = π means hanging straight down. Training starts from a small random perturbation of the all-down state (`θᵢ = π` for every link). The bounded reward's angle factor is a **product** over links of `0.5 + 0.5·cos θᵢ`, peaking at 1.0 only when every pole is upright.

**Laptop defaults / speed**: The gradient update runs on CPU (`device="auto"` → CPU); forcing `--device mps` is much slower for this tiny network (see "Device selection" below).

**Goal-conditioned transitions** (`EnvConfig.goal_conditioned`, `--goals`): one network moves between every up/down configuration — for 2 links `UU`, `DU` (link 1 down, link 2 up), `UD`, `DD` (labels are one `U`/`D` per link, base link first; `goals.py`). Only `DD` is passively stable; the rest must be actively balanced. What changes when it's on:
- **Obs** gains `n` dims, `cos` of each link's target angle (±1), placed *between* the kinematics and the sysID block: `[kinematics, goal, sysID]`. Mirror sign +1 (targets are 0/π). Sensor noise is never applied to the goal.
- **Reward** uses `cos(θᵢ − θᵢ*)` instead of `cos θᵢ`; every other factor (velocity/cart gates, shaping) is unchanged, so holding DD means damping the swing. `goal_reward()` is the vectorized form and must stay in lockstep with `_compute_reward` (`test_goal_reward_matches_env_reward` guards this).
- **Goal switches** (every `goal_hold_steps`, default 400–800 steps, or `env.set_goal("DU")`) recompute `_prev_potential` under the new goal so the switch itself gives no shaping spike. Each finished goal period is a *segment* reported as `info["segment"] = (from, to, success)` (success = settled within `goal_tol_*` for `goal_settle_steps`); `from = -1` means it didn't start settled at an equilibrium.
- **Curriculum**: the env oversamples transitions with a low success EMA (`goal_curriculum`). The stats live per env instance.
- **TQC relabeling** (`TQCConfig.relabel_frac`): the buffer also stores raw `state/next_state/force/dforce` so a fraction of each batch gets a random goal with its reward recomputed — dynamics don't depend on the goal, so every transition trains every goal. Old non-goal buffers can't be loaded into a goal run.
- **Episodes** default to 3000 steps with `--goals`; checkpoints go to `checkpoints/<links>/<algo>-goal/`. `load_policy()` exposes `bundle.goal_conditioned`/`bundle.env_config`, and `bundle.encode(states, goal=...)` needs the goal. Old checkpoints (no goal fields in their pickled `EnvConfig`) load as non-goal via class-attribute defaults.

## Things that will bite you

**Device selection**: `_resolve_device()` maps `device="auto"` to **CPU**, not MPS. This is deliberate: the actor/critic are small and minibatches are small, so GPU per-op dispatch overhead outweighs the math (measured ~3x slower on Apple Silicon MPS). Keep `auto`/`cpu` unless you substantially enlarge the network, in which case `--device mps`/`--device cuda` may pay off. Environment stepping always runs on CPU regardless.

**Angle wrapping**: The internal state angles are not wrapped — they can accumulate beyond ±π during long episodes with rapid spinning. This is intentional: the RK45 integrator handles it correctly and the cos/sin observation encoding is already periodic. Do not wrap angles in dynamics.py or you'll break energy conservation.

**Mass matrix near singular configurations**: The `(n+1)×(n+1)` mass matrix M(θ) is guaranteed positive definite for any θ (provable from the kinetic energy). However, numerical conditioning degrades when adjacent links align (θᵢ ≈ θⱼ), and worsens as n grows. `np.linalg.solve` handles this fine for the default parameters up to a few links; if you change params significantly or push n higher, verify `np.linalg.cond(M)` stays below ~1e6. If you edit `dynamics.py`, re-run the symbolic cross-check (see `tests/test_n_links.py` energy-conservation tests, which catch any term error via drift).

**Torch version and MPS**: MPS support requires PyTorch ≥ 2.1. Not all ops are MPS-supported; the code uses `.to(device)` before ops and falls back to CPU automatically when `torch.backends.mps.is_available()` returns False.
