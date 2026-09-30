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
task eval-transitions -- --checkpoint checkpoints/double/tqc-goal/tqc_latest.pt  # start×goal success matrix + settle time / hold force / peak force+speed / off-rail (--randomize: perturbed true plant)
task train -- --config config/rig.toml --randomize            # buildable reference rig (rigid-body links, delay, randomized true plant, reward shaping)
task play -- --checkpoint checkpoints/double/tqc-goal/tqc_latest.pt --start DD --goals "UU@0,DU@6,DD@12"
task play -- --checkpoint checkpoints/double/tqc/tqc_latest.pt   # interactive HTML replay (auto link count)
task plot -- --csv checkpoints/double/tqc/metrics.csv            # interactive training dashboard
task policy-map -- --checkpoint checkpoints/double/tqc/tqc_latest.pt  # input→output force/value control-surface map
task export-web                                 # swing-up actor → ../personal-site/assets/models/ (browser demo; see below)
task format                                     # ruff format + ruff check --fix + mypy
task test                                       # pytest with coverage over n_cartpole/
task ci                                         # format + test (local CI mirror)
task clean                                      # remove .venv, caches, checkpoints
```

The only learner is **TQC** (`scripts/train.py`, `training/off_policy.py`, `policy/tqc.py`). PPO was removed; `load_policy()` rejects old PPO checkpoints with a clear error. `--links N` (`n_cartpole/env/factory.py` builds the env and its obs dim/mirror-sign) defaults to 2 (double).

**Checkpoint naming and layout**: checkpoints live under `checkpoints/<single|double>/<tqc|tqc-goal>/`, chosen automatically from `--links`/`--goals` (override with `--checkpoint-dir`). Filenames are `tqc_latest.pt` and `tqc_NNNNNNN.pt`. Every checkpoint embeds its full `TQCConfig` (including `env.n_links`) plus an `"algo": "tqc"` key, so `policy/loader.py::load_policy()` and therefore `scripts/play.py` / `scripts/policy_map.py` reconstruct the right network shape (`n_cartpole/env/factory.py::env_spec()`) from the checkpoint alone — replay and the policy map work for any checkpoint regardless of which folder it's opened from. `policy/tqc.py`'s `SquashedGaussianActor`/`QuantileCritic` take an `obs_dim` override for this (default 8, for backward compatibility with pre-existing double-link checkpoints and `tests/test_tqc.py`).

**Publishing checkpoints**: `checkpoints/` is gitignored — never commit `.pt` files, and nothing trains in CI (no GitHub Actions workflow does model training; `ci.yml` only runs lint/tests). Pretrained models are distributed via GitHub Releases, built entirely locally via Taskfile targets: `task package-models` zips whatever exists under `checkpoints/<single|double>/<tqc|tqc-goal>/` into `dist/{single,double}-{tqc,tqc-goal}.zip` (skipping combinations that haven't been trained — see the guard in `Taskfile.yml`), and `task release-models -- <tag>` runs that, then `git tag`/`git push`, then `gh release create <tag> dist/*.zip --generate-notes` (requires the `gh` CLI, authenticated). `task download-models [-- <tag>]` reverses it: `gh release download` + unzip back into `checkpoints/`. If you change the checkpoint directory convention (`scripts/train.py`'s `--checkpoint-dir` default), update `package-models`'s `for links in single double; for algo in tqc tqc-goal` loop in `Taskfile.yml` to match.

**Browser export** (`scripts/export_web.py`, `task export-web`): writes `swingup-tqc.{json,bin}` + `swingup-tqc-fixture.json` for the personal site's `/controls/swingup/` page, which runs the actor in vanilla JS. The JSON carries the checkpoint's `PhysicsParams`, `force_slew`, and success tolerances — the page reads physics from it, so the browser plant always matches training. The `.bin` is float16 with `RunningNorm` folded into `in_proj` and only the mean row of `out_proj`. Plain swing-up checkpoints only (no `--goals`, no `hardware`). If you change `dynamics.py`, the obs encoding, the SimBa block, or the actuator (slew/clip), port the change to `personal-site/assets/js/swingup.js` too and re-export — the site's `script/test-swingup` (net/physics parity + closed-loop swing-up against the fixture) is what catches drift.

Run a single test:
```bash
uv run pytest tests/test_dynamics.py::test_energy_conservation -v
```

## Architecture

```
n_cartpole/
  env/
    dynamics.py         — GENERAL n-link mass_matrix(theta), rhs(), step() with friction — pure numpy + scipy; PhysicsParams (per-link masses/lengths/com/inertia/joint_friction as scalar-or-sequence LinkSpec); link_from_parts(); batched total_energy_batch()
    randomization.py    — PlantRandomization: per-episode perturbation of the TRUE plant (masses, lengths, com, inertia, friction, force gain)
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
  eval_transitions.py   — goal-conditioned checkpoint → start×goal success matrix + speed/steadiness/effort matrices (task eval-transitions)
config/
  default.toml          — point-mass defaults
  rig.toml              — reference buildable rig: Al-bar links + tip assemblies (com/inertia derived via link_from_parts), 1 m rail (x_lim 0.40), 15 N belt drive, 1-step delay, [randomize], [reward]; loaded by `train.py --config`
  export_web.py         — swing-up checkpoint → float16 weights + physics JSON + parity fixture for the website (task export-web)
tests/
  test_dynamics.py      — energy conservation (frictionless), friction dissipation, equilibria, mass matrix PD (2-link)
  test_env.py           — gym API contract, obs shape, bounded reward, termination (double)
  test_single.py        — single-link env/dynamics/factory
  test_n_links.py       — n = 3, 4: energy conservation, PD mass matrix, gym contract, reward bounds; PhysicsParams broadcasting + legacy-checkpoint upgrade
  test_tqc.py           — actor/critic shapes, quantile loss, TQC update + save/resume
  test_goals.py         — goal conditioning: obs layout, reward/relabel parity, segments, curriculum, trainers, play, transition metrics
  test_rigid_links.py   — com/inertia links: exact parity with the point-mass model, finite-difference Euler–Lagrange check, energy conservation, fast-path (_ode) == rhs/mass_matrix
  test_rig.py           — rig.toml plausibility, link_from_parts, plant randomization (ranges, mirror symmetry), no-sysID obs layout
  test_reward_shape.py  — RewardShape: zero weights == original reward, energy potential (no DD dead zone), wall/effort terms, env↔goal_reward parity, relabel
```

**n links**: The env, dynamics, obs encoding, mirror-symmetry, and reward are all generalized to an arbitrary number of pendulum links, selected by `EnvConfig.n_links` (`--links 1|2|3|4|…`; anything ≥1 works). `dynamics.py` builds the `(n+1)×(n+1)` mass matrix and `rhs` from closed forms (absolute angles, so the velocity coupling is purely centrifugal), verified term-by-term against the old hand-derived single/double models and symbolically for n up to 4. Checkpoints go under `checkpoints/<single|double|triple|quadruple|Nlink>/<tqc|tqc-goal>/` via `factory.links_name()`. Per-link physics (`PhysicsParams.masses/lengths/joint_friction`) accept a scalar (shared across links) or a per-link sequence; a shorter sequence is extended by repeating its last value, so `--links 3` just works with the 2-link defaults.

**Rigid-body links and the reference rig**: `PhysicsParams.com` (joint→centre-of-mass, default = `lengths`) and `.inertia` (about the COM, default 0) make each link a rigid body; the defaults are the original point-mass bob. `dynamics.py` generalizes the closed forms with `a_k = m_k c_k + l_k Σ_{i>k} m_i` and `d_k = I_k + m_k c_k² + l_k² Σ_{i>k} m_i` (see its module docstring). **`_ode` (the integrator's hot path, coefficients precomputed once per `step()`) duplicates the equations in `rhs`/`mass_matrix`** — change all three together; `tests/test_rigid_links.py` pins them to each other, to the old point-mass formulas, and to a finite-difference Euler–Lagrange residual. `config/rig.toml` is the buildable reference rig (every number has a part-level comment; link mass/com/inertia are derived from bar dimensions + tip masses). `train.py --config` takes the *env* (physics, hardware pipeline, `[randomize]`, `[reward]`) from the TOML; training hyperparameters stay CLI flags, and checkpoints go to a `-<config name>` folder (e.g. `checkpoints/double/tqc-goal-rig/`) so they never overwrite the point-mass ones (web export + `package-models` only use `tqc`/`tqc-goal`).

**True plant vs nominal model**: `EnvConfig.randomize` perturbs the plant the dynamics integrate each episode (`env._plant`, `env._force_gain`), but rewards, energy shaping, `goal_reward`/replay relabeling, and the sysID context all use the *nominal* `cfg.physics` — so relabeled rewards stay consistent with the buffer. `HardwareConfig.sysid_context=False` keeps the action delay/sensor noise but leaves the observation purely kinematic (+ goal), which is what the rig config uses (the deployed actor needs no parameter inputs).

**Reward shaping** (`EnvConfig.reward_shape`, `RewardShape`; `None` = the original reward exactly): (1) an energy-gap potential, faded by `1 − r_angle`, fixes the zero-value/zero-slope dead zone of the angle-product potential at hanging-down for any goal with an upright link (the cause of the ~10 s idle on DD→UD/UU); (2) a rail-margin factor keeps transitions off the end-stops; (3) a hold-gated force penalty and a sharper velocity gate make holds quiet. All are mirror-even. `goal_reward()`/`_relabel` take the shape; the env keeps a second potential (`_prev_energy_potential`, re-anchored by `_reset_potentials()` on reset and every goal switch). Scripts that set `env._prev_potential` by hand (`eval_swingup.py`, `export_web.py`) are for the unshaped swing-up only.

**State vs. observation**: The internal physics state is `[x, ẋ, θ₁, θ̇₁, …, θₙ, θ̇ₙ]` (`2 + 2n`-D, angles in radians, 0 = upright). The observation fed to the net is `[x, ẋ, cos θ₁, sin θ₁, θ̇₁, …]` (`2 + 3n`-D) to remove the angle discontinuity at ±π. So the double link is 6D state / 8D obs, single is 4D/5D, triple is 8D/11D.

**Angle convention**: θ = 0 means upright; θ = π means hanging straight down. Training starts from a small random perturbation of the all-down state (`θᵢ = π` for every link). The bounded reward's angle factor is a **product** over links of `0.5 + 0.5·cos θᵢ`, peaking at 1.0 only when every pole is upright.

**Laptop defaults / speed**: The gradient update runs on CPU (`device="auto"` → CPU); forcing `--device mps` is much slower for this tiny network (see "Device selection" below).

**Goal-conditioned transitions** (`EnvConfig.goal_conditioned`, `--goals`): one network moves between every up/down configuration — for 2 links `UU`, `DU` (link 1 down, link 2 up), `UD`, `DD` (labels are one `U`/`D` per link, base link first; `goals.py`). Only `DD` is passively stable; the rest must be actively balanced. What changes when it's on:
- **Obs** gains `n` dims, `cos` of each link's target angle (±1), placed *between* the kinematics and the sysID block: `[kinematics, goal, sysID]`. Mirror sign +1 (targets are 0/π). Sensor noise is never applied to the goal.
- **Reward** uses `cos(θᵢ − θᵢ*)` instead of `cos θᵢ`; every other factor (velocity/cart gates, shaping) is unchanged, so holding DD means damping the swing. `goal_reward()` is the vectorized form and must stay in lockstep with `_compute_reward` (`test_goal_reward_matches_env_reward` guards this).
- **Goal switches** (every `goal_hold_steps`, default 400–800 steps, or `env.set_goal("DU")`) recompute `_prev_potential` under the new goal so the switch itself gives no shaping spike. Each finished goal period is a *segment* reported as `info["segment"] = (from, to, success)` (success = settled within `goal_tol_*` for `goal_settle_steps`); `from = -1` means it didn't start settled at an equilibrium.
- **Curriculum**: the env oversamples transitions with a low success EMA (`goal_curriculum`). The stats live per env instance.
- **Hold curriculum** (`EnvConfig.hold_prob`, `--hold-prob`; `TQCConfig.hold_phase_steps`, `--hold-phase-steps`): a hold reset starts *near* a random equilibrium and commands that same goal. The perturbation is per equilibrium (`env.hold_scale`, within `hold_noise`), ×1.05 after a held segment and ÷1.1 after a failure (~2/3 success at equilibrium), and is logged each `log_every`. During the hold phase the trainer sets `env.hold_only`: every unpinned reset is a hold and the episode truncates when that segment closes. Resets with `options={"start"/"goal"}` (eval, play) never become holds.
- **Warm start from swing-up** (`--init-from <plain ckpt>`, `TQCTrainer.warm_start`): copies actor/critics/norm/α from a non-goal checkpoint, inserting the goal input columns with zero weight so the net starts as the swing-up policy; network size/physics are taken from the source. Then `seed_buffer` (`--seed-steps`, default 100k with `--init-from`) fills the buffer with that policy's rollouts (raw states included, separate env so the curriculum isn't skewed), and relabeling spreads them to every goal. Recommended: `task train -- --goals --init-from checkpoints/double/tqc/tqc_latest.pt --hold-phase-steps 100000 --hold-prob 0.25 --steps 1000000`.
- **Goal switches in the buffer**: the stored `next_obs` keeps the *old* goal on a switch step (the reward was for it), so every stored transition bootstraps under a fixed goal, matching relabeled rows.
- **TQC relabeling** (`TQCConfig.relabel_frac`): the buffer also stores raw `state/next_state/force/dforce` so a fraction of each batch gets a random goal with its reward recomputed — dynamics don't depend on the goal, so every transition trains every goal. Old non-goal buffers can't be loaded into a goal run.
- **Episodes** default to 3000 steps with `--goals`; checkpoints go to `checkpoints/<links>/<algo>-goal/`. `load_policy()` exposes `bundle.goal_conditioned`/`bundle.env_config`, and `bundle.encode(states, goal=...)` needs the goal. Old checkpoints (no goal fields in their pickled `EnvConfig`) load as non-goal via class-attribute defaults.

## Things that will bite you

**Entropy target units**: the actor's `log_prob` is a density over force in newtons, not over the [-1, 1] action. `TQCConfig.target_entropy=None` therefore means `log(force_max) - 1` (≈2.0 for 20 N, SAC's -1 convention on the scaled action, ~1.8 N exploration std). The old hard-coded `-1` was in newtons (~0.09 N std) and let α collapse to ~1e-3. Resumed old checkpoints pick up the new default (pickled configs lack the field), so α will climb after resuming.

**Throughput**: the gradient update (~27 ms at `hidden=256` on CPU) dominates the ~0.6 ms env step, so training is ~36 steps/s. `--hidden 128` roughly halves the update, but `--init-from` requires the source's hidden size.

**Device selection**: `_resolve_device()` maps `device="auto"` to **CPU**, not MPS. This is deliberate: the actor/critic are small and minibatches are small, so GPU per-op dispatch overhead outweighs the math (measured ~3x slower on Apple Silicon MPS). Keep `auto`/`cpu` unless you substantially enlarge the network, in which case `--device mps`/`--device cuda` may pay off. Environment stepping always runs on CPU regardless.

**Angle wrapping**: The internal state angles are not wrapped — they can accumulate beyond ±π during long episodes with rapid spinning. This is intentional: the RK45 integrator handles it correctly and the cos/sin observation encoding is already periodic. Do not wrap angles in dynamics.py or you'll break energy conservation.

**Mass matrix near singular configurations**: The `(n+1)×(n+1)` mass matrix M(θ) is guaranteed positive definite for any θ (provable from the kinetic energy). However, numerical conditioning degrades when adjacent links align (θᵢ ≈ θⱼ), and worsens as n grows. `np.linalg.solve` handles this fine for the default parameters up to a few links; if you change params significantly or push n higher, verify `np.linalg.cond(M)` stays below ~1e6. If you edit `dynamics.py`, re-run the symbolic cross-check (see `tests/test_n_links.py` energy-conservation tests, which catch any term error via drift).

**Torch version and MPS**: MPS support requires PyTorch ≥ 2.1. Not all ops are MPS-supported; the code uses `.to(device)` before ops and falls back to CPU automatically when `torch.backends.mps.is_available()` returns False.
