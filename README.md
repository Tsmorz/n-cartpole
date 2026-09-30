# n-cartpole

[![CI](https://github.com/Tsmorz/n-cartpole/actions/workflows/ci.yml/badge.svg)](https://github.com/Tsmorz/n-cartpole/actions/workflows/ci.yml)
[![Coverage Status](https://coveralls.io/repos/github/Tsmorz/n-cartpole/badge.svg?branch=main)](https://coveralls.io/github/Tsmorz/n-cartpole?branch=main)
[![Python](<https://img.shields.io/badge/python-3.13%20%7C%203.14-blue.svg>)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

TQC policy learning for double pendulum cartpole swing-up. A cart on a frictionless track carries two pendulums in series; the policy learns to apply horizontal forces to swing both poles from hanging to upright and balance them there.

<img src="docs/assets/tqc_swingup.gif" alt="TQC policy swinging up and balancing both poles" width="900" height="522">

*A trained TQC policy swinging up and balancing both links upright, cart re-centered — return 817.5 over the episode.*

## Install

```bash
# requires uv and go-task
task init
```

## Quickstart

```bash
# TQC (off-policy, distributional) — defaults to 2 links (double)
task train

# single-link warm-up run, or a longer double-link run
task train -- --links 1
task train -- --links 2 --steps 300000

# watch a trained policy — interactive HTML replay (auto-detects link count)
task play -- --checkpoint checkpoints/double/tqc/tqc_latest.pt
task play -- --checkpoint checkpoints/single/tqc/tqc_latest.pt

# interactive training dashboard from a metrics.csv
task plot -- --csv checkpoints/double/tqc/metrics.csv
```

**Checkpoint layout.** Checkpoints are organized `checkpoints/<single|double>/<tqc|tqc-goal>/`,
so the folder alone tells you the link count and whether the policy is
goal-conditioned — e.g. `checkpoints/double/tqc/tqc_latest.pt`. Within each folder:
`tqc_latest.pt` (rolling most-recent, plus its replay buffer) and
`tqc_NNNNNNN.pt` (periodic snapshots by step). `task play` and
`task policy-map` auto-detect link count from the checkpoint itself (via the
saved `TQCConfig`), so any checkpoint path works regardless of which folder it
lives in.

**Pretrained models.** `checkpoints/` is gitignored (binary, environment-specific,
trivially reproducible) — pretrained weights are published as
[GitHub Releases](https://github.com/Tsmorz/n-cartpole/releases) instead, each
release carrying up to four zips (`single-tqc.zip`, `double-tqc.zip`,
`single-tqc-goal.zip`, `double-tqc-goal.zip`), one per combination you've
trained locally. Nothing trains in CI — releases are built and published
entirely from your machine. Fetch and unpack the latest release into
`checkpoints/` with:

```bash
task download-models                 # latest release
task download-models -- models-v1    # a specific tag
```

To cut a new release: train locally as usual, then run

```bash
task release-models -- models-v1
```

which zips whatever's under `checkpoints/<single|double>/<tqc|tqc-goal>/`
(`task package-models` alone, if you just want the zips without
tagging/publishing), tags and pushes `models-v1`, and creates the GitHub
Release — only the combinations you actually trained are included.

The last step requires the [GitHub CLI](https://cli.github.com/) (`gh auth login`).
Without it, do the first two steps manually and upload via the web UI:

```bash
task package-models              # produces dist/*.zip
git tag models-v1
git push origin models-v1
# then go to github.com/Tsmorz/n-cartpole/releases/new, pick the tag, and attach dist/*.zip
```

**Visualization.** Every view renders to a self-contained, interactive **Plotly**
HTML page (hover, zoom, play/scrub) — no static PNGs. `task play` gives a scrub-able
cart-and-poles replay with a synced telemetry sidebar (uprightness, cart position,
applied force, per-step reward). `task policy-map` turns the trained MLP into a
readable control surface — a contour phase-portrait of the force the policy applies
across a plane of states (blue = push right, red = push left), with a slider over a
third dimension. `task plot` renders the training curves as small multiples.

**Learner.** **TQC** (Truncated Quantile Critics) is the off-policy,
distributional actor-critic that Lee et al. used for real multi-pendulum
hardware. It trains with the bounded reward, symmetric data augmentation, and
the diverse initial-state distribution described below.

> **Device note:** the networks are small, so the whole run is fastest on CPU —
> GPU per-op dispatch overhead outweighs the tiny matmuls. `--device auto`
> therefore selects CPU. Use `--device mps`/`--device cuda` only after
> substantially enlarging the network.

## Transitions between configurations

With `--goals`, one network learns to move between every up/down configuration
and to switch between them on command, not just to swing up. Each configuration
is labeled with one letter per link, base link first:

| Label  | Link 1 (base) | Link 2 (tip) | Passively stable?             |
| ------ | ------------- | ------------ | ----------------------------- |
| `UU` | up            | up           | no — actively balanced       |
| `DU` | down          | up           | no — actively balanced       |
| `UD` | up            | down         | no — actively balanced       |
| `DD` | down          | down         | yes — policy damps the swing |

That gives 12 transitions plus 4 "hold" cases, all learned by the same network.
The target configuration is an extra input to the policy (±1 per link), and the
reward measures closeness to that target instead of to upright.

```bash
# train (checkpoints go to checkpoints/double/tqc-goal/)
task train -- --goals --steps 2000000

# success rate of every start → goal transition
task eval-transitions -- --checkpoint checkpoints/double/tqc-goal/tqc_latest.pt

# replay a commanded sequence: start hanging, then UU at 0 s, DU at 6 s, DD at 12 s
task play -- --checkpoint checkpoints/double/tqc-goal/tqc_latest.pt --start DD --goals "UU@0,DU@6,DD@12"
```

During training, the goal changes every 4–8 s, episodes start at any
configuration, and transitions that keep failing are practiced more often.
Training logs a start × goal success table, and `task plot` adds a
transition-success panel. On hardware, `env.set_goal("DU")` is the command
input.

**Recommended: start from the swing-up policy.** Rather than learning from
scratch, warm-start the goal network from a plain swing-up checkpoint. The goal
inputs start at zero weight, so it begins as the swing-up policy, and its
rollouts seed the replay buffer (relabeled to every goal). A hold phase then
teaches it to stabilize each configuration from growing perturbations before
the transitions are mixed in:

```bash
task train -- --goals --init-from checkpoints/double/tqc/tqc_latest.pt \
  --hold-phase-steps 100000 --hold-prob 0.25 --steps 1000000

# swing-up timing for a goal-conditioned checkpoint (goal fixed to UU)
uv run python scripts/eval_swingup.py --checkpoint checkpoints/double/tqc-goal/tqc_latest.pt
```

## Development

| Command                   | Description                                                                                         |
| ------------------------- | --------------------------------------------------------------------------------------------------- |
| `task init`             | Create virtual environment and install deps                                                         |
| `task train`            | Train the TQC policy (`--links {1,2}`, `--goals`)                                               |
| `task play`             | Interactive HTML replay of a trained policy (`--goals` schedule for goal-conditioned checkpoints) |
| `task eval-transitions` | Start × goal success table for a goal-conditioned checkpoint                                       |
| `task download-models`  | Download a models release into`checkpoints/`                                                      |
| `task package-models`   | Zip local checkpoints into`dist/`                                                                 |
| `task release-models`   | Package, tag, and publish local checkpoints as a GitHub Release                                     |
| `task plot`             | Interactive training dashboard from`metrics.csv`                                                  |
| `task policy-map`       | Interactive input→output (force/value) control-surface map                                         |
| `task test`             | Run tests with coverage                                                                             |
| `task format`           | Ruff format + lint + mypy                                                                           |
| `task ci`               | Full local CI (format + test)                                                                       |
| `task clean`            | Remove venv, caches, checkpoints                                                                    |

## System

- **Dynamics**: Lagrangian EOM for cart + 2 pendulums (point-mass bobs) with viscous cart/joint friction; RK45 at **100 Hz**. Buildable bench-scale defaults: 1 kg cart, ~0.2/0.15 kg bobs, 0.25 m rods, ±0.5 m rail, ±20 N motor.
- **Reward**: bounded, multiplicative shaping — `r_angle · r_pos · r_vel · r_act` (Lee et al. [2, 3]), nearly always in (0, 1]. Product over links forces *both* poles upright; the velocity term rewards balancing over spinning. On top of that, a potential-based shaping term (Ng et al. [1]) rewards progress toward upright every step, so swing-up is discovered sooner — see [Appendix: papers](#appendix-papers) below.
- **Learner**: **TQC** (off-policy, 2 quantile critics × 25 atoms with top-drop truncation, SAC-style auto-entropy, replay buffer).
- **Sample efficiency**: left-right **symmetry augmentation** (mirror every transition) and a **diverse initial-state distribution** (30% fully-random resets for recovery from any configuration).
- **Observation**: `[x, ẋ, cos θ₁, sin θ₁, θ̇₁, cos θ₂, sin θ₂, θ̇₂]` — cos/sin encoding eliminates angle discontinuities.

## Appendix: papers

Design choices in this repo that come from published research, cited where used above:

1. A. Y. Ng, D. Harada, and S. Russell, "Policy Invariance Under Reward
   Transformations: Theory and Application to Reward Shaping," in *Proc. 16th
   International Conference on Machine Learning (ICML)*, 1999. — Source of the
   potential-based reward shaping term (`F(s,s') = γ·Φ(s') − Φ(s)`) added to the
   base reward in `n_cartpole/env/double_cartpole.py` / `single_cartpole.py`,
   to make swing-up progress rewarding sooner without changing the optimal
   policy.
2. T. Lee, D. Ju, and Y. S. Lee, "Transition Control of a Double-Inverted
   Pendulum System Using Sim2Real Reinforcement Learning," *Machines*, vol. 13,
   no. 3, p. 186, 2025. https://doi.org/10.3390/machines13030186 — PDF at
   `docs/machines-13-00186.pdf`. Source of the bounded, multiplicative reward
   structure (product of per-objective terms in [0, 1]) and the "recovery
   characteristics" diverse initial-state distribution.
3. Y. Oh, T. Lee, S. Ryoo, K. C. Koh, S. Han, and Y. S. Lee, "Reinforcement
   Learning to Achieve Real-time Control of a Quadruple Inverted Pendulum,"
   *International Journal of Control, Automation, and Systems*, vol. 23, no. 9,
   pp. 2797-2806, 2025. https://doi.org/10.1007/s12555-025-0235-y — PDF at
   `docs/s12555-025-0235-y.pdf`. Source of the TQC (Truncated Quantile Critics)
   learner and Virtual Experience Replay (VER), the left-right mirror-symmetry
   data augmentation used during off-policy training.
