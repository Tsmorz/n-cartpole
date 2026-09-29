# n-cartpole

[![CI](https://github.com/Tsmorz/n-cartpole/actions/workflows/ci.yml/badge.svg)](https://github.com/Tsmorz/n-cartpole/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

PPO policy learning for double pendulum cartpole swing-up. A cart on a frictionless track carries two pendulums in series; the policy learns to apply horizontal forces to swing both poles from hanging to upright and balance them there.

![Double pendulum cartpole simulation](docs/assets/demo.gif)

## Install

```bash
# requires uv and go-task
task init
```

## Quickstart

```bash
# train with all available CPU workers, gradient updates on MPS (Apple Silicon)
task train

# watch a trained policy
task play -- --checkpoint checkpoints/latest.pt
```

## Development

| Command | Description |
|---|---|
| `task init` | Create virtual environment and install deps |
| `task train` | Train the PPO policy (default settings) |
| `task play` | Run a trained policy and visualize |
| `task test` | Run tests with coverage |
| `task format` | Ruff format + lint + mypy |
| `task ci` | Full local CI (format + test) |
| `task clean` | Remove venv, caches, checkpoints |

## System

- **Dynamics**: Lagrangian EOM for cart + 2 pendulums (point masses); RK45 integration at 50 Hz
- **Policy**: Separate Actor and Critic MLPs (8 → 64 → 64 → 1), Gaussian policy with learnable log-std
- **Training**: PPO-clip (ε=0.2), GAE (λ=0.95), parallel CPU rollout workers → MPS gradient update
- **Observation**: `[x, ẋ, cos θ₁, sin θ₁, θ̇₁, cos θ₂, sin θ₂, θ̇₂]` — cos/sin encoding eliminates angle discontinuities
