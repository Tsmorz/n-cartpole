"""CLI: render the policy's input→output map as an interactive HTML page.

Sweeps two state dimensions (default θ₁ vs θ̇₁) and draws the deterministic force
output — and the value function, where available — as contour phase portraits,
with a slider over a third dimension (default θ₂).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from n_cartpole.policy.loader import load_policy
from n_cartpole.viz.policy_map import plot_policy_map

# Friendly names for the raw-state indices, accepted on the CLI.
_DIM_NAMES = {
    "x": 0,
    "xdot": 1,
    "theta1": 2,
    "theta1dot": 3,
    "theta2": 4,
    "theta2dot": 5,
}


def _dim(value: str) -> int:
    """Parse a dimension given as a name or a raw index."""
    if value in _DIM_NAMES:
        return _DIM_NAMES[value]
    return int(value)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    choices = list(_DIM_NAMES) + [str(i) for i in range(6)]
    parser = argparse.ArgumentParser(
        description="Visualize the policy input→output map."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/double/ppo/ppo_latest.pt"),
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--x",
        type=_dim,
        default="theta1",
        metavar="DIM",
        help=f"x-axis state dim ({', '.join(choices)})",
    )
    parser.add_argument(
        "--y",
        type=_dim,
        default="theta1dot",
        metavar="DIM",
        help="y-axis state dim",
    )
    parser.add_argument(
        "--slider",
        type=str,
        default="theta2",
        metavar="DIM",
        help="slider state dim, or 'none'",
    )
    parser.add_argument("--resolution", type=int, default=81, help="grid points/axis")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("checkpoints/policy_map.html"),
        help="Output HTML path",
    )
    parser.add_argument("--no-open", action="store_true", help="Do not open in browser")
    return parser.parse_args()


def main() -> None:
    """Load the policy and render its control surface."""
    args = parse_args()
    bundle = load_policy(args.checkpoint)
    logger.info(
        f"Loaded {bundle.algo.upper()} policy from {args.checkpoint} "
        f"({bundle.n_links}-link)"
    )
    slider = None if args.slider.lower() == "none" else _dim(args.slider)
    plot_policy_map(
        bundle,
        dim_x=args.x,
        dim_y=args.y,
        slider_dim=slider,
        resolution=args.resolution,
        save_path=args.out,
        open_browser=not args.no_open,
    )


if __name__ == "__main__":
    main()
