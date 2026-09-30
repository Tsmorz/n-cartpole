"""CLI: render the training dashboard (interactive HTML) from a metrics CSV.

Prefers the rich ``metrics.csv`` (return + PPO diagnostics) and falls back to the
legacy ``returns.csv``. Both are written by the trainer into the checkpoint dir.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from n_cartpole.viz.plots import plot_training_curves


def _default_csv() -> Path:
    """Pick the newest metrics.csv (else returns.csv) under checkpoints/**/."""
    candidates = sorted(
        Path("checkpoints").glob("*/*/metrics.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    candidates = sorted(
        Path("checkpoints").glob("*/*/returns.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    return Path("checkpoints") / "double" / "ppo" / "metrics.csv"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Plot the PPO training dashboard.")
    parser.add_argument(
        "--csv", type=Path, default=None, help="Metrics CSV (default: auto-detect)"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("checkpoints/training.html"),
        help="Output HTML path",
    )
    parser.add_argument("--title", type=str, default="PPO training", help="Title")
    parser.add_argument(
        "--no-open", action="store_true", help="Do not open the HTML in a browser"
    )
    return parser.parse_args()


def main() -> None:
    """Read the CSV and render the dashboard."""
    args = parse_args()
    csv_path = args.csv or _default_csv()
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No metrics CSV at {csv_path}. Run training first (task train)."
        )
    logger.info(f"Reading metrics ← {csv_path}")
    plot_training_curves(
        csv_path, args.out, title=args.title, open_browser=not args.no_open
    )


if __name__ == "__main__":
    main()
