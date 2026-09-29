"""Plot a training-return curve from a returns.csv produced by the trainer."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Plot PPO training return curve.")
    parser.add_argument("--csv", type=Path, required=True, help="Path to returns.csv")
    parser.add_argument("--out", type=Path, required=True, help="Output image path")
    parser.add_argument("--title", type=str, default="Training return", help="Title")
    return parser.parse_args()


def _smooth(y: np.ndarray, window: int = 11) -> np.ndarray:
    """Centered moving average (odd window), edges shrink the window."""
    if len(y) < 3:
        return y
    window = min(window, len(y) if len(y) % 2 else len(y) - 1)
    if window < 3:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def main() -> None:
    """Read the CSV and render a return-vs-iteration plot."""
    args = parse_args()
    iters, rets = [], []
    with args.csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            iters.append(int(row["iteration"]))
            rets.append(float(row["mean_return"]))

    x = np.array(iters)
    y = np.array(rets)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, y, color="#adb5bd", linewidth=1, alpha=0.8, label="per-iteration")
    ax.plot(x, _smooth(y), color="#1d3557", linewidth=2.5, label="smoothed")
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("Mean episode return")
    ax.set_title(args.title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.patch.set_facecolor("white")
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    logger.info(f"Saved return plot → {args.out}")


if __name__ == "__main__":
    main()
