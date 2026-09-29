"""Matplotlib animation for the double pendulum cartpole."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

from n_cartpole.env.dynamics import PhysicsParams


def animate_episode(
    states: np.ndarray,
    params: PhysicsParams | None = None,
    dt: float = 0.02,
    save_path: str | Path | None = None,
    speed: float = 1.0,
) -> None:
    """Animate a recorded episode of the single- or double-pendulum cartpole.

    Args:
        states:     (T, 4) or (T, 6) array of raw physics states.
                    Single link: [x, x_dot, theta1, theta1_dot]
                    Double link: [x, x_dot, theta1, theta1_dot, theta2, theta2_dot]
        params:     physics parameters (uses defaults if None)
        dt:         simulation timestep in seconds
        save_path:  if given, saves to .gif or .mp4 instead of showing live
        speed:      playback speed multiplier (2.0 = 2x faster)

    """
    if params is None:
        params = PhysicsParams()
    p = params
    T = len(states)
    n_links = 1 if states.shape[1] == 4 else 2

    track_half = p.x_lim + 0.5
    pole_total = p.l1 + (p.l2 if n_links == 2 else 0.0)
    x_range = 2 * track_half
    y_range = 2 * (pole_total + 0.3)

    fig_width = 12.0
    fig_height = max(2.0, fig_width * y_range / x_range)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(-track_half, track_half)
    ax.set_ylim(-(pole_total + 0.3), pole_total + 0.3)
    ax.set_aspect("equal")
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#f8f9fa")

    # Track
    ax.axhline(0, color="#adb5bd", linewidth=1.5, zorder=0)
    ax.axvline(-p.x_lim, color="#e63946", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.axvline(p.x_lim, color="#e63946", linewidth=1.0, linestyle="--", alpha=0.6)

    cart_w, cart_h = 0.3, 0.1
    cart_patch = Rectangle(
        (-cart_w / 2, -cart_h / 2),
        cart_w,
        cart_h,
        fc="#1d3557",
        ec="#457b9d",
        linewidth=2,
        zorder=3,
    )
    ax.add_patch(cart_patch)

    (pole1_line,) = ax.plot(
        [], [], color="#e63946", linewidth=4, solid_capstyle="round", zorder=4
    )
    (pole2_line,) = ax.plot(
        [], [], color="#f4a261", linewidth=3, solid_capstyle="round", zorder=4
    )
    (joint1_dot,) = ax.plot([], [], "o", color="#457b9d", markersize=8, zorder=5)
    (joint2_dot,) = ax.plot([], [], "o", color="#e63946", markersize=6, zorder=5)
    (tip_dot,) = ax.plot([], [], "o", color="#f4a261", markersize=5, zorder=5)

    title = ax.set_title("", fontsize=12)
    ax.set_xlabel("Cart position (m)", fontsize=10)

    def _geometry(state: np.ndarray) -> tuple[float, float, float, float, float]:
        x, th1 = state[0], state[2]
        x1 = x + p.l1 * np.sin(th1)
        y1 = p.l1 * np.cos(th1)
        if n_links == 2:
            th2 = state[4]
            x2 = x1 + p.l2 * np.sin(th2)
            y2 = y1 + p.l2 * np.cos(th2)
        else:
            x2, y2 = x1, y1
        return x, x1, y1, x2, y2

    def _update(frame: int) -> list[Any]:
        state = states[frame]
        x, x1, y1, x2, y2 = _geometry(state)

        cart_patch.set_xy((x - cart_w / 2, -cart_h / 2))
        pole1_line.set_data([x, x1], [0.0, y1])
        joint1_dot.set_data([x], [0.0])
        tip_dot.set_data([x2], [y2])
        if n_links == 2:
            pole2_line.set_data([x1, x2], [y1, y2])
            joint2_dot.set_data([x1], [y1])

        th1_deg = np.degrees(state[2] % (2 * np.pi))
        t_sec = frame * dt
        if n_links == 2:
            th2_deg = np.degrees(state[4] % (2 * np.pi))
            title.set_text(
                f"t={t_sec:.2f}s  |  x={x:.2f}m  |  "
                f"θ₁={th1_deg:.0f}°  |  θ₂={th2_deg:.0f}°"
            )
        else:
            title.set_text(f"t={t_sec:.2f}s  |  x={x:.2f}m  |  θ₁={th1_deg:.0f}°")

        artists = [cart_patch, pole1_line, joint1_dot, tip_dot, title]
        if n_links == 2:
            artists += [pole2_line, joint2_dot]
        return artists

    interval_ms = max(1, int(dt * 1000 / speed))
    anim = FuncAnimation(fig, _update, frames=T, interval=interval_ms, blit=False)

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        writer: PillowWriter | FFMpegWriter
        if save_path.suffix == ".gif":
            writer = PillowWriter(fps=int(1.0 / dt * speed))
        else:
            writer = FFMpegWriter(fps=int(1.0 / dt * speed))
        anim.save(save_path, writer=writer)
        logger.info(f"Saved animation → {save_path}")
    else:
        plt.show()

    plt.close(fig)
