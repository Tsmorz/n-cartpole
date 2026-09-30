"""Interactive episode replay for the single-/double-pendulum cartpole (Plotly).

Renders a self-contained HTML page: on the left, a play/scrub animation of the
cart and poles; on the right, a synced telemetry stack (uprightness, cart
position, applied force, per-step reward) with a moving time cursor. Being able
to scrub the physics against the force the policy applied is what makes the
controller's behavior legible.
"""

from __future__ import annotations

import webbrowser
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from loguru import logger
from plotly.subplots import make_subplots

from n_cartpole.env.dynamics import PhysicsParams
from n_cartpole.viz import style

_MAX_FRAMES = 300  # cap animation frames so the HTML stays light and smooth


def _joint_coords(
    state: np.ndarray, p: PhysicsParams, n_links: int
) -> tuple[list[float], list[float]]:
    """Return the ``(xs, ys)`` of every joint for one raw state.

    ``xs[0], ys[0]`` is the cart pivot; ``xs[k+1], ys[k+1]`` is the tip of link
    ``k``. Length ``n_links + 1``.
    """
    lengths = p.link_lengths(n_links)
    px, py = float(state[0]), 0.0
    xs, ys = [px], [py]
    for k in range(n_links):
        th = float(state[2 + 2 * k])
        px = px + lengths[k] * np.sin(th)
        py = py + lengths[k] * np.cos(th)
        xs.append(px)
        ys.append(py)
    return xs, ys


def _cart_shape(cx: float, w: float, h: float) -> tuple[list[float], list[float]]:
    """Return a closed rectangle outline centered at (cx, 0)."""
    xs = [cx - w / 2, cx + w / 2, cx + w / 2, cx - w / 2, cx - w / 2]
    ys = [-h / 2, -h / 2, h / 2, h / 2, -h / 2]
    return xs, ys


def animate_episode(
    states: np.ndarray,
    params: PhysicsParams | None = None,
    dt: float = 0.02,
    save_path: str | Path | None = None,
    speed: float = 1.0,
    *,
    actions: np.ndarray | None = None,
    rewards: np.ndarray | None = None,
    title: str | None = None,
    open_browser: bool = True,
    goals: np.ndarray | None = None,
) -> go.Figure:
    """Render an interactive replay of a recorded episode.

    Args:
        states:   (T, 2+2n) raw physics states (n links).
        params:   physics parameters (defaults if None).
        dt:       simulation timestep (s).
        save_path: if given, write an ``.html`` file here.
        speed:    playback speed multiplier for the auto-play button.
        actions:  (T-1,) applied force per step; adds the force telemetry panel.
        rewards:  (T-1,) per-step reward; adds the reward telemetry panel.
        title:    page title.
        goals:    (T, n) commanded target angles (goal-conditioned policies);
                  drawn as dashed target lines on the uprightness panel.
        open_browser: open the written HTML in a browser.

    Returns:
        The Plotly figure.

    """
    p = params or PhysicsParams()
    states = np.asarray(states)
    T = len(states)
    n_links = (states.shape[1] - 2) // 2
    t = np.arange(T) * dt
    t_ctrl = np.arange(T - 1) * dt

    # --- Telemetry panels (right column): (name, x, y, color, extra) ---------
    panels: list[dict] = []
    up_traces: list[tuple] = [
        (f"θ{k + 1} upright", np.cos(states[:, 2 + 2 * k]), style.link_color(k))
        for k in range(n_links)
    ]
    if goals is not None:
        g = np.cos(np.asarray(goals)[:T])
        up_traces += [
            (f"θ{k + 1} target", g[:, k], style.link_color(k), "dash")
            for k in range(n_links)
        ]
    panels.append(
        {
            "title": "Uprightness (cos θ)",
            "series": up_traces,
            "t": t,
            "range": [-1.08, 1.08],
            "band": (0.9, 1.0),
        }
    )
    panels.append(
        {
            "title": "Cart position (m)",
            "series": [("x", states[:, 0], style.ACCENT)],
            "t": t,
            "range": None,
            "limit": p.x_lim,
        }
    )
    if actions is not None:
        a = np.asarray(actions).reshape(-1)[: T - 1]
        panels.append(
            {
                "title": "Applied force (N)",
                "series": [("F", a, style.FORCE)],
                "t": t_ctrl[: len(a)],
                "range": [-p.force_max * 1.1, p.force_max * 1.1],
                "limit": p.force_max,
            }
        )
    if rewards is not None:
        r = np.asarray(rewards).reshape(-1)[: T - 1]
        panels.append(
            {
                "title": "Reward (per step)",
                "series": [("r", r, style.REWARD)],
                "t": t_ctrl[: len(r)],
                "range": [0, 1.05],
                "fill": True,
            }
        )

    n_panels = len(panels)
    specs: list[list] = [[{"rowspan": n_panels}, {}]] + [
        [None, {}] for _ in range(n_panels - 1)
    ]
    fig = make_subplots(
        rows=n_panels,
        cols=2,
        specs=specs,
        column_widths=[0.54, 0.46],
        horizontal_spacing=0.09,
        vertical_spacing=0.09,
        subplot_titles=[""] + [pl["title"] for pl in panels],
    )

    # --- Physical scene (left) — static scenery via shapes -------------------
    track_half = p.x_lim + 0.4
    reach = float(np.sum(p.link_lengths(n_links))) + 0.25
    fig.add_shape(
        type="line",
        x0=-track_half,
        x1=track_half,
        y0=0,
        y1=0,
        line={"color": style.BASELINE, "width": 4},
        row=1,
        col=1,
    )
    for xr in (-p.x_lim, p.x_lim):
        fig.add_shape(
            type="line",
            x0=xr,
            x1=xr,
            y0=-0.1,
            y1=0.1,
            line={"color": style.INK_2, "width": 5},
            row=1,
            col=1,
        )

    cart_w, cart_h = 0.34, 0.12
    xs0, ys0 = _joint_coords(states[0], p, n_links)
    cxs, cys = _cart_shape(xs0[0], cart_w, cart_h)

    # Animated physical traces (order fixed → referenced in frames by index).
    fig.add_trace(
        go.Scatter(
            x=[xs0[0], xs0[0]],
            y=[0, reach - 0.1],
            mode="lines",
            line={
                "color": style.rgba(style.ACCENT, 0.35),
                "width": 1,
                "dash": "dash",
            },
            hoverinfo="skip",
            showlegend=False,
            name="upright",
        ),
        row=1,
        col=1,
    )
    idx_upright = len(fig.data) - 1
    fig.add_trace(
        go.Scatter(
            x=cxs,
            y=cys,
            mode="lines",
            fill="toself",
            fillcolor=style.ACCENT,
            line={"color": style.ACCENT, "width": 1},
            hoverinfo="skip",
            showlegend=False,
            name="cart",
        ),
        row=1,
        col=1,
    )
    idx_cart = len(fig.data) - 1
    # One trace per link: dark rod, hollow pivot dot, colored bob at the tip.
    idx_poles: list[int] = []
    for k in range(n_links):
        fig.add_trace(
            go.Scatter(
                x=[xs0[k], xs0[k + 1]],
                y=[ys0[k], ys0[k + 1]],
                mode="lines+markers",
                line={"color": style.INK, "width": 4},
                marker={
                    "size": [6, max(12, 18 - 2 * k)],
                    "color": [style.SURFACE, style.link_color(k)],
                    "line": {"width": 0},
                },
                hoverinfo="skip",
                showlegend=False,
                name=f"link {k + 1}",
            ),
            row=1,
            col=1,
        )
        idx_poles.append(len(fig.data) - 1)

    fig.update_xaxes(
        range=[-track_half, track_half], showgrid=False, zeroline=False, row=1, col=1
    )
    fig.update_yaxes(
        range=[-0.4, reach],
        showgrid=False,
        zeroline=False,
        scaleanchor="x",
        scaleratio=1,
        row=1,
        col=1,
    )

    # --- Telemetry traces (right) — full-resolution static curves ------------
    cursor_idx: list[int] = []
    for i, pl in enumerate(panels):
        row = i + 1
        band = pl.get("band")
        if band is not None:  # shade the "upright" zone
            fig.add_hrect(
                y0=band[0],
                y1=band[1],
                line_width=0,
                fillcolor=style.rgba(style.ACCENT, 0.12),
                row=row,
                col=2,
            )
        for name, y, color, *dash in pl["series"]:
            fig.add_trace(
                go.Scatter(
                    x=pl["t"],
                    y=y,
                    mode="lines",
                    name=name,
                    line={
                        "color": color,
                        "width": 1.5 if dash else 2,
                        "dash": dash[0] if dash else None,
                    },
                    fill="tozeroy" if pl.get("fill") else None,
                    fillcolor=style.rgba(style.REWARD, 0.15)
                    if pl.get("fill")
                    else None,
                    showlegend=len(pl["series"]) > 1,
                    legendgroup=f"panel{i}",
                    hovertemplate="t=%{x:.2f}s<br>%{y:.3f}<extra></extra>",
                ),
                row=row,
                col=2,
            )
        if pl.get("limit") is not None:  # rail / force limit lines
            for lim in (-pl["limit"], pl["limit"]):
                fig.add_hline(
                    y=lim,
                    line={"color": style.CRITICAL, "width": 1, "dash": "dash"},
                    opacity=0.6,
                    row=row,
                    col=2,
                )
        yr = pl["range"]
        if yr is None:
            allv = np.concatenate([s[1] for s in pl["series"]])
            pad = 0.1 * (allv.max() - allv.min() + 1e-6)
            yr = [float(allv.min() - pad), float(allv.max() + pad)]
        fig.update_yaxes(range=yr, row=row, col=2)
        if row == n_panels:
            fig.update_xaxes(title_text="time (s)", row=row, col=2)

        # Moving cursor for this panel (animated).
        fig.add_trace(
            go.Scatter(
                x=[0, 0],
                y=yr,
                mode="lines",
                line={"color": style.INK_2, "width": 1, "dash": "dot"},
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=2,
        )
        cursor_idx.append(len(fig.data) - 1)

    # --- Animation frames ----------------------------------------------------
    stride = max(1, T // _MAX_FRAMES)
    frame_ids = list(range(0, T, stride))
    if frame_ids[-1] != T - 1:
        frame_ids.append(T - 1)

    animated = [idx_upright, idx_cart, *idx_poles]
    animated += cursor_idx

    frames = []
    for fi in frame_ids:
        xs, ys = _joint_coords(states[fi], p, n_links)
        cxs, cys = _cart_shape(xs[0], cart_w, cart_h)
        data = [
            go.Scatter(x=[xs[0], xs[0]], y=[0, reach - 0.1]),
            go.Scatter(x=cxs, y=cys),
        ]
        for k in range(n_links):
            data.append(go.Scatter(x=[xs[k], xs[k + 1]], y=[ys[k], ys[k + 1]]))
        tt = float(fi * dt)
        for pl in panels:
            data.append(go.Scatter(x=[tt, tt], y=_cursor_range(pl)))
        frames.append(go.Frame(name=str(fi), data=data, traces=animated))
    fig.frames = frames

    # --- Play/scrub controls -------------------------------------------------
    frame_ms = max(10, int(dt * 1000 / max(speed, 1e-6)))
    fig.update_layout(
        **style.base_layout(
            title={
                "text": title or f"Episode replay · {n_links}-link cartpole",
                "font": {
                    "family": style.FONT,
                    "color": style.INK,
                    "size": 18,
                },
                "x": 0.02,
            },
            height=max(560, 175 * n_panels + 90),
            showlegend=True,
            legend={
                "orientation": "h",
                "y": 1.03,
                "x": 1,
                "xanchor": "right",
                "yanchor": "bottom",
            },
            updatemenus=[
                {
                    "type": "buttons",
                    "showactive": False,
                    "x": 0.0,
                    "y": -0.04,
                    "xanchor": "left",
                    "yanchor": "top",
                    "direction": "left",
                    "pad": {"t": 4},
                    "buttons": [
                        {
                            "label": "▶ Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {"duration": frame_ms, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": 0},
                                },
                            ],
                        },
                        {
                            "label": "⏸ Pause",
                            "method": "animate",
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                },
                            ],
                        },
                    ],
                }
            ],
            sliders=[
                {
                    "active": 0,
                    "x": 0.0,
                    "len": 0.52,
                    "y": -0.02,
                    "yanchor": "top",
                    "pad": {"t": 30, "b": 8},
                    "currentvalue": {
                        "prefix": "t = ",
                        "suffix": " s",
                        "font": {
                            "color": style.INK,
                            "family": style.MONO,
                            "size": 12,
                        },
                    },
                    "steps": [
                        {
                            "label": f"{fi * dt:.1f}",
                            "method": "animate",
                            "args": [
                                [str(fi)],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                },
                            ],
                        }
                        for fi in frame_ids
                    ],
                }
            ],
        )
    )
    axis = {
        "showgrid": True,
        "gridcolor": style.GRID,
        "zeroline": False,
        "linecolor": style.BASELINE,
        "tickcolor": style.BASELINE,
        "ticks": "outside",
        "ticklen": 4,
        "tickfont": {"family": style.MONO, "size": 10},
    }
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    fig.update_annotations(font={"family": style.FONT, "color": style.INK, "size": 13})
    fig.update_xaxes(
        showgrid=False, zeroline=False, showticklabels=False, ticks="", row=1, col=1
    )
    fig.update_yaxes(
        showgrid=False, zeroline=False, showticklabels=False, ticks="", row=1, col=1
    )

    if save_path is not None:
        save_path = Path(save_path).with_suffix(".html")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(save_path, include_plotlyjs="cdn", auto_play=False)
        logger.info(f"Saved episode replay → {save_path}")
        if open_browser:
            webbrowser.open(save_path.resolve().as_uri())
    return fig


def _cursor_range(panel: dict) -> list[float]:
    """Return the y-range a panel's cursor line should span."""
    yr = panel["range"]
    if yr is not None:
        return list(yr)
    allv = np.concatenate([s[1] for s in panel["series"]])
    pad = 0.1 * (allv.max() - allv.min() + 1e-6)
    return [float(allv.min() - pad), float(allv.max() + pad)]
