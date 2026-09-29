"""Visualize the policy's input→output mapping as an interactive phase portrait.

The controller is a function ``state → force``. This sweeps two chosen state
dimensions over a grid (holding the rest at a nominal state), evaluates the
policy's deterministic force — and the critic's value, where available — and
draws them as contour heatmaps. A slider walks a third dimension, so you can
watch the control law deform as, say, the second pole swings around.

Force uses a diverging ramp centered at zero (blue = push right, red = push
left); value uses a sequential ramp. This turns an opaque MLP into a readable
control surface.
"""

from __future__ import annotations

import webbrowser
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from loguru import logger
from plotly.subplots import make_subplots

from n_cartpole.policy.loader import PolicyBundle
from n_cartpole.viz import style

# Raw-state layout: [x, ẋ, θ₁, θ̇₁, θ₂, θ̇₂]. Per-dim label + default sweep range.
_DIMS: dict[int, tuple[str, tuple[float, float]]] = {
    0: ("cart x (m)", (-0.5, 0.5)),
    1: ("cart ẋ (m/s)", (-3.0, 3.0)),
    2: ("θ₁ (rad)", (-np.pi, np.pi)),
    3: ("θ̇₁ (rad/s)", (-8.0, 8.0)),
    4: ("θ₂ (rad)", (-np.pi, np.pi)),
    5: ("θ̇₂ (rad/s)", (-8.0, 8.0)),
}


def _evaluate(
    bundle: PolicyBundle,
    dim_x: int,
    dim_y: int,
    gx: np.ndarray,
    gy: np.ndarray,
    nominal: np.ndarray,
    fixed: dict[int, float],
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (action_grid, value_grid|None) of shape (len(gy), len(gx))."""
    xx, yy = np.meshgrid(gx, gy)
    n = xx.size
    raw = np.tile(nominal, (n, 1)).astype(np.float64)
    raw[:, dim_x] = xx.ravel()
    raw[:, dim_y] = yy.ravel()
    for d, v in fixed.items():
        raw[:, d] = v

    obs_norm = bundle.normalize(bundle.encode(raw))
    action = bundle.select_action(obs_norm).reshape(xx.shape)
    value = None
    if bundle.value is not None:
        value = bundle.value(obs_norm).reshape(xx.shape)
    return action, value


def plot_policy_map(
    bundle: PolicyBundle,
    dim_x: int = 2,
    dim_y: int = 3,
    slider_dim: int | None = 4,
    resolution: int = 81,
    save_path: str | Path | None = None,
    *,
    title: str | None = None,
    open_browser: bool = True,
) -> go.Figure:
    """Render the policy (and value) as contour maps over two state dimensions.

    Args:
        bundle:     loaded policy (see :func:`n_cartpole.policy.loader.load_policy`).
        dim_x:      raw-state index for the x-axis (default θ₁).
        dim_y:      raw-state index for the y-axis (default θ̇₁).
        slider_dim: raw-state index swept by a slider, or None. Ignored when it
                    exceeds the link count (e.g. θ₂ for a single-link policy).
        resolution: grid points per axis.
        save_path:  if given, write an ``.html`` file here.
        title:      page title.
        open_browser: open the written HTML in a browser.

    Returns:
        The Plotly figure.

    """
    state_dim = 2 + 2 * bundle.n_links
    if slider_dim is not None and slider_dim >= state_dim:
        slider_dim = None  # not present for this link count

    nominal = np.zeros(state_dim)  # upright, centered, at rest
    gx = np.linspace(*_DIMS[dim_x][1], resolution)
    gy = np.linspace(*_DIMS[dim_y][1], resolution)

    if slider_dim is not None:
        slider_vals = np.linspace(_DIMS[slider_dim][1][0], _DIMS[slider_dim][1][1], 11)
    else:
        slider_vals = np.array([0.0])

    show_value = bundle.value is not None
    ncols = 2 if show_value else 1
    fig = make_subplots(
        rows=1,
        cols=ncols,
        subplot_titles=(
            ["Force output (N)", "Value V(s)"] if show_value else ["Force output (N)"]
        ),
        horizontal_spacing=0.12,
    )

    # Compute every slider slice up front (an exported HTML is static).
    action_slices, value_slices = [], []
    for sv in slider_vals:
        fixed = {slider_dim: float(sv)} if slider_dim is not None else {}
        a, v = _evaluate(bundle, dim_x, dim_y, gx, gy, nominal, fixed)
        action_slices.append(a)
        value_slices.append(v)

    amax = max(float(np.abs(a).max()) for a in action_slices) or 1.0

    def _action_trace(a: np.ndarray) -> go.Contour:
        return go.Contour(
            x=gx,
            y=gy,
            z=a,
            colorscale=style.DIVERGING,
            zmid=0,
            zmin=-amax,
            zmax=amax,
            colorbar={"title": "N", "len": 0.9, "x": (0.44 if show_value else 1.02)},
            contours={"showlines": True, "coloring": "heatmap"},
            line={"width": 0.5, "color": "rgba(11,11,11,0.15)"},
            hovertemplate=f"{_DIMS[dim_x][0]}=%{{x:.2f}}<br>"
            f"{_DIMS[dim_y][0]}=%{{y:.2f}}<br>F=%{{z:.2f}} N<extra></extra>",
        )

    def _value_trace(v: np.ndarray | None) -> go.Contour:
        assert v is not None  # only called when a value surface exists
        return go.Contour(
            x=gx,
            y=gy,
            z=v,
            colorscale=style.SEQUENTIAL,
            colorbar={"title": "V", "len": 0.9, "x": 1.02},
            contours={"showlines": True, "coloring": "heatmap"},
            line={"width": 0.5, "color": "rgba(11,11,11,0.12)"},
            hovertemplate=f"{_DIMS[dim_x][0]}=%{{x:.2f}}<br>"
            f"{_DIMS[dim_y][0]}=%{{y:.2f}}<br>V=%{{z:.2f}}<extra></extra>",
        )

    fig.add_trace(_action_trace(action_slices[0]), row=1, col=1)
    if show_value:
        fig.add_trace(_value_trace(value_slices[0]), row=1, col=2)

    # Mark the upright balance point if it lies inside the swept plane.
    for c in range(1, ncols + 1):
        if _DIMS[dim_x][1][0] <= 0 <= _DIMS[dim_x][1][1] and (
            _DIMS[dim_y][1][0] <= 0 <= _DIMS[dim_y][1][1]
        ):
            fig.add_trace(
                go.Scatter(
                    x=[0],
                    y=[0],
                    mode="markers",
                    marker={
                        "symbol": "star",
                        "size": 13,
                        "color": style.GOOD,
                        "line": {"color": "white", "width": 1},
                    },
                    name="upright",
                    showlegend=(c == 1),
                    hovertemplate="upright equilibrium<extra></extra>",
                ),
                row=1,
                col=c,
            )

    if slider_dim is not None:
        frames = []
        for i, sv in enumerate(slider_vals):
            data = [_action_trace(action_slices[i])]
            if show_value:
                data.append(_value_trace(value_slices[i]))
            frames.append(
                go.Frame(name=f"{sv:.2f}", data=data, traces=list(range(ncols)))
            )
        fig.frames = frames
        sliders = [
            {
                "active": len(slider_vals) // 2,
                "x": 0.0,
                "len": 1.0,
                "y": -0.08,
                "pad": {"t": 30, "b": 10},
                "currentvalue": {
                    "prefix": f"{_DIMS[slider_dim][0]} = ",
                    "font": {"color": style.INK},
                },
                "steps": [
                    {
                        "label": f"{sv:.1f}",
                        "method": "animate",
                        "args": [
                            [f"{sv:.2f}"],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                            },
                        ],
                    }
                    for sv in slider_vals
                ],
            }
        ]
        # Start on the middle (nominal) slice.
        mid = len(slider_vals) // 2
        fig.data[0].z = action_slices[mid]
        if show_value:
            fig.data[1].z = value_slices[mid]
    else:
        sliders = []

    for c in range(1, ncols + 1):
        fig.update_xaxes(title_text=_DIMS[dim_x][0], row=1, col=c)
        fig.update_yaxes(title_text=_DIMS[dim_y][0], row=1, col=c)

    subtitle = f"{bundle.algo.upper()} · {bundle.n_links}-link · nominal state = 0"
    fig.update_layout(
        **style.base_layout(
            title={
                "text": title
                or f"Policy input→output map<br>"
                f"<span style='font-size:13px;color:{style.MUTED}'>{subtitle}</span>"
            },
            height=560,
            sliders=sliders,
            showlegend=True,
            legend={
                "orientation": "h",
                "y": 1.02,
                "x": 1,
                "xanchor": "right",
                "yanchor": "bottom",
            },
        )
    )

    if save_path is not None:
        save_path = Path(save_path).with_suffix(".html")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(save_path, include_plotlyjs="cdn", auto_play=False)
        logger.info(f"Saved policy map → {save_path}")
        if open_browser:
            webbrowser.open(save_path.resolve().as_uri())
    return fig
