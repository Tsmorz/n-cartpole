"""Training-curve dashboard: turn a metrics CSV into an interactive HTML page.

A metrics CSV holds one row per iteration (return + learner diagnostics). This
renders them as small multiples — one measure per panel, never a shared dual axis — so
each curve keeps its own scale and stays legible, with hover read-outs and zoom.
"""

from __future__ import annotations

import csv
import webbrowser
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from loguru import logger
from plotly.subplots import make_subplots

from n_cartpole.viz import style

# Column -> (panel title, y-axis label, series color). Order here is panel order.
_PANELS: dict[str, tuple[str, str, str]] = {
    "mean_return": ("Episode return", "mean return", style.SERIES[0]),
    "value_loss": ("Value loss", "critic loss", style.SERIES[1]),
    "policy_loss": ("Policy loss", "actor loss", style.SERIES[6]),
    "entropy": ("Policy entropy", "nats", style.SERIES[2]),
    "approx_kl": ("Approx. KL", "KL / update", style.SERIES[3]),
    "clip_fraction": ("Clip fraction", "fraction", style.SERIES[4]),
    # Goal-conditioned runs only (empty/NaN otherwise, and then not drawn).
    "goal_success": ("Transition success", "fraction", style.ACCENT),
}


def _smooth(y: np.ndarray, window: int = 11) -> np.ndarray:
    """Centered moving average (odd window); edges reuse the valid mean."""
    n = len(y)
    if n < 5:
        return y
    window = min(window, n if n % 2 else n - 1)
    if window < 3:
        return y
    kernel = np.ones(window) / window
    smoothed = np.convolve(y, kernel, mode="same")
    half = window // 2
    for i in range(half):  # repair the convolution's shrinking edges
        smoothed[i] = y[: i + half + 1].mean()
        smoothed[-(i + 1)] = y[-(i + half + 1) :].mean()
    return smoothed


def _read_metrics(csv_path: Path) -> dict[str, np.ndarray]:
    """Read a metrics/returns CSV into column arrays (all float)."""
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {csv_path}")
    return {
        k: np.array([float(r[k]) if r[k] not in ("", None) else np.nan for r in rows])
        for k in rows[0]
    }


def _to_rgba(hex_color: str, alpha: float) -> str:
    """`#rrggbb` -> `rgba(r,g,b,alpha)`."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def plot_training_curves(
    csv_path: str | Path,
    save_path: str | Path | None = None,
    *,
    title: str = "Training",
    target_kl: float | None = 0.03,
    open_browser: bool = True,
) -> go.Figure:
    """Render a training dashboard from a metrics CSV as an interactive HTML page.

    Handles both the rich ``metrics.csv`` (return + diagnostics) and the
    legacy two-column ``returns.csv`` — it plots whichever measures are present.

    Args:
        csv_path:     CSV written by the trainer.
        save_path:    if given, write an ``.html`` file here; else keep in memory.
        title:        page title / suptitle.
        target_kl:    if the KL panel is drawn, mark this early-stop threshold.
        open_browser: open the written HTML in a browser.

    Returns:
        The Plotly figure (also useful for tests / notebooks).

    """
    cols = _read_metrics(Path(csv_path))
    x = cols.get("iteration", np.arange(1, len(next(iter(cols.values()))) + 1))

    panels = [c for c in _PANELS if c in cols and np.isfinite(cols[c]).any()]
    if not panels:
        raise ValueError(f"No plottable columns in {csv_path}: found {list(cols)}")

    ncols = 1 if len(panels) == 1 else min(3, len(panels))
    nrows = (len(panels) + ncols - 1) // ncols
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=[_PANELS[c][0] for c in panels],
        horizontal_spacing=0.08,
        vertical_spacing=0.16,
    )

    for i, col in enumerate(panels):
        row, cell = divmod(i, ncols)
        row += 1
        cell += 1
        _, ylabel, color = _PANELS[col]
        y = cols[col]

        if col == "mean_return":
            # Return is the headline: raw + smoothed trend + best-so-far envelope.
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    name="per iter",
                    line={"color": style.MUTED, "width": 1},
                    opacity=0.6,
                    hovertemplate="iter %{x}<br>%{y:.3f}<extra></extra>",
                ),
                row=row,
                col=cell,
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=_smooth(y),
                    mode="lines",
                    name="smoothed",
                    line={"color": color, "width": 3},
                    hovertemplate="iter %{x}<br>%{y:.3f}<extra></extra>",
                ),
                row=row,
                col=cell,
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=np.maximum.accumulate(y),
                    mode="lines",
                    name="best so far",
                    line={"color": style.GOOD, "width": 1.5, "dash": "dash"},
                    opacity=0.8,
                    hovertemplate="iter %{x}<br>best %{y:.3f}<extra></extra>",
                ),
                row=row,
                col=cell,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    showlegend=False,
                    line={"color": _to_rgba(color, 0.45), "width": 1.2},
                    hoverinfo="skip",
                ),
                row=row,
                col=cell,
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=_smooth(y),
                    mode="lines",
                    showlegend=False,
                    line={"color": color, "width": 2.4},
                    hovertemplate="iter %{x}<br>%{y:.4f}<extra></extra>",
                ),
                row=row,
                col=cell,
            )

        if col == "approx_kl" and target_kl is not None:
            fig.add_hline(
                y=target_kl,
                line={"color": style.CRITICAL, "width": 1.2, "dash": "dot"},
                annotation_text=f"target {target_kl:g}",
                annotation_font_color=style.CRITICAL,
                row=row,
                col=cell,
            )

        fig.update_yaxes(title_text=ylabel, row=row, col=cell)
        if row == nrows:
            fig.update_xaxes(title_text="iteration", row=row, col=cell)

    fig.update_layout(
        **style.base_layout(
            title={"text": title},
            height=280 * nrows + 80,
            margin={"l": 60, "r": 24, "t": 90, "b": 48},
            legend={  # top margin, level with the title — clear of subplot titles
                "orientation": "h",
                "yref": "container",
                "yanchor": "top",
                "y": 0.99,
                "xanchor": "right",
                "x": 1,
            },
        )
    )
    style.style_axes(fig)

    if save_path is not None:
        save_path = Path(save_path).with_suffix(".html")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(save_path, include_plotlyjs="cdn")
        logger.info(f"Saved training dashboard → {save_path}")
        if open_browser:
            webbrowser.open(save_path.resolve().as_uri())
    return fig
