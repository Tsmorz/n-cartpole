"""Interactive HTML report of a system-identification fit (plotly, repo style).

Measured vs simulated readings over every shooting window (fitted and config
model), the fitted model's residuals, the delay grid, and the parameter table.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from n_cartpole.sysid.fit import FitResult
from n_cartpole.viz import style

_UNITS = {"x": "m", "th": "rad"}


def _channel_names(n: int) -> list[str]:
    return ["x"] + [f"th{k}" for k in range(n)]


def _series(result: FitResult, values: np.ndarray, gap: float = 1.0):
    """Concatenate windows into one trace (None between windows), recordings offset."""
    w = result.windows
    offsets: dict[int, float] = {}
    t_end = 0.0
    for r in sorted(set(w.rec.tolist())):
        offsets[r] = t_end
        t_end += float(w.t[w.rec == r].max()) + gap
    xs: list[float | None] = []
    ys: list[list[float | None]] = [[] for _ in range(values.shape[2])]
    for b in range(values.shape[0]):
        xs.extend((w.t[b] + offsets[int(w.rec[b])]).tolist())
        xs.append(None)
        for c in range(values.shape[2]):
            ys[c].extend(values[b, :, c].tolist())
            ys[c].append(None)
    return xs, ys


def _fit_figure(result: FitResult) -> go.Figure:
    n = result.n_links
    names = _channel_names(n)
    rows = len(names) * 2
    titles = []
    for nm in names:
        unit = _UNITS["x" if nm == "x" else "th"]
        titles += [f"{nm} ({unit}): measured vs simulated", f"{nm} residual ({unit})"]
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=titles,
    )
    t, meas = _series(result, result.windows.meas)
    _, fit = _series(result, result.sim)
    _, nom = _series(result, result.sim_nominal)
    _, err = _series(result, result.sim - result.windows.meas)
    for c, nm in enumerate(names):
        color = style.CART if c == 0 else style.link_color(c - 1)
        row = 2 * c + 1
        show = c == 0
        fig.add_trace(
            go.Scatter(
                x=t,
                y=meas[c],
                name="measured",
                legendgroup="m",
                showlegend=show,
                mode="lines",
                line={"color": style.INK, "width": 1},
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=nom[c],
                name="config model",
                legendgroup="n",
                showlegend=show,
                mode="lines",
                line={"color": style.CRITICAL, "width": 1, "dash": "dot"},
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=fit[c],
                name="fitted model",
                legendgroup="f",
                showlegend=show,
                mode="lines",
                line={"color": color, "width": 1.5},
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=err[c],
                name=f"{nm} residual",
                showlegend=False,
                mode="lines",
                line={"color": color, "width": 1},
            ),
            row=row + 1,
            col=1,
        )
    fig.update_xaxes(title_text="time (s; recordings back to back)", row=rows, col=1)
    fig.update_layout(
        **style.base_layout(
            title={"text": "Shooting windows: readings vs simulation"},
            height=230 * rows,
            hovermode="x unified",
        )
    )
    style.style_axes(fig)
    return fig


def _delay_figure(result: FitResult) -> go.Figure:
    ds = sorted(result.delay_cost)
    fig = go.Figure(
        go.Bar(
            x=[str(d) for d in ds],
            y=[result.delay_cost[d] for d in ds],
            marker_color=[
                style.ACCENT if d == result.delay else style.MUTED for d in ds
            ],
            hovertemplate="delay %{x} steps<br>mean sq. residual %{y:.4g}<extra></extra>",
        )
    )
    fig.update_layout(
        **style.base_layout(
            title={"text": "Actuator delay grid (lower is better)"},
            xaxis_title="delay (control steps)",
            yaxis_title="mean squared scaled residual",
            height=320,
        )
    )
    style.style_axes(fig)
    return fig


def _param_table(result: FitResult) -> go.Figure:
    rows = [
        (
            e.name,
            f"{e.nominal:.5g}",
            f"{e.value:.5g}",
            f"{e.se:.2g}" if np.isfinite(e.se) else "-",
            f"{100 * e.rel_change:+.1f}%" if np.isfinite(e.rel_change) else "-",
            "fitted" if e.fitted else ("derived" if np.isfinite(e.se) else "fixed"),
        )
        for e in result.estimates
    ]
    rows.append(
        ("delay_steps", str(result.nominal_delay), str(result.delay), "-", "-", "grid")
    )
    cols = list(zip(*rows, strict=True))
    fig = go.Figure(
        go.Table(
            header={
                "values": ["parameter", "config", "fitted", "± se", "change", ""],
                "fill_color": style.SURFACE,
                "font": {"family": style.FONT, "color": style.INK},
                "align": "left",
            },
            cells={
                "values": cols,
                "font": {"family": style.MONO, "size": 12, "color": style.INK},
                "align": "left",
                "fill_color": style.PAGE,
            },
        )
    )
    fig.update_layout(
        **style.base_layout(title={"text": "Parameters"}, height=40 + 28 * len(rows))
    )
    return fig


def write_report(
    result: FitResult, path: str | Path, sources: Sequence[str] = ()
) -> Path:
    """Write the HTML report; return its path."""
    names = _channel_names(result.n_links)
    rms = "".join(
        f"<li>{nm}: {result.rms_nominal[nm]:.3g} &rarr; <b>{result.rms[nm]:.3g}</b> "
        f"{_UNITS['x' if nm == 'x' else 'th']}</li>"
        for nm in names
    )
    warn = "".join(f"<p class='warn'>{html.escape(w)}</p>" for w in result.warnings)
    src = ", ".join(html.escape(s) for s in sources) or "(in memory)"
    figs = [_param_table(result), _delay_figure(result), _fit_figure(result)]
    body = "\n".join(
        f.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False)
        for i, f in enumerate(figs)
    )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>System identification</title>
<style>
body {{ font-family: {style.FONT}; color: {style.INK}; background: {style.PAGE};
       max-width: 1200px; margin: 0 auto; padding: 16px; }}
.warn {{ color: {style.CRITICAL}; }}
code {{ font-family: {style.MONO}; }}
</style></head><body>
<h1>System identification</h1>
<p>Recordings: {src}. {result.windows.meas.shape[0]} windows of
{result.windows.meas.shape[1]} samples. Actuator delay {result.delay} step(s)
(config {result.nominal_delay}); force gain {result.force_gain:.4f}.</p>
<p>Reading RMS, config model &rarr; fitted model:</p><ul>{rms}</ul>
{warn}
{body}
</body></html>
"""
    out = Path(path)
    out.write_text(page)
    return out
