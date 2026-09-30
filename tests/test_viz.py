"""Tests for n_cartpole.viz: style helpers, animate, and plots."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest

import n_cartpole.viz  # noqa: F401 — triggers __init__ coverage
from n_cartpole.env.dynamics import PhysicsParams
from n_cartpole.viz import animate_episode, plot_training_curves
from n_cartpole.viz.animate import _cart_shape, _joint_coords
from n_cartpole.viz.plots import _smooth
from n_cartpole.viz.style import base_layout, link_color, rgba, style_axes


# ---------------------------------------------------------------------------
# style.py helpers
# ---------------------------------------------------------------------------


def test_link_color_first_two_slots() -> None:
    from n_cartpole.viz.style import POLE1, POLE2

    assert link_color(0) == POLE1
    assert link_color(1) == POLE2


def test_link_color_higher_indices() -> None:
    # should cycle without raising
    for k in range(2, 10):
        c = link_color(k)
        assert c.startswith("#")
        assert len(c) == 7


def test_rgba_output_format() -> None:
    result = rgba("#ff0000", 0.5)
    assert result == "rgba(255,0,0,0.5)"


def test_rgba_black() -> None:
    result = rgba("#000000", 1.0)
    assert result == "rgba(0,0,0,1.0)"


def test_base_layout_keys() -> None:
    layout = base_layout()
    assert "paper_bgcolor" in layout
    assert "plot_bgcolor" in layout
    assert "font" in layout


def test_base_layout_override() -> None:
    layout = base_layout(height=400)
    assert layout["height"] == 400
    assert "paper_bgcolor" in layout


def test_style_axes_runs() -> None:
    import plotly.graph_objects as go

    fig = go.Figure()
    style_axes(fig)  # should not raise


# ---------------------------------------------------------------------------
# animate.py geometry helpers
# ---------------------------------------------------------------------------


def test_joint_coords_upright_single_link() -> None:
    p = PhysicsParams(lengths=0.5)
    # theta=0 means upright: pole points straight up
    state = np.array([0.0, 0.0, 0.0, 0.0])  # single link
    xs, ys = _joint_coords(state, p, n_links=1)
    assert len(xs) == 2
    assert len(ys) == 2
    assert xs[0] == pytest.approx(0.0)
    assert ys[0] == pytest.approx(0.0)
    # upright: sin(0)=0, cos(0)=1 → tip directly above
    assert xs[1] == pytest.approx(0.0, abs=1e-9)
    assert ys[1] == pytest.approx(0.5, abs=1e-9)


def test_joint_coords_double_link() -> None:
    p = PhysicsParams(lengths=0.25)
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    xs, ys = _joint_coords(state, p, n_links=2)
    assert len(xs) == 3
    assert len(ys) == 3


def test_joint_coords_hanging_down() -> None:
    p = PhysicsParams(lengths=1.0)
    state = np.array([0.0, 0.0, math.pi, 0.0])  # single link, hanging
    xs, ys = _joint_coords(state, p, n_links=1)
    # sin(pi)~0, cos(pi)=-1 → tip below pivot
    assert xs[1] == pytest.approx(0.0, abs=1e-6)
    assert ys[1] == pytest.approx(-1.0, abs=1e-6)


def test_cart_shape_dimensions() -> None:
    xs, ys = _cart_shape(0.0, 0.4, 0.1)
    assert len(xs) == 5  # closed rectangle
    assert len(ys) == 5
    assert xs[0] == pytest.approx(-0.2)
    assert xs[1] == pytest.approx(0.2)


def test_cart_shape_centered() -> None:
    cx = 1.5
    xs, ys = _cart_shape(cx, 0.3, 0.1)
    assert min(xs) == pytest.approx(cx - 0.15)
    assert max(xs) == pytest.approx(cx + 0.15)


# ---------------------------------------------------------------------------
# plots.py: _smooth
# ---------------------------------------------------------------------------


def test_smooth_short_array() -> None:
    y = np.array([1.0, 2.0, 3.0])
    out = _smooth(y)
    np.testing.assert_array_equal(out, y)


def test_smooth_even_length() -> None:
    y = np.arange(10.0)
    out = _smooth(y, window=3)
    assert len(out) == len(y)


def test_smooth_longer_array() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(size=50)
    out = _smooth(y, window=5)
    assert len(out) == len(y)
    assert not np.any(np.isnan(out))


# ---------------------------------------------------------------------------
# animate_episode (full figure build, no browser/file I/O)
# ---------------------------------------------------------------------------


def _make_states(T: int = 20, n_links: int = 2) -> np.ndarray:
    """Synthetic state sequence: all zeros (poles upright)."""
    state_dim = 2 + 2 * n_links
    return np.zeros((T, state_dim), dtype=np.float64)


def test_animate_episode_returns_figure() -> None:
    import plotly.graph_objects as go

    states = _make_states()
    fig = animate_episode(states, open_browser=False)
    assert isinstance(fig, go.Figure)


def test_animate_episode_single_link() -> None:
    import plotly.graph_objects as go

    states = _make_states(n_links=1)
    fig = animate_episode(states, open_browser=False)
    assert isinstance(fig, go.Figure)


def test_animate_episode_with_actions_and_rewards() -> None:
    import plotly.graph_objects as go

    T = 15
    states = _make_states(T=T)
    actions = np.zeros(T - 1)
    rewards = np.ones(T - 1) * 0.8
    fig = animate_episode(
        states,
        actions=actions,
        rewards=rewards,
        title="Test episode",
        open_browser=False,
    )
    assert isinstance(fig, go.Figure)


def test_animate_episode_saves_html(tmp_path: Path) -> None:
    import plotly.graph_objects as go

    states = _make_states()
    save_path = tmp_path / "test_replay.html"
    fig = animate_episode(states, save_path=save_path, open_browser=False)
    assert isinstance(fig, go.Figure)
    assert save_path.exists()


# ---------------------------------------------------------------------------
# plot_training_curves (full figure build)
# ---------------------------------------------------------------------------


def _write_metrics_csv(path: Path, n_rows: int = 30) -> None:
    cols = ["iteration", "mean_return", "value_loss", "policy_loss", "entropy", "approx_kl", "clip_fraction"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        rng = np.random.default_rng(42)
        for i in range(1, n_rows + 1):
            w.writerow({
                "iteration": i,
                "mean_return": float(rng.uniform(-5, 5)),
                "value_loss": float(rng.uniform(0.01, 2.0)),
                "policy_loss": float(rng.uniform(-0.1, 0.1)),
                "entropy": float(rng.uniform(0.5, 2.0)),
                "approx_kl": float(rng.uniform(0.001, 0.05)),
                "clip_fraction": float(rng.uniform(0.0, 0.3)),
            })


def test_plot_training_curves_returns_figure(tmp_path: Path) -> None:
    import plotly.graph_objects as go

    csv_path = tmp_path / "metrics.csv"
    _write_metrics_csv(csv_path)
    fig = plot_training_curves(csv_path, open_browser=False)
    assert isinstance(fig, go.Figure)


def test_plot_training_curves_saves_html(tmp_path: Path) -> None:
    csv_path = tmp_path / "metrics.csv"
    _write_metrics_csv(csv_path)
    save_path = tmp_path / "dash.html"
    plot_training_curves(csv_path, save_path=save_path, open_browser=False)
    assert save_path.exists()


def test_plot_training_curves_no_plottable_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("foo,bar\n1,2\n")
    with pytest.raises(ValueError, match="No plottable columns"):
        plot_training_curves(csv_path, open_browser=False)


def test_plot_training_curves_minimal_csv(tmp_path: Path) -> None:
    import plotly.graph_objects as go

    csv_path = tmp_path / "minimal.csv"
    csv_path.write_text("iteration,mean_return\n1,10.0\n2,12.0\n")
    fig = plot_training_curves(csv_path, open_browser=False)
    assert isinstance(fig, go.Figure)
