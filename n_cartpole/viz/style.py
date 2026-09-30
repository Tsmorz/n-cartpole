"""Shared visual style for all figures — one design system, built on Plotly.

Centralizing the palette here keeps the training dashboard, the episode replay,
and the policy map reading as one system. Colors are the validated
categorical / ink / status slots from the data-viz reference palette (CVD-checked
in fixed order — never cycled); pulling a color means naming the *role* it plays.
"""

from __future__ import annotations

from typing import Any

# --- Categorical series (fixed order; assign by slot, never cycle) -----------
SERIES: tuple[str, ...] = (
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
)

# --- Ink / chrome ------------------------------------------------------------
INK = "#0b0b0b"  # primary text
INK_2 = "#52514e"  # secondary text
MUTED = "#898781"  # axis labels / de-emphasized marks
GRID = "#e1e0d9"  # hairline gridlines
BASELINE = "#c3c2b7"  # zero line / axis spine
SURFACE = "#fcfcfb"  # chart surface
PAGE = "#f9f9f7"  # page plane behind the chart

# --- Status (reserved — never reused as a series) ----------------------------
GOOD = "#0ca30c"
WARNING = "#fab219"
CRITICAL = "#d03b3b"

# --- Semantic roles used across the cartpole figures -------------------------
POLE1 = SERIES[0]  # first link — blue
POLE2 = SERIES[1]  # second link — orange
CART = "#3a3a38"  # the cart body: a neutral dark, not a data series
FORCE = SERIES[6]  # applied force — violet (distinct from the link hues)
REWARD = SERIES[2]  # per-step reward — aqua
LIMIT = CRITICAL  # rail / force limits

# Diverging ramp for signed fields (e.g. force output): red ← neutral → blue.
# Blue = push right (+F), red = push left (-F); gray reads as "do nothing".
DIVERGING = [
    [0.0, "#d03b3b"],
    [0.25, "#eb9d8c"],
    [0.5, "#f0efec"],
    [0.75, "#86b6ef"],
    [1.0, "#184f95"],
]
# Sequential ramp for magnitude fields (e.g. value function): light → dark blue.
SEQUENTIAL = [
    [0.0, "#cde2fb"],
    [0.25, "#9ec5f4"],
    [0.5, "#5598e7"],
    [0.75, "#2a78d6"],
    [1.0, "#104281"],
]

_FONT = "-apple-system, Segoe UI, Helvetica, Arial, sans-serif"


def link_color(k: int) -> str:
    """Return the categorical color for pendulum link ``k`` (0-indexed).

    Links 0 and 1 reuse the POLE1/POLE2 roles; higher links cycle the remaining
    categorical slots (skipping the reward/force hues where possible).
    """
    if k == 0:
        return POLE1
    if k == 1:
        return POLE2
    # Remaining slots for links 3+: yellow, magenta, green, violet, red, aqua.
    extra = (SERIES[3], SERIES[4], SERIES[5], SERIES[6], SERIES[7], SERIES[2])
    return extra[(k - 2) % len(extra)]


def rgba(hex_color: str, alpha: float) -> str:
    """`#rrggbb` -> `rgba(r,g,b,alpha)` for translucent fills/bands."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def base_layout(**overrides: Any) -> dict[str, Any]:
    """Return a Plotly layout dict with the shared chrome; merge in overrides."""
    layout: dict[str, Any] = {
        "paper_bgcolor": PAGE,
        "plot_bgcolor": SURFACE,
        "font": {"family": _FONT, "color": INK_2, "size": 13},
        "title": {"font": {"family": _FONT, "color": INK, "size": 18}, "x": 0.02},
        "legend": {"bgcolor": "rgba(0,0,0,0)"},
        "margin": {"l": 60, "r": 24, "t": 60, "b": 48},
    }
    layout.update(overrides)
    return layout


def style_axes(fig: Any) -> None:
    """Apply the recessive grid / baseline treatment to every axis in a figure."""
    fig.update_xaxes(
        showgrid=True,
        gridcolor=GRID,
        gridwidth=1,
        zeroline=True,
        zerolinecolor=BASELINE,
        zerolinewidth=1,
        linecolor=BASELINE,
        ticks="outside",
        tickcolor=BASELINE,
        ticklen=4,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRID,
        gridwidth=1,
        zeroline=True,
        zerolinecolor=BASELINE,
        zerolinewidth=1,
        linecolor=BASELINE,
        ticks="outside",
        tickcolor=BASELINE,
        ticklen=4,
    )
