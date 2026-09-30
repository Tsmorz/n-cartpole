"""Shared visual style for all figures — one design system, built on Plotly.

The palette mirrors the light theme of the personal site
(``personal-site/assets/css/main.css``), so the training dashboard and the
episode replay read like the browser demo. Pulling a color means naming the
*role* it plays.
"""

from __future__ import annotations

from typing import Any

# --- Brand ------------------------------------------------------------------
ACCENT = "#0d7a75"  # --accent (teal): cart, cart position, "good"

# --- Categorical series (fixed order; assign by slot, never cycle) -----------
# Slots 1-4 are the site's --viz-1..4; the rest extend the set for dashboards.
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
INK = "#14181d"  # --text: primary text, rods
INK_2 = "#5b636d"  # --text-muted: secondary text
MUTED = "#8a929c"  # axis labels / de-emphasized marks
GRID = "#e3e6ea"  # --border: hairline gridlines
BASELINE = "#cfd4da"  # --border-strong: zero line / axis spine / rail
SURFACE = "#f5f6f8"  # --bg-soft: plot / scene surface
PAGE = "#ffffff"  # --bg: page plane behind the chart

# --- Status (reserved — never reused as a series) ----------------------------
GOOD = ACCENT
WARNING = "#eda100"
CRITICAL = "#c9524b"  # --nn-act-neg

# --- Semantic roles used across the cartpole figures -------------------------
POLE1 = SERIES[0]  # first link — blue
POLE2 = SERIES[1]  # second link — orange
CART = ACCENT  # the cart body
FORCE = SERIES[3]  # applied force — yellow
REWARD = SERIES[2]  # per-step reward — aqua
LIMIT = CRITICAL  # rail / force limits

# Diverging ramp for signed fields (e.g. force output): red <- neutral -> teal.
DIVERGING = [
    [0.0, "#c9524b"],
    [0.25, "#e2a5a1"],
    [0.5, "#d3d8de"],
    [0.75, "#7fbdb9"],
    [1.0, "#0d7a75"],
]
# Sequential ramp for magnitude fields (e.g. value function): light -> dark teal.
SEQUENTIAL = [
    [0.0, "#d9efed"],
    [0.25, "#a5d6d2"],
    [0.5, "#5fb1ab"],
    [0.75, "#0d7a75"],
    [1.0, "#084744"],
]

FONT = '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif'
MONO = '"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace'


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
        "font": {"family": FONT, "color": INK_2, "size": 13},
        "title": {"font": {"family": FONT, "color": INK, "size": 18}, "x": 0.02},
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
        tickfont={"family": MONO, "size": 10},
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
        tickfont={"family": MONO, "size": 10},
    )
