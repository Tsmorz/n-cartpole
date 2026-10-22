"""Write a fitted config TOML for ``train.py --config`` (or ``--resume --config``).

The output is the source config with:

- ``[physics]`` replaced by the fitted nominal plant as direct values (``M``,
  ``masses``, ``com``, ``inertia``, ``joint_friction``, ...; the part-level
  ``[physics.drive]`` / ``[physics.links]`` tables are dropped, ``force_max`` and
  ``v_noload`` keep the values they derived),
- ``[hardware.pipeline].delay_steps`` set to the fitted delay,
- ``[randomize]`` narrowed to the fit's uncertainty: each identified range becomes
  ``sigma`` standard errors (never wider than the source, never narrower than a
  floor that covers model error the fit cannot see).
"""

from __future__ import annotations

import dataclasses
import json
import math
import tomllib
from pathlib import Path
from typing import Any

import numpy as np

from n_cartpole.env.randomization import PlantRandomization
from n_cartpole.sysid.fit import FitResult

REL_FLOOR = 0.02  # +-2 %
SCALE_FLOOR = 1.1  # x/÷ 1.1


def _fmt_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int | np.integer):
        return str(int(v))
    if isinstance(v, float | np.floating):
        f = float(v)
        if not math.isfinite(f):
            raise ValueError(f"cannot write non-finite value {f} to TOML")
        return repr(f)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, list | tuple | np.ndarray):
        return "[" + ", ".join(_fmt_value(x) for x in v) + "]"
    raise TypeError(f"unsupported TOML value {v!r}")


def dump_toml(data: dict[str, Any], header: str = "") -> str:
    """Serialize nested dicts of scalars/lists (the subset config files use)."""
    out: list[str] = [f"# {line}".rstrip() for line in header.splitlines()]

    def table(prefix: str, d: dict[str, Any]) -> None:
        scalars = {k: v for k, v in d.items() if not isinstance(v, dict)}
        tables = {k: v for k, v in d.items() if isinstance(v, dict)}
        if prefix and (scalars or not tables):
            out.append("")
            out.append(f"[{prefix}]")
        for k, v in scalars.items():
            out.append(f"{k} = {_fmt_value(v)}")
        for k, v in tables.items():
            table(f"{prefix}.{k}" if prefix else k, v)

    table("", data)
    return "\n".join(out).lstrip("\n") + "\n"


def _narrow_rel(src: float, se_rel: float, sigma: float) -> float:
    return float(min(src, max(REL_FLOOR, sigma * se_rel)))


def _narrow_scale(src: tuple[float, float], se_log: float, sigma: float) -> list[float]:
    src_factor = max(src[1], 1.0 / src[0])
    f = min(src_factor, max(SCALE_FLOOR, math.exp(sigma * se_log)))
    return [1.0 / f, f]


def narrowed_randomization(
    result: FitResult, src: PlantRandomization, sigma: float = 3.0
) -> dict[str, Any]:
    """Return the ``[randomize]`` table narrowed to ``sigma`` standard errors."""
    out = {
        k: (list(v) if isinstance(v, tuple) else v)
        for k, v in dataclasses.asdict(src).items()
    }

    def rel_se(name: str) -> float | None:
        try:
            e = result.estimate(name)
        except KeyError:
            return None
        if not np.isfinite(e.se) or e.value == 0.0:
            return None
        return abs(e.se / e.value)

    def links_rel(prefix: str) -> float | None:
        vals = [rel_se(e.name) for e in result.estimates if e.name.startswith(prefix)]
        found = [v for v in vals if v is not None]
        return max(found) if found else None

    # The dynamics see the links only through a_k, d_k: once those are fitted, a
    # wider link-mass randomization would undo the narrowing (lengths stay as set).
    moments = [links_rel("a"), links_rel("d")]
    moments_rel = max((m for m in moments if m is not None), default=None)
    pairs_rel = (
        ("cart_mass_rel", rel_se("M")),
        ("link_mass_rel", moments_rel),
        ("force_gain_rel", rel_se("force_gain")),
        ("link_com_rel", links_rel("com")),
        ("link_inertia_rel", links_rel("inertia")),
    )
    for key, se in pairs_rel:
        if se is not None:
            out[key] = _narrow_rel(getattr(src, key), se, sigma)
    log_se = result.log_se
    jf = [v for k, v in log_se.items() if k.startswith("joint_friction")]
    pairs_scale = (
        ("cart_friction_scale", log_se.get("b")),
        ("cart_coulomb_scale", log_se.get("cart_coulomb")),
        ("joint_friction_scale", max(jf) if jf else None),
    )
    for key, se in pairs_scale:
        if se is not None and np.isfinite(se):
            out[key] = _narrow_scale(getattr(src, key), se, sigma)
    return out


def fitted_config(
    source: str | Path, result: FitResult, sigma: float = 3.0
) -> dict[str, Any]:
    """Return the source config's tables with the fitted physics, delay, ranges."""
    with open(source, "rb") as f:
        data = tomllib.load(f)
    p = result.physics
    n = result.n_links
    src_phys = dict(data.get("physics", {}))
    phys: dict[str, Any] = {
        k: v for k, v in src_phys.items() if not isinstance(v, dict)
    }
    for k in ("masses", "com", "inertia", "lengths"):
        phys.pop(k, None)
    phys.update(
        {
            "dt": p.dt,
            "g": p.g,
            "x_lim": p.x_lim,
            "M": p.M,
            "force_max": p.force_max,
            "b": p.b,
            "cart_coulomb": p.cart_coulomb,
            "cart_coulomb_vscale": p.cart_coulomb_vscale,
            "lengths": [float(x) for x in p.link_lengths(n)],
            "masses": [float(x) for x in p.link_masses(n)],
            "com": [float(x) for x in p.link_coms(n)],
            "inertia": [float(x) for x in p.link_inertias(n)],
            "joint_friction": [float(x) for x in p.joint_frictions(n)],
        }
    )
    if p.v_noload is not None:
        phys["v_noload"] = p.v_noload
    data["physics"] = phys
    hw = data.setdefault("hardware", {})
    pipe = hw.setdefault("pipeline", {})
    pipe["delay_steps"] = int(result.delay)
    pipe.setdefault("sysid_context", False)
    src_kwargs: dict[str, Any] = {
        k: (tuple(v) if isinstance(v, list) else v)
        for k, v in data.get("randomize", {}).items()
    }
    src_rnd = PlantRandomization(**src_kwargs)
    data["randomize"] = narrowed_randomization(result, src_rnd, sigma)
    return data


def write_fitted_config(
    source: str | Path, result: FitResult, out: str | Path, sigma: float = 3.0
) -> str:
    """Write the fitted config TOML; return its text."""
    data = fitted_config(source, result, sigma)
    gain = result.force_gain
    header = (
        f"Fitted by scripts/sysid.py from {source}.\n"
        "Physics: identified nominal plant (direct values; edit with care).\n"
        f"Actuator delay: {result.delay} step(s). Force gain: {gain:.4f}"
        + (
            " -> scale the firmware's PWM full-scale force by this factor."
            if abs(gain - 1.0) > 1e-12
            else " (fixed)."
        )
        + f"\n[randomize]: narrowed to {sigma:g} standard errors of the fit."
    )
    text = dump_toml(data, header)
    Path(out).write_text(text)
    return text
