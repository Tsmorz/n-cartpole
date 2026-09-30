"""Tests for config.py (TOML loading) and hardware_config.py."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from n_cartpole.env.hardware_config import HardwareConfig

_DEFAULT_TOML = Path(__file__).parent.parent / "config" / "default.toml"


# ---------------------------------------------------------------------------
# HardwareConfig dataclass
# ---------------------------------------------------------------------------


def test_hardware_config_defaults() -> None:
    """HardwareConfig default field values match documented constants."""
    hw = HardwareConfig()
    assert hw.M_noise == pytest.approx(0.05)
    assert hw.delay_steps == 1
    assert hw.sensor_noise_std == pytest.approx(0.0)


def test_hardware_config_from_toml(tmp_path: Path) -> None:
    """from_toml reads physics, noise, and pipeline sections correctly."""
    toml_text = textwrap.dedent("""
        [hardware.physics]
        M = 1.5
        [hardware.noise]
        M_noise = 0.02
        b_noise = 0.01
        [hardware.pipeline]
        delay_steps = 2
        sensor_noise_std = 0.005
    """)
    p = tmp_path / "hw.toml"
    p.write_text(toml_text)
    hw = HardwareConfig.from_toml(p)
    assert hw.physics.M == pytest.approx(1.5)
    assert hw.M_noise == pytest.approx(0.02)
    assert hw.delay_steps == 2
    assert hw.sensor_noise_std == pytest.approx(0.005)


def test_hardware_config_from_toml_missing_section(tmp_path: Path) -> None:
    """from_toml raises KeyError when no [hardware] section is present."""
    p = tmp_path / "no_hw.toml"
    p.write_text("[env]\nn_links = 2\n")
    with pytest.raises(KeyError, match="hardware"):
        HardwareConfig.from_toml(p)


def test_hardware_config_from_toml_defaults(tmp_path: Path) -> None:
    """from_toml falls back to HardwareConfig defaults for missing keys."""
    toml_text = "[hardware]\n"
    p = tmp_path / "hw_empty.toml"
    p.write_text(toml_text)
    hw = HardwareConfig.from_toml(p)
    assert hw.delay_steps == 1
    assert hw.M_noise == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# config.py: load_ppo_config / load_tqc_config
# ---------------------------------------------------------------------------


def test_load_ppo_config_default_toml() -> None:
    """load_ppo_config reads the shipped default.toml without error."""
    from n_cartpole.config import load_ppo_config

    cfg = load_ppo_config(_DEFAULT_TOML)
    assert cfg.env.n_links == 2
    assert cfg.gamma == pytest.approx(0.99)
    assert cfg.lr > 0


def test_load_tqc_config_default_toml() -> None:
    """load_tqc_config reads the shipped default.toml without error."""
    from n_cartpole.config import load_tqc_config

    cfg = load_tqc_config(_DEFAULT_TOML)
    assert cfg.env.n_links == 2
    assert cfg.gamma == pytest.approx(0.99)
    assert cfg.n_critics >= 1


def test_load_ppo_config_with_hardware(tmp_path: Path) -> None:
    """load_ppo_config builds HardwareConfig when [hardware] section is present."""
    from n_cartpole.config import load_ppo_config

    toml_text = textwrap.dedent("""
        [physics]
        M = 1.0
        [env]
        n_links = 1
        [hardware.physics]
        M = 1.1
        [hardware.noise]
        M_noise = 0.01
        [hardware.pipeline]
        delay_steps = 1
    """)
    p = tmp_path / "hw_ppo.toml"
    p.write_text(toml_text)
    cfg = load_ppo_config(p)
    assert cfg.env.hardware is not None
    assert cfg.env.hardware.physics.M == pytest.approx(1.1)
    assert cfg.env.n_links == 1


def test_load_tqc_config_with_hardware(tmp_path: Path) -> None:
    """load_tqc_config builds HardwareConfig when [hardware] section is present."""
    from n_cartpole.config import load_tqc_config

    toml_text = textwrap.dedent("""
        [physics]
        M = 1.0
        [env]
        n_links = 2
        [hardware.physics]
        M = 0.9
        [hardware.noise]
        [hardware.pipeline]
        delay_steps = 0
    """)
    p = tmp_path / "hw_tqc.toml"
    p.write_text(toml_text)
    cfg = load_tqc_config(p)
    assert cfg.env.hardware is not None
    assert cfg.env.hardware.delay_steps == 0


def test_load_ppo_config_zero_workers(tmp_path: Path) -> None:
    """n_workers=0 in TOML triggers auto-detection and resolves to >= 1."""
    from n_cartpole.config import load_ppo_config

    toml_text = textwrap.dedent("""
        [physics]
        [env]
        [ppo]
        n_workers = 0
    """)
    p = tmp_path / "zero_workers.toml"
    p.write_text(toml_text)
    cfg = load_ppo_config(p)
    assert cfg.n_workers >= 1
