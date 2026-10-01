"""Load training configuration from a TOML file.

``config/default.toml`` is the single source of truth for all physical
parameters, environment settings, and training hyperparameters.  The functions
here read that file and construct the typed dataclasses consumed by the
trainers and environment.

Usage::

    from n_cartpole.config import load_tqc_config

    tqc_cfg = load_tqc_config("config/default.toml")
"""

from __future__ import annotations

import math
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from n_cartpole.training.off_policy import TQCConfig

from n_cartpole.env.cartpole import EnvConfig, RewardShape
from n_cartpole.env.dynamics import PhysicsParams, link_from_parts
from n_cartpole.env.hardware_config import HardwareConfig, SensorModel
from n_cartpole.env.randomization import PlantRandomization


def _link_specs_from_bars(d: dict) -> dict:
    """Derive per-link ``masses/com/inertia`` from real bar dimensions + tip masses.

    ``d`` is the ``[physics.links]`` table: a rectangular bar
    (``bar_width * bar_thickness``, ``bar_density``) spans each link's ``lengths``
    joint to joint, and ``tip_masses`` are point masses at each distal joint.
    """
    tips = [float(t) for t in d["links"]["tip_masses"]]
    n = len(tips)
    lengths = PhysicsParams(lengths=d.get("lengths", 0.25)).link_lengths(n)
    links = d["links"]
    area = links["bar_width"] * links["bar_thickness"]
    rod = area * links["bar_density"] * lengths
    parts = [link_from_parts(lengths[i], rod[i], tips[i]) for i in range(n)]
    return {
        "masses": tuple(p[0] for p in parts),
        "com": tuple(p[1] for p in parts),
        "inertia": tuple(p[2] for p in parts),
        "lengths": tuple(float(x) for x in lengths),
    }


def drive_from_parts(
    cart_mass: float,
    rotor_inertia: float,
    pulley_radius: float,
    peak_torque: float,
    noload_rpm: float,
    derate: float = 1.0,
) -> tuple[float, float, float]:
    """Effective cart mass, peak belt force and no-load cart speed of a belt drive.

    The motor rotor (and pulley) inertia ``J`` reflects to the cart as
    ``J / r^2`` — for a small pulley this is often the largest part of the moving
    mass — so ``M_eff = cart_mass + J/r^2``. The belt force is ``derate * tau / r``
    and the cart speed at which the motor's back-EMF reaches the supply is
    ``omega_noload * r`` (see ``PhysicsParams.v_noload``).
    """
    m_eff = cart_mass + rotor_inertia / pulley_radius**2
    force = derate * peak_torque / pulley_radius
    v_noload = noload_rpm * 2.0 * math.pi / 60.0 * pulley_radius
    return m_eff, force, v_noload


def _physics_from(d: dict) -> PhysicsParams:
    d = dict(d)
    if "drive" in d:  # derive cart mass, force limit and speed limit from parts
        dr = d["drive"]
        d["M"], d["force_max"], d["v_noload"] = drive_from_parts(
            dr["cart_mass"],
            dr["rotor_inertia"],
            dr["pulley_radius"],
            dr["peak_torque"],
            dr["noload_rpm"],
            dr.get("derate", 1.0),
        )
    specs = (
        _link_specs_from_bars(d)
        if "links" in d
        else {
            "masses": d.get("masses", (0.20, 0.15)),
            "lengths": d.get("lengths", 0.25),
            "com": d.get("com"),
            "inertia": d.get("inertia", 0.0),
        }
    )
    return PhysicsParams(
        M=d.get("M", 1.0),
        g=d.get("g", 9.81),
        dt=d.get("dt", 0.01),
        force_max=d.get("force_max", 20.0),
        x_lim=d.get("x_lim", 0.5),
        b=d.get("b", 0.10),
        joint_friction=d.get("joint_friction", 0.002),
        cart_coulomb=d.get("cart_coulomb", 0.0),
        cart_coulomb_vscale=d.get("cart_coulomb_vscale", 0.005),
        v_noload=d.get("v_noload"),
        **specs,
    )


def _randomization_from(d: dict) -> PlantRandomization:
    """Build ``PlantRandomization`` from the ``[randomize]`` table (lists → tuples)."""
    kwargs: dict[str, Any] = {
        k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()
    }
    return PlantRandomization(**kwargs)


def rig_summary(physics: PhysicsParams, n_links: int) -> str:
    """Human-readable table of the rig parameters a build must match."""
    m = physics.link_masses(n_links)
    ln = physics.link_lengths(n_links)
    c = physics.link_coms(n_links)
    inertia = physics.link_inertias(n_links)
    lines = [
        f"cart: M={physics.M:.3f} kg, b={physics.b:.3f} N·s/m, rail ±{physics.x_lim:.2f} m, "
        f"force ±{physics.force_max:.1f} N ({physics.force_max / physics.M:.1f} m/s² on the cart), "
        f"dt={physics.dt * 1000:.0f} ms",
        f"drive: Coulomb friction {physics.cart_coulomb:.2f} N, no-load speed "
        + ("unlimited" if physics.v_noload is None else f"{physics.v_noload:.2f} m/s"),
    ]
    for i in range(n_links):
        lines.append(
            f"link {i}: m={m[i]:.4f} kg, length={ln[i]:.3f} m, com={c[i]:.3f} m, "
            f"I_com={inertia[i]:.3e} kg·m², joint friction="
            f"{physics.joint_frictions(n_links)[i]:.4f} N·m·s/rad"
        )
    return "\n".join(lines)


def _env_from(data: dict) -> EnvConfig:
    phys = _physics_from(data.get("physics", {}))
    env_d = data.get("env", {})

    hw: HardwareConfig | None = None
    if "hardware" in data:
        hw_d = data["hardware"]
        hw_phys = _physics_from(hw_d.get("physics", data.get("physics", {})))
        noise = hw_d.get("noise", {})
        pipe = hw_d.get("pipeline", {})
        hw = HardwareConfig(
            physics=hw_phys,
            M_noise=noise.get("M_noise", 0.05),
            b_noise=noise.get("b_noise", 0.03),
            masses_noise=noise.get("masses_noise", 0.005),
            lengths_noise=noise.get("lengths_noise", 0.005),
            joint_friction_noise=noise.get("joint_friction_noise", 0.0005),
            delay_steps=pipe.get("delay_steps", 1),
            sensor_noise_std=pipe.get("sensor_noise_std", 0.0),
            sysid_context=pipe.get("sysid_context", True),
            sensors=SensorModel(**hw_d["sensors"]) if "sensors" in hw_d else None,
        )

    hold = env_d.get("goal_hold_steps", [400, 800])
    return EnvConfig(
        physics=phys,
        n_links=env_d.get("n_links", 2),
        max_steps=env_d.get("max_steps", 1000),
        init_noise=env_d.get("init_noise", 0.05),
        init_random_prob=env_d.get("init_random_prob", 0.3),
        init_vel_noise=env_d.get("init_vel_noise", 2.0),
        hardware=hw,
        randomize=_randomization_from(data["randomize"])
        if "randomize" in data
        else None,
        reward_shape=RewardShape(**data["reward"]) if "reward" in data else None,
        force_slew=env_d.get("force_slew", 800.0),
        goal_conditioned=env_d.get("goal_conditioned", False),
        goal_hold_steps=tuple(hold) if hold else None,
        goal_curriculum=env_d.get("goal_curriculum", True),
    )


def load_tqc_config(path: str | Path) -> TQCConfig:
    """Construct a :class:`~n_cartpole.training.off_policy.TQCConfig` from TOML."""
    from n_cartpole.training.off_policy import TQCConfig

    with open(path, "rb") as f:
        data = tomllib.load(f)

    env = _env_from(data)
    tqc = data.get("tqc", {})

    return TQCConfig(
        env=env,
        device=tqc.get("device", "auto"),
        hidden=tqc.get("hidden", 256),
        n_critics=tqc.get("n_critics", 2),
        n_quantiles=tqc.get("n_quantiles", 25),
        top_quantiles_to_drop=tqc.get("top_quantiles_to_drop", 2),
        gamma=tqc.get("gamma", 0.99),
        tau=tqc.get("tau", 0.005),
        lr=tqc.get("lr", 3e-4),
        batch_size=tqc.get("batch_size", 256),
        buffer_size=tqc.get("buffer_size", 1_000_000),
        start_steps=tqc.get("start_steps", 5_000),
        updates_per_step=tqc.get("updates_per_step", 1),
        symmetry_augment=tqc.get("symmetry_augment", True),
        total_steps=tqc.get("total_steps", 200_000),
        checkpoint_every=tqc.get("checkpoint_every", 25_000),
        log_every=tqc.get("log_every", 5_000),
    )
