"""Hardware configuration for physical cartpole deployment.

Fill in ``physics`` with bench-measured values from your specific rig.  The
``*_noise`` fields set the Gaussian std-dev added to each parameter when
building the sysID context vector that the network receives.  During training
this noise forces the policy to be robust to measurement error; at deploy time
you can set them to zero if you trust your measurements.

``delay_steps`` and ``sensor_noise_std`` model the hardware pipeline rather
than the pendulum mechanics.  Action delay is the number of 100 Hz control
steps between issuing a command and the actuator applying it (typically 1-2
steps for an ESP32 over USB/UART).  Sensor noise is the Gaussian std-dev added
to each observation dimension to match encoder / IMU noise floor.

The canonical source of these values is ``config/default.toml``.  Load it
with :meth:`HardwareConfig.from_toml` or via :func:`n_cartpole.config.load_tqc_config`.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from n_cartpole.env.dynamics import PhysicsParams


@dataclass
class HardwareConfig:
    """Measured physical parameters and pipeline properties for a real rig.

    ``physics`` holds the best-estimate nominal values from bench sysID.  The
    ``*_noise`` fields are std-devs that perturb the sysID context vector fed
    to the network — sized to match your actual measurement confidence.

    The sysID context is resampled once per episode so the network sees
    different plausible measurements each rollout, training it to cope with the
    gap between your measurement and the true hardware state.
    """

    # Best-estimate physical parameters from bench measurement.
    physics: PhysicsParams = field(default_factory=PhysicsParams)

    # Measurement uncertainty (Gaussian std dev, same units as the parameter).
    M_noise: float = 0.05
    b_noise: float = 0.03
    masses_noise: float = 0.005
    lengths_noise: float = 0.005
    joint_friction_noise: float = 0.0005

    # Number of control loop steps between issuing a command and the actuator
    # applying it.  At 100 Hz, 1 step = 10 ms.  Typical for ESP32 USB/UART: 1-2.
    delay_steps: int = 1

    # Gaussian std dev added independently to each observation dimension to
    # model encoder / IMU noise.  Set to 0.0 if your sensors are clean.
    sensor_noise_std: float = 0.0

    # Append the (noisy) sysID context vector to every observation. Turn off to
    # keep the observation purely kinematic (+ goal) while still modelling the
    # action delay and sensor noise — the actor then needs no parameter inputs,
    # and robustness to model error comes from ``EnvConfig.randomize`` instead.
    sysid_context: bool = True

    @classmethod
    def from_toml(cls, path: str | Path) -> HardwareConfig:
        """Load from the ``[hardware]`` section of a TOML config file.

        Returns ``None`` if the file has no ``[hardware]`` section, so callers
        can do ``hw = HardwareConfig.from_toml(path)`` and get ``None`` when
        hardware mode is not configured.
        """
        with open(path, "rb") as f:
            data = tomllib.load(f)
        hw = data.get("hardware")
        if hw is None:
            raise KeyError(f"No [hardware] section found in {path}")
        phys_data = hw.get("physics", {})
        noise_data = hw.get("noise", {})
        pipe_data = hw.get("pipeline", {})
        return cls(
            physics=PhysicsParams(**phys_data) if phys_data else PhysicsParams(),
            M_noise=noise_data.get("M_noise", 0.05),
            b_noise=noise_data.get("b_noise", 0.03),
            masses_noise=noise_data.get("masses_noise", 0.005),
            lengths_noise=noise_data.get("lengths_noise", 0.005),
            joint_friction_noise=noise_data.get("joint_friction_noise", 0.0005),
            delay_steps=pipe_data.get("delay_steps", 1),
            sensor_noise_std=pipe_data.get("sensor_noise_std", 0.0),
            sysid_context=pipe_data.get("sysid_context", True),
        )
