"""Hardware configuration for physical cartpole deployment.

Fill in ``physics`` with bench-measured values from your specific rig.  The
``*_noise`` fields set the Gaussian std-dev added to each parameter when
building the sysID context vector that the network receives.  During training
this noise forces the policy to be robust to measurement error; at deploy time
you can set them to zero if you trust your measurements.

``delay_steps`` and ``sensor_noise_std`` model the hardware pipeline rather
than the pendulum mechanics.  Action delay is the number of 100 Hz control
steps between issuing a command and the actuator applying it (typically 1–2
steps for an ESP32 over USB/UART).  Sensor noise is the Gaussian std-dev added
to each observation dimension to match encoder / IMU noise floor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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

    Example measurement guidance:
      - Mass: bench scale ±1 g → M_noise ≈ 0.001, masses_noise ≈ 0.001
      - Length: steel ruler ±0.5 mm → lengths_noise ≈ 0.0005
      - Cart friction: free-deceleration fit, rough (±30 %) → b_noise ≈ 0.03
      - Joint friction: free-oscillation decay fit, rough → joint_friction_noise ≈ 0.0005
    """

    # Best-estimate physical parameters from bench measurement.
    physics: PhysicsParams = field(default_factory=PhysicsParams)

    # Measurement uncertainty (Gaussian std dev, same units as the parameter).
    M_noise: float = 0.05           # cart mass (kg)
    b_noise: float = 0.03           # cart viscous friction (N·s/m)
    masses_noise: float = 0.005     # per-link bob mass (kg)
    lengths_noise: float = 0.005    # per-link rod length (m)
    joint_friction_noise: float = 0.0005  # per-link joint friction (N·m·s/rad)

    # Number of control loop steps between issuing a command and the actuator
    # applying it.  At 100 Hz, 1 step = 10 ms.  Typical for ESP32 USB/UART: 1–2.
    delay_steps: int = 1

    # Gaussian std dev added independently to each observation dimension to
    # model encoder / IMU noise.  Set to 0.0 if your sensors are clean.
    sensor_noise_std: float = 0.0
