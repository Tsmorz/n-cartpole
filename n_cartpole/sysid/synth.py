"""Simulated identification experiments: a known plant -> a :class:`Recording`.

The plant is integrated with :func:`n_cartpole.env.dynamics.step` through the
env's actuator chain (delay -> slew -> available_force -> gain) and read through
an encoder model (Gaussian noise + quantization, like ``env._read``). The drive
command is the firmware's MANUAL mode: a PD on the cart (differenced readings)
plus an excitation, capped like the dashboard's force cap. Used to test the fit
on parameters that are known, and to rehearse an experiment before running it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from n_cartpole.env.dynamics import PhysicsParams, available_force, step
from n_cartpole.sysid.excite import Excitation, ExciteParams
from n_cartpole.sysid.recording import Recording, make_recording


@dataclass
class Encoders:
    """Reading noise std and quantization step for the cart (m) and the joints (rad)."""

    x_noise: float = 2e-5
    angle_noise: float = 5e-4
    x_res: float = 1e-5
    angle_res: float = 2 * np.pi / 16384


def simulate_recording(
    plant: PhysicsParams,
    n_links: int,
    seconds: float,
    *,
    excite: ExciteParams | None = None,
    armed: bool = True,
    start: np.ndarray | None = None,
    delay: int = 1,
    force_slew: float | None = 300.0,
    force_gain: float = 1.0,
    kp: float = 150.0,
    kd: float = 20.0,
    cap: float = 10.0,
    encoders: Encoders | None = None,
    seed: int = 0,
) -> Recording:
    """Run one experiment on ``plant`` and return what the device would record."""
    rng = np.random.default_rng(seed)
    enc = encoders or Encoders()
    n, dt = n_links, plant.dt
    s = np.zeros(2 + 2 * n)
    s[2::2] = np.pi
    if start is not None:
        s = np.asarray(start, dtype=float).copy()
    noise = np.array([enc.x_noise] + [enc.angle_noise] * n)
    res = np.array([enc.x_res] + [enc.angle_res] * n)
    exc = Excitation(excite or ExciteParams(), dt)
    buf = [0.0] * delay
    prev_applied = 0.0
    prev_x = None
    N = int(round(seconds / dt))
    pos_log, cmd_log, exc_log, truth = [], [], [], []
    for _ in range(N):
        true_pos = np.concatenate(([s[0]], s[2::2]))
        reading = true_pos + rng.normal(size=n + 1) * noise
        reading = np.where(
            res > 0, np.round(reading / np.where(res > 0, res, 1)) * res, reading
        )
        xd = 0.0 if prev_x is None else (reading[0] - prev_x) / dt
        prev_x = reading[0]
        ext = exc.next() if armed else 0.0
        cmd = float(
            np.clip(
                np.clip(-kp * reading[0] - kd * xd, -cap, cap) + ext,
                -plant.force_max,
                plant.force_max,
            )
        )
        pos_log.append(reading)
        cmd_log.append(cmd)
        exc_log.append(ext)
        truth.append(s.copy())
        u = cmd if armed else 0.0
        if delay:
            buf.append(u)
            u = buf.pop(0)
        if force_slew is not None:
            dF = force_slew * dt
            u = prev_applied + float(np.clip(u - prev_applied, -dF, dF))
        prev_applied = u
        F = available_force(u, float(s[1]), plant) * force_gain
        s = step(s, F, plant)
    rec = make_recording(
        dt,
        np.array(pos_log),
        np.array(cmd_log),
        armed=armed,
        excite=np.array(exc_log),
        truth=np.array(truth),
        note="simulated",
    )
    rec.excite_kind = (excite or ExciteParams()).kind
    return rec
