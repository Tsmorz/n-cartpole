"""TQC checkpoint -> ``.ncpm`` model blob for the ESP32-S3 firmware (and back, for tests).

The byte layout is ``ncp_header_t`` in
``firmware/components/cartpole_core/include/ncp_model.h``: a 512-byte little-endian
header (dims, flags, actuator limits, nominal physics, sensor/EKF parameters,
label) followed by the actor weights as float32 or float16, in the order of
:func:`n_cartpole.deploy.actor_export.actor_tensors` (``RunningNorm`` folded into
``in_w``/``in_b``, only the mean row of the output head). ``crc32`` is
``zlib.crc32`` of the whole file with the crc field read as zero.

The firmware reproduces the training env's observation pipeline exactly, so only
configurations it implements are exported: no observation history, no sysID
context, at most 4 links / 4 residual blocks / 8 past commands / 8 delay steps.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger

from n_cartpole.deploy.actor_export import actor_tensors, n_blocks
from n_cartpole.policy.loader import load_policy

MAGIC = b"NCPM"
VERSION = 1
HEADER_SIZE = 512
MAX_LINKS = 4
MAX_HIST = 8
MAX_DELAY = 8
MAX_BLOCKS = 4
LABEL_BYTES = 64

DTYPE_F32 = 0
DTYPE_F16 = 1
DTYPES = {"f32": DTYPE_F32, "f16": DTYPE_F16}

FLAG_GOAL = 1 << 0
FLAG_DELTA = 1 << 1
FLAG_EKF = 1 << 2

# (field name, struct code); per-link arrays are MAX_LINKS doubles.
_LINK_FIELDS = ("masses", "lengths", "com", "inertia", "joint_friction")
_INT_FIELDS: tuple[tuple[str, str], ...] = (
    ("magic", "4s"),
    ("version", "H"),
    ("header_size", "H"),
    ("total_size", "I"),
    ("crc32", "I"),
    ("train_step", "I"),
    ("n_links", "B"),
    ("n_blocks", "B"),
    ("expand", "B"),
    ("dtype", "B"),
    ("hidden", "H"),
    ("obs_dim", "H"),
    ("flags", "B"),
    ("action_history", "B"),
    ("delay_steps", "B"),
    ("n_goals", "B"),
    ("goal_settle_steps", "H"),
    ("reserved0", "H"),
    ("reserved1", "I"),
)
_SCALARS_A = ("dt", "force_max", "force_slew", "x_lim")
_SCALARS_B = ("M", "g", "b", "cart_coulomb", "cart_coulomb_vscale", "v_noload")
_SCALARS_C = (
    "ekf_force_std",
    "ekf_accel_std",
    "velocity_filter",
    "x_noise_std",
    "angle_noise_std",
    "x_resolution",
    "angle_resolution",
    "goal_tol_angle",
    "goal_tol_vel",
)
HEADER_FORMAT = (
    "<"
    + "".join(code for _, code in _INT_FIELDS)
    + "d" * (len(_SCALARS_A) + len(_SCALARS_B))
    + "d" * (MAX_LINKS * len(_LINK_FIELDS))
    + "d" * len(_SCALARS_C)
    + f"{LABEL_BYTES}s96x"
)
assert struct.calcsize(HEADER_FORMAT) == HEADER_SIZE
CRC_OFFSET = 12  # offsetof(ncp_header_t, crc32)


def pack_header(fields: dict[str, Any]) -> bytes:
    """Pack a header dict (the keys :func:`read_header` returns) into 512 bytes.

    Missing integer/scalar fields default to 0; per-link lists are zero-padded to
    ``MAX_LINKS``; ``label`` is a ``str`` (truncated to 63 bytes, NUL-terminated).
    """
    vals: list[Any] = []
    for name, _ in _INT_FIELDS:
        default: Any = MAGIC if name == "magic" else 0
        vals.append(fields.get(name, default))
    vals += [float(fields.get(k, 0.0)) for k in _SCALARS_A + _SCALARS_B]
    for name in _LINK_FIELDS:
        arr = [float(v) for v in fields.get(name, [])]
        if len(arr) > MAX_LINKS:
            raise ValueError(f"{name}: at most {MAX_LINKS} links")
        vals += arr + [0.0] * (MAX_LINKS - len(arr))
    vals += [float(fields.get(k, 0.0)) for k in _SCALARS_C]
    vals.append(_label_bytes(str(fields.get("label", ""))))
    return struct.pack(HEADER_FORMAT, *vals)


def _label_bytes(label: str) -> bytes:
    """UTF-8 label cut to 63 bytes on a character boundary (room for the NUL)."""
    raw = label.encode("utf-8")[: LABEL_BYTES - 1]
    return raw.decode("utf-8", errors="ignore").encode("utf-8")


def unpack_header(buf: bytes) -> dict[str, Any]:
    """Decode the first 512 bytes into a dict without validating anything."""
    if len(buf) < HEADER_SIZE:
        raise ValueError(f"blob too short for a header: {len(buf)} bytes")
    vals = list(struct.unpack(HEADER_FORMAT, buf[:HEADER_SIZE]))
    out: dict[str, Any] = {}
    for name, _ in _INT_FIELDS:
        out[name] = vals.pop(0)
    for k in _SCALARS_A + _SCALARS_B:
        out[k] = vals.pop(0)
    n = int(out["n_links"])
    for name in _LINK_FIELDS:
        arr = [vals.pop(0) for _ in range(MAX_LINKS)]
        out[name] = arr[:n]
    for k in _SCALARS_C:
        out[k] = vals.pop(0)
    out["label"] = vals.pop(0).split(b"\0", 1)[0].decode("utf-8", errors="replace")
    return out


def weight_count(h: dict[str, Any]) -> int:
    """Return the number of weights after the header (ncp_weight_count)."""
    H, D, E = int(h["hidden"]), int(h["obs_dim"]), int(h["hidden"]) * int(h["expand"])
    return H * D + H + int(h["n_blocks"]) * (2 * H + 2 * E * H + E + H) + 3 * H + 1


def blob_crc(blob: bytes) -> int:
    """CRC-32 of a blob with its crc32 field read as zero (ncp_blob_crc)."""
    zeroed = blob[:CRC_OFFSET] + b"\0\0\0\0" + blob[CRC_OFFSET + 4 :]
    return zlib.crc32(zeroed) & 0xFFFFFFFF


def read_header(blob: bytes) -> dict[str, Any]:
    """Decode and verify a blob's header: magic, version, sizes and CRC.

    Raises:
        ValueError: if the blob is not a valid ``.ncpm`` file.

    """
    h = unpack_header(blob)
    if h["magic"] != MAGIC:
        raise ValueError("not an .ncpm blob (bad magic)")
    if h["version"] != VERSION or h["header_size"] != HEADER_SIZE:
        raise ValueError(
            f"unsupported blob version {h['version']} / header size {h['header_size']}"
        )
    width = 2 if h["dtype"] == DTYPE_F16 else 4
    expected = HEADER_SIZE + weight_count(h) * width
    if h["total_size"] != expected or len(blob) != expected:
        raise ValueError(
            f"blob size {len(blob)} disagrees with its header "
            f"(total_size {h['total_size']}, dims say {expected})"
        )
    if blob_crc(blob) != h["crc32"]:
        raise ValueError("blob CRC mismatch (corrupted)")
    return h


def read_weights(blob: bytes) -> np.ndarray:
    """Return the blob's weights as float32 (after verifying the header)."""
    h = read_header(blob)
    dt = "<f2" if h["dtype"] == DTYPE_F16 else "<f4"
    return np.frombuffer(blob, dtype=dt, offset=HEADER_SIZE).astype(np.float32)


def _check_supported(cfg: Any, n: int, blocks: int) -> None:
    """Reject configurations the firmware's controller does not reproduce."""
    hw = cfg.hardware
    problems = []
    if int(getattr(cfg, "obs_history", 0)) > 0:
        problems.append("obs_history > 0 (past sensor frames) is not implemented")
    if hw is not None and hw.sysid_context:
        problems.append("sysid_context=True (sysID observation block) is not supported")
    if n > MAX_LINKS:
        problems.append(f"n_links {n} > {MAX_LINKS}")
    if int(getattr(cfg, "action_history", 0)) > MAX_HIST:
        problems.append(f"action_history {cfg.action_history} > {MAX_HIST}")
    if hw is not None and hw.delay_steps > MAX_DELAY:
        problems.append(f"delay_steps {hw.delay_steps} > {MAX_DELAY}")
    if blocks > MAX_BLOCKS:
        problems.append(f"{blocks} residual blocks > {MAX_BLOCKS}")
    if problems:
        raise ValueError("cannot export for the firmware: " + "; ".join(problems))


@dataclass
class _Sensing:
    estimator: str = "diff"
    ekf_force_std: float = 0.0
    ekf_accel_std: float = 0.0
    velocity_filter: float = 0.0
    x_noise_std: float = 0.0
    angle_noise_std: float = 0.0
    x_resolution: float = 0.0
    angle_resolution: float = 0.0


def _sensing(cfg: Any) -> _Sensing:
    """Return the encoder model and estimator as configured (realism 1)."""
    hw = cfg.hardware
    sensors = hw.sensors if hw is not None else None
    if sensors is None:
        logger.warning(
            "checkpoint trained on exact states (no hardware.sensors): the device "
            "will difference encoder positions for the velocities"
        )
        return _Sensing()
    return _Sensing(
        estimator=sensors.estimator,
        ekf_force_std=sensors.ekf_force_std,
        ekf_accel_std=sensors.ekf_accel_std,
        velocity_filter=sensors.velocity_filter,
        x_noise_std=sensors.x_noise_std,
        angle_noise_std=sensors.angle_noise_std,
        x_resolution=sensors.x_resolution,
        angle_resolution=sensors.angle_resolution,
    )


def default_label(checkpoint: Path | str, step: int) -> str:
    """``<checkpoint dir name>@<step>``, cut to what fits the header."""
    label = f"{Path(checkpoint).resolve().parent.name}@{step}"
    return _label_bytes(label).decode("utf-8")


def export_blob(
    checkpoint: Path | str, dtype: str = "f32", label: str | None = None
) -> bytes:
    """Build the ``.ncpm`` blob of a TQC checkpoint.

    Args:
        checkpoint: path to a ``tqc_*.pt`` checkpoint.
        dtype: ``"f32"`` or ``"f16"`` weights.
        label: display name; default ``<checkpoint dir name>@<step>``.

    Raises:
        ValueError: for a dtype or a training configuration the firmware cannot run.

    """
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}, got {dtype!r}")
    bundle = load_policy(checkpoint)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = bundle.env_config
    p = bundle.physics
    n = bundle.n_links
    blocks = n_blocks(ckpt)
    _check_supported(cfg, n, blocks)

    tensors = actor_tensors(ckpt)
    weights = dict(tensors)
    hidden = int(weights["in_w"].shape[0])
    expand = int(weights["b0_fc1_w"].shape[0]) // hidden if blocks else 4
    goal = bool(bundle.goal_conditioned)
    delta = cfg.action_mode == "delta"
    sens = _sensing(cfg)
    flags = (
        (FLAG_GOAL if goal else 0)
        | (FLAG_DELTA if delta else 0)
        | (FLAG_EKF if sens.estimator == "ekf" else 0)
    )
    hw = cfg.hardware
    step = int(ckpt.get("step", 0))
    if label is None:
        label = default_label(checkpoint, step)

    np_dtype = "<f2" if dtype == "f16" else "<f4"
    payload = b"".join(
        np.ascontiguousarray(arr.ravel(), dtype=np_dtype).tobytes()
        for _, arr in tensors
    )
    fields: dict[str, Any] = {
        "magic": MAGIC,
        "version": VERSION,
        "header_size": HEADER_SIZE,
        "total_size": HEADER_SIZE + len(payload),
        "crc32": 0,
        "train_step": step & 0xFFFFFFFF,
        "n_links": n,
        "n_blocks": blocks,
        "expand": expand,
        "dtype": DTYPES[dtype],
        "hidden": hidden,
        "obs_dim": bundle.obs_dim,
        "flags": flags,
        "action_history": int(cfg.action_history),
        "delay_steps": int(hw.delay_steps) if hw is not None else 0,
        "n_goals": 2**n if goal else 1,
        "goal_settle_steps": int(cfg.goal_settle_steps),
        "dt": p.dt,
        "force_max": p.force_max,
        "force_slew": cfg.force_slew or 0.0,
        "x_lim": p.x_lim,
        "M": p.M,
        "g": p.g,
        "b": p.b,
        "cart_coulomb": p.cart_coulomb,
        "cart_coulomb_vscale": p.cart_coulomb_vscale,
        "v_noload": p.v_noload or 0.0,
        "masses": p.link_masses(n).tolist(),
        "lengths": p.link_lengths(n).tolist(),
        "com": p.link_coms(n).tolist(),
        "inertia": p.link_inertias(n).tolist(),
        "joint_friction": p.joint_frictions(n).tolist(),
        "ekf_force_std": sens.ekf_force_std,
        "ekf_accel_std": sens.ekf_accel_std,
        "velocity_filter": sens.velocity_filter,
        "x_noise_std": sens.x_noise_std,
        "angle_noise_std": sens.angle_noise_std,
        "x_resolution": sens.x_resolution,
        "angle_resolution": sens.angle_resolution,
        "goal_tol_angle": cfg.goal_tol_angle,
        "goal_tol_vel": cfg.goal_tol_vel,
        "label": label,
    }
    if weight_count(fields) * (2 if dtype == "f16" else 4) != len(payload):
        raise ValueError(
            "actor tensors do not match the SimBa layout"
        )  # pragma: no cover
    blob = bytearray(pack_header(fields) + payload)
    struct.pack_into("<I", blob, CRC_OFFSET, blob_crc(bytes(blob)))
    return bytes(blob)


@dataclass
class InferenceCost:
    """Size and a rough ESP32-S3 timing estimate of one forward pass."""

    params: int
    macs: int
    weight_bytes_f32: int
    fits_internal: bool
    est_us: float
    dt_us: float

    @property
    def budget_fraction(self) -> float:
        """Estimated inference time as a fraction of the control period."""
        return self.est_us / self.dt_us


# ESP32-S3 rules of thumb: ~200 KB of internal SRAM left for weights next to
# Wi-Fi; ~3 CPU cycles per float32 MAC at 240 MHz from internal SRAM; weights in
# PSRAM are streamed every inference at ~40 MB/s (memory bound).
INTERNAL_WEIGHT_BUDGET = 200 * 1024
CYCLES_PER_MAC = 3.0
CPU_HZ = 240e6
PSRAM_BYTES_PER_S = 40e6


def inference_cost(h: dict[str, Any]) -> InferenceCost:
    """Parameter count, multiply-accumulates and estimated time for a header dict."""
    H, D = int(h["hidden"]), int(h["obs_dim"])
    E = H * int(h["expand"])
    params = weight_count(h)
    macs = H * D + int(h["n_blocks"]) * 2 * H * E + H
    nbytes = params * 4  # decoded to float32 on the device whatever the file dtype
    fits = nbytes <= INTERNAL_WEIGHT_BUDGET
    compute_us = macs * CYCLES_PER_MAC / CPU_HZ * 1e6
    est = compute_us if fits else max(compute_us, nbytes / PSRAM_BYTES_PER_S * 1e6)
    return InferenceCost(params, macs, nbytes, fits, est, float(h["dt"]) * 1e6)
