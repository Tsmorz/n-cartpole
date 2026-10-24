#!/usr/bin/env python3
"""A stand-in n-cartpole device, for developing the dashboard without hardware.

Serves ``firmware/web/`` and the whole HTTP / WebSocket API of
``firmware/docs/design.md`` from the Python standard library (``http.server``
plus a hand-written RFC 6455 WebSocket), backed by a genuine 200 Hz simulation:
the training env (``NPendulumCartpole``, with the checkpoint's own env config:
encoder model, EKF, action delay, slew limit) driven by the checkpoint's actor,
so telemetry, observations and commands are the real thing. Telemetry frames are
packed exactly as ``ncp_telemetry.h`` lays them out.

    unset VIRTUAL_ENV; uv run python firmware/tools/mock_device.py   # http://127.0.0.1:8080
    ... --rig            # pretend to be the rig backend (no truth, no pose reset; zeroing)
    ... --blob m.ncpm    # serve this blob at /api/model instead of exporting one
    ... --no-model       # a device with no model stored (/api/model 404, AI refused)

``GET /api/model`` serves ``n_cartpole.deploy.firmware_blob.export_blob(ckpt,
dtype="f32")`` (imported lazily; a clear 404 if that exporter is missing). An
uploaded ``.ncpm`` (``POST /api/model``) is validated (header, size, CRC) and, if
its observation layout matches the env, drives the plant through a NumPy port of
the firmware's forward pass. A firmware image (``POST /api/ota``) is checked like
the device does (0xE9, app descriptor project ``n_cartpole_fw``) and "installed":
the mock drops every connection for a few seconds, then comes back reporting the
image's version.

Recordings for system identification (``{"record": ...}``, ``{"excite": ...}``,
``GET /api/record``) follow ``ncp_record.h`` / ``ncp_excite.h``: one 128-byte
sample per 200 Hz tick (the env's encoder readings, command, its excitation part,
applied force, estimate, true state), the excitation added to the drive force
while armed (capped at 10 N).

Mock-only commands, for checking how the dashboard shows trouble:
``POST /api/cmd {"inject_fault": "e-stop"}`` (any fault reason) and, with ``--rig``,
``{"inject_io": "weak-magnet"}`` / ``"bad-reads"`` / ``"ok"`` (encoder diagnostics).
``--uncalibrated`` starts a rig whose zero references are not set (arming is refused).
"""

from __future__ import annotations

import argparse
import base64
import collections
import dataclasses
import hashlib
import json
import logging
import queue
import re
import socket
import struct
import sys
import threading
import time
import zlib
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
WEB = REPO / "firmware" / "web"
DEFAULT_CKPT = REPO / "checkpoints/double/tqc-rail05-goals-ft-weighted/tqc_2000000.pt"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

log = logging.getLogger("mock")

# ---- ncp_telemetry.h -------------------------------------------------------
TLM_VERSION = 1
MAX_LINKS = 4
MAX_STATE = 2 + 2 * MAX_LINKS
MAX_POS = 1 + MAX_LINKS
MAX_HIST = 8
MAX_OBS = 2 + 3 * MAX_LINKS + MAX_LINKS + MAX_HIST
FRAME = struct.Struct(f"<IBBBB{MAX_STATE}f{MAX_POS}f{MAX_STATE}f{MAX_OBS}ffffHHHH")
MSG_HDR = struct.Struct("<BBH")
assert FRAME.size == 232, FRAME.size
RUN_IDLE, RUN_ARMED, RUN_FAULT, RUN_MAINT = range(4)
RUN_NAMES = ("idle", "armed", "fault", "maint")
F_SETTLED, F_RANDOM, F_SIM, F_TRUTH = 1, 2, 4, 8
MODE_AI, MODE_MANUAL = 0, 1

# ---- ncp_model.h -----------------------------------------------------------
HEADER = struct.Struct("<4sHHIII BBBB HH BBBB HH I4d 6d 4d 4d 4d 4d 4d 7d 2d 64s 96x")
assert HEADER.size == 512, HEADER.size
FLAG_GOAL, FLAG_DELTA, FLAG_EKF = 1, 2, 4

# ---- ncp_record.h / ncp_excite.h -------------------------------------------
REC_HEADER = struct.Struct("<4sHHIIIId BBBB 6f Q 32s 64s 64s 28x")
REC_SAMPLE = struct.Struct(f"<II{MAX_POS}ffff{MAX_STATE}f{MAX_STATE}fBBBBHH")
assert REC_HEADER.size == 256, REC_HEADER.size
assert REC_SAMPLE.size == 128, REC_SAMPLE.size
REC_ARMED, REC_EXCITE, REC_TRUTH = 1, 2, 4
EXC_KINDS = ("none", "chirp", "prbs", "steps", "sine")
REC_MAX_S = 120.0

TLM_RATE_HZ = 50
MSGS_PER_S = 25
STATUS_PERIOD = 0.5
SETTLE_RESET_S = 1.5
RANDOM_DWELL_S = 1.0
REBOOT_S = 4.0
PD_KP, PD_KD, PD_FMAX = 150.0, 20.0, 10.0
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def goal_labels(n: int) -> list[str]:
    """Goal labels in index order: bit k set = link k down (goals.goal_configs)."""
    return [
        "".join("D" if g >> k & 1 else "U" for k in range(n)) for g in range(1 << n)
    ]


def parse_header(blob: bytes) -> dict[str, Any]:
    """Decode an .ncpm header into a dict (raises ValueError if it is not one)."""
    if len(blob) < HEADER.size or blob[:4] != b"NCPM":
        raise ValueError("not an .ncpm model blob (magic)")
    v = HEADER.unpack_from(blob)
    keys = (
        "magic version header_size total_size crc32 train_step n_links n_blocks "
        "expand dtype hidden obs_dim flags action_history delay_steps n_goals "
        "goal_settle_steps reserved0 reserved1"
    ).split()
    h: dict[str, Any] = dict(zip(keys, v[:19], strict=True))
    d = list(v[19:58])
    for name in "dt force_max force_slew x_lim M g b cart_coulomb cart_coulomb_vscale v_noload".split():
        h[name] = d.pop(0)
    for name in "masses lengths com inertia joint_friction".split():
        h[name], d = d[:MAX_LINKS], d[MAX_LINKS:]
    for name in (
        "ekf_force_std ekf_accel_std velocity_filter x_noise_std angle_noise_std "
        "x_resolution angle_resolution goal_tol_angle goal_tol_vel"
    ).split():
        h[name] = d.pop(0)
    h["label"] = v[58].split(b"\0", 1)[0].decode("utf-8", "replace")
    return h


def weight_count(h: dict[str, Any]) -> int:
    """Floats after the header: in_w .. out_b (ncp_weight_count)."""
    H, D = h["hidden"], h["obs_dim"]
    E = H * h["expand"]
    block = 2 * H + E * H + E + H * E + H
    return H * D + H + h["n_blocks"] * block + 2 * H + H + 1


DTYPE_NAMES = {0: "f32", 1: "f16", 2: "i8"}


def _tensor_shapes(h: dict[str, Any]) -> list[tuple[bool, tuple[int, ...]]]:
    """(int8 in an i8 blob?, shape) of each tensor in file order (ncp_model.h)."""
    H, D = h["hidden"], h["obs_dim"]
    E = H * h["expand"]
    block = [(False, (H,)), (False, (H,)), (True, (E, H)), (False, (E,))]
    block += [(True, (H, E)), (False, (H,))]
    return (
        [(False, (H, D)), (False, (H,))]
        + block * h["n_blocks"]
        + [(False, (H,)), (False, (H,)), (False, (H,)), (False, (1,))]
    )


def weight_bytes(h: dict[str, Any]) -> int:
    """Bytes of the weights in the file (ncp_weight_bytes)."""
    if h["dtype"] != 2:
        return weight_count(h) * (4 if h["dtype"] == 0 else 2)
    total = 0
    for quant, shape in _tensor_shapes(h):
        n = int(np.prod(shape))
        total += 4 * shape[0] + (n + 3) // 4 * 4 if quant else 4 * n
    return total


def weight_ram_bytes(h: dict[str, Any]) -> int:
    """Bytes the device holds the weights in (ncp_weight_ram_bytes)."""
    return weight_bytes(h) if h["dtype"] == 2 else 4 * weight_count(h)


def decode_weights(blob: bytes, h: dict[str, Any]) -> np.ndarray:
    """Every weight as float32 in file order (int8: q * row scale)."""
    if h["dtype"] != 2:
        dt = np.dtype("<f4") if h["dtype"] == 0 else np.dtype("<f2")
        return np.frombuffer(blob, dtype=dt, offset=HEADER.size).astype(np.float32)
    out, off = [], HEADER.size
    for quant, shape in _tensor_shapes(h):
        n = int(np.prod(shape))
        if quant:
            scale = np.frombuffer(blob, "<f4", shape[0], off)
            q = np.frombuffer(blob, np.int8, n, off + 4 * shape[0]).reshape(shape)
            out.append((q.astype(np.float32) * scale[:, None]).ravel())
            off += 4 * shape[0] + (n + 3) // 4 * 4
        else:
            out.append(np.frombuffer(blob, "<f4", n, off).astype(np.float32))
            off += 4 * n
    return np.concatenate(out)


def validate_blob(blob: bytes) -> dict[str, Any]:
    """Header, size and CRC checks of ncp_blob_validate; returns the header."""
    h = parse_header(blob)
    if h["version"] != 1 or h["header_size"] != HEADER.size:
        raise ValueError(f"unsupported blob version {h['version']}")
    if not 1 <= h["n_links"] <= MAX_LINKS or h["dtype"] not in DTYPE_NAMES:
        raise ValueError("dims out of range")
    want = HEADER.size + weight_bytes(h)
    if h["total_size"] != len(blob) or want != len(blob):
        raise ValueError(
            f"size mismatch: file {len(blob)}, header {h['total_size']}, dims {want}"
        )
    crc = zlib.crc32(blob[:12] + b"\0\0\0\0" + blob[16:])
    if crc != h["crc32"]:
        raise ValueError("CRC mismatch (corrupted)")
    return h


def model_info(h: dict[str, Any]) -> dict[str, Any]:
    """Build the /api/info "model" object from a blob header."""
    n = h["n_links"]
    nbytes = h["total_size"]
    return {
        "label": h["label"],
        "step": h["train_step"],
        "n_links": n,
        "obs_dim": h["obs_dim"],
        "hidden": h["hidden"],
        "blocks": h["n_blocks"],
        "expand": h["expand"],
        "params": weight_count(h),
        "bytes": nbytes,
        "crc32": f"{h['crc32']:08x}",
        "dtype": DTYPE_NAMES[h["dtype"]],
        "weights_in": "psram" if weight_ram_bytes(h) > 256 * 1024 else "internal",
        "goal_conditioned": bool(h["flags"] & FLAG_GOAL),
        "action_mode": "delta" if h["flags"] & FLAG_DELTA else "absolute",
        "estimator": "ekf" if h["flags"] & FLAG_EKF else "diff",
        "action_history": h["action_history"],
        "delay_steps": h["delay_steps"],
        "dt": h["dt"],
        "force_max": h["force_max"],
        "force_slew": h["force_slew"],
        "x_lim": h["x_lim"],
        "goals": goal_labels(n) if h["flags"] & FLAG_GOAL else [],
    }


def blob_policy(blob: bytes, h: dict[str, Any]) -> Callable[[np.ndarray], float]:
    """NumPy port of ncp_policy_act: the blob's deterministic actor, in newtons."""
    w = decode_weights(blob, h)
    H, D, E = h["hidden"], h["obs_dim"], h["hidden"] * h["expand"]
    pos = 0

    def take(*shape: int) -> np.ndarray:
        nonlocal pos
        size = int(np.prod(shape))
        out = w[pos : pos + size].reshape(shape)
        pos += size
        return out

    in_w, in_b = take(H, D), take(H)
    blocks = [
        (take(H), take(H), take(E, H), take(E), take(H, E), take(H))
        for _ in range(h["n_blocks"])
    ]
    fl_w, fl_b, out_w, out_b = take(H), take(H), take(H), take(1)
    fmax = h["force_max"]

    def ln(x: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
        m = x.mean()
        return (x - m) / np.sqrt(((x - m) ** 2).mean() + 1e-5) * g + b

    def act(obs: np.ndarray) -> float:
        x = in_w @ obs.astype(np.float32) + in_b
        for g, b, w1, b1, w2, b2 in blocks:
            x = x + w2 @ np.maximum(w1 @ ln(x, g, b) + b1, 0.0) + b2
        mean = float(out_w @ ln(x, fl_w, fl_b) + out_b[0])
        return float(np.tanh(mean) * fmax)

    return act


def app_descriptor(image: bytes) -> dict[str, str]:
    """esp_app_desc_t of an ESP32 app image (first byte 0xE9), or raise ValueError."""
    if len(image) < 256 or image[0] != 0xE9:
        raise ValueError("not an ESP32 app image (first byte must be 0xE9)")
    magic = struct.unpack_from("<I", image, 32)[0]
    if magic != 0xABCD5432:
        raise ValueError("no app descriptor in the image")

    def cstr(a: int, n: int) -> str:
        return image[a : a + n].split(b"\0", 1)[0].decode("utf-8", "replace")

    return {
        "version": cstr(48, 32),
        "project": cstr(80, 32),
        "time": cstr(112, 16),
        "date": cstr(128, 16),
        "idf": cstr(144, 32),
    }


class Excite:
    """An ncp_excite.h signal: chirp, PRBS, steps or sine, sampled every dt."""

    def __init__(self, p: dict[str, Any], dt: float) -> None:
        self.kind = str(p.get("kind", "none"))
        self.amp = float(min(abs(float(p.get("amp", 0.0))), PD_FMAX))
        self.f0 = float(p.get("f0", 0.2))
        self.f1 = float(p.get("f1", 6.0))
        self.period = max(float(p.get("period", 0.05)), dt)
        self.duration = float(p.get("duration", 30.0))
        self.seed = int(p.get("seed", 1))
        self.dt, self.t, self.phase, self.value = dt, 0.0, 0.0, 0.0
        self.rng = np.random.default_rng(self.seed)
        self.active = self.kind != "none"
        self._slot = -1

    def params(self) -> list[float]:
        """Return amp, f0, f1, period, duration, seed: the header's excite[6]."""
        return [
            self.amp,
            self.f0,
            self.f1,
            self.period,
            self.duration,
            float(self.seed),
        ]

    def next(self) -> float:
        """Return this tick's force offset (N) and advance; 0 once duration has elapsed."""
        if not self.active or self.t >= self.duration:
            self.active, self.value = False, 0.0
            return 0.0
        t = self.t
        if self.kind == "chirp":  # log sweep f0 -> f1 over duration
            f = self.f0 * (self.f1 / self.f0) ** (t / self.duration)
            self.value = self.amp * np.sin(self.phase)
            self.phase += 2 * np.pi * f * self.dt
        elif self.kind == "sine":
            self.value = self.amp * np.sin(2 * np.pi * self.f0 * t)
        else:
            slot = int(t / self.period)
            if slot != self._slot:
                self._slot = slot
                if self.kind == "prbs":
                    self.value = self.amp * (1.0 if self.rng.random() < 0.5 else -1.0)
                else:  # steps: +amp, 0, -amp, 0, ...
                    self.value = self.amp * (1.0, 0.0, -1.0, 0.0)[slot % 4]
        self.t += self.dt
        return float(self.value)

    def status(self) -> dict[str, Any]:
        """Build the status JSON's "excite" object."""
        return {"kind": self.kind, "active": self.active, "t": round(self.t, 2)}


# ---- WebSocket (RFC 6455, server side) --------------------------------------
class WsClient:
    """One WebSocket connection; sends are serialized by a lock."""

    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self.lock = threading.Lock()
        self.alive = True

    def send(self, opcode: int, payload: bytes) -> bool:
        """Send one unmasked, unfragmented frame; False once the peer is gone."""
        n = len(payload)
        if n < 126:
            head = struct.pack("!BB", 0x80 | opcode, n)
        elif n < 1 << 16:
            head = struct.pack("!BBH", 0x80 | opcode, 126, n)
        else:
            head = struct.pack("!BBQ", 0x80 | opcode, 127, n)
        try:
            with self.lock:
                self.sock.sendall(head + payload)
        except OSError:
            self.alive = False
        return self.alive

    def close(self) -> None:
        """Send a close frame and shut the socket (the reader thread then exits)."""
        self.send(0x8, struct.pack("!H", 1001))
        self.alive = False
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


def ws_read_frame(rfile: Any) -> tuple[int, bytes] | None:
    """Read one (masked) client frame; None when the connection ends."""
    head = rfile.read(2)
    if len(head) < 2:
        return None
    opcode, n = head[0] & 0x0F, head[1] & 0x7F
    if n == 126:
        n = struct.unpack("!H", rfile.read(2))[0]
    elif n == 127:
        n = struct.unpack("!Q", rfile.read(8))[0]
    mask = rfile.read(4) if head[1] & 0x80 else b"\0\0\0\0"
    data = bytearray(rfile.read(n))
    for i in range(len(data)):
        data[i] ^= mask[i % 4]
    return opcode, bytes(data)


# ---- The device -------------------------------------------------------------
@dataclasses.dataclass
class LoopStats:
    """Per-tick timings accumulated between two status messages."""

    exec_us: list[float] = dataclasses.field(default_factory=list)
    policy_us: list[float] = dataclasses.field(default_factory=list)
    est_us: list[float] = dataclasses.field(default_factory=list)
    period_us: list[float] = dataclasses.field(default_factory=list)


class Device:
    """The simulated controller: plant, policy, run state, telemetry, uploads."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.rig = bool(args.rig)
        self.strict_budget = bool(getattr(args, "strict_budget", False))
        self.boot = time.monotonic()
        self.rng = np.random.default_rng(0)
        self.cmds: queue.Queue[dict[str, Any]] = queue.Queue()
        self.clients: list[WsClient] = []
        self.clients_lock = threading.Lock()
        self.frames: collections.deque[bytes] = collections.deque(maxlen=400)
        self.stats = LoopStats()
        self.stats_lock = threading.Lock()
        self.down_until = 0.0
        self.version = self._repo_version()
        self.built = time.strftime("%b %d %Y %H:%M:%S")
        self.ota_running, self.ota_next, self.ota_state = "ota_0", "ota_1", "valid"
        self.pending_until = 0.0
        self.hostname = "cartpole"
        self.boot_count = 7
        self.events: collections.deque[dict[str, Any]] = collections.deque(maxlen=40)
        self.events_seq = 0
        self.fault_count = 0
        self.last_fault = ""
        self.last_fault_t = 0.0
        self.calib: dict[str, Any] = (
            {"angles": False, "cart": False}
            if args.uncalibrated
            else {
                "angles": True,
                "cart": True,
                "off1": 2.81734,
                "off2": -0.41263,
                "angles_boot": 5,
                "angles_unix": int(time.time()) - 3 * 86400,
                "x_zero": 0.01275,
                "cart_boot": 5,
                "cart_unix": int(time.time()) - 3 * 86400,
            }
        )
        self.enc_bad = [0, 0]
        self.enc_agc = [72, 88]
        self.enc_weak = False
        self.forget_wifi = False
        self.wifi = {
            "mode": "sta",
            "ssid": "workshop",
            "ip": args.host,
            "rssi": -52,
            "join": {"state": "idle", "ssid": "", "ip": "", "error": ""},
        }

        self._load_plant(Path(args.checkpoint))
        self.blob: bytes | None = None
        self.blob_hdr: dict[str, Any] | None = None
        self._blob_error = ""
        if not args.no_model:
            self._load_blob(Path(args.blob) if args.blob else None)

        # Controller state.
        self.run = RUN_IDLE
        self.fault = ""
        self.fault_t = 0.0
        self.mode = MODE_AI if self.can_ai() else MODE_MANUAL
        self.goal = self.labels.index("DD")
        self.random = False
        self.random_at: float | None = None
        self.x_ref = 0.0
        self.tick = 0
        self.excite: Excite | None = None
        self.base_cmd = 0.0  # the controller's own command, before the excitation
        self.rec_state = "idle"
        self.rec: list[bytes] = []
        self.rec_count = 0
        self.rec_capacity = REC_MAX_S
        self.rec_note = ""
        self.rec_excite: Excite | None = None
        self.rec_t0 = 0.0
        self.rec_start_us = 0
        self.rec_file: bytes | None = None
        self._reset_plant("DD")
        self.event(
            "info",
            f"boot #{self.boot_count}: v{self.version}, "
            f"{'rig' if self.rig else 'sim'} backend, reset: power-on",
        )
        if self.rig and not self.calib["angles"]:
            self.event(
                "warn", "rig angles not zeroed: hang the links at rest and Zero angles"
            )
        if self.rig and not self.calib["cart"]:
            self.event("warn", "cart not zeroed: centre it and Zero cart")

    # -- the event log (events.c) --
    def event(self, level: str, msg: str) -> None:
        """Append to the device event log."""
        self.events_seq += 1
        self.events.append(
            {
                "seq": self.events_seq,
                "boot": self.boot_count,
                # max: the reboot logs its boot line just before the new boot's clock starts
                "t_ms": max(0, int((time.monotonic() - self.boot) * 1000)),
                "unix": int(time.time()),  # the device's clock is set over SNTP
                "level": level,
                "msg": msg,
            }
        )
        log.info("event %s: %s", level, msg)

    def events_json(self) -> dict[str, Any]:
        """GET /api/events."""
        return {
            "seq": self.events_seq,
            "boot": self.boot_count,
            "uptime_ms": int((time.monotonic() - self.boot) * 1000),
            "events": list(self.events),
        }

    def _trip(self, why: str) -> None:
        self.run, self.fault, self.fault_t = RUN_FAULT, why, time.monotonic()
        self.fault_count += 1
        self.last_fault, self.last_fault_t = why, self.fault_t
        self.event("error", f"FAULT: {why}")

    # -- setup --
    def _repo_version(self) -> str:
        p = REPO / "firmware" / "version.txt"
        try:
            return p.read_text().strip() or "0.1.0-dev"
        except OSError:
            return "0.1.0-dev"

    def _load_plant(self, ckpt: Path) -> None:
        import torch

        from n_cartpole.env.cartpole import NPendulumCartpole
        from n_cartpole.policy.loader import load_policy

        self.ckpt = ckpt
        self.bundle = load_policy(ckpt)
        cfg = dataclasses.replace(
            self.bundle.env_config,
            goal_hold_steps=None,  # goals only change on command
            hold_prob=0.0,
            max_steps=1 << 40,
            randomize=None,  # the nominal plant, like ncp_sim
        )
        if not cfg.goal_conditioned:
            raise SystemExit(
                f"{ckpt}: the dashboard needs a goal-conditioned checkpoint"
            )

        class Plant(NPendulumCartpole):
            """The env, recording what the controller would see each tick."""

            last_meas: np.ndarray | None = None
            last_est: np.ndarray | None = None
            sense_us: float = 0.0

            def _sense(self) -> None:
                t0 = time.perf_counter()
                super()._sense()
                self.sense_us = (time.perf_counter() - t0) * 1e6

            def _read(self, true_pos, noise, res):  # type: ignore[no-untyped-def]
                out = super()._read(true_pos, noise, res)
                self.last_meas = np.array(out, dtype=np.float64)
                return out

            def _measured_state(self, sensors):  # type: ignore[no-untyped-def]
                est = super()._measured_state(sensors)
                self.last_est = np.array(est, dtype=np.float64)
                return est

        self.env = Plant(cfg)
        self.cfg = cfg
        self.p = cfg.physics
        self.n = cfg.n_links
        self.labels = goal_labels(self.n)
        self.delta = cfg.action_mode == "delta"
        norm, select = self.bundle.normalize, self.bundle.select_action

        def torch_policy(obs: np.ndarray) -> float:
            with torch.no_grad():
                return float(np.asarray(select(norm(obs[None, :]))).reshape(-1)[0])

        self.torch_policy = torch_policy
        self.policy: Callable[[np.ndarray], float] = torch_policy
        torch.set_num_threads(1)
        log.info(
            "plant: %s (%d links, obs %d, %s)",
            ckpt.name,
            self.n,
            self.bundle.obs_dim,
            cfg.action_mode,
        )

    def _load_blob(self, path: Path | None) -> None:
        if path is not None:
            blob = path.read_bytes()
        else:
            try:
                from n_cartpole.deploy.firmware_blob import (
                    export_blob,
                )
            except ImportError as e:
                self._blob_error = f"n_cartpole.deploy.firmware_blob unavailable ({e})"
                log.warning("/api/model will 404: %s", self._blob_error)
                return
            blob = export_blob(self.ckpt, dtype="f32")
        try:
            self.blob_hdr = validate_blob(blob)
        except ValueError as e:
            self._blob_error = f"exported blob is invalid: {e}"
            log.error("/api/model will 404: %s", self._blob_error)
            return
        self.blob = blob
        log.info("model blob: %s, %d bytes", self.blob_hdr["label"], len(blob))

    # -- plant helpers --
    def _reset_plant(self, start: str) -> None:
        self.obs, _ = self.env.reset(
            options={"start": start, "goal": self.labels[self.goal]}
        )
        self.settle_ok = False
        self.random_at = None

    def _force_to_action(self, force: float) -> float:
        """Return the env action that commands `force` (delta: as fast as the slew allows)."""
        if not self.delta:
            return force
        assert self.cfg.force_slew is not None
        rate = (force - self.env._prev_cmd) / (self.cfg.force_slew * self.p.dt)
        return float(np.clip(rate, -1.0, 1.0) * self.p.force_max)

    def can_ai(self) -> bool:
        """AI mode needs a stored model."""
        return not self.args.no_model or self.blob is not None

    # -- commands (validated in the HTTP thread, applied by the control loop) --
    def check_cmd(self, cmd: dict[str, Any]) -> str:  # noqa: PLR0911
        """Return an error string for an invalid command, '' if it can be queued."""
        if len(cmd) != 1:
            return "one key per command"
        key, val = next(iter(cmd.items()))
        ok = {
            "arm": lambda: isinstance(val, bool),
            "mode": lambda: val in ("ai", "manual"),
            "goal": lambda: val in self.labels,
            "random": lambda: isinstance(val, bool),
            "x_ref": lambda: isinstance(val, int | float) and np.isfinite(val),
            "reset": lambda: val in self.labels,
            "zero": lambda: val in ("angles", "cart"),
            "inject_fault": lambda: isinstance(val, str) and bool(val),
            "inject_io": lambda: val in ("weak-magnet", "bad-reads", "ok"),
            "record": lambda: isinstance(val, dict)
            and val.get("action") in ("start", "stop"),
            "excite": lambda: val is None
            or (isinstance(val, dict) and val.get("kind") in EXC_KINDS),
        }
        if key not in ok:
            return f"unknown command {key!r}"
        if not ok[key]():
            return f"bad value for {key!r}"
        if key == "arm" and val:
            if self.run == RUN_MAINT:
                return "an upload is writing flash"
            if self.rig and self.strict_budget and not self.budget_ok:
                return "loop budget exceeded: the model is too slow for the rig"
            if self.mode == MODE_AI and not self.can_ai():
                return "no model on the device"
            if self.rig and not self.calib["angles"]:
                return "angles not zeroed: hang the links at rest and Zero angles"
            if self.rig and not self.calib["cart"]:
                return "cart not zeroed: centre it and Zero cart"
        if key == "inject_io" and not self.rig:
            return "inject_io is rig-only (--rig)"
        if key == "mode" and val == "ai" and not self.can_ai():
            return "no model on the device"
        if key == "reset" and self.rig:
            return "reset is simulator-only"
        if key == "record" and val["action"] == "start":
            exc = val.get("excite")
            if exc is not None and (
                not isinstance(exc, dict) or exc.get("kind") not in EXC_KINDS
            ):
                return "bad excitation"
            secs = val.get("seconds", REC_MAX_S)
            if not isinstance(secs, int | float) or secs <= 0:
                return "bad record length"
        if key == "zero":
            if not self.rig:
                return "zeroing is rig-only"
            if self.run == RUN_ARMED:
                return "disarm first"
        return ""

    def _apply(self, cmd: dict[str, Any]) -> None:
        key, val = next(iter(cmd.items()))
        if key == "arm":
            if val and self.run != RUN_ARMED:
                self.run, self.fault = RUN_ARMED, ""
                ai = self.mode == MODE_AI
                self.event(
                    "info",
                    f"armed: {'AI -> ' + self.labels[self.goal] if ai else 'manual'}",
                )
            elif not val and self.run in (RUN_ARMED, RUN_FAULT):
                self.event(
                    "info",
                    "fault cleared (disarmed)" if self.run == RUN_FAULT else "disarmed",
                )
                self.run, self.fault = RUN_IDLE, ""
        elif key == "mode":
            new = MODE_AI if val == "ai" else MODE_MANUAL
            if new != self.mode:
                self.event("info", f"mode: {'AI' if new == MODE_AI else 'manual'}")
            self.mode = new
        elif key == "goal":
            self.random = False
            self._set_goal(self.labels.index(val))
        elif key == "random":
            self.random = bool(val)
            self.random_at = None
        elif key == "x_ref":
            lim = 0.5 * self.p.x_lim
            self.x_ref = float(np.clip(val, -lim, lim))
        elif key == "reset":
            self._reset_plant(val)
            if self.run == RUN_FAULT:
                self.run = RUN_IDLE
        elif key == "zero":
            if val == "angles":
                self.calib.update(
                    angles=True,
                    off1=round(float(self.rng.uniform(-3, 3)), 5),
                    off2=round(float(self.rng.uniform(-3, 3)), 5),
                    angles_boot=self.boot_count,
                    angles_unix=int(time.time()),
                )
                self.event(
                    "info",
                    f"angles zeroed: off1 {self.calib['off1']:.4f}, off2 {self.calib['off2']:.4f} rad",
                )
            else:
                self.calib.update(
                    cart=True,
                    x_zero=round(float(self.rng.uniform(-0.02, 0.02)), 5),
                    cart_boot=self.boot_count,
                    cart_unix=int(time.time()),
                )
                self.event("info", f"cart zeroed: x_zero {self.calib['x_zero']:.4f} m")
        elif key == "record":
            if val["action"] == "start":
                self._record_start(val)
            elif self.rec_state == "recording":
                self._record_stop()
        elif key == "excite":
            self.excite = (
                Excite(val, self.p.dt) if val and val.get("kind") != "none" else None
            )
            self.base_cmd = float(self.env._prev_cmd)
        elif key == "inject_fault":
            self._trip(val)
        elif key == "inject_io":
            if val == "weak-magnet":
                self.enc_weak = True
            if val == "bad-reads":
                self.enc_bad[1] += 37
            if val == "ok":
                self.enc_weak = False
                self.enc_bad = [0, 0]

    def _set_goal(self, goal: int) -> None:
        self.goal = goal
        self.obs = self.env.set_goal(self.labels[goal])
        self.random_at = None

    # -- recording (ncp_record.h) --
    def _record_start(self, val: dict[str, Any]) -> None:
        exc = val.get("excite")
        self.excite = (
            Excite(exc, self.p.dt) if exc and exc.get("kind") != "none" else None
        )
        self.rec_excite = self.excite
        self.base_cmd = float(self.env._prev_cmd)
        self.rec_capacity = float(min(float(val.get("seconds", REC_MAX_S)), REC_MAX_S))
        self.rec_note = str(val.get("note", ""))[:63]
        self.rec, self.rec_file = [], None
        self.rec_count += 1
        self.rec_t0 = time.perf_counter()
        self.rec_start_us = int((time.monotonic() - self.boot) * 1e6)
        self.rec_state = "recording"
        log.info(
            "recording %d started (%.0f s, %s)",
            self.rec_count,
            self.rec_capacity,
            self.excite.kind if self.excite else "no excitation",
        )

    def _record_stop(self) -> None:
        if self.rec_excite is not None and self.excite is self.rec_excite:
            self.excite = None
        exc = self.rec_excite
        crc = self.blob_hdr["crc32"] if self.blob_hdr else 0
        label = self.blob_hdr["label"] if self.blob_hdr else ""
        head = REC_HEADER.pack(
            b"NCPR",
            1,
            REC_HEADER.size,
            REC_SAMPLE.size,
            len(self.rec),
            0,
            crc,
            self.p.dt,
            self.n,
            1 if self.rig else 0,
            EXC_KINDS.index(exc.kind) if exc else 0,
            0,
            *(exc.params() if exc else [0.0] * 6),
            self.rec_start_us,
            self.version.encode()[:31],
            label.encode()[:63],
            self.rec_note.encode()[:63],
        )
        self.rec_file = head + b"".join(self.rec)
        self.rec_state = "done"
        log.info(
            "recording %d done: %d samples (%d bytes)",
            self.rec_count,
            len(self.rec),
            len(self.rec_file),
        )

    def record_status(self) -> dict[str, Any]:
        """Build the status JSON's "record" object."""
        n = len(self.rec)
        return {
            "state": self.rec_state,
            "samples": n,
            "seconds": round(n * self.p.dt, 2),
            "capacity_s": self.rec_capacity,
            "dropped": 0,
            "note": self.rec_note,
        }

    # -- the control loop (the `ctrl` task) --
    def control_loop(self) -> None:
        """200 Hz (the model's dt) loop: commands, policy, plant, telemetry."""
        dt = self.p.dt
        every = max(1, round(1.0 / (dt * TLM_RATE_HZ)))
        nxt = time.perf_counter()
        last = nxt
        while True:
            nxt += dt
            now = time.perf_counter()
            if now < nxt:
                time.sleep(nxt - now)
            elif now - nxt > 0.1:  # fell far behind (debugger, sleep): resync
                nxt = now
            t0 = time.perf_counter()
            period = (t0 - last) * 1e6
            last = t0
            if time.monotonic() < self.down_until:
                continue  # rebooting
            self.last_period_us = period
            frame = self._tick()
            ex = (time.perf_counter() - t0) * 1e6
            with self.stats_lock:
                self.stats.exec_us.append(ex)
                self.stats.policy_us.append(self.policy_us)
                self.stats.est_us.append(self.env.sense_us)
                self.stats.period_us.append(period)
            if self.tick % every == 0:
                self.frames.append(frame(ex, period))

    def _tick(self) -> Callable[[float, float], bytes]:
        while True:
            try:
                self._apply(self.cmds.get_nowait())
            except queue.Empty:
                break
        now = time.monotonic()
        self.tick += 1
        if self.ota_state == "pending" and now > self.pending_until:
            self.ota_state = "valid"
        if (
            self.run == RUN_FAULT
            and self.fault == "off rail"
            and now - self.fault_t > SETTLE_RESET_S
        ):
            self._reset_plant("DD")
            self.fault = "off rail (reset to DD)"

        obs = self.obs.copy()
        obs[0] -= self.x_ref  # the AI balances around x_ref (translation invariance)
        est = (
            self.env.last_est if self.env.last_est is not None else self.env.get_state()
        )
        meas = self.env.last_meas
        truth = self.env.get_state().copy()

        t0 = time.perf_counter()
        a_ai = self.policy(obs)  # runs every tick, armed or not (known timing)
        self.policy_us = (time.perf_counter() - t0) * 1e6

        exc = self.excite.next() if self.excite is not None else 0.0
        exc_on = (
            self.excite is not None and self.excite.active and self.run == RUN_ARMED
        )
        fmax = self.p.force_max
        if self.run != RUN_ARMED:
            action = self._force_to_action(0.0)
            self.base_cmd = 0.0
        elif self.mode == MODE_AI and not exc_on:
            action = a_ai
            self.base_cmd = float(self.env._prev_cmd)
        else:
            if self.mode == MODE_AI:  # the policy's own command, then the offset on top
                if self.delta:
                    assert self.cfg.force_slew is not None
                    rate = float(np.clip(a_ai / fmax, -1.0, 1.0))
                    base = self.base_cmd + rate * self.cfg.force_slew * self.p.dt
                else:
                    base = a_ai
                self.base_cmd = float(np.clip(base, -fmax, fmax))
            else:
                f = PD_KP * (self.x_ref - est[0]) - PD_KD * est[1]
                self.base_cmd = float(np.clip(f, -PD_FMAX, PD_FMAX))
            total = self.base_cmd + (exc if exc_on else 0.0)
            action = self._force_to_action(float(np.clip(total, -fmax, fmax)))

        frozen = self.run == RUN_FAULT and self.fault == "off rail"
        applied = 0.0
        if not frozen:
            self.obs, _, terminated, _, info = self.env.step(
                np.array([action], np.float32)
            )
            applied = float(info["applied_force"])
            if terminated:
                self._trip("off rail")
        cmd = float(self.env._prev_cmd)
        if self.rec_state == "recording":
            nan = [float("nan")] * MAX_STATE
            pos = list(meas if meas is not None else [truth[0], *truth[2::2]])
            rflags = (
                (REC_ARMED if self.run == RUN_ARMED else 0)
                | (REC_EXCITE if exc_on else 0)
                | (0 if self.rig else REC_TRUTH)
            )
            self.rec.append(
                REC_SAMPLE.pack(
                    self.tick & 0xFFFFFFFF,
                    int((time.perf_counter() - self.rec_t0) * 1e6) & 0xFFFFFFFF,
                    *(pos + [0.0] * MAX_POS)[:MAX_POS],
                    cmd,
                    applied,
                    cmd - self.base_cmd if exc_on else 0.0,
                    *(list(est) + [0.0] * MAX_STATE)[:MAX_STATE],
                    *(
                        nan
                        if self.rig
                        else (list(truth) + [0.0] * MAX_STATE)[:MAX_STATE]
                    ),
                    self.run,
                    self.mode,
                    self.goal,
                    rflags,
                    int(min(max(self.last_period_us, 0), 65535)),
                    0,
                )
            )
            if len(self.rec) * self.p.dt >= self.rec_capacity:
                self._record_stop()

        settled = bool(self.env.settled)
        if self.random and settled:
            if self.random_at is None:
                self.random_at = now + RANDOM_DWELL_S
            elif now >= self.random_at:
                others = [g for g in range(len(self.labels)) if g != self.goal]
                self._set_goal(int(self.rng.choice(others)))
        self.settle_ok = settled

        flags = (F_SETTLED if settled else 0) | (F_RANDOM if self.random else 0)
        if not self.rig:
            flags |= F_SIM | F_TRUTH
        tick, run, mode, goal, x_ref = (
            self.tick,
            self.run,
            self.mode,
            self.goal,
            self.x_ref,
        )
        pol_us, est_us = self.policy_us, self.env.sense_us

        def pack(ex: float, period: float) -> bytes:
            def pad(a: Any, n: int) -> list[float]:
                v = [] if a is None else [float(x) for x in np.asarray(a).ravel()]
                return (v + [0.0] * n)[:n]

            def u16(v: float) -> int:
                return int(min(max(v, 0), 65535))

            return FRAME.pack(
                tick & 0xFFFFFFFF,
                run,
                mode,
                goal,
                flags,
                *pad(est, MAX_STATE),
                *pad(meas, MAX_POS),
                *pad(truth if not self.rig else None, MAX_STATE),
                *pad(obs, MAX_OBS),
                cmd,
                applied,
                x_ref,
                u16(ex),
                u16(pol_us),
                u16(est_us),
                u16(period),
            )

        return pack

    policy_us: float = 0.0
    settle_ok: bool = False
    last_period_us: float = 0.0

    # -- status --
    @property
    def budget_ok(self) -> bool:
        """Execution within 80 % of the period (over the last status window).

        This is the mock's own Python loop, so it measures the machine running the mock, not the
        device: a slow or shared CI runner can exceed the budget. It is only reported by default;
        ``--strict-budget`` also makes it refuse to arm the rig, as the firmware does.
        """
        return self._last_loop.get("exec_avg_us", 0) <= 0.8 * self.p.dt * 1e6

    _last_loop: dict[str, Any] = {}  # noqa: RUF012  (replaced, never mutated)

    def status(self) -> dict[str, Any]:
        """Build the status JSON (design.md); timings cover the time since the last call."""
        with self.stats_lock:
            st, self.stats = self.stats, LoopStats()
        period_nom = self.p.dt * 1e6

        def avg(v: list[float]) -> float:
            return float(np.mean(v)) if v else 0.0

        if st.exec_us:
            ex_avg, per_avg = avg(st.exec_us), avg(st.period_us)
            self._last_loop = {
                "period_us": round(period_nom),
                "exec_avg_us": round(ex_avg),
                "exec_max_us": round(max(st.exec_us)),
                "policy_avg_us": round(avg(st.policy_us)),
                "est_avg_us": round(avg(st.est_us)),
                "misses": self._last_loop.get("misses", 0)
                + sum(1 for p in st.period_us if p > 1.5 * period_nom),
                "rt_factor": round(
                    min(1.0, period_nom / per_avg) if per_avg else 1.0, 3
                ),
                "budget_ok": ex_avg <= 0.8 * period_nom,
            }
        loop = self._last_loop
        ctrl_cpu = 100.0 * loop.get("exec_avg_us", 0) / period_nom
        j = lambda s: float(self.rng.normal(0, s))  # jitter
        n_ws = len(self.clients)
        tasks = [
            {
                "name": "ctrl",
                "core": 1,
                "prio": 22,
                "cpu": ctrl_cpu,
                "stack_free": 2104,
            },
            {
                "name": "IDLE1",
                "core": 1,
                "prio": 0,
                "cpu": 100 - ctrl_cpu - 0.4,
                "stack_free": 812,
            },
            {"name": "ipc1", "core": 1, "prio": 24, "cpu": 0.0, "stack_free": 520},
            {
                "name": "wifi",
                "core": 0,
                "prio": 23,
                "cpu": 3.1 + j(0.4),
                "stack_free": 3420,
            },
            {
                "name": "tiT",
                "core": 0,
                "prio": 18,
                "cpu": 2.2 + 0.6 * n_ws + j(0.3),
                "stack_free": 1890,
            },
            {"name": "sys_evt", "core": 0, "prio": 20, "cpu": 0.0, "stack_free": 1210},
            {
                "name": "esp_timer",
                "core": 0,
                "prio": 22,
                "cpu": 0.1,
                "stack_free": 2660,
            },
            {
                "name": "httpd",
                "core": 0,
                "prio": 5,
                "cpu": 0.8 + j(0.2),
                "stack_free": 4210,
            },
            {
                "name": "tlm",
                "core": 0,
                "prio": 4,
                "cpu": 1.4 + 0.9 * n_ws + j(0.2),
                "stack_free": 2380,
            },
            {"name": "ipc0", "core": 0, "prio": 24, "cpu": 0.0, "stack_free": 524},
        ]
        busy0 = sum(t["cpu"] for t in tasks if t["core"] == 0)
        tasks.append(
            {
                "name": "IDLE0",
                "core": 0,
                "prio": 0,
                "cpu": 100 - busy0,
                "stack_free": 830,
            }
        )
        for t in tasks:
            t["cpu"] = round(max(0.0, t["cpu"]), 1)
        model_bytes = self.blob_hdr["total_size"] if self.blob_hdr else 0
        self.wifi["rssi"] = int(-52 + j(1.5))
        return {
            "run": RUN_NAMES[self.run],
            "fault": self.fault if self.run == RUN_FAULT else "",
            "mode": "ai" if self.mode == MODE_AI else "manual",
            "goal": self.labels[self.goal],
            "random": self.random,
            "x_ref": round(self.x_ref, 4),
            "settled": self.settle_ok,
            "backend": "rig" if self.rig else "sim",
            "model": self.blob_hdr["label"]
            if self.blob_hdr
            else None,  # telemetry.c: the label
            "loop": loop,
            "heap": {
                "internal_free": int(182000 + j(800)),
                "internal_min": 151320,
                "psram_free": int(8 * 1024 * 1024 - 210_000 - model_bytes),
                "internal_largest": 110592,
                "alloc_failed": 0,
            },
            "wifi": dict(self.wifi),
            "uptime_s": int(time.monotonic() - self.boot),
            "boot": self.boot_count,
            "events_seq": self.events_seq,
            "unix": int(time.time()),
            "faults": {"count": self.fault_count, "last": self.last_fault}
            | (
                {"ago_s": round(time.monotonic() - self.last_fault_t, 1)}
                if self.fault_count
                else {}
            ),
            "io": self.io_status(),
            **({"calib": dict(self.calib)} if self.rig else {}),
            "cores": [
                {"id": 0, "load": round(min(100.0, busy0), 1)},
                {"id": 1, "load": round(min(100.0, ctrl_cpu + 0.4), 1)},
            ],
            "tasks": tasks,
            "record": self.record_status(),
            "excite": self.excite.status()
            if self.excite
            else {"kind": "none", "active": False, "t": 0.0},
        }

    def io_status(self) -> dict[str, Any]:
        """Report the rig's sensors and drive (hal.h ncp_hal_io_t); {"rig": false} on the sim."""
        if not self.rig:
            return {"rig": False}
        meas = self.env.last_meas
        pos = [0.0, np.pi, np.pi] if meas is None else list(meas)
        raws = [pos[1], pos[2] - pos[1]]  # joint 2 reads link 2 relative to link 1
        cmd = float(self.env._prev_cmd) if self.run == RUN_ARMED else 0.0
        enc = []
        for i, a in enumerate(raws):
            agc = (
                236
                if self.enc_weak and i == 1
                else self.enc_agc[i] + int(self.rng.integers(-2, 3))
            )
            enc.append(
                {
                    "raw": int((a % (2 * np.pi)) / (2 * np.pi) * 16384) % 16384,
                    "bad": self.enc_bad[i],
                    "agc": agc,
                    "ocf": True,
                    "cof": False,
                    "comp_low": False,
                    "comp_high": self.enc_weak and i == 1,
                }
            )
        return {
            "rig": True,
            "estop": self.run == RUN_FAULT and self.fault == "e-stop",
            "drive": self.run == RUN_ARMED,
            "duty": round(min(1.0, abs(cmd) / 15.35), 3),
            "dir": 1 if cmd >= 0 else -1,
            "cart_count": int(pos[0] / (2 * np.pi * 6.35e-3 / 4000)),
            "read_errors": 0,
            "enc": enc,
        }

    def info(self) -> dict[str, Any]:
        """Build the /api/info JSON."""
        return {
            "fw": {
                "project": "n_cartpole_fw",
                "version": self.version,
                "idf": "v5.3.1",
                "built": self.built,
                "repo": "Tsmorz/n-cartpole",
            },
            "chip": {
                "model": "ESP32-S3",
                "cores": 2,
                "revision": 2,
                "mhz": 240,
                "flash_mb": 16,
                "psram_mb": 8,
            },
            "ota": {
                "running": self.ota_running,
                "next": self.ota_next,
                "state": self.ota_state,
            },
            "backend": "rig" if self.rig else "sim",
            "hostname": self.hostname,
            "boot": self.boot_count,
            "reset_reason": "power-on",
            "telemetry": {
                "version": TLM_VERSION,
                "frame_size": FRAME.size,
                "rate_hz": TLM_RATE_HZ,
            },
            "model": model_info(self.blob_hdr) if self.blob_hdr else None,
        }

    # -- telemetry task (`tlm`) --
    def telemetry_loop(self) -> None:
        """Send queued frames ~25 times a second and the status at 2 Hz."""
        next_status = 0.0
        while True:
            time.sleep(1.0 / MSGS_PER_S)
            if time.monotonic() < self.down_until:
                self.frames.clear()
                continue
            frames: list[bytes] = []
            while self.frames and len(frames) < 255:
                frames.append(self.frames.popleft())
            msg = MSG_HDR.pack(TLM_VERSION, len(frames), FRAME.size) + b"".join(frames)
            text = None
            if time.monotonic() >= next_status:
                next_status = time.monotonic() + STATUS_PERIOD
                text = json.dumps(self.status()).encode()
            with self.clients_lock:
                clients = list(self.clients)
            for c in clients:
                ok = (not frames or c.send(0x2, msg)) and (
                    text is None or c.send(0x1, text)
                )
                if not ok:
                    with self.clients_lock:
                        if c in self.clients:
                            self.clients.remove(c)

    # -- joining a Wi-Fi network (no reboot; an SSID containing "wrong" fails) --
    def start_join(self, ssid: str, password: str) -> None:
        """Simulate POST /api/wifi: connecting, then ok or failed, then the AP closes."""
        join = self.wifi["join"]
        join.update(state="connecting", ssid=ssid, ip="", error="")

        def go() -> None:
            time.sleep(2.0)
            if "wrong" in ssid.lower():
                join.update(
                    state="failed", error="authentication failed (wrong password?)"
                )
                self.event("warn", f"Wi-Fi: could not join '{ssid}': {join['error']}")
                return
            join.update(state="ok", ip="192.168.1.77")
            self.event("info", f"Wi-Fi: joined '{ssid}' at 192.168.1.77")
            time.sleep(5.0)
            self.wifi.update(mode="sta", ssid=ssid, ip="192.168.1.77")
            join.update(state="idle")

        threading.Thread(target=go, daemon=True).start()

    # -- uploads and reboots --
    def reboot(self, delay: float, new_version: str | None = None) -> None:
        """Drop every connection, stay dark for REBOOT_S, come back fresh."""

        def go() -> None:
            time.sleep(delay)
            log.info("rebooting%s", f" into {new_version}" if new_version else "")
            self.down_until = time.monotonic() + REBOOT_S
            with self.clients_lock:
                clients, self.clients = self.clients, []
            for c in clients:
                c.close()
            if new_version:
                self.version = new_version
                self.built = time.strftime("%b %d %Y %H:%M:%S")
                self.ota_running, self.ota_next = self.ota_next, self.ota_running
                self.ota_state = "pending"
                self.pending_until = time.monotonic() + REBOOT_S + 10
            self.boot = time.monotonic() + REBOOT_S
            self.run, self.fault, self.mode, self.random, self.x_ref = (
                RUN_IDLE,
                "",
                MODE_AI,
                False,
                0.0,
            )
            self.goal = self.labels.index("DD")
            self._last_loop = {}
            self._reset_plant("DD")
            self.boot_count += 1
            self.fault_count, self.last_fault = 0, ""
            self.event(
                "info",
                f"boot #{self.boot_count}: v{self.version}, "
                f"{'rig' if self.rig else 'sim'} backend, reset: software restart",
            )
            if new_version:
                self.event(
                    "warn",
                    "new firmware pending: it is kept if it runs healthy for 10 s",
                )
            if self.forget_wifi:
                self.forget_wifi = False
                self.wifi.update(mode="ap", ssid="cartpole-3F2A", ip="192.168.4.1")

        threading.Thread(target=go, daemon=True).start()

    def write_flash(self, nbytes: int) -> None:
        """Pretend to erase + write flash: MAINT for a moment (drive disabled)."""
        prev = self.run
        self.run = RUN_MAINT
        time.sleep(min(3.0, 0.4 + nbytes / 2e6))
        self.run = prev if prev != RUN_ARMED else RUN_IDLE

    def store_model(self, blob: bytes) -> dict[str, Any]:
        """Validate and hot-load an uploaded model."""
        h = validate_blob(blob)
        if h["n_links"] != self.n:
            raise ValueError(f"model has {h['n_links']} links, the plant {self.n}")
        self.event("info", f"receiving model '{h['label'][:48]}' ({len(blob)} bytes)")
        self.write_flash(len(blob))
        self.blob, self.blob_hdr = blob, h
        self.event("info", f"model: {h['label']}")
        self.args.no_model = False
        if h["obs_dim"] == self.bundle.obs_dim:
            self.policy = blob_policy(blob, h)
            log.info("model %s loaded: the blob now drives the plant", h["label"])
        else:
            self.policy = self.torch_policy
            log.warning(
                "model obs_dim %d != env %d: keep the checkpoint policy",
                h["obs_dim"],
                self.bundle.obs_dim,
            )
        return h


# ---- HTTP -------------------------------------------------------------------
# The read-only GETs web.c answers with Access-Control-Allow-Origin: * (a dashboard served
# from another origin, e.g. the personal site, reads these; POSTs stay same-origin only).
CORS_GET = ("/api/info", "/api/status", "/api/events", "/api/model")
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
}
STATIC = {
    "/": "index.html",
    "/index.html": "index.html",
    "/app.js": "app.js",
    "/style.css": "style.css",
}
MAX_UPLOAD = 8 * 1024 * 1024


def make_handler(dev: Device) -> type[BaseHTTPRequestHandler]:
    """Bind a request handler class to the device."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "n-cartpole-mock/1"
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            if not str(args[0]).startswith(("GET /api/status", "GET /api/info")):
                log.debug(format, *args)

        # -- plumbing --
        def _down(self) -> bool:
            if time.monotonic() < dev.down_until:
                self.close_connection = True
                return True
            return False

        def _send(
            self,
            code: int,
            body: bytes,
            ctype: str,
            extra: dict[str, str] | None = None,
        ) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            if self.command == "GET" and self.path.split("?")[0] in CORS_GET:
                self.send_header("Access-Control-Allow-Origin", "*")  # as web.c does
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, obj: Any, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _err(self, code: int, msg: str) -> None:
            self._json({"ok": False, "error": msg}, code)

        def _guard(self, path: str) -> str:
            """Return why a POST is refused as cross-site (web.c web_post_guard), or ''."""
            want = (
                "application/octet-stream"
                if path in ("/api/model", "/api/ota")
                else "application/json"
            )
            if not (self.headers.get("Content-Type") or "").lower().startswith(want):
                return f"Content-Type must be {want}"
            origin = self.headers.get("Origin")
            if origin is not None:
                _, sep, rest = origin.partition("://")
                if not sep or rest.lower() != (self.headers.get("Host") or "").lower():
                    return "cross-origin request refused"
            return ""

        def _body(self) -> bytes | None:
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_UPLOAD:
                self._err(413, "too large")
                return None
            return self.rfile.read(n) if n else b""

        # -- GET --
        def do_GET(self) -> None:  # noqa: N802
            """Serve static files, API reads and the WebSocket upgrade."""
            if self._down():
                return
            path = self.path.split("?", 1)[0]
            if path in STATIC:
                f = WEB / STATIC[path]
                try:
                    body = f.read_bytes()
                except OSError:
                    self._err(404, f"{f.name} missing")
                    return
                self._send(200, body, CONTENT_TYPES[f.suffix])
            elif path == "/api/info":
                self._json(dev.info())
            elif path == "/api/status":
                self._json(dev.status())
            elif path == "/api/events":
                self._json(dev.events_json())
            elif path == "/api/model":
                if dev.blob is None:
                    self._err(
                        404,
                        "no model stored"
                        if dev.args.no_model
                        else (dev._blob_error or "no model stored"),
                    )
                else:
                    name = (dev.blob_hdr or {}).get("label", "model").replace("@", "-")
                    self._send(
                        200,
                        dev.blob,
                        "application/octet-stream",
                        {"Content-Disposition": f'attachment; filename="{name}.ncpm"'},
                    )
            elif path == "/api/record":
                if dev.rec_state == "recording":
                    self._err(409, "still recording")
                elif dev.rec_file is None:
                    self._err(404, "no recording")
                else:
                    name = f"cartpole-{dev.rec_count}.ncpr"
                    self._send(
                        200,
                        dev.rec_file,
                        "application/octet-stream",
                        {"Content-Disposition": f'attachment; filename="{name}"'},
                    )
            elif path == "/ws":
                self._websocket()
            elif path == "/favicon.ico":
                self._send(204, b"", "image/x-icon")
            else:
                self._err(404, "not found")

        do_HEAD = do_GET  # noqa: N815

        def _websocket(self) -> None:
            key = self.headers.get("Sec-WebSocket-Key")
            if self.headers.get("Upgrade", "").lower() != "websocket" or not key:
                self._err(400, "websocket upgrade expected")
                return
            accept = base64.b64encode(
                hashlib.sha1((key + WS_GUID).encode()).digest()  # noqa: S324 (RFC 6455)
            ).decode()
            self.send_response(101, "Switching Protocols")
            self.send_header("Upgrade", "websocket")
            self.send_header("Connection", "Upgrade")
            self.send_header("Sec-WebSocket-Accept", accept)
            self.end_headers()
            self.wfile.flush()
            client = WsClient(self.connection)
            with dev.clients_lock:
                dev.clients.append(client)
            client.send(0x1, json.dumps(dev.status()).encode())
            try:
                while client.alive:
                    fr = ws_read_frame(self.rfile)
                    if fr is None or fr[0] == 0x8:
                        break
                    if fr[0] == 0x9:
                        client.send(0xA, fr[1])
            except OSError:
                pass
            finally:
                client.alive = False
                with dev.clients_lock:
                    if client in dev.clients:
                        dev.clients.remove(client)
                self.close_connection = True

        # -- POST --
        def do_POST(self) -> None:  # noqa: N802, PLR0911
            """Handle commands, uploads, Wi-Fi and reboot."""
            if self._down():
                return
            path = self.path.split("?", 1)[0]
            why = self._guard(path)
            if why:
                self._err(403, why)
                return
            body = self._body()
            if body is None:
                return
            if path == "/api/cmd":
                try:
                    cmd = json.loads(body or b"{}")
                except ValueError:
                    self._err(400, "bad JSON")
                    return
                err = (
                    dev.check_cmd(cmd)
                    if isinstance(cmd, dict)
                    else "expected an object"
                )
                if err:
                    # Like the device: a valid command refused in the current state is 409.
                    conflict = (
                        "disarm" in err
                        or "no model" in err
                        or "not zeroed" in err
                        or (
                            "arm" in cmd
                            and cmd["arm"] is True
                            and "bad value" not in err
                        )
                    )
                    self._err(409 if conflict else 400, err)
                    return
                dev.cmds.put(cmd)
                self._json({"ok": True})
            elif path in ("/api/model", "/api/ota"):
                if dev.run == RUN_ARMED:
                    self._err(409, "disarm first: writing flash stalls both cores")
                    return
                if dev.run == RUN_MAINT:
                    self._err(409, "another upload is in progress")
                    return
                try:
                    if path == "/api/model":
                        h = dev.store_model(body)
                        self._json({"ok": True, "label": h["label"]})
                    else:
                        desc = app_descriptor(body)
                        if desc["project"] != "n_cartpole_fw":
                            raise ValueError(
                                f"wrong project {desc['project']!r} (expected n_cartpole_fw)"
                            )
                        dev.write_flash(len(body))
                        dev.event(
                            "info",
                            f"firmware {desc['version']} written to {dev.ota_next}; rebooting",
                        )
                        self._json({"ok": True, "version": desc["version"]})
                        dev.reboot(1.0, desc["version"])
                except ValueError as e:
                    self._err(400, str(e))
            elif path == "/api/wifi":
                try:
                    w = json.loads(body or b"{}")
                    ssid = str(w["ssid"])
                except (ValueError, KeyError, TypeError):
                    self._err(400, "expected {ssid, password}")
                    return
                if not 0 < len(ssid) <= 32:
                    self._err(400, "SSID must be 1-32 characters")
                    return
                self._json({"ok": True})
                dev.start_join(ssid, str(w.get("password", "")))
            elif path == "/api/wifi/forget":
                if dev.run == RUN_ARMED:
                    self._err(409, "disarm first")
                    return
                dev.forget_wifi = True
                dev.event(
                    "warn",
                    "Wi-Fi network forgotten: rebooting into the setup access point",
                )
                self._json({"ok": True})
                dev.reboot(0.8)
            elif path == "/api/hostname":
                try:
                    name = str(json.loads(body or b"{}")["hostname"])
                except (ValueError, KeyError, TypeError):
                    self._err(400, "expected {hostname}")
                    return
                if not re.fullmatch(r"[a-z0-9]([a-z0-9-]{0,30}[a-z0-9])?", name):
                    self._err(
                        400,
                        "hostname: 1-32 of a-z, 0-9 and '-', not starting or ending with '-'",
                    )
                    return
                if dev.run == RUN_ARMED:
                    self._err(409, "disarm first")
                    return
                dev.hostname = name
                dev.event("info", f"hostname: {name}.local")
                self._json({"ok": True, "hostname": name})
            elif path == "/api/reboot":
                dev.event("info", "reboot requested")
                self._json({"ok": True})
                dev.reboot(0.5)
            else:
                self._err(404, "not found")

    return Handler


def main() -> None:
    """Parse the CLI and serve until interrupted."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument(
        "--host",
        default="127.0.0.1",
        help="bind address (0.0.0.0 to reach it from a phone)",
    )
    ap.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    ap.add_argument("--rig", action="store_true", help="pretend to be the rig backend")
    ap.add_argument(
        "--blob", help="serve this .ncpm at /api/model instead of exporting one"
    )
    ap.add_argument(
        "--no-model", action="store_true", help="no model stored on the device"
    )
    ap.add_argument(
        "--uncalibrated",
        action="store_true",
        help="rig: the zero references are not set yet (arming is refused)",
    )
    ap.add_argument(
        "--strict-budget",
        action="store_true",
        help="rig: refuse to arm while this process's own loop exceeds 80 %% of the period "
        "(depends on the machine, so off by default and in the tests)",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    dev = Device(args)
    threading.Thread(target=dev.control_loop, name="ctrl", daemon=True).start()
    threading.Thread(target=dev.telemetry_loop, name="tlm", daemon=True).start()
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(dev))
    srv.daemon_threads = True
    log.info(
        "dashboard on http://%s:%d/ (%s backend)",
        args.host,
        args.port,
        "rig" if args.rig else "sim",
    )
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
