"""The firmware's recording file (``.ncpr``), byte for byte as ``ncp_record.h``.

A 256-byte ``ncp_rec_header_t`` followed by ``n_samples`` packed 128-byte
``ncp_rec_sample_t`` records, little-endian. Arrays are sized for the firmware's
maxima (4 links: 5 readings, 10 state entries); :class:`Recording` trims them to
the recording's link count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

MAGIC = b"NCPR"
VERSION = 1
HEADER_SIZE = 256
SAMPLE_SIZE = 128
MAX_LINKS = 4
MAX_POS = 1 + MAX_LINKS
MAX_STATE = 2 + 2 * MAX_LINKS

# ncp_exc_kind_t
EXCITE_KINDS = ("none", "chirp", "prbs", "steps", "sine")
EXCITE_PARAMS = ("amp", "f0", "f1", "period", "duration", "seed")

# NCP_REC_* sample flags
FLAG_ARMED = 1 << 0
FLAG_EXCITE = 1 << 1
FLAG_TRUTH = 1 << 2
FLAG_ENC_ERR = 1 << 3

HEADER_DTYPE = np.dtype(
    [
        ("magic", "S4"),
        ("version", "<u2"),
        ("header_size", "<u2"),
        ("sample_size", "<u4"),
        ("n_samples", "<u4"),
        ("dropped", "<u4"),
        ("model_crc32", "<u4"),
        ("dt", "<f8"),
        ("n_links", "u1"),
        ("backend", "u1"),
        ("excite_kind", "u1"),
        ("reserved0", "u1"),
        ("excite", "<f4", (6,)),
        ("start_us", "<u8"),
        ("fw_version", "S32"),
        ("model_label", "S64"),
        ("note", "S64"),
        ("reserved1", "u1", (28,)),
    ]
)
SAMPLE_DTYPE = np.dtype(
    [
        ("tick", "<u4"),
        ("t_us", "<u4"),
        ("pos", "<f4", (MAX_POS,)),
        ("cmd", "<f4"),
        ("applied", "<f4"),
        ("excite", "<f4"),
        ("est", "<f4", (MAX_STATE,)),
        ("truth", "<f4", (MAX_STATE,)),
        ("run", "u1"),
        ("mode", "u1"),
        ("goal", "u1"),
        ("flags", "u1"),
        ("period_us", "<u2"),
        ("reserved", "<u2"),
    ]
)
assert HEADER_DTYPE.itemsize == HEADER_SIZE
assert SAMPLE_DTYPE.itemsize == SAMPLE_SIZE


def _text(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("utf-8", errors="replace")


@dataclass
class Recording:
    """One recording: header fields plus per-tick arrays (``N`` samples, ``n`` links).

    ``pos`` is ``(N, n + 1)`` readings ``[x, th0, ...]``; ``est`` / ``truth`` are
    ``(N, 2 + 2n)`` states (``truth`` is NaN on the rig). ``t`` is the actual
    sample time in seconds since the first sample.
    """

    dt: float
    n_links: int
    tick: np.ndarray
    t: np.ndarray
    pos: np.ndarray
    cmd: np.ndarray
    applied: np.ndarray
    excite: np.ndarray
    est: np.ndarray
    truth: np.ndarray
    run: np.ndarray
    mode: np.ndarray
    goal: np.ndarray
    flags: np.ndarray
    period_us: np.ndarray
    backend: int = 0
    excite_kind: str = "none"
    excite_params: dict[str, float] = field(default_factory=dict)
    start_us: int = 0
    fw_version: str = ""
    model_label: str = ""
    model_crc32: int = 0
    note: str = ""
    dropped: int = 0
    source: str = ""

    def __len__(self) -> int:
        """Return the number of samples."""
        return int(self.tick.size)

    @property
    def armed(self) -> np.ndarray:
        """Per sample: whether the drive was enabled (the command reached it)."""
        return (self.flags & FLAG_ARMED) != 0

    @property
    def enc_err(self) -> np.ndarray:
        """Per sample: an encoder read failed (the reading repeats the last good one)."""
        return (self.flags & FLAG_ENC_ERR) != 0

    @property
    def has_truth(self) -> bool:
        """Whether the simulator's true state was recorded."""
        return bool(np.any(self.flags & FLAG_TRUTH))

    def segments(self) -> list[tuple[int, int]]:
        """``[start, stop)`` index ranges of consecutive ticks without encoder errors."""
        bad = self.enc_err
        out: list[tuple[int, int]] = []
        start = None
        for i in range(len(self)):
            if bad[i]:
                if start is not None:
                    out.append((start, i))
                start = None
                continue
            if start is None:
                start = i
            elif self.tick[i] != self.tick[i - 1] + 1:
                out.append((start, i))
                start = i
        if start is not None:
            out.append((start, len(self)))
        return out

    def trimmed(self, t0: float = 0.0, t1: float | None = None) -> Recording:
        """Return the samples with ``t0 <= t < t1`` (seconds since the first one)."""
        keep = (self.t >= t0) & (self.t < (np.inf if t1 is None else t1))
        arrays = {
            k: getattr(self, k)[keep]
            for k in (
                "tick",
                "t",
                "pos",
                "cmd",
                "applied",
                "excite",
                "est",
                "truth",
                "run",
                "mode",
                "goal",
                "flags",
                "period_us",
            )
        }
        meta = {
            k: getattr(self, k)
            for k in (
                "backend",
                "excite_kind",
                "excite_params",
                "start_us",
                "fw_version",
                "model_label",
                "model_crc32",
                "note",
                "dropped",
                "source",
            )
        }
        return Recording(dt=self.dt, n_links=self.n_links, **arrays, **meta)

    # --- file I/O ------------------------------------------------------------

    @classmethod
    def from_bytes(cls, data: bytes, source: str = "") -> Recording:
        """Parse a ``.ncpr`` file's contents.

        Raises:
            ValueError: on a bad magic/version/size, or a truncated file.

        """
        if len(data) < HEADER_SIZE:
            raise ValueError(f"recording too short for a header: {len(data)} bytes")
        h = np.frombuffer(data, HEADER_DTYPE, count=1)[0]
        if bytes(h["magic"]) != MAGIC:
            raise ValueError("not an .ncpr recording (bad magic)")
        if int(h["version"]) != VERSION or int(h["header_size"]) != HEADER_SIZE:
            raise ValueError(f"unsupported recording version {int(h['version'])}")
        if int(h["sample_size"]) != SAMPLE_SIZE:
            raise ValueError(f"unexpected sample size {int(h['sample_size'])}")
        n_samples = int(h["n_samples"])
        if len(data) < HEADER_SIZE + n_samples * SAMPLE_SIZE:
            raise ValueError(
                f"recording truncated: header says {n_samples} samples, file holds "
                f"{(len(data) - HEADER_SIZE) // SAMPLE_SIZE}"
            )
        n = int(h["n_links"])
        if not 1 <= n <= MAX_LINKS:
            raise ValueError(f"n_links {n} out of range")
        s = np.frombuffer(data, SAMPLE_DTYPE, count=n_samples, offset=HEADER_SIZE)
        kind = int(h["excite_kind"])
        dim = 2 + 2 * n
        t_us = s["t_us"].astype(np.int64)
        return cls(
            dt=float(h["dt"]),
            n_links=n,
            tick=s["tick"].astype(np.int64),
            t=(t_us - (t_us[0] if t_us.size else 0)) * 1e-6,
            pos=s["pos"][:, : n + 1].astype(np.float64),
            cmd=s["cmd"].astype(np.float64),
            applied=s["applied"].astype(np.float64),
            excite=s["excite"].astype(np.float64),
            est=s["est"][:, :dim].astype(np.float64),
            truth=s["truth"][:, :dim].astype(np.float64),
            run=s["run"].copy(),
            mode=s["mode"].copy(),
            goal=s["goal"].copy(),
            flags=s["flags"].copy(),
            period_us=s["period_us"].copy(),
            backend=int(h["backend"]),
            excite_kind=EXCITE_KINDS[kind] if kind < len(EXCITE_KINDS) else str(kind),
            excite_params={
                k: float(v) for k, v in zip(EXCITE_PARAMS, h["excite"], strict=True)
            },
            start_us=int(h["start_us"]),
            fw_version=_text(bytes(h["fw_version"])),
            model_label=_text(bytes(h["model_label"])),
            model_crc32=int(h["model_crc32"]),
            note=_text(bytes(h["note"])),
            dropped=int(h["dropped"]),
            source=source,
        )

    def to_bytes(self) -> bytes:
        """Serialize to the ``.ncpr`` layout."""
        n, N = self.n_links, len(self)
        h = np.zeros(1, HEADER_DTYPE)
        kind = (
            EXCITE_KINDS.index(self.excite_kind)
            if self.excite_kind in EXCITE_KINDS
            else 0
        )
        fields: dict[str, Any] = {
            "magic": MAGIC,
            "version": VERSION,
            "header_size": HEADER_SIZE,
            "sample_size": SAMPLE_SIZE,
            "n_samples": N,
            "dropped": self.dropped,
            "model_crc32": self.model_crc32,
            "dt": self.dt,
            "n_links": n,
            "backend": self.backend,
            "excite_kind": kind,
            "excite": [self.excite_params.get(k, 0.0) for k in EXCITE_PARAMS],
            "start_us": self.start_us,
            "fw_version": self.fw_version.encode()[:31],
            "model_label": self.model_label.encode()[:63],
            "note": self.note.encode()[:63],
        }
        for k, v in fields.items():
            h[k] = v
        s = np.zeros(N, SAMPLE_DTYPE)
        dim = 2 + 2 * n
        s["tick"] = self.tick
        s["t_us"] = np.round(self.t * 1e6).astype(np.int64)
        s["pos"][:, : n + 1] = self.pos
        s["est"][:, :dim] = self.est
        s["truth"][:, :dim] = self.truth
        for k in ("cmd", "applied", "excite", "run", "mode", "goal", "flags"):
            s[k] = getattr(self, k)
        s["period_us"] = self.period_us
        return h.tobytes() + s.tobytes()

    @classmethod
    def read(cls, path: str | Path) -> Recording:
        """Read a ``.ncpr`` file."""
        return cls.from_bytes(Path(path).read_bytes(), source=str(path))

    def write(self, path: str | Path) -> None:
        """Write a ``.ncpr`` file."""
        Path(path).write_bytes(self.to_bytes())


def make_recording(
    dt: float,
    pos: np.ndarray,
    cmd: np.ndarray,
    *,
    armed: np.ndarray | bool = True,
    excite: np.ndarray | None = None,
    truth: np.ndarray | None = None,
    est: np.ndarray | None = None,
    note: str = "",
) -> Recording:
    """Build a :class:`Recording` from arrays (e.g. a simulated experiment)."""
    pos = np.asarray(pos, dtype=float)
    N, n = pos.shape[0], pos.shape[1] - 1
    dim = 2 + 2 * n
    arm = np.broadcast_to(np.asarray(armed, dtype=bool), (N,))
    flags = np.where(arm, FLAG_ARMED, 0).astype(np.uint8)
    exc = np.zeros(N) if excite is None else np.asarray(excite, dtype=float)
    flags |= np.where(exc != 0.0, FLAG_EXCITE, 0).astype(np.uint8)
    if truth is not None:
        flags |= FLAG_TRUTH
    return Recording(
        dt=dt,
        n_links=n,
        tick=np.arange(N, dtype=np.int64),
        t=np.arange(N) * dt,
        pos=pos,
        cmd=np.asarray(cmd, dtype=float),
        applied=np.zeros(N),
        excite=exc,
        est=np.zeros((N, dim)) if est is None else np.asarray(est, dtype=float),
        truth=np.full((N, dim), np.nan) if truth is None else np.asarray(truth),
        run=arm.astype(np.uint8),
        mode=np.ones(N, dtype=np.uint8),
        goal=np.zeros(N, dtype=np.uint8),
        flags=flags,
        period_us=np.full(N, int(round(dt * 1e6)), dtype=np.uint16),
        note=note,
    )
