"""Excitation signals, sample for sample as the firmware's ``ncp_excite.c``.

Used to simulate an experiment exactly as the device would run it (and as the
reference the C implementation is tested against).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

MASK64 = (1 << 64) - 1
KINDS = ("none", "chirp", "prbs", "steps", "sine")


def _seed_mix(seed: int) -> int:
    """splitmix64 finalizer (never 0), as ncp_sim / ncp_excite seed their RNG."""
    z = (seed + 0x9E3779B97F4A7C15) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    z ^= z >> 31
    return z or 0x2545F4914F6CDD1D


@dataclass
class ExciteParams:
    """``ncp_excite_params_t``: kind, amplitude (N), frequencies (Hz), period, duration (s)."""

    kind: str = "none"
    amp: float = 3.0
    f0: float = 0.2
    f1: float = 6.0
    period: float = 0.05
    duration: float = 0.0  # <= 0: until stopped
    seed: int = 1

    def __post_init__(self) -> None:
        """Reject an unknown kind."""
        if self.kind not in KINDS:
            raise ValueError(
                f"excitation kind must be one of {KINDS}, got {self.kind!r}"
            )


class Excitation:
    """A running excitation sampled every ``dt`` (``ncp_excite_t``)."""

    def __init__(self, p: ExciteParams, dt: float) -> None:
        """Start the signal (kind ``none``: inactive, always 0)."""
        self.p = p
        self.dt = dt if dt > 0 else 1.0
        self.t = 0.0
        self.phase = 0.0
        self.value = 0.0
        self._rng = _seed_mix(int(p.seed) & 0xFFFFFFFF)
        self.active = p.kind != "none"

    def _rand(self) -> int:
        x = self._rng
        x ^= x >> 12
        x ^= (x << 25) & MASK64
        x ^= x >> 27
        self._rng = x
        return (x * 0x2545F4914F6CDD1D) & MASK64

    def _hold(self) -> int:
        return max(1, round_half_away(self.p.period / self.dt))

    def next(self) -> float:
        """Return this tick's force offset (N); 0 once the duration has elapsed."""
        if not self.active:
            self.value = 0.0
            return 0.0
        p = self.p
        k = round_half_away(self.t / self.dt)
        if p.duration > 0.0 and k >= round_half_away(p.duration / self.dt):
            self.active = False
            self.value = 0.0
            return 0.0
        if p.kind == "chirp":
            v = p.amp * math.sin(self.phase)
            f = p.f0
            if p.duration > 0.0 and p.f0 > 0.0 and p.f1 > 0.0:
                f = p.f0 * math.pow(p.f1 / p.f0, self.t / p.duration)
            elif p.duration > 0.0:
                f = p.f0 + (p.f1 - p.f0) * self.t / p.duration
            self.phase += 2.0 * math.pi * f * self.dt
        elif p.kind == "prbs":
            if k % self._hold() == 0:
                self.value = p.amp if self._rand() >> 63 else -p.amp
            v = self.value
        elif p.kind == "steps":
            v = p.amp * (1.0, 0.0, -1.0, 0.0)[(k // self._hold()) % 4]
        else:  # sine
            v = p.amp * math.sin(2.0 * math.pi * p.f0 * self.t)
        self.value = v
        self.t = (k + 1) * self.dt
        return v


def round_half_away(x: float) -> int:
    """C ``lround``: nearest integer, halves away from zero."""
    return int(math.floor(abs(x) + 0.5)) * (1 if x >= 0 else -1)


def excitation_signal(p: ExciteParams, dt: float, n: int) -> list[float]:
    """Return the first ``n`` samples of an excitation."""
    e = Excitation(p, dt)
    return [e.next() for _ in range(n)]
