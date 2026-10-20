"""Temporally correlated (colored) exploration noise.

White Gaussian exploration noise re-draws every control step, so on a system
integrating at 100 Hz it averages out: a pendulum can only be pumped by pushing in
a consistent direction for many steps, and the jitter it adds is not something a
real actuator could follow anyway. Pink noise (power spectrum ``1/f``) keeps the
per-step marginal ``N(0, 1)`` that the policy's std expects but correlates it over
time, so exploratory pushes are sustained and smooth (Eberhard et al., "Pink noise
is all you need", ICLR 2023). ``beta = 0`` is white, ``1`` pink, ``2`` red.
"""

from __future__ import annotations

import numpy as np


def colored_noise(n: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    """Return ``n`` samples of unit-variance Gaussian noise with PSD ``∝ 1/f**beta``.

    Built in the Fourier domain; the lowest non-zero frequency (``1/n``) also
    stands in for DC so a sequence has no unbounded drift over its length.
    """
    if n < 2:
        return rng.normal(size=n)
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    scale = f ** (-beta / 2.0)
    spec = (rng.normal(size=f.size) + 1j * rng.normal(size=f.size)) * scale
    spec[0] = spec[0].real  # DC must be real
    if n % 2 == 0:
        spec[-1] = spec[-1].real  # so must Nyquist
    # Normalize analytically (the sequence variance is fixed by the spectrum).
    w = scale[1:].copy()
    if n % 2 == 0:
        w[-1] *= 0.5
    sigma = 2.0 * np.sqrt(np.sum(w**2)) / n
    return np.fft.irfft(spec, n) / sigma


class ExplorationNoise:
    """Per-episode stream of ``(act_dim,)`` unit-variance noise samples.

    ``beta = 0`` reproduces ordinary white noise, so the trainer can use one code
    path. Call :meth:`reset` at every episode start; each dimension gets an
    independent sequence, regenerated if an episode outlasts ``horizon``.
    """

    def __init__(
        self,
        act_dim: int = 1,
        beta: float = 1.0,
        horizon: int = 4096,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Create the stream (``horizon`` is the sequence length to pre-generate)."""
        self.act_dim, self.beta, self.horizon = act_dim, beta, horizon
        self.rng = rng if rng is not None else np.random.default_rng()
        self._seq = np.zeros((act_dim, 0))
        self._i = 0
        self.reset()

    def reset(self) -> None:
        """Start a new independent noise sequence (call at every episode start)."""
        self._seq = np.stack(
            [
                colored_noise(self.horizon, self.beta, self.rng)
                for _ in range(self.act_dim)
            ]
        )
        self._i = 0

    def next(self) -> np.ndarray:
        """Return the next noise sample, shape ``(act_dim,)``."""
        if self._i >= self._seq.shape[1]:
            self.reset()
        out = self._seq[:, self._i]
        self._i += 1
        return out.astype(np.float32)
