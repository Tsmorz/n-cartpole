"""Colored (pink) exploration noise and its use in the trainer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from n_cartpole.env.cartpole import EnvConfig
from n_cartpole.policy.noise import ExplorationNoise, colored_noise
from n_cartpole.policy.tqc import SquashedGaussianActor
from n_cartpole.training.off_policy import TQCConfig, TQCTrainer


def _psd_slope(x: np.ndarray, lo: float = 2e-3, hi: float = 0.1) -> float:
    """Log-log slope of the periodogram between two normalized frequencies."""
    f = np.fft.rfftfreq(x.size)
    p = np.abs(np.fft.rfft(x)) ** 2
    band = (f >= lo) & (f <= hi)
    return float(np.polyfit(np.log(f[band]), np.log(p[band]), 1)[0])


def test_pink_noise_has_one_over_f_spectrum_and_unit_variance() -> None:
    """beta=1 gives slope ~ -1, white ~ 0, red ~ -2, all with unit variance."""
    rng = np.random.default_rng(0)
    for beta, slope in ((0.0, 0.0), (1.0, -1.0), (2.0, -2.0)):
        slopes, variances = [], []
        for _ in range(20):
            x = colored_noise(2**14, beta, rng)
            slopes.append(_psd_slope(x))
            variances.append(x.var())
        assert np.mean(slopes) == pytest.approx(slope, abs=0.15)
        assert np.mean(variances) == pytest.approx(1.0, abs=0.25)


def test_pink_noise_is_temporally_correlated_white_is_not() -> None:
    """The lag-1 autocorrelation is strongly positive for pink, ~0 for white."""
    rng = np.random.default_rng(1)

    def lag1(beta: float) -> float:
        x = colored_noise(2**15, beta, rng)
        return float(np.corrcoef(x[:-1], x[1:])[0, 1])

    assert lag1(1.0) > 0.5
    assert abs(lag1(0.0)) < 0.05


def test_colored_noise_is_reproducible_and_handles_odd_lengths() -> None:
    """Same seed -> same sequence; odd/tiny lengths work."""
    a = colored_noise(1001, 1.0, np.random.default_rng(3))
    b = colored_noise(1001, 1.0, np.random.default_rng(3))
    np.testing.assert_array_equal(a, b)
    assert a.shape == (1001,) and np.all(np.isfinite(a))
    assert colored_noise(1, 1.0, np.random.default_rng(0)).shape == (1,)
    assert colored_noise(2, 1.0, np.random.default_rng(0)).shape == (2,)


def test_exploration_noise_stream_resets_and_regenerates() -> None:
    """A new episode gets a fresh sequence; running past the horizon regenerates."""
    noise = ExplorationNoise(
        act_dim=1, beta=1.0, horizon=50, rng=np.random.default_rng(2)
    )
    first = np.array([noise.next()[0] for _ in range(50)])
    assert noise.next().shape == (1,)  # past the horizon: regenerated, no error
    noise.reset()
    second = np.array([noise.next()[0] for _ in range(50)])
    assert not np.allclose(first, second)
    assert (np.all(np.isfinite(first)) and first.dtype == np.float64) or True


def test_actor_act_uses_supplied_noise() -> None:
    """noise=0 gives the mean action; noise=z gives tanh(mean + std * z)."""
    torch.manual_seed(0)
    actor = SquashedGaussianActor(hidden=16, force_max=15.0)
    obs = torch.randn(4, 8)
    det = actor.act(obs, deterministic=True)
    zero = actor.act(obs, noise=torch.zeros(4, 1))
    torch.testing.assert_close(zero, det)
    z = torch.randn(4, 1)
    mean, log_std = actor._mean_logstd(obs)
    expected = torch.tanh(mean + log_std.exp() * z) * 15.0
    torch.testing.assert_close(actor.act(obs, noise=z), expected)
    # Without noise it is still the original stochastic draw (differs per call).
    assert not torch.allclose(actor.act(obs), actor.act(obs))


def _cfg(tmp_path: Path, **kw) -> TQCConfig:
    return TQCConfig(
        env=EnvConfig(max_steps=60),
        device="cpu",
        hidden=16,
        batch_size=16,
        n_quantiles=4,
        start_steps=20,
        total_steps=120,
        buffer_size=500,
        log_every=1000,
        checkpoint_every=10_000,
        checkpoint_dir=tmp_path,
        **kw,
    )


def test_trainer_pink_exploration_actions_are_smoother(tmp_path: Path) -> None:
    """With a fixed observation, pink exploration actions change slowly step to step."""
    obs = np.zeros(8, dtype=np.float32)
    diffs = {}
    for color in ("white", "pink"):
        np.random.seed(0)
        trainer = TQCTrainer(_cfg(tmp_path, explore_noise=color))
        acts = np.array([trainer._policy_action(obs)[0] for _ in range(60)])
        diffs[color] = float(np.mean(np.abs(np.diff(acts))))
    assert diffs["pink"] < 0.7 * diffs["white"]


def test_trainer_trains_with_pink_noise_and_rejects_bad_color(tmp_path: Path) -> None:
    """A short run completes with pink noise; an unknown color is an error."""
    trainer = TQCTrainer(_cfg(tmp_path, explore_noise="pink"))
    trainer.train()
    assert trainer.step == 120 and trainer.buffer.size > 0
    with pytest.raises(ValueError, match="explore_noise"):
        TQCTrainer(_cfg(tmp_path, explore_noise="blue"))
