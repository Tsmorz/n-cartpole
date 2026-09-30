"""Tests for policy/loader.py: load_policy() and PolicyBundle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from n_cartpole.env.dynamics import PhysicsParams
from n_cartpole.policy.loader import load_policy
from n_cartpole.policy.running_norm import RunningNorm
from n_cartpole.policy.tqc import QuantileCritic, SquashedGaussianActor
from n_cartpole.training.off_policy import TQCConfig

_OBS_DIM = 8  # double-link default
_HIDDEN = 16


def _make_tqc_checkpoint(path: Path, *, with_critic: bool = True) -> None:
    """Write a minimal TQC checkpoint to disk."""
    phys = PhysicsParams()
    actor = SquashedGaussianActor(
        obs_dim=_OBS_DIM, hidden=_HIDDEN, force_max=phys.force_max
    )
    norm = RunningNorm(_OBS_DIM)
    cfg = TQCConfig(hidden=_HIDDEN)
    ckpt: dict = {
        "actor": actor.state_dict(),
        "norm": norm.state_dict(),
        "algo": "tqc",
        "cfg": cfg,
    }
    if with_critic:
        critic = QuantileCritic(obs_dim=_OBS_DIM, hidden=_HIDDEN)
        ckpt["critic"] = critic.state_dict()
    torch.save(ckpt, path)


# ---------------------------------------------------------------------------
# load_policy: file-not-found
# ---------------------------------------------------------------------------


def test_load_policy_missing_file(tmp_path: Path) -> None:
    """load_policy raises FileNotFoundError for a non-existent path."""
    with pytest.raises(FileNotFoundError):
        load_policy(tmp_path / "nonexistent.pt")


# ---------------------------------------------------------------------------
# Non-TQC checkpoints
# ---------------------------------------------------------------------------


def test_load_policy_rejects_ppo_checkpoint(tmp_path: Path) -> None:
    """A checkpoint tagged with another algorithm raises a clear ValueError."""
    path = tmp_path / "ppo.pt"
    norm = RunningNorm(_OBS_DIM)
    torch.save({"actor": {}, "norm": norm.state_dict(), "algo": "ppo"}, path)
    with pytest.raises(ValueError, match="only TQC is supported"):
        load_policy(path)


# ---------------------------------------------------------------------------
# TQC checkpoints
# ---------------------------------------------------------------------------


def test_load_tqc_policy_with_critic(tmp_path: Path) -> None:
    """TQC checkpoint with critic populates bundle.value."""
    path = tmp_path / "tqc.pt"
    _make_tqc_checkpoint(path, with_critic=True)
    bundle = load_policy(path)
    assert bundle.algo == "tqc"
    assert bundle.value is not None


def test_load_tqc_policy_without_critic(tmp_path: Path) -> None:
    """TQC checkpoint without critic leaves bundle.value as None."""
    path = tmp_path / "tqc_no_critic.pt"
    _make_tqc_checkpoint(path, with_critic=False)
    bundle = load_policy(path)
    assert bundle.algo == "tqc"
    assert bundle.value is None


# ---------------------------------------------------------------------------
# PolicyBundle.encode and .normalize
# ---------------------------------------------------------------------------


def test_policy_bundle_encode(tmp_path: Path) -> None:
    """encode() converts raw 2+2n states to 2+3n cos/sin observations."""
    path = tmp_path / "tqc.pt"
    _make_tqc_checkpoint(path)
    bundle = load_policy(path)
    # 2-link state: [x, xd, th1, thd1, th2, thd2]
    states = np.zeros((4, 6), dtype=np.float32)
    states[:, 2] = np.pi  # th1 = pi (hanging down)
    states[:, 4] = np.pi  # th2 = pi
    enc = bundle.encode(states)
    assert enc.shape == (4, _OBS_DIM)
    # cos(pi) = -1
    assert enc[0, 2] == pytest.approx(-1.0, abs=1e-5)


def test_policy_bundle_normalize(tmp_path: Path) -> None:
    """normalize() applies the running normalizer and returns a tensor."""
    path = tmp_path / "tqc.pt"
    _make_tqc_checkpoint(path)
    bundle = load_policy(path)
    obs = np.zeros((3, _OBS_DIM), dtype=np.float32)
    t = bundle.normalize(obs)
    assert t.shape == (3, _OBS_DIM)
    assert isinstance(t, torch.Tensor)


# ---------------------------------------------------------------------------
# select_action / value inference
# ---------------------------------------------------------------------------


def test_tqc_select_action(tmp_path: Path) -> None:
    """TQC select_action returns an (B, 1) numpy array."""
    path = tmp_path / "tqc.pt"
    _make_tqc_checkpoint(path)
    bundle = load_policy(path)
    obs = torch.zeros(2, _OBS_DIM)
    actions = bundle.select_action(obs)
    assert actions.shape == (2, 1)


def test_tqc_value(tmp_path: Path) -> None:
    """TQC value function returns a (B,) numpy array."""
    path = tmp_path / "tqc.pt"
    _make_tqc_checkpoint(path, with_critic=True)
    bundle = load_policy(path)
    obs = torch.zeros(2, _OBS_DIM)
    vals = bundle.value(obs)  # type: ignore[misc]
    assert vals.shape == (2,)
