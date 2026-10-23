"""Int8 actor weights (n_cartpole/deploy/quantize.py) and the NumPy forward pass."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from n_cartpole.deploy import firmware_blob as fb
from n_cartpole.deploy.actor_export import actor_tensors
from n_cartpole.deploy.quantize import (
    actor_forward,
    dequantize,
    is_quantized,
    numpy_bundle,
    quantize_rows,
    quantize_tensors,
)
from n_cartpole.policy.loader import load_policy
from tests.firmware_util import make_checkpoint


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Tiny rig-config checkpoint (hidden 32, 2 blocks)."""
    return make_checkpoint(tmp_path_factory.mktemp("q") / "tqc_1.pt")


def test_quantize_rows_range_and_error() -> None:
    """Q spans [-127, 127] per row, the error is at most half a step, zero rows stay zero."""
    rng = np.random.default_rng(0)
    w = rng.normal(size=(6, 9)) * np.array([1e-3, 0.1, 1, 10, 100, 0])[:, None]
    q, scale = quantize_rows(w)
    assert q.dtype == np.int8 and scale.dtype == np.float32
    assert np.all(np.abs(q).max(axis=1)[:5] == 127)
    assert scale[5] == 0 and not q[5].any()
    err = np.abs(dequantize(q, scale) - w)
    assert np.all(err <= scale[:, None] * 0.5 * (1 + 1e-5) + 1e-12)


def test_only_block_matrices_are_quantized(ckpt: Path) -> None:
    """fc1/fc2 become (q, scale); every other tensor is float32 and unchanged."""
    t = actor_tensors(torch.load(ckpt, map_location="cpu", weights_only=False))
    for (name, a), (name_q, b) in zip(t, quantize_tensors(t), strict=True):
        assert name == name_q
        assert isinstance(b, tuple) == is_quantized(name)
        if not isinstance(b, tuple):
            np.testing.assert_array_equal(b, a.astype(np.float32))
    assert {n for n, _ in t if is_quantized(n)} == {
        "b0_fc1_w",
        "b0_fc2_w",
        "b1_fc1_w",
        "b1_fc2_w",
    }


def test_numpy_forward_matches_torch_and_int8_is_close(ckpt: Path) -> None:
    """The float32 port equals the torch actor; int8 stays within a few % of force_max."""
    bundle = load_policy(ckpt)
    t = actor_tensors(torch.load(ckpt, map_location="cpu", weights_only=False))
    rng = np.random.default_rng(1)
    obs = rng.normal(size=(300, bundle.obs_dim)).astype(np.float32)
    ref = bundle.select_action(bundle.normalize(obs))[:, 0]
    fmax = bundle.physics.force_max
    np.testing.assert_allclose(actor_forward(t, obs, fmax), ref, atol=1e-4)
    i8 = actor_forward(quantize_tensors(t), obs, fmax)
    assert np.mean(np.abs(i8 - ref)) < 0.02 * fmax


def test_numpy_bundle_is_a_drop_in(ckpt: Path) -> None:
    """select_action(normalize(raw obs)) on the NumPy bundle == the torch bundle."""
    bundle = load_policy(ckpt)
    t = actor_tensors(torch.load(ckpt, map_location="cpu", weights_only=False))
    nb = numpy_bundle(bundle, t)
    obs = np.random.default_rng(2).normal(size=(20, bundle.obs_dim)).astype(np.float32)
    np.testing.assert_allclose(
        nb.select_action(nb.normalize(obs)),
        bundle.select_action(bundle.normalize(obs)),
        atol=1e-4,
    )


def test_i8_blob_layout(ckpt: Path) -> None:
    """An i8 blob is ~1/4 the size, reads back dequantized, and matches the quantizer."""
    b32, b8 = fb.export_blob(ckpt, dtype="f32"), fb.export_blob(ckpt, dtype="i8")
    h8 = fb.read_header(b8)
    assert h8["dtype"] == fb.DTYPE_I8 and fb.DTYPE_NAMES[h8["dtype"]] == "i8"
    assert len(b8) == 512 + fb.weight_bytes(h8) and len(b8) % 4 == 0
    assert fb.weight_ram_bytes(h8) == fb.weight_bytes(h8) < 0.35 * (len(b32) - 512)
    t = actor_tensors(torch.load(ckpt, map_location="cpu", weights_only=False))
    want = np.concatenate(
        [
            (dequantize(*a) if isinstance(a, tuple) else a).ravel()
            for _, a in quantize_tensors(t)
        ]
    )
    np.testing.assert_array_equal(fb.read_weights(b8), want)
    # The f32 weights' cost estimate counts decoded floats, the i8 one the int8 image.
    assert fb.inference_cost(h8).weight_ram_bytes == fb.weight_bytes(h8)
