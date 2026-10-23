"""Int8 weights for the firmware actor, and a NumPy forward pass that matches the C core.

Only the residual blocks' ``fc1_w``/``fc2_w`` matrices are quantized: they hold ~96%
of a SimBa actor's parameters (36,864 of 38,305 for the h48x2 student). Each output
row gets its own symmetric scale, ``w ≈ q * scale`` with ``q`` in [-127, 127] and
``scale = max|row| / 127``. Activations stay float32 ("weight-only int8"), so the
dot product is ``scale[o] * Σ q[o, i] x[i]``. Everything else stays float32: the
input projection carries the folded ``RunningNorm`` (per-input ``1/std`` makes its
columns span orders of magnitude, which one row scale cannot cover) and is tiny, and
the biases, LayerNorm parameters and output head are vectors.

:func:`actor_forward` runs exactly the function ``ncp_policy_act`` computes, for
either weight set, so the quantized network can be evaluated in the Python
simulation (``scripts/eval_quantized.py``) before it goes near the device.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np
import torch

from n_cartpole.policy.loader import PolicyBundle

QUANT_SUFFIXES = ("_fc1_w", "_fc2_w")
# A float32 tensor, or an int8 matrix as (q [out, in], scale [out]).
Weight = np.ndarray | tuple[np.ndarray, np.ndarray]
LN_EPS = 1e-5


def is_quantized(name: str) -> bool:
    """Whether tensor ``name`` (an :func:`actor_tensors` name) is stored as int8."""
    return name.endswith(QUANT_SUFFIXES)


def quantize_rows(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-row int8: return ``(q int8 [out, in], scale float32 [out])``.

    An all-zero row gets scale 0 (and q 0), so it dequantizes to zero exactly.
    """
    w = np.asarray(w, dtype=np.float64)
    amax = np.max(np.abs(w), axis=1)
    scale = (amax / 127.0).astype(np.float32)
    safe = np.where(scale > 0, scale, 1.0).astype(np.float64)
    q = np.clip(np.rint(w / safe[:, None]), -127, 127).astype(np.int8)
    return q, scale


def quantize_tensors(
    tensors: Sequence[tuple[str, np.ndarray]],
) -> list[tuple[str, Weight]]:
    """Replace each fc1/fc2 matrix by ``(q, scale)``; other tensors become float32."""
    out: list[tuple[str, Weight]] = []
    for name, arr in tensors:
        if is_quantized(name):
            out.append((name, quantize_rows(arr)))
        else:
            out.append((name, np.asarray(arr, dtype=np.float32)))
    return out


def dequantize(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """``q * scale`` per row, float32."""
    return q.astype(np.float32) * scale.astype(np.float32)[:, None]


def _linear(w: Weight, x: np.ndarray) -> np.ndarray:
    """``x @ W.T`` for a float32 matrix or an int8 ``(q, scale)`` pair (batched x)."""
    if isinstance(w, tuple):
        q, scale = w
        return (x @ q.astype(np.float32).T) * scale[None, :]
    return x @ w.T


def _layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=1, keepdims=True)
    return (x - mean) / np.sqrt(var + LN_EPS) * w + b


def actor_forward(
    tensors: Sequence[tuple[str, Weight]],
    obs: np.ndarray,
    force_max: float,
) -> np.ndarray:
    """Force (N) of the deterministic actor for raw observations ``(B, obs_dim)``.

    ``tensors`` is :func:`~n_cartpole.deploy.actor_export.actor_tensors` (the
    normalizer folded in, so ``obs`` is NOT normalized) or its
    :func:`quantize_tensors` form. float32 throughout, like the firmware.
    """
    t = dict(tensors)

    def v(name: str) -> np.ndarray:
        a = t[name]
        assert not isinstance(a, tuple), f"{name} is a vector, never quantized"
        return a

    x = np.atleast_2d(np.asarray(obs, dtype=np.float32))
    h = _linear(t["in_w"], x) + v("in_b")
    i = 0
    while f"b{i}_fc1_w" in t:
        u = _layer_norm(h, v(f"b{i}_ln_w"), v(f"b{i}_ln_b"))
        u = np.maximum(_linear(t[f"b{i}_fc1_w"], u) + v(f"b{i}_fc1_b"), 0.0)
        h = h + _linear(t[f"b{i}_fc2_w"], u) + v(f"b{i}_fc2_b")
        i += 1
    h = _layer_norm(h, v("final_ln_w"), v("final_ln_b"))
    mean = _linear(t["out_w"], h)[:, 0] + v("out_b")[0]
    return (np.tanh(mean) * np.float32(force_max)).astype(np.float32)


class _Identity:
    """Stand-in normalizer: the folded tensors take raw observations."""

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x


def numpy_bundle(
    bundle: PolicyBundle,
    tensors: Sequence[tuple[str, Weight]],
) -> PolicyBundle:
    """Return a copy of ``bundle`` whose actor is :func:`actor_forward` over ``tensors``.

    Drop-in for the evaluation code (``evaluate_transition_metrics``), which calls
    ``select_action(normalize(obs))``.
    """
    f32: list[tuple[str, Weight]] = [
        (n, a if isinstance(a, tuple) else np.asarray(a, np.float32))
        for n, a in tensors
    ]
    force_max = float(bundle.physics.force_max)

    def select_action(obs: torch.Tensor) -> np.ndarray:
        return actor_forward(f32, obs.cpu().numpy(), force_max)[:, None]

    return dataclasses.replace(
        bundle,
        norm=_Identity(),  # type: ignore[arg-type]
        select_action=select_action,
        value=None,
    )
