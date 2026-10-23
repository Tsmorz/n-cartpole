"""Actor weights in a flat, normalizer-free form for non-PyTorch runtimes.

Shared by the browser export (``scripts/export_web.py``) and the ESP32 firmware
blob (:mod:`n_cartpole.deploy.firmware_blob`): both run the SimBa actor's
deterministic head, ``F = tanh(mean) * force_max``, so the ``RunningNorm`` is
folded into the input projection and only the mean row of the output head is
kept.
"""

from __future__ import annotations

import numpy as np


def actor_tensors(ckpt: dict) -> list[tuple[str, np.ndarray]]:
    """Return the actor's tensors (float64) in forward order, normalizer folded in.

    Order: ``in_w, in_b``, then per block ``ln_w, ln_b, fc1_w, fc1_b, fc2_w,
    fc2_b``, then ``final_ln_w, final_ln_b, out_w (1 x H), out_b (1,)``.
    """
    sd = {k: v.double().numpy() for k, v in ckpt["actor"].items()}
    mean = ckpt["norm"]["mean"].double().numpy()
    std = np.sqrt(ckpt["norm"]["var"].double().numpy()) + 1e-8

    # W (x - mu) / sigma + b  ==  (W / sigma) x + (b - (W / sigma) mu)
    w_in = sd["net.in_proj.weight"] / std
    b_in = sd["net.in_proj.bias"] - w_in @ mean
    out: list[tuple[str, np.ndarray]] = [("in_w", w_in), ("in_b", b_in)]
    for i in range(n_blocks(ckpt)):
        p = f"net.blocks.{i}."
        out += [
            (f"b{i}_ln_w", sd[p + "norm.weight"]),
            (f"b{i}_ln_b", sd[p + "norm.bias"]),
            (f"b{i}_fc1_w", sd[p + "fc1.weight"]),
            (f"b{i}_fc1_b", sd[p + "fc1.bias"]),
            (f"b{i}_fc2_w", sd[p + "fc2.weight"]),
            (f"b{i}_fc2_b", sd[p + "fc2.bias"]),
        ]
    out += [
        ("final_ln_w", sd["net.final_norm.weight"]),
        ("final_ln_b", sd["net.final_norm.bias"]),
        # Row 0 is the Gaussian mean; row 1 (log-std) is unused when deterministic.
        ("out_w", sd["net.out_proj.weight"][:1]),
        ("out_b", sd["net.out_proj.bias"][:1]),
    ]
    return out


def n_blocks(ckpt: dict) -> int:
    """Return the number of residual blocks in the checkpoint's actor."""
    return len({k.split(".")[2] for k in ckpt["actor"] if k.startswith("net.blocks.")})
