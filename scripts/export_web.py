"""CLI: export a TQC actor for in-browser inference (the personal site).

Takes a goal-conditioned checkpoint (``--goals``; the default) or a plain swing-up
one. For a goal net the JSON also lists every target configuration, and the net's
extra inputs (``cos`` of each link's target angle) follow the kinematic ones.

Writes three files to ``--out-dir``:

- ``<name>.json``: physics, actuator limits, success criterion, network shape, and
  a tensor table. The web page reads its physics from here, so the browser sim
  always matches what the policy was trained on.
- ``<name>.bin``: the actor weights as little-endian float16, laid out as the
  table says. The ``RunningNorm`` is folded into the input projection and only
  the mean row of the output head is kept (deterministic action = tanh(mean) *
  force_max), so the browser needs no normalizer and no log-std.
- ``<name>-fixture.json``: parity data for the site's Node test: observation →
  torch action pairs, closed-loop starts with their Python times (near-hanging
  swing-ups, or for a goal net one transition per ``(start, goal)`` pair), and
  one open-loop trajectory from :func:`n_cartpole.env.dynamics.step`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from n_cartpole.env.cartpole import encode_obs
from n_cartpole.env.dynamics import step
from n_cartpole.env.factory import make_env
from n_cartpole.env.goals import goal_configs, goal_labels
from n_cartpole.policy.loader import load_policy

FORMAT_VERSION = 2  # 2: optional "goals" block + goal inputs


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export a TQC actor (goal-conditioned or swing-up) + physics "
        "for the browser."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/double/tqc-goal/tqc_latest.pt"),
        help="Goal-conditioned or plain swing-up checkpoint without sysID inputs",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("../personal-site/assets/models"),
        help="Directory to write <name>.json, <name>.bin, <name>-fixture.json",
    )
    parser.add_argument("--name", type=str, default="swingup-tqc", help="File stem")
    parser.add_argument(
        "--trials", type=int, default=6, help="Closed-loop fixture starts"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    return parser.parse_args()


def _actor_tensors(ckpt: dict) -> list[tuple[str, np.ndarray]]:
    """Return the actor's tensors (float64) with the normalizer folded in."""
    sd = {k: v.double().numpy() for k, v in ckpt["actor"].items()}
    mean = ckpt["norm"]["mean"].double().numpy()
    std = np.sqrt(ckpt["norm"]["var"].double().numpy()) + 1e-8

    # W (x - mu) / sigma + b  ==  (W / sigma) x + (b - (W / sigma) mu)
    w_in = sd["net.in_proj.weight"] / std
    b_in = sd["net.in_proj.bias"] - w_in @ mean
    out: list[tuple[str, np.ndarray]] = [("in_w", w_in), ("in_b", b_in)]
    n_blocks = len({k.split(".")[2] for k in sd if k.startswith("net.blocks.")})
    for i in range(n_blocks):
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


def main() -> None:
    """Export weights, metadata, and parity fixtures."""
    args = parse_args()
    bundle = load_policy(args.checkpoint)
    cfg = bundle.env_config
    if cfg.hardware is not None or cfg.randomize is not None:
        raise ValueError(
            "export_web supports simulation checkpoints only "
            "(no sysID/hardware inputs, no plant randomization)"
        )
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    p = bundle.physics
    n = bundle.n_links
    goals = bundle.goal_conditioned
    if p.com is not None or np.any(p.link_inertias(n) != 0.0):
        raise ValueError("the browser port models point-mass links only")

    tensors = _actor_tensors(ckpt)
    table, blobs, offset = [], [], 0
    for name, arr in tensors:
        half = np.ascontiguousarray(arr, dtype="<f2")
        table.append({"name": name, "shape": list(arr.shape), "offset": offset})
        blobs.append(half.tobytes())
        offset += half.size

    meta = {
        "format": FORMAT_VERSION,
        "source": {"checkpoint": str(args.checkpoint), "step": int(ckpt["step"])},
        "n_links": n,
        "physics": {
            "M": p.M,
            "g": p.g,
            "dt": p.dt,
            "force_max": p.force_max,
            "x_lim": p.x_lim,
            "b": p.b,
            "masses": p.link_masses(n).tolist(),
            "lengths": p.link_lengths(n).tolist(),
            "joint_friction": p.joint_frictions(n).tolist(),
        },
        "force_slew": cfg.force_slew,
        "integrator": {"method": "rk4", "substeps": 4},
        "success": {
            "tol_angle": cfg.goal_tol_angle,
            "tol_vel": cfg.goal_tol_vel,
            "settle_steps": cfg.goal_settle_steps,
        },
        "network": {
            "obs": "x, xd, then cos(th), sin(th), thd per link (th = 0 upright)"
            + (", then cos(target th) per link" if goals else ""),
            "obs_dim": bundle.obs_dim,
            "hidden": bundle.hidden,
            "expand": 4,
            "blocks": sum(1 for name, _ in tensors if name.endswith("_ln_w")) - 1,
            "ln_eps": 1e-5,
            "dtype": "float16",
            "tensors": table,
        },
    }

    if goals:
        # Labels are one U/D per link, base link first; targets are 0 (up) / pi (down).
        meta["goals"] = {
            "labels": goal_labels(n),
            "targets": goal_configs(n).tolist(),
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = args.out_dir / f"{args.name}.bin"
    bin_path.write_bytes(b"".join(blobs))
    (args.out_dir / f"{args.name}.json").write_text(json.dumps(meta, indent=2) + "\n")
    logger.info(f"wrote {bin_path} ({bin_path.stat().st_size / 1e6:.2f} MB)")

    # --- Parity fixtures -----------------------------------------------------
    rng = np.random.default_rng(args.seed)

    n_goals = len(goal_configs(n))

    def obs_of(raw: np.ndarray, goal: int | None) -> np.ndarray:
        return bundle.encode(np.atleast_2d(raw), goal=goal)[0]

    # 1) Network parity over states spread across the whole operating range
    #    (and, for a goal net, every goal).
    states = np.empty((64, 2 + 2 * n))
    states[:, 0] = rng.uniform(-p.x_lim, p.x_lim, 64)
    states[:, 1] = rng.uniform(-2, 2, 64)
    states[:, 2::2] = rng.uniform(-np.pi, np.pi, (64, n))
    states[:, 3::2] = rng.uniform(-8, 8, (64, n))
    net_goals = [i % n_goals if goals else None for i in range(64)]
    obs_batch = np.stack([obs_of(s, g) for s, g in zip(states, net_goals, strict=True)])
    actions = bundle.select_action(bundle.normalize(obs_batch))[:, 0]
    net_cases = [
        {"obs": o.tolist(), "action": float(a)}
        for o, a in zip(obs_batch, actions, strict=True)
    ]
    assert np.allclose(
        obs_batch[:, : 2 + 3 * n], np.stack([encode_obs(s) for s in states])
    )

    # 2) Physics parity: open-loop RK45 trajectory under a known force sequence.
    s = np.zeros(2 + 2 * n)
    s[2::2] = np.pi - 0.3
    s[3::2] = 0.5
    forces = [float(12.0 * np.sin(2 * np.pi * 1.5 * k * p.dt)) for k in range(100)]
    traj = [s.tolist()]
    for f in forces:
        s = step(s, f, p)
        traj.append(s.tolist())

    # 3) Closed loop. A plain net: near-hanging swing-ups, sampled like
    #    scripts/eval_swingup.py. A goal net: one episode per (start, goal) pair
    #    with start != goal, from a perturbed equilibrium like evaluate.py.
    seconds = 20.0 if not goals else 10.0
    env = make_env(
        dataclasses.replace(
            cfg,
            max_steps=int(round(seconds / p.dt)),
            goal_hold_steps=None if goals else cfg.goal_hold_steps,
        )
    )
    pairs: list[tuple[int, int] | None]
    if goals:
        pairs = [(a, b) for a in range(n_goals) for b in range(n_goals) if a != b]
    else:
        pairs = [None] * args.trials
    labels = goal_labels(n)
    starts = []
    for pair in pairs:
        seed = int(rng.integers(2**31))
        if pair is not None:
            frm, to = pair
            env.reset(seed=seed, options={"start": frm, "goal": to})
            st = env.get_state()
        else:
            env.reset(seed=seed)
            st = env.get_state()
            st[0] = 0.0
            st[1] = rng.uniform(-0.2, 0.2)
            st[2::2] = np.pi + rng.uniform(-0.2, 0.2, n)
            st[3::2] = rng.uniform(-0.2, 0.2, n)
            env._state = st.copy()
            env._prev_potential = env._angle_potential(st)
        obs = env._make_obs()
        up_time, settled, k = None, 0, 0
        terminated = truncated = False
        while not (terminated or truncated):
            a = bundle.select_action(bundle.normalize(np.asarray(obs)[None]))
            obs, _, terminated, truncated, _ = env.step(a[0])
            k += 1
            if up_time is None:
                settled = settled + 1 if env._at_goal(env._state) else 0
                if settled >= cfg.goal_settle_steps:
                    up_time = (k - cfg.goal_settle_steps + 1) * p.dt
        entry: dict = {"state": st.tolist(), "python_swingup_s": up_time}
        if pair is not None:
            entry["from"], entry["goal"] = labels[pair[0]], labels[pair[1]]
        starts.append(entry)
        tag = f"{entry['from']} -> {entry['goal']}" if pair else "swing-up"
        logger.info(f"fixture start ({tag}): python {up_time}")

    fixture = {
        "net": net_cases,
        "open_loop": {"forces": forces, "states": traj},
        "swingup": {"seconds": seconds, "starts": starts},
    }
    fx_path = args.out_dir / f"{args.name}-fixture.json"
    fx_path.write_text(json.dumps(fixture) + "\n")
    logger.info(f"wrote {fx_path}")


if __name__ == "__main__":
    main()
