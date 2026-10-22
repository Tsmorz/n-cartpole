"""LQR hold study: design, catch region, and the learned-policy + LQR hybrid.

uv run python scripts/lqr_study.py design   --checkpoint <ckpt>
uv run python scripts/lqr_study.py basin    --checkpoint <ckpt> [--samples 30]
uv run python scripts/lqr_study.py hybrid   --checkpoint <ckpt> [--trials 20]
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np

from n_cartpole.control.hybrid import HoldController
from n_cartpole.control.lqr import LQRBank, LQRWeights, equilibrium
from n_cartpole.env.factory import make_env
from n_cartpole.env.goals import TransitionStats
from n_cartpole.policy.loader import PolicyBundle, load_policy
from n_cartpole.training.evaluate import TransitionMetrics, _rollout

np.set_printoptions(precision=3, suppress=True, linewidth=150)


def eval_env(bundle: PolicyBundle, seconds: float, nominal: bool = False):
    """Build the checkpoint's training env: one goal per episode, judged at the end."""
    cfg = dataclasses.replace(
        bundle.env_config,
        goal_hold_steps=None,
        max_steps=int(round(seconds / bundle.physics.dt)),
        randomize=None if nominal else bundle.env_config.randomize,
    )
    return make_env(cfg)


def inject(env, state: np.ndarray) -> np.ndarray:
    """Replace the true state after a reset and give the EKF that state (known velocities)."""
    env._state = np.asarray(state, float).copy()
    env._meas_prev = None
    env._ekf = None
    env._make_obs()  # builds the EKF from a first reading
    if env._ekf is not None:
        env._ekf.x = env._state.copy()
        env._ekf.P = 1e-6 * np.eye(env._ekf.dim)
    return env._make_obs()


def cmd_design(args: argparse.Namespace) -> None:
    """Print the gains and poles of every goal."""
    bundle = load_policy(args.checkpoint)
    bank = LQRBank(bundle.physics, bundle.n_links, _weights(args), delay=args.delay)
    for d in bank.designs:
        print(
            f"{d.label}: controllable {d.controllability_rank}/{d.ns + d.m}  "
            f"max|pole| {np.abs(d.poles).max():.4f}  K = {d.k.ravel()}"
        )


def _weights(args: argparse.Namespace) -> LQRWeights:
    """Collect the LQR weights from the CLI."""
    return LQRWeights(
        q_x=args.q_x, q_xd=args.q_xd, q_th=args.q_th, q_thd=args.q_thd, r_dcmd=args.r
    )


def sample_on_level(rng: np.random.Generator, bank: LQRBank, gi: int, level: float):
    """Draw a random state with cost-to-go ``level`` (command history zero)."""
    d = bank.designs[gi]
    ns = d.ns
    pss = d.p[:ns, :ns]
    u = rng.normal(size=ns)
    u *= np.sqrt(level / (u @ pss @ u))
    return equilibrium(d.goal) + u


def sample_scaled(
    rng: np.random.Generator, bank: LQRBank, gi: int, scale: float
) -> np.ndarray:
    """Draw a physical perturbation like the env's hold curriculum: angles +-scale, rates +-2 scale.

    The cart is perturbed too (x +-scale/2 m, xd +-scale m/s), so it does not depend on the LQR
    weights and different designs can be compared.
    """
    goal = bank.designs[gi].goal
    s = equilibrium(goal)
    s[0] += rng.uniform(-0.5 * scale, 0.5 * scale)
    s[1] += rng.uniform(-scale, scale)
    s[2::2] += rng.uniform(-scale, scale, goal.size)
    s[3::2] += rng.uniform(-2.0 * scale, 2.0 * scale, goal.size)
    return s


def hold_trial(
    env, ctrl: HoldController, gi: int, state0: np.ndarray, seed: int
) -> dict:
    """One hold episode from ``state0`` near goal ``gi``; no policy, LQR throughout."""
    env.reset(seed=seed, options={"start": gi, "goal": gi})
    obs = inject(env, state0)
    done = False
    forces: list[float] = []
    err: list[float] = []
    term = False
    while not done:
        act = ctrl.select_action(obs[None])
        obs, _, term, trunc, info = env.step(act[0])
        done = term or trunc
        forces.append(info["applied_force"])
        s = env.get_state()
        err.append(np.max(np.abs(np.angle(np.exp(1j * (s[2::2] - env.goal_angles))))))
    ok = (not term) and env._is_settled()
    half = len(forces) // 2
    return {
        "ok": ok,
        "term": term,
        "rms": float(np.sqrt(np.mean(np.square(forces[half:])))),
        "dev": float(np.max(err[half:])),
    }


def cmd_basin(args: argparse.Namespace) -> None:
    """Success vs perturbation size (cost-to-go level or physical scale) per goal."""
    bundle = load_policy(args.checkpoint)
    bank = load_bank(args, bundle)
    env = eval_env(bundle, args.seconds, nominal=args.nominal)
    ctrl = HoldController(env, bank)
    rng = np.random.default_rng(args.seed)
    physical = args.scales is not None
    levels = [float(v) for v in (args.scales or args.levels).split(",")]
    goals = [
        g
        for g in range(len(bank.designs))
        if not args.goal or bank.designs[g].label in args.goal
    ]
    for gi in goals:
        lab = bank.designs[gi].label
        print(
            f"\n{lab}   {'scale' if physical else 'level'}  success  rms_force  max_dev(rad)"
        )
        for level in levels:
            res = []
            for i in range(args.samples):
                s0 = (
                    sample_scaled(rng, bank, gi, level)
                    if physical
                    else sample_on_level(rng, bank, gi, level)
                )
                if abs(s0[0]) > 0.45:
                    s0[0] = np.sign(s0[0]) * 0.45
                res.append(hold_trial(env, ctrl, gi, s0, args.seed * 1000 + i))
            ok = np.mean([r["ok"] for r in res])
            good = [r for r in res if r["ok"]]
            rms = np.mean([r["rms"] for r in good]) if good else float("nan")
            dev = np.max([r["dev"] for r in good]) if good else float("nan")
            print(f"      {level:8.2f}  {ok:6.0%}  {rms:9.2f}  {dev:9.3f}", flush=True)


_W: dict = {}


def _tune_init(checkpoint: str, seconds: float, nominal: bool) -> None:
    bundle = load_policy(checkpoint)
    _W["bundle"] = bundle
    _W["env"] = eval_env(bundle, seconds, nominal)


def _tune_eval(job: tuple[int, dict, list[float], int, int]) -> dict:
    """Score one weight set per goal: mean success over the scales, plus mean hold force."""
    idx, w, scales, samples, seed = job
    bundle, env = _W["bundle"], _W["env"]
    weights = LQRWeights(**w)
    bank = LQRBank(bundle.physics, bundle.n_links, weights)
    if max(np.abs(d.poles).max() for d in bank.designs) >= 1.0:
        return {"idx": idx, "w": w, "score": {}, "rms": {}, "bad": True}
    ctrl = HoldController(env, bank)
    score: dict[str, float] = {}
    rms: dict[str, float] = {}
    for gi, d in enumerate(bank.designs):
        oks, forces = [], []
        for si, sc in enumerate(scales):
            rng = np.random.default_rng(
                [seed, gi, si]
            )  # same states for every weight set
            for i in range(samples):
                s0 = sample_scaled(rng, bank, gi, sc)
                r = hold_trial(env, ctrl, gi, s0, seed * 7919 + gi * 101 + si * 13 + i)
                oks.append(r["ok"])
                if r["ok"]:
                    forces.append(r["rms"])
        score[d.label] = float(np.mean(oks))
        rms[d.label] = float(np.mean(forces)) if forces else float("nan")
    return {"idx": idx, "w": w, "score": score, "rms": rms, "bad": False}


def cmd_tune(args: argparse.Namespace) -> None:
    """Random search over LQR weights, scored per goal."""
    import json
    import multiprocessing as mp

    rng = np.random.default_rng(args.seed)
    base = dataclasses.asdict(LQRWeights())
    jobs = [(0, base)]
    for i in range(1, args.configs):
        jobs.append(
            (
                i,
                {
                    "q_x": float(10 ** rng.uniform(-1, 2)),
                    "q_xd": float(10 ** rng.uniform(-1.5, 1.5)),
                    "q_th": float(10 ** rng.uniform(1, 3)),
                    "q_thd": float(10 ** rng.uniform(-1, 2)),
                    "q_cmd": float(10 ** rng.uniform(-3, 0)),
                    "r_dcmd": float(10 ** rng.uniform(-1.5, 1.5)),
                },
            )
        )
    scales = [float(v) for v in args.scales.split(",")]
    work = [(i, w, scales, args.samples, args.seed + 1) for i, w in jobs]
    with mp.Pool(
        args.workers, _tune_init, (str(args.checkpoint), args.seconds, args.nominal)
    ) as pool:
        results = pool.map(_tune_eval, work, chunksize=1)
    Path(args.out).write_text("\n".join(json.dumps(r) for r in results) + "\n")
    labels = ["UU", "DU", "UD", "DD"]
    for lab in labels:
        ranked = sorted(
            (r for r in results if not r["bad"]),
            key=lambda r: (-r["score"][lab], r["rms"][lab]),
        )
        print(f"\n{lab}: top 3 of {len(results)} (config 0 = defaults)")
        for r in ranked[:3]:
            w = {k: round(v, 3) for k, v in r["w"].items()}
            print(
                f"  #{r['idx']:3d} success {r['score'][lab]:.0%} rms {r['rms'][lab]:.2f} N  {w}"
            )
        base_r = results[0]
        print(
            f"  defaults: success {base_r['score'][lab]:.0%} rms {base_r['rms'][lab]:.2f} N"
        )


def load_bank(args: argparse.Namespace, bundle: PolicyBundle) -> LQRBank:
    """Design the bank: ``--weights-json`` (label -> weights) overrides the CLI weights."""
    import json

    base = _weights(args)
    if args.weights_json:
        spec = json.loads(Path(args.weights_json).read_text())
        weights = {
            lab: dataclasses.replace(base, **spec[lab]) if lab in spec else base
            for lab in ("UU", "DU", "UD", "DD")
        }
        return LQRBank(bundle.physics, bundle.n_links, weights, delay=args.delay)
    return LQRBank(bundle.physics, bundle.n_links, base, delay=args.delay)


def calibrate(
    env, bank: LQRBank, samples: int, seed: int, need: float = 0.95
) -> list[float]:
    """Largest cost-to-go level the LQR catches ``need`` of the time, per goal.

    Levels double from 10; the result is the last level before success first drops below
    ``need`` (the basin is not monotone in general, so this is conservative).
    """
    ctrl = HoldController(env, bank)
    out = []
    for gi, d in enumerate(bank.designs):
        rng = np.random.default_rng([seed, gi])
        best, level = 0.0, 10.0
        while level < 1e6:
            ok = 0
            for i in range(samples):
                s0 = sample_on_level(rng, bank, gi, level)
                s0[0] = float(np.clip(s0[0], -0.45, 0.45))
                ok += hold_trial(env, ctrl, gi, s0, seed * 1000 + i)["ok"]
            if ok / samples < need:
                break
            best, level = level, level * 2.0
        out.append(best)
        print(f"  {d.label}: catch level {best:.0f}", flush=True)
    return out


def cmd_hybrid(args: argparse.Namespace) -> None:
    """Compare the policy alone with the policy plus LQR hold on every transition."""
    bundle = load_policy(args.checkpoint)
    bank = load_bank(args, bundle)
    env = eval_env(bundle, args.seconds, nominal=args.nominal)
    labels = [d.label for d in bank.designs]
    lqr_goals = set(args.lqr_goals)
    if args.engage:
        levels = dict(kv.split("=") for kv in args.engage.split(","))
        catch = [float(levels.get(lab, 0.0)) for lab in labels]
    else:
        print("calibrating the catch level per goal...")
        catch = calibrate(env, bank, args.calib_samples, args.seed)
    engage = [
        c * args.engage_frac if lab in lqr_goals else -1.0
        for c, lab in zip(catch, labels, strict=True)
    ]
    release = [
        c * args.release_mult if lab in lqr_goals else -1.0
        for c, lab in zip(catch, labels, strict=True)
    ]
    print(
        "engage below",
        dict(zip(labels, np.round(engage, 1), strict=False)),
        "release above",
        dict(zip(labels, np.round(release, 1), strict=False)),
    )
    n_goals = len(labels)
    for name, make in (
        ("policy only", lambda: bundle),
        (
            "policy + LQR hold",
            lambda: HoldController(env, bank, bundle, engage, release),
        ),
    ):
        if args.only and args.only not in name:
            continue
        stats, metrics = TransitionStats(n_goals), TransitionMetrics(n_goals)
        share = np.zeros((n_goals, n_goals))
        for start in range(n_goals):
            for goal in range(n_goals):
                shares = []
                for trial in range(args.trials):
                    agent = make()
                    result, seg = _rollout(agent, env, start, goal, args.seed + trial)  # type: ignore[arg-type]  # duck-typed bundle
                    metrics.record(start, goal, result)
                    if seg is not None:
                        stats.record(*seg)
                    if isinstance(agent, HoldController):
                        shares.append(agent.lqr_steps / max(agent.steps, 1))
                share[start, goal] = np.mean(shares) if shares else 0.0
        print(f"\n===== {name} ({args.trials} trials per pair) =====")
        print(stats.format_matrix(labels))
        print(metrics.format(labels))
        if name != "policy only":
            print("share of steps under LQR")
            print("from\\to  " + "".join(f"{lab:>7}" for lab in labels))
            for i, lab in enumerate(labels):
                print(
                    lab.ljust(8)
                    + "".join(f"{share[i, j]:7.0%}" for j in range(n_goals))
                )


def cmd_calibrate(args: argparse.Namespace) -> None:
    """Print the catch level per goal."""
    bundle = load_policy(args.checkpoint)
    bank = load_bank(args, bundle)
    env = eval_env(bundle, args.seconds, nominal=args.nominal)
    catch = calibrate(env, bank, args.calib_samples, args.seed)
    print(
        "--engage",
        ",".join(
            f"{d.label}={c:.0f}" for d, c in zip(bank.designs, catch, strict=True)
        ),
    )


def main() -> None:
    """Parse the command line and run the chosen study."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument("cmd", choices=["design", "basin", "tune", "calibrate", "hybrid"])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--delay", type=int, default=1)
    ap.add_argument("--q-x", type=float, default=10.0)
    ap.add_argument("--q-xd", type=float, default=1.0)
    ap.add_argument("--q-th", type=float, default=100.0)
    ap.add_argument("--q-thd", type=float, default=1.0)
    ap.add_argument("--r", type=float, default=1.0)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--levels", default="1,3,10,30,100,300,1000")
    ap.add_argument(
        "--scales",
        default=None,
        help="physical perturbation scales (rad), instead of --levels",
    )
    ap.add_argument("--configs", type=int, default=40, help="tune: weight sets to try")
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--out", default="lqr_tune.jsonl")
    ap.add_argument(
        "--weights-json", default=None, help="per-goal LQR weights {label: {q_th: ..}}"
    )
    ap.add_argument(
        "--engage",
        default=None,
        help="catch levels 'UU=300,DU=1000,...' (else calibrate)",
    )
    ap.add_argument(
        "--engage-frac",
        type=float,
        default=0.5,
        help="engage at this fraction of the catch level",
    )
    ap.add_argument(
        "--release-mult",
        type=float,
        default=3.0,
        help="release at this multiple of the catch level",
    )
    ap.add_argument("--lqr-goals", nargs="*", default=["UU", "DU", "UD"])
    ap.add_argument("--calib-samples", type=int, default=20)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument(
        "--only", default=None, help="hybrid: only the run whose name contains this"
    )
    ap.add_argument("--goal", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--nominal", action="store_true", help="exact nominal plant (no randomization)"
    )
    args = ap.parse_args()
    {
        "design": cmd_design,
        "basin": cmd_basin,
        "tune": cmd_tune,
        "calibrate": cmd_calibrate,
        "hybrid": cmd_hybrid,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
