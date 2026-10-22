"""CLI: fit the plant parameters to device recordings and write a config TOML.

    task sysid -- recordings/*.ncpr --config config/rig.toml --out config/rig-fitted.toml

Reads ``.ncpr`` recordings (``ncp_record.h``; download from the dashboard), fits the
nominal plant of ``--config`` to them (``n_cartpole.sysid.fit``), prints the
parameter table, writes the fitted config for ``train.py --config`` / ``--resume
--config`` and, with ``--report``, an interactive HTML report.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from n_cartpole.config import load_tqc_config
from n_cartpole.sysid.config_out import write_fitted_config
from n_cartpole.sysid.fit import DEFAULT_FIT, FIT_GROUPS, FitOptions, fit_recordings
from n_cartpole.sysid.recording import Recording
from n_cartpole.sysid.report import write_report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fit the cartpole plant to .ncpr recordings (multiple shooting) "
        "and write a fitted config TOML."
    )
    parser.add_argument("recordings", type=Path, nargs="+", help=".ncpr files")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/rig.toml"),
        help="Nominal config: start values, link masses/lengths, actuator limits",
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Fitted config TOML to write"
    )
    parser.add_argument(
        "--fit",
        default=",".join(DEFAULT_FIT),
        help=f"Comma-separated parameters to fit, from {','.join(FIT_GROUPS)} "
        "(M, force_gain and links cannot all be fitted: they are only identified "
        "up to a common scale)",
    )
    parser.add_argument(
        "--delay-max", type=int, default=3, help="Largest actuator delay tried (steps)"
    )
    parser.add_argument(
        "--window", type=float, default=0.25, help="Shooting window length (s)"
    )
    parser.add_argument(
        "--trim",
        type=float,
        nargs=2,
        metavar=("START", "END"),
        default=None,
        help="Use only START <= t < END seconds of every recording (e.g. to drop "
        "the moment a hand releases the links)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=3.0,
        help="[randomize] ranges = this many standard errors of the fit",
    )
    parser.add_argument(
        "--loss",
        choices=("soft_l1", "linear", "huber", "cauchy"),
        default="soft_l1",
        help="least_squares loss (robust by default)",
    )
    parser.add_argument("--report", type=Path, default=None, help="HTML report")
    return parser.parse_args()


def main() -> None:
    """Load recordings, fit, write the TOML and the report."""
    args = parse_args()
    cfg = load_tqc_config(args.config).env
    hw = cfg.hardware
    recs = [Recording.read(p) for p in args.recordings]
    if args.trim is not None:
        recs = [r.trimmed(args.trim[0], args.trim[1]) for r in recs]
    for path, r in zip(args.recordings, recs, strict=True):
        logger.info(
            f"{path}: {len(r)} samples ({len(r) * r.dt:.1f} s), excitation "
            f"{r.excite_kind}, {int(r.armed.sum())} armed, note {r.note!r}"
        )
    opts = FitOptions(
        fit=tuple(s.strip() for s in args.fit.split(",") if s.strip()),
        delay_max=args.delay_max,
        window=args.window,
        loss=args.loss,
    )
    result = fit_recordings(
        recs,
        cfg.physics,
        cfg.n_links,
        cfg.force_slew,
        nominal_delay=hw.delay_steps if hw is not None else 0,
        opts=opts,
    )
    logger.info("fit:\n" + result.summary())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_fitted_config(args.config, result, args.out, sigma=args.sigma)
    logger.info(f"wrote {args.out}")
    if args.report is not None:
        write_report(result, args.report, [str(p) for p in args.recordings])
        logger.info(f"wrote {args.report}")


if __name__ == "__main__":
    main()
