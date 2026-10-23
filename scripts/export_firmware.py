"""CLI: export a TQC checkpoint as an ``.ncpm`` model blob for the ESP32-S3 firmware.

The blob (``n_cartpole.deploy.firmware_blob``) carries the actor weights (float32
or float16, ``RunningNorm`` folded in), the nominal physics, the actuator limits,
the sensor/EKF model and the observation layout. Drop it on the device dashboard
or flash it to the ``model`` partition. Also prints the network's size and a
rough ESP32-S3 inference-time estimate against the checkpoint's control period.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from loguru import logger

from n_cartpole.deploy.firmware_blob import (
    CPU_HZ,
    INTERNAL_WEIGHT_BUDGET,
    export_blob,
    inference_cost,
    read_header,
)

# Execution budget of the control loop (firmware/docs/design.md).
LOOP_BUDGET = 0.8


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export a TQC checkpoint as an .ncpm model for the ESP32-S3 "
        "firmware and estimate whether it runs at the control rate."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/double/tqc-rail05-goals-ft-weighted/tqc_2000000.pt"),
        help="TQC checkpoint (no obs history, no sysID inputs)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output file (default: dist/firmware/<label>.ncpm)",
    )
    parser.add_argument(
        "--dtype",
        choices=("f32", "f16"),
        default="f32",
        help="Weight storage in the file (the device decodes to float32)",
    )
    parser.add_argument(
        "--label", type=str, default=None, help="Display name (max 63 bytes)"
    )
    return parser.parse_args()


def safe_name(label: str) -> str:
    """File-system-safe version of a model label."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_") or "model"


def main() -> None:
    """Export the blob, write it, and report size and feasibility."""
    args = parse_args()
    blob = export_blob(args.checkpoint, dtype=args.dtype, label=args.label)
    h = read_header(blob)
    out = args.out or Path("dist/firmware") / f"{safe_name(h['label'])}.ncpm"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(blob)

    cost = inference_cost(h)
    logger.info(f"wrote {out} ({len(blob) / 1e6:.2f} MB, {args.dtype}): {h['label']}")
    logger.info(
        f"network: obs {h['obs_dim']} -> hidden {h['hidden']} x {h['n_blocks']} "
        f"blocks (expand {h['expand']}) -> 1; {cost.params:,} parameters, "
        f"{cost.macs:,} MACs per inference"
    )
    where = "internal SRAM" if cost.fits_internal else "PSRAM"
    logger.info(
        f"ESP32-S3: float32 weights {cost.weight_bytes_f32 / 1024:.0f} KB -> {where} "
        f"(~{INTERNAL_WEIGHT_BUDGET // 1024} KB usable internally next to Wi-Fi); "
        f"estimated inference {cost.est_us / 1e3:.2f} ms at {CPU_HZ / 1e6:.0f} MHz "
        f"= {100 * cost.budget_fraction:.0f}% of the {cost.dt_us / 1e3:.1f} ms period"
    )
    if cost.budget_fraction > LOOP_BUDGET:
        logger.warning(
            f"TOO SLOW FOR REAL TIME: inference alone (~{cost.est_us / 1e3:.1f} ms) "
            f"exceeds {LOOP_BUDGET:.0%} of the {cost.dt_us / 1e3:.1f} ms control "
            "period. The rig backend will refuse to arm; the sim backend runs in "
            "slow motion. Train a smaller network (e.g. --hidden 32-64) to deploy."
        )


if __name__ == "__main__":
    main()
