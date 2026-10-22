#!/usr/bin/env python3
"""Flash a merged factory image over USB without touching the NVS partition.

`esptool merge_bin` pads the gaps between partitions with 0xFF, so writing the merged image
at 0x0 also overwrites NVS: the saved Wi-Fi network, the rig's zero references. This writes the
image in two pieces around the nvs partition (read from partitions.csv) so they survive;
`--erase-settings` writes the whole image instead (a factory reset of the settings).
The model partition lies beyond the image and is never touched either.

    task fw-flash -- <port> [image] [--erase-settings]
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

FIRMWARE = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE = FIRMWARE / "build" / "factory.bin"
PARTITIONS = FIRMWARE / "partitions.csv"


def _size(text: str) -> int:
    text = text.strip()
    scale = {"K": 1024, "M": 1024 * 1024}.get(text[-1:].upper(), 1)
    return int(text.rstrip("kKmM"), 0) * scale


def nvs_range(partitions: Path = PARTITIONS) -> tuple[int, int]:
    """Return (offset, end) of the nvs partition."""
    lines = [
        ln
        for ln in partitions.read_text().splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    for row in csv.reader(lines):
        cells = [c.strip() for c in row]
        if cells[0] == "nvs":
            offset = int(cells[3], 0)
            return offset, offset + _size(cells[4])
    raise SystemExit(f"no nvs partition in {partitions}")


def split_around(image: bytes, start: int, end: int) -> list[tuple[int, bytes]]:
    """Split the image into the bytes below `start` and from `end` on, as (flash offset, data)."""
    pieces = [(0, image[:start])]
    if len(image) > end:
        pieces.append((end, image[end:]))
    return [(off, data) for off, data in pieces if data]


def esptool_cmd() -> list[str]:
    """Find esptool: on PATH (ESP-IDF's or pip's), else through uvx."""
    for name in ("esptool.py", "esptool"):
        if shutil.which(name):
            return [name]
    if shutil.which("uvx"):
        return ["uvx", "esptool"]
    raise SystemExit(
        "esptool not found: install ESP-IDF (export.sh), `pip install esptool` or uv"
    )


def main() -> None:
    """Parse the arguments and flash."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument("port", help="serial port, e.g. /dev/cu.usbmodem11301")
    ap.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=DEFAULT_IMAGE,
        help="merged factory image",
    )
    ap.add_argument("--baud", type=int, default=460800)
    ap.add_argument(
        "--erase-settings",
        action="store_true",
        help="write the whole image, including NVS (forgets Wi-Fi)",
    )
    args = ap.parse_args()

    image = args.image.read_bytes()
    start, end = nvs_range()
    with tempfile.TemporaryDirectory() as tmp:
        if args.erase_settings:
            pieces = [(0, image)]
            print(
                "writing the whole image: the saved Wi-Fi network and zero references are erased"
            )
        else:
            pieces = split_around(image, start, end)
            print(
                f"keeping NVS (0x{start:x}-0x{end:x}): the saved Wi-Fi network survives"
            )
        files: list[str] = []
        for off, data in pieces:
            path = Path(tmp) / f"piece_{off:x}.bin"
            path.write_bytes(data)
            files += [hex(off), str(path)]
        cmd = [
            *esptool_cmd(),
            "--chip",
            "esp32s3",
            "-p",
            args.port,
            "-b",
            str(args.baud),
            "write_flash",
            *files,
        ]
        sys.exit(subprocess.run(cmd, check=False).returncode)


if __name__ == "__main__":
    main()
