#!/usr/bin/env python3
"""Write the browser installer's files (esp-web-tools, Web Serial) for a firmware release.

Writes `manifest.json` and the factory image in two pieces around the nvs partition (as
`flash.py` does), so installing from the browser keeps the saved Wi-Fi network and the rig's
zero references. The default output is the personal site's `swingup/real/`, next to the page itself
(`index.md`, served at `/swingup/real/`), which reads that manifest from its own
origin (release downloads send no CORS headers, so the files must live with the page).

    task fw-installer -- fw-v0.1.0                      # download the release image
    python firmware/tools/web_install.py path/to/factory.bin 0.1.0
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import urllib.request
from pathlib import Path

REPO = "Tsmorz/n-cartpole"
DEFAULT_OUT = Path(__file__).resolve().parents[3] / "personal-site" / "swingup" / "real"

_spec = importlib.util.spec_from_file_location(
    "ncp_flash", Path(__file__).resolve().parent / "flash.py"
)
assert _spec is not None and _spec.loader is not None
flash = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(flash)


def build(image: Path, version: str, out: Path) -> dict:
    """Write the manifest and image pieces into `out` (replacing older pieces); return the manifest."""
    start, end = flash.nvs_range()
    pieces = flash.split_around(image.read_bytes(), start, end)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("n-cartpole-fw-*.bin"):
        old.unlink()
    parts = []
    for off, data in pieces:
        name = f"n-cartpole-fw-{version}-{off:x}.bin"
        (out / name).write_bytes(data)
        parts.append({"path": name, "offset": off})
    manifest = {
        "name": "n-cartpole",
        "version": version,
        "new_install_prompt_erase": False,
        "builds": [{"chipFamily": "ESP32-S3", "parts": parts}],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def download(tag: str, dest: Path) -> tuple[Path, str]:
    """Fetch the factory image of release `tag` (e.g. fw-v0.1.0); return (file, version)."""
    version = tag.removeprefix("fw-v")
    url = f"https://github.com/{REPO}/releases/download/{tag}/n-cartpole-fw-{version}-factory.bin"
    path = dest / f"n-cartpole-fw-{version}-factory.bin"
    with urllib.request.urlopen(url) as resp:  # noqa: S310 (fixed https URL)
        path.write_bytes(resp.read())
    return path, version


def main() -> None:
    """Parse the arguments and write the files."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", help="a release tag (fw-v0.1.0) or a local factory.bin")
    ap.add_argument(
        "version", nargs="?", help="version label (required for a local file)"
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    if args.source.startswith("fw-v"):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            image, version = download(args.source, Path(tmp))
            manifest = build(image, version, args.out)
    else:
        if not args.version:
            ap.error("a local image needs a version")
        manifest = build(Path(args.source), args.version, args.out)
    for part in manifest["builds"][0]["parts"]:
        print(f"{args.out / part['path']}  @ 0x{part['offset']:x}")


if __name__ == "__main__":
    main()
