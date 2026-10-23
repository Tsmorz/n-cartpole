#!/usr/bin/env python3
"""Build the browser-install page (esp-web-tools, Web Serial) for a firmware release.

Writes a static site: `index.html`, `manifest.json` and the factory image in two pieces
around the nvs partition (as `flash.py` does), so installing from the browser keeps the
saved Wi-Fi network and the rig's zero references. The images must be served from the
same origin as the page (release assets send no CORS headers), so the firmware workflow
deploys this directory to GitHub Pages on `fw-v*` tags.

    python firmware/tools/web_install.py <factory.bin> <version> <out-dir>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "ncp_flash", Path(__file__).resolve().parent / "flash.py"
)
assert _spec is not None and _spec.loader is not None
flash = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(flash)

ESP_WEB_TOOLS = "https://unpkg.com/esp-web-tools@10/dist/web/install-button.js?module"

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>n-cartpole firmware install</title>
<style>
  body {{ font: 16px/1.5 system-ui, sans-serif; max-width: 38rem; margin: 2rem auto; padding: 0 1rem; }}
  h1 {{ font-size: 1.4rem; }}
  code {{ background: #8882; padding: 0 .25em; border-radius: 3px; }}
</style>
<script type="module" src="{script}"></script>
</head>
<body>
<h1>n-cartpole firmware {version}</h1>
<p>Flash the ESP32-S3 from this page: plug the board in over USB, click install and pick its serial port.
Needs Chrome or Edge on a desktop (Web Serial).</p>
<esp-web-install-button manifest="manifest.json">
  <span slot="unsupported">This browser has no Web Serial. Use Chrome or Edge, or <code>task fw-flash</code>.</span>
  <span slot="not-allowed">Web Serial needs HTTPS.</span>
</esp-web-install-button>
<ul>
  <li>The saved Wi-Fi network and the rig's zero references are kept. Do not tick any erase option:
      it also wipes the model.</li>
  <li>The model is separate: after installing, drag the <code>.ncpm</code> onto the dashboard.</li>
  <li>If the board does not enter download mode by itself, hold <b>BOOT</b> while pressing <b>RESET</b>.</li>
  <li>A board that is already set up updates over Wi-Fi by dragging the app image onto its dashboard.</li>
</ul>
</body>
</html>
"""


def build(image: Path, version: str, out: Path) -> dict:
    """Write the site into `out` and return the manifest."""
    start, end = flash.nvs_range()
    pieces = flash.split_around(image.read_bytes(), start, end)
    out.mkdir(parents=True, exist_ok=True)
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
    (out / "index.html").write_text(PAGE.format(version=version, script=ESP_WEB_TOOLS))
    return manifest


def main() -> None:
    """Parse the arguments and build the site."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", type=Path, help="merged factory image")
    ap.add_argument("version")
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    manifest = build(args.image, args.version, args.out)
    for part in manifest["builds"][0]["parts"]:
        print(f"{args.out / part['path']}  @ 0x{part['offset']:x}")


if __name__ == "__main__":
    main()
