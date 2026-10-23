#!/usr/bin/env python3
"""Write the browser installer's files (esp-web-tools, Web Serial) for a firmware release.

Writes `manifest.json` and the factory image in two pieces around the nvs partition (as
`flash.py` does), so installing from the browser keeps the saved Wi-Fi network and the rig's
zero references. The default output is the personal site's `controls/swingup/real/`, next to the page itself
(`index.md`, served at `/controls/swingup/real/`), which reads that manifest from its own
origin (release downloads send no CORS headers, so the files must live with the page).

    task fw-installer -- fw-v0.1.0                      # download the release image
    python firmware/tools/web_install.py path/to/factory.bin 0.1.0
    task fw-installer -- fw-v0.1.0 --model dist/firmware/<name>-i8.ncpm   # also publish the model
    task fw-installer -- --model dist/firmware/<name>-i8.ncpm             # only swap the model

The model is not part of a release (the firmware and the model are separate files), so `--model` copies an
exported `.ncpm` next to the page as `MODEL_NAME`, plus `MODEL_MANIFEST`, which makes it a second
esp-web-tools install: the page's "Install model" button writes the blob into the `model` partition over USB.
The same file is linked for download (for the dashboard's drag and drop). What the page shows
about the firmware and the model (version, model label, size) lives under the manifest's `n_cartpole` key,
which esp-web-tools ignores.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import urllib.request
from pathlib import Path

REPO = "Tsmorz/n-cartpole"
MODEL_NAME = "n-cartpole-model.ncpm"
MODEL_MANIFEST = "manifest-model.json"
DEFAULT_OUT = (
    Path(__file__).resolve().parents[3]
    / "personal-site"
    / "controls"
    / "swingup"
    / "real"
)

_spec = importlib.util.spec_from_file_location(
    "ncp_flash", Path(__file__).resolve().parent / "flash.py"
)
assert _spec is not None and _spec.loader is not None
flash = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(flash)


def _read_manifest(out: Path) -> dict:
    """Return the manifest already in `out`, or {} if there is none."""
    path = out / "manifest.json"
    return json.loads(path.read_text()) if path.exists() else {}


def build(image: Path, version: str, out: Path) -> dict:
    """Write the manifest and image pieces into `out` (replacing older pieces); return the manifest.

    The model info of an earlier run is kept: a new firmware does not change the model.
    """
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
    if info := _read_manifest(out).get("n_cartpole"):
        manifest["n_cartpole"] = info
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def publish_model(model: Path, out: Path) -> dict:
    """Publish the model next to the page; return its record.

    Copies the `.ncpm` into `out` as MODEL_NAME, records what it is in the manifest and writes the model's own
    installer manifest (the blob goes to the start of the `model` partition, as `task fw-flash-model` does).
    """
    from n_cartpole.deploy.firmware_blob import DTYPE_NAMES, read_header

    blob = model.read_bytes()
    h = read_header(blob)  # raises ValueError for anything the device would refuse
    info = {
        "file": MODEL_NAME,
        "label": h["label"],
        "dtype": DTYPE_NAMES[h["dtype"]],
        "hidden": h["hidden"],
        "blocks": h["n_blocks"],
        "bytes": len(blob),
    }
    manifest = _read_manifest(out)
    if not manifest:
        raise SystemExit(f"no manifest.json in {out}: install a firmware release first")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(model, out / MODEL_NAME)
    manifest["n_cartpole"] = {**manifest.get("n_cartpole", {}), "model": info}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    offset, size = flash.partition("model")
    if len(blob) > size:
        raise SystemExit(
            f"model is {len(blob)} bytes, the model partition holds {size}"
        )
    model_manifest = {
        "name": "n-cartpole model",
        "version": info["label"],
        "new_install_prompt_erase": False,
        "builds": [
            {
                "chipFamily": "ESP32-S3",
                "parts": [{"path": MODEL_NAME, "offset": offset}],
            }
        ],
    }
    (out / MODEL_MANIFEST).write_text(json.dumps(model_manifest, indent=2) + "\n")
    return info


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
    ap.add_argument(
        "source",
        nargs="?",
        help="a release tag (fw-v0.1.0) or a local factory.bin (omit with --model to only swap the model)",
    )
    ap.add_argument(
        "version", nargs="?", help="version label (required for a local file)"
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--model", type=Path, help="an exported .ncpm to publish next to the page"
    )
    args = ap.parse_args()
    if args.source is None:
        if not args.model:
            ap.error("give a release tag or image, or --model")
    elif args.source.startswith("fw-v"):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            image, version = download(args.source, Path(tmp))
            manifest = build(image, version, args.out)
    else:
        if not args.version:
            ap.error("a local image needs a version")
        manifest = build(Path(args.source), args.version, args.out)
    if args.source is not None:
        for part in manifest["builds"][0]["parts"]:
            print(f"{args.out / part['path']}  @ 0x{part['offset']:x}")
    if args.model:
        info = publish_model(args.model, args.out)
        print(
            f"{args.out / MODEL_NAME}  (model {info['label']}, {info['bytes']} bytes)"
        )


if __name__ == "__main__":
    main()
