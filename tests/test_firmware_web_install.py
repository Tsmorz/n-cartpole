"""firmware/tools/web_install.py: the browser installer must not touch the NVS partition."""

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "ncp_web_install", REPO / "firmware" / "tools" / "web_install.py"
)
assert _spec is not None and _spec.loader is not None
web_install = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(web_install)


def test_manifest_parts_skip_nvs_and_reassemble_the_image(tmp_path: Path) -> None:
    """Parts are verbatim slices of the image at their offsets; nothing in NVS is written."""
    start, end = web_install.flash.nvs_range()
    image = bytes(range(256)) * ((end + 5000) // 256 + 1)
    src = tmp_path / "factory.bin"
    src.write_bytes(image)
    out = tmp_path / "site"
    web_install.build(src, "1.2.3", out)

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["version"] == "1.2.3"
    assert manifest["new_install_prompt_erase"] is False
    build = manifest["builds"][0]
    assert build["chipFamily"] == "ESP32-S3"
    chip = bytearray(b"\xff" * len(image))
    for part in build["parts"]:
        data = (out / part["path"]).read_bytes()
        chip[part["offset"] : part["offset"] + len(data)] = data
    assert bytes(chip[start:end]) == b"\xff" * (end - start)
    assert bytes(chip[:start]) == image[:start] and bytes(chip[end:]) == image[end:]


def test_page_points_at_the_manifest(tmp_path: Path) -> None:
    """The page loads esp-web-tools and the manifest next to it."""
    src = tmp_path / "factory.bin"
    src.write_bytes(b"\x01" * 100)
    web_install.build(src, "0.0.1", tmp_path / "site")
    html = (tmp_path / "site" / "index.html").read_text()
    assert 'manifest="manifest.json"' in html and "esp-web-tools" in html
