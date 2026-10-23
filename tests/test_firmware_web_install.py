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


def test_new_release_replaces_old_pieces(tmp_path: Path) -> None:
    """Pieces of an earlier version are removed so the site does not accumulate images."""
    src = tmp_path / "factory.bin"
    src.write_bytes(b"\x01" * 100)
    out = tmp_path / "site"
    web_install.build(src, "0.0.1", out)
    web_install.build(src, "0.0.2", out)
    assert sorted(p.name for p in out.glob("*.bin")) == ["n-cartpole-fw-0.0.2-0.bin"]


def test_model_is_published_and_survives_a_new_firmware(tmp_path: Path) -> None:
    """The page's model link and details outlive a firmware update; an invalid blob is refused."""
    import pytest

    from n_cartpole.deploy.firmware_blob import export_blob, read_header
    from tests.firmware_util import make_checkpoint

    model = tmp_path / "model.ncpm"
    model.write_bytes(export_blob(make_checkpoint(tmp_path / "tqc_5.pt", step=5)))
    src = tmp_path / "factory.bin"
    src.write_bytes(b"\x01" * 100)
    out = tmp_path / "site"
    web_install.build(src, "0.0.1", out)
    info = web_install.publish_model(model, out)
    assert (out / web_install.MODEL_NAME).read_bytes() == model.read_bytes()
    assert info["label"] == read_header(model.read_bytes())["label"]
    web_install.build(src, "0.0.2", out)
    kept = json.loads((out / "manifest.json").read_text())
    assert kept["version"] == "0.0.2" and kept["n_cartpole"]["model"] == info

    bad = tmp_path / "bad.ncpm"
    bad.write_bytes(b"not a model")
    with pytest.raises(ValueError):
        web_install.publish_model(bad, out)
