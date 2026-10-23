"""firmware/tools/flash.py: flashing a merged image must not touch the NVS partition."""

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "ncp_flash", REPO / "firmware" / "tools" / "flash.py"
)
assert _spec is not None and _spec.loader is not None
flash = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(flash)


def test_nvs_range_comes_from_partitions_csv() -> None:
    """The nvs partition is 24 KiB at 0x9000 in the shipped table."""
    assert flash.nvs_range() == (0x9000, 0xF000)


def test_split_skips_exactly_the_nvs_bytes() -> None:
    """Both pieces are verbatim slices of the image; nothing in [start, end) is written."""
    start, end = flash.nvs_range()
    image = bytes(range(256)) * ((end + 5000) // 256 + 1)
    low, high = flash.split_around(image, start, end)
    assert low == (0, image[:start])
    assert high == (end, image[end:])
    # Reassembling over an erased (0xFF) chip leaves the NVS range untouched.
    chip = bytearray(b"\xff" * len(image))
    for off, data in (low, high):
        chip[off : off + len(data)] = data
    assert bytes(chip[start:end]) == b"\xff" * (end - start)
    assert bytes(chip[:start]) == image[:start] and bytes(chip[end:]) == image[end:]


def test_short_image_has_no_upper_piece() -> None:
    """An image that ends inside or before NVS yields only the lower piece."""
    start, end = flash.nvs_range()
    assert flash.split_around(b"\x01" * (start - 10), start, end) == [
        (0, b"\x01" * (start - 10))
    ]
    assert [off for off, _ in flash.split_around(b"\x01" * (end - 1), start, end)] == [
        0
    ]
