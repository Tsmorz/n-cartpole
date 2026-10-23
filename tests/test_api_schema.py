"""The device API against firmware/docs/api-schema.json.

The schema is written from the firmware's JSON builders (web.c, telemetry.c, events.c);
these tests hold the mock to it, so the two cannot drift apart unnoticed. Point
``NCP_DEVICE`` at a real device to check its firmware too (GETs only, nothing changes):

    NCP_DEVICE=http://192.168.1.109 uv run pytest tests/test_api_schema.py -k device
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from tests.firmware_util import REPO, make_checkpoint
from tests.test_dashboard import Mock

SCHEMA = json.loads((REPO / "firmware" / "docs" / "api-schema.json").read_text())
ENDPOINTS = {"api/info": "info", "api/status": "status", "api/events": "events"}


def _check(doc: Any, part: str) -> None:
    """Fail with every schema violation (path: message) of one response."""
    schema = {**SCHEMA, "$ref": f"#/$defs/{part}"}
    errors = sorted(
        Draft202012Validator(schema).iter_errors(doc), key=lambda e: list(e.path)
    )
    assert not errors, "\n".join(
        f"{part}.{'.'.join(map(str, e.path))}: {e.message}" for e in errors
    )


def test_schema_is_valid() -> None:
    """The schema itself is well-formed draft 2020-12."""
    Draft202012Validator.check_schema(SCHEMA)


@pytest.fixture(
    scope="module",
    params=[(), ("--rig", "--uncalibrated"), ("--rig",), ("--no-model",)],
)
def mock(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[Mock]:
    """Run the mock as sim, unzeroed rig, zeroed rig and a device with no model."""
    ckpt: Path = make_checkpoint(tmp_path_factory.mktemp("schema") / "tqc_1234.pt")
    m = Mock(ckpt, *request.param)
    yield m
    m.close()


@pytest.mark.parametrize("path", list(ENDPOINTS))
def test_mock_matches_schema(mock: Mock, path: str) -> None:
    """Every mock flavour answers in the firmware's shape."""
    _check(mock.get(path), ENDPOINTS[path])


def test_mock_matches_schema_after_events(mock: Mock) -> None:
    """Faults, zeroing and fault history keep the shape (optional fields included)."""
    mock.cmd({"inject_fault": "e-stop"})
    mock.wait(lambda st: st["faults"]["count"] >= 1)
    if mock.get("api/info")["backend"] == "rig":
        mock.cmd({"zero": "angles"})
        mock.cmd({"zero": "cart"})
        mock.wait(lambda st: st["calib"]["angles"] and st["calib"]["cart"])
    mock.cmd({"arm": False})
    for path, part in ENDPOINTS.items():
        _check(mock.get(path), part)


@pytest.mark.skipif(
    not os.environ.get("NCP_DEVICE"),
    reason="set NCP_DEVICE=http://<device> to check real firmware",
)
@pytest.mark.parametrize("path", list(ENDPOINTS))
def test_device_matches_schema(path: str) -> None:
    """A real device answers in the documented shape (read-only)."""
    base = os.environ["NCP_DEVICE"].rstrip("/") + "/"
    with urllib.request.urlopen(base + path, timeout=5) as r:  # noqa: S310
        _check(json.loads(r.read()), ENDPOINTS[path])
