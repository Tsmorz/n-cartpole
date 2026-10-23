"""The stand-in device's arm gate must not depend on the speed of the machine running it."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

from tests.firmware_util import REPO, make_checkpoint

_spec = importlib.util.spec_from_file_location(
    "ncp_mock_device", REPO / "firmware" / "tools" / "mock_device.py"
)
assert _spec is not None and _spec.loader is not None
mock_device = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mock_device  # its dataclasses look their module up
_spec.loader.exec_module(mock_device)


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a tiny untrained goal-conditioned rig checkpoint."""
    return make_checkpoint(tmp_path_factory.mktemp("mockdev") / "tqc_1234.pt")


def _rig(ckpt: Path, strict_budget: bool) -> object:
    args = argparse.Namespace(
        rig=True,
        checkpoint=str(ckpt),
        blob=None,
        no_model=False,
        uncalibrated=False,
        strict_budget=strict_budget,
        host="127.0.0.1",
        port=0,
    )
    dev = mock_device.Device(args)
    dev._last_loop = {
        "exec_avg_us": 99_999
    }  # far over the 80 % budget, as on a slow runner
    return dev


def test_slow_machine_does_not_stop_the_rig_arming_by_default(ckpt: Path) -> None:
    """A slow loop is reported, but a calibrated rig still arms (CI runners are slower)."""
    dev = _rig(ckpt, strict_budget=False)
    assert not dev.budget_ok  # type: ignore[attr-defined]
    assert dev.check_cmd({"arm": True}) == ""  # type: ignore[attr-defined]


def test_strict_budget_refuses_to_arm_when_over_budget(ckpt: Path) -> None:
    """With --strict-budget the mock behaves like the firmware's 80 % arming gate."""
    dev = _rig(ckpt, strict_budget=True)
    assert "loop budget exceeded" in dev.check_cmd({"arm": True})  # type: ignore[attr-defined]


@pytest.mark.parametrize("dtype", ["f32", "f16", "i8"])
def test_mock_reads_every_blob_dtype(ckpt: Path, dtype: str) -> None:
    """The mock validates and decodes each weight dtype exactly as firmware_blob does."""
    import numpy as np

    from n_cartpole.deploy import firmware_blob as fb

    blob = fb.export_blob(ckpt, dtype=dtype)
    h = mock_device.validate_blob(blob)
    assert mock_device.weight_bytes(h) == fb.weight_bytes(h)
    np.testing.assert_array_equal(
        mock_device.decode_weights(blob, h), fb.read_weights(blob)
    )
    assert mock_device.model_info(h)["dtype"] == dtype
