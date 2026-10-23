"""The .ncpm firmware model blob: header layout, CRC, dtypes, unsupported configs."""

from __future__ import annotations

import dataclasses
import struct
from pathlib import Path

import numpy as np
import pytest
import torch

from n_cartpole.deploy import firmware_blob as fb
from n_cartpole.deploy.actor_export import actor_tensors
from n_cartpole.env.cartpole import EnvConfig
from n_cartpole.env.hardware_config import HardwareConfig
from n_cartpole.policy.loader import load_policy
from tests.firmware_util import make_checkpoint, rig_env


@pytest.fixture(scope="module")
def rig_ckpt(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Tiny goal-conditioned delta-mode checkpoint on the rig config."""
    return make_checkpoint(tmp_path_factory.mktemp("rig-goal") / "tqc_1234.pt")


@pytest.fixture(scope="module")
def plain_ckpt(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Tiny swing-up checkpoint without hardware: exact states, absolute actions."""
    return make_checkpoint(
        tmp_path_factory.mktemp("plain") / "tqc_77.pt", env=EnvConfig(), step=77
    )


def test_header_layout_matches_c_struct() -> None:
    """512 bytes, doubles at offset 40, label at 352 (ncp_model.h)."""
    assert struct.calcsize(fb.HEADER_FORMAT) == fb.HEADER_SIZE == 512
    ints = "<" + "".join(code for _, code in fb._INT_FIELDS)
    assert struct.calcsize(ints) == 40
    n_doubles = 4 + 6 + 4 * 5 + 9
    assert struct.calcsize(ints + "d" * n_doubles) == 352
    assert fb.CRC_OFFSET == struct.calcsize("<4sHHI")


def test_export_header_fields_and_round_trip(rig_ckpt: Path) -> None:
    """Every header field comes from the checkpoint; pack(unpack(h)) is identity."""
    blob = fb.export_blob(rig_ckpt)
    h = fb.read_header(blob)
    bundle = load_policy(rig_ckpt)
    cfg = bundle.env_config
    p = bundle.physics
    assert cfg.hardware is not None and cfg.hardware.sensors is not None
    s = cfg.hardware.sensors
    assert h["magic"] == b"NCPM" and h["version"] == 1 and h["header_size"] == 512
    assert h["total_size"] == len(blob)
    assert h["train_step"] == 1234
    assert (h["n_links"], h["n_blocks"], h["expand"], h["hidden"]) == (2, 2, 4, 32)
    assert h["obs_dim"] == bundle.obs_dim == 2 + 3 * 2 + 2 + 2
    assert h["flags"] == fb.FLAG_GOAL | fb.FLAG_DELTA | fb.FLAG_EKF
    assert (h["action_history"], h["delay_steps"], h["n_goals"]) == (2, 1, 4)
    assert h["goal_settle_steps"] == cfg.goal_settle_steps
    assert h["dt"] == p.dt and h["force_max"] == p.force_max
    assert h["force_slew"] == cfg.force_slew and h["x_lim"] == p.x_lim
    assert (h["M"], h["g"], h["b"]) == (p.M, p.g, p.b)
    assert h["cart_coulomb"] == p.cart_coulomb > 0
    assert h["v_noload"] == p.v_noload
    np.testing.assert_array_equal(h["masses"], p.link_masses(2))
    np.testing.assert_array_equal(h["lengths"], p.link_lengths(2))
    np.testing.assert_array_equal(h["com"], p.link_coms(2))
    np.testing.assert_array_equal(h["inertia"], p.link_inertias(2))
    np.testing.assert_array_equal(h["joint_friction"], p.joint_frictions(2))
    assert (h["ekf_force_std"], h["ekf_accel_std"]) == (
        s.ekf_force_std,
        s.ekf_accel_std,
    )
    assert h["x_resolution"] == s.x_resolution
    assert h["angle_noise_std"] == s.angle_noise_std
    assert h["goal_tol_angle"] == cfg.goal_tol_angle
    assert h["label"] == f"{rig_ckpt.parent.name}@1234"
    assert fb.pack_header(h) == blob[: fb.HEADER_SIZE]


def test_weights_are_actor_tensors_in_order(rig_ckpt: Path) -> None:
    """f32 weights are exactly actor_tensors(), flattened back to back."""
    blob = fb.export_blob(rig_ckpt)
    ckpt = torch.load(rig_ckpt, map_location="cpu", weights_only=False)
    flat = np.concatenate([a.ravel() for _, a in actor_tensors(ckpt)])
    np.testing.assert_array_equal(fb.read_weights(blob), flat.astype(np.float32))
    assert fb.weight_count(fb.read_header(blob)) == flat.size


def test_f16_blob_is_half_size_and_close(rig_ckpt: Path) -> None:
    """f16 halves the payload and decodes within float16 rounding of f32."""
    b32 = fb.export_blob(rig_ckpt, dtype="f32")
    b16 = fb.export_blob(rig_ckpt, dtype="f16")
    h16 = fb.read_header(b16)
    assert h16["dtype"] == fb.DTYPE_F16
    assert len(b16) - 512 == (len(b32) - 512) // 2
    w32, w16 = fb.read_weights(b32), fb.read_weights(b16)
    np.testing.assert_allclose(w16, w32, rtol=1e-3, atol=1e-4)
    with pytest.raises(ValueError, match="dtype"):
        fb.export_blob(rig_ckpt, dtype="int8")


def test_crc_and_size_checks_detect_corruption(rig_ckpt: Path) -> None:
    """A flipped bit anywhere, truncation, bad magic or version are rejected."""
    blob = fb.export_blob(rig_ckpt)
    for pos in (5, 100, 600, len(blob) - 1):  # header and weights
        bad = bytearray(blob)
        bad[pos] ^= 0x01
        with pytest.raises(ValueError):
            fb.read_header(bytes(bad))
    with pytest.raises(ValueError, match="CRC"):
        bad = bytearray(blob)
        bad[700] ^= 0x10
        fb.read_header(bytes(bad))
    with pytest.raises(ValueError, match="size"):
        fb.read_header(blob[:-4])
    with pytest.raises(ValueError, match="too short"):
        fb.read_header(blob[:100])
    with pytest.raises(ValueError, match="magic"):
        fb.read_header(b"XXXX" + blob[4:])
    bad = bytearray(blob)
    struct.pack_into("<H", bad, 4, 9)
    with pytest.raises(ValueError, match="version"):
        fb.read_header(bytes(bad))
    assert fb.blob_crc(blob) == fb.read_header(blob)["crc32"]


def test_label_truncation(rig_ckpt: Path) -> None:
    """Labels are cut to 63 bytes without splitting a UTF-8 character."""
    h = fb.read_header(fb.export_blob(rig_ckpt, label="x" * 100))
    assert h["label"] == "x" * 63
    h = fb.read_header(fb.export_blob(rig_ckpt, label="é" * 40))  # 2 bytes each
    assert h["label"] == "é" * 31
    long_dir = Path("ckpts") / ("d" * 80) / "tqc_5.pt"
    assert fb.default_label(long_dir, 5) == "d" * 63
    with pytest.raises(ValueError, match="links"):
        fb.pack_header({"masses": [1.0] * 5})


def test_plain_checkpoint_without_hardware(plain_ckpt: Path) -> None:
    """No hardware: no delay, no sensors, differenced velocities, one goal."""
    h = fb.read_header(fb.export_blob(plain_ckpt))
    assert h["flags"] == 0
    assert (h["delay_steps"], h["n_goals"], h["action_history"]) == (0, 1, 0)
    assert h["obs_dim"] == 8
    assert h["x_noise_std"] == h["velocity_filter"] == h["ekf_force_std"] == 0.0
    assert h["v_noload"] == 0.0
    assert h["label"] == f"{plain_ckpt.parent.name}@77"


def test_rejects_unsupported_configs() -> None:
    """Obs history, sysID, >4 links/blocks, long histories and delays are refused."""
    ok = rig_env()
    fb._check_supported(ok, 2, 2)
    hw = ok.hardware
    assert hw is not None
    cases = [
        (dataclasses.replace(ok, obs_history=1), 2, 2, "obs_history"),
        (
            dataclasses.replace(
                ok, hardware=dataclasses.replace(hw, sysid_context=True)
            ),
            2,
            2,
            "sysid",
        ),
        (ok, 5, 2, "n_links"),
        (dataclasses.replace(ok, action_history=9), 2, 2, "action_history"),
        (
            dataclasses.replace(ok, hardware=dataclasses.replace(hw, delay_steps=9)),
            2,
            2,
            "delay_steps",
        ),
        (ok, 2, 5, "blocks"),
    ]
    for cfg, n, blocks, msg in cases:
        with pytest.raises(ValueError, match=msg):
            fb._check_supported(cfg, n, blocks)


def test_sensing_without_sensor_model() -> None:
    """Hardware without sensors exports the 'difference exact positions' model."""
    cfg = dataclasses.replace(
        rig_env(), hardware=HardwareConfig(sensors=None, sysid_context=False)
    )
    s = fb._sensing(cfg)
    assert s.estimator == "diff" and s.x_resolution == 0.0
    assert fb._sensing(rig_env()).estimator == "ekf"


def test_inference_cost_estimate(rig_ckpt: Path) -> None:
    """A tiny net fits internal SRAM; a 256-wide one is PSRAM-bound and too slow."""
    h = fb.read_header(fb.export_blob(rig_ckpt))
    small = fb.inference_cost(h)
    assert small.fits_internal
    assert small.params == fb.weight_count(h)
    assert small.macs == 32 * 12 + 2 * 2 * 32 * 128 + 32
    assert small.budget_fraction < 0.1
    big = fb.inference_cost(dict(h, hidden=256))
    assert not big.fits_internal
    assert big.est_us == pytest.approx(big.weight_ram_bytes / 40e6 * 1e6)
    assert big.budget_fraction > 1.0
