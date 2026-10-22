"""The device dashboard (firmware/web/) against the stand-in device (mock_device.py).

The mock serves the same HTTP/WebSocket API as the firmware, backed by a tiny untrained
checkpoint, so these run without hardware or trained models:

- API checks: the event log, the rig's calibration gate on arming, fault history,
  hostname validation (the shapes the page reads, docs/design.md).
- A browser smoke test (Playwright + Chromium; skipped when they are not installed):
  every tab renders without page errors and the key fields fill in.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.firmware_util import REPO, make_checkpoint

MOCK = REPO / "firmware" / "tools" / "mock_device.py"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class Mock:
    """A running mock device on a free port."""

    def __init__(self, ckpt: Path, *flags: str) -> None:
        self.port = _free_port()
        self.base = f"http://127.0.0.1:{self.port}/"
        env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        self.proc = subprocess.Popen(
            [
                sys.executable,
                str(MOCK),
                "--port",
                str(self.port),
                "--checkpoint",
                str(ckpt),
                *flags,
            ],
            cwd=REPO,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"mock exited: {self.proc.stderr.read() if self.proc.stderr else ''}"
                )
            try:
                self.get("api/info")
                return
            except OSError:
                time.sleep(0.3)
        self.close()
        raise RuntimeError("mock did not come up")

    def get(self, path: str) -> Any:
        """GET a JSON endpoint."""
        with urllib.request.urlopen(self.base + path, timeout=5) as r:  # noqa: S310
            return json.loads(r.read())

    def post(self, path: str, body: Any) -> tuple[int, Any]:
        """POST JSON; return (HTTP status, JSON reply), errors included."""
        req = urllib.request.Request(  # noqa: S310
            self.base + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as r:  # noqa: S310
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def raw(self, path: str, data: bytes, headers: dict[str, str]) -> int:
        """POST bytes with exactly these headers; return the HTTP status."""
        req = urllib.request.Request(self.base + path, data=data, headers=headers)  # noqa: S310
        try:
            with urllib.request.urlopen(req, timeout=5) as r:  # noqa: S310
                return int(r.status)
        except urllib.error.HTTPError as e:
            return int(e.code)

    def cmd(self, body: dict[str, Any]) -> tuple[int, Any]:
        """Send one /api/cmd command."""
        return self.post("api/cmd", body)

    def wait(self, pred: Any, timeout: float = 5.0) -> Any:
        """Poll the status until pred(status) is truthy (commands apply on the next tick)."""
        deadline = time.monotonic() + timeout
        while True:
            st = self.get("api/status")
            if pred(st) or time.monotonic() > deadline:
                return st
            time.sleep(0.05)

    def close(self) -> None:
        """Stop the mock."""
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Tiny goal-conditioned rig-config checkpoint (untrained)."""
    return make_checkpoint(tmp_path_factory.mktemp("dash") / "tqc_1234.pt")


@pytest.fixture(scope="module")
def sim(ckpt: Path) -> Iterator[Mock]:
    """Run the simulator backend."""
    m = Mock(ckpt)
    yield m
    m.close()


@pytest.fixture(scope="module")
def rig(ckpt: Path) -> Iterator[Mock]:
    """Run the rig backend, zero references not set yet."""
    m = Mock(ckpt, "--rig", "--uncalibrated")
    yield m
    m.close()


# ---- API --------------------------------------------------------------------


def test_events_and_status_shape(sim: Mock) -> None:
    """The event log, its sequence number in the status, and the boot info."""
    ev = sim.get("api/events")
    assert ev["seq"] >= 1 and ev["events"], ev
    assert ev["events"][0]["msg"].startswith("boot #")
    assert {"seq", "t_ms", "level", "msg"} <= set(ev["events"][0])
    st = sim.get("api/status")
    assert st["events_seq"] == ev["seq"]
    assert st["io"] == {"rig": False}
    assert "calib" not in st
    assert st["faults"]["count"] == 0
    info = sim.get("api/info")
    assert info["boot"] >= 1 and info["reset_reason"]


def test_fault_history_and_events(sim: Mock) -> None:
    """A fault is counted and logged, and its history outlives clearing it."""
    seq0 = sim.get("api/status")["events_seq"]
    assert sim.cmd({"inject_fault": "e-stop"})[0] == 200
    st = sim.wait(lambda s: s["faults"]["count"] >= 1)
    assert st["run"] == "fault" and st["faults"]["last"] == "e-stop"
    assert sim.cmd({"arm": False})[0] == 200
    st = sim.wait(lambda s: s["run"] == "idle")
    assert (
        st["faults"]["count"] == 1 and "ago_s" in st["faults"]
    )  # history outlives the fault
    msgs = [e["msg"] for e in sim.get("api/events")["events"] if e["seq"] > seq0]
    assert "FAULT: e-stop" in msgs and "fault cleared (disarmed)" in msgs


def test_rig_refuses_to_arm_until_zeroed(rig: Mock) -> None:
    """Arming the rig is a 409 until both zero references are set."""
    st = rig.get("api/status")
    assert st["io"]["rig"] and len(st["io"]["enc"]) == 2
    assert st["calib"] == {"angles": False, "cart": False}
    code, body = rig.cmd({"arm": True})
    assert code == 409 and "not zeroed" in body["error"]
    assert rig.cmd({"zero": "angles"})[0] == 200
    assert rig.cmd({"zero": "cart"})[0] == 200
    st = rig.wait(lambda s: s["calib"]["angles"] and s["calib"]["cart"])
    assert st["calib"]["angles_boot"] == st["boot"] and "x_zero" in st["calib"]
    assert rig.cmd({"arm": True})[0] == 200
    assert rig.wait(lambda s: s["run"] == "armed")["io"]["drive"]
    rig.cmd({"arm": False})
    rig.wait(lambda s: s["run"] == "idle")


@pytest.mark.parametrize(
    ("name", "ok"),
    [
        ("bench-2", True),
        ("a", True),
        ("Bad", False),
        ("-x", False),
        ("x-", False),
        ("a" * 33, False),
    ],
)
def test_hostname_validation(sim: Mock, name: str, ok: bool) -> None:
    """Only mDNS-safe names are accepted."""
    code, body = sim.post("api/hostname", {"hostname": name})
    assert (code == 200) == ok, body
    if ok:
        assert sim.get("api/info")["hostname"] == name


def test_cross_site_posts_are_refused(sim: Mock) -> None:
    """A POST another site could send without a preflight is a 403; the page's own is not."""
    body = json.dumps({"x_ref": 0.0}).encode()
    host = f"127.0.0.1:{sim.port}"
    assert sim.raw("api/cmd", body, {"Content-Type": "text/plain"}) == 403
    assert (
        sim.raw("api/cmd", body, {"Content-Type": "application/x-www-form-urlencoded"})
        == 403
    )
    evil = {"Content-Type": "application/json", "Origin": "http://evil.example"}
    assert sim.raw("api/cmd", body, evil) == 403
    assert sim.raw("api/reboot", b"{}", evil) == 403
    assert sim.raw("api/model", b"NCPM", {"Content-Type": "text/plain"}) == 403
    assert sim.raw("api/ota", b"\xe9", {"Content-Type": "application/json"}) == 403
    same = {
        "Content-Type": "application/json; charset=utf-8",
        "Origin": f"http://{host}",
    }
    assert sim.raw("api/cmd", body, same) == 200
    assert (
        sim.raw("api/cmd", body, {"Content-Type": "application/json"}) == 200
    )  # no Origin: curl


# ---- browser ----------------------------------------------------------------


@pytest.fixture(scope="module")
def browser() -> Iterator[Any]:
    """Headless Chromium, or skip (pip/uv: playwright; then `playwright install chromium`)."""
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as e:
            pytest.skip(f"Chromium not available: {e}")
        yield b
        b.close()


def _open(browser: Any, url: str, width: int = 1440) -> tuple[Any, list[str]]:
    page = browser.new_page(viewport={"width": width, "height": 900})
    # No internet in these tests: the page asks GitHub for the firmware releases and Google
    # for its fonts. An unauthenticated GitHub API call can be rate-limited (403) on a shared
    # runner IP, which the browser logs as a console error, so answer both here.
    page.route(
        "https://api.github.com/**",
        lambda r: r.fulfill(status=200, content_type="application/json", body="[]"),
    )
    page.route(
        "https://fonts.googleapis.com/**",
        lambda r: r.fulfill(status=200, content_type="text/css", body=""),
    )
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on(
        "console",
        lambda m: errors.append(m.text)
        if m.type == "error" and "fonts.g" not in m.text
        else None,
    )
    page.goto(url)
    page.wait_for_function(
        "document.getElementById('conn').dataset.state === 'live'", timeout=15000
    )
    return page, errors


def _text(page: Any, sel: str) -> str:
    return str(page.locator(sel).inner_text())


@pytest.mark.parametrize("width", [1440, 390])
def test_every_tab_renders(sim: Mock, browser: Any, width: int) -> None:
    """Each tab fills in without page errors, on a desktop and a phone."""
    page, errors = _open(browser, sim.base + "#live", width)
    page.wait_for_function(
        "document.querySelectorAll('#ro-rows tr').length >= 6", timeout=10000
    )
    assert _text(page, "#run-chip-text") in ("IDLE", "ARMED", "FAULT")
    assert "Hz" in _text(page, "#fact-rate")
    assert page.locator("#ro-rows tr").first.locator("td").first.inner_text() != "—"
    page.click("#tab-btn-health")
    page.wait_for_function(
        "document.querySelector('#dev-kv').textContent.includes('fps')", timeout=15000
    )
    page.wait_for_selector("#events li.ev", timeout=15000)
    assert "sim" in _text(page, "#io-tag")
    page.click("#tab-btn-record")
    page.locator(".rec-ctrl [data-arm]").wait_for(state="visible", timeout=15000)
    page.click("#tab-btn-settings")
    page.wait_for_function(
        "document.querySelector('#model-kv').textContent.includes('label')",
        timeout=10000,
    )
    assert "boot" in _text(page, "#sys-kv")
    assert "clients" not in _text(page, "#wifi-kv")
    links = page.eval_on_selector_all("#wifi-kv a", "els => els.map(e => e.href)")
    assert any(h.endswith(".local/") for h in links) and len(links) >= 2, links
    assert page.locator("#stale-banner").is_hidden()
    page.close()
    assert not errors, errors


def test_arm_from_record_tab_and_disarm_from_header(sim: Mock, browser: Any) -> None:
    """Record's Arm arms; the header chip disarms."""
    page, errors = _open(browser, sim.base + "#record")
    page.click(".rec-ctrl [data-arm]")
    page.wait_for_function(
        "document.getElementById('run-chip-text').textContent === 'ARMED'",
        timeout=15000,
    )
    page.click("#run-chip")
    page.wait_for_function(
        "document.getElementById('run-chip-text').textContent === 'IDLE'", timeout=15000
    )
    page.close()
    assert not errors, errors


def test_rig_sensors_card(rig: Mock, browser: Any) -> None:
    """The rig's encoder diagnostics reach Health → Sensors & drive."""
    rig.cmd({"inject_io": "weak-magnet"})
    page, errors = _open(browser, rig.base + "#health")
    page.wait_for_selector(".io-table tbody tr", timeout=10000)
    page.wait_for_function(
        "document.querySelector('#io-body').textContent.includes('too weak')",
        timeout=15000,
    )
    assert "check sensors" in _text(page, "#io-tag") or "not zeroed" in _text(
        page, "#io-tag"
    )
    rig.cmd({"inject_io": "ok"})
    page.close()
    assert not errors, errors


def test_record_warning_fixes_the_setup(sim: Mock, browser: Any) -> None:
    """Record's setup warning carries a button that does what it asks (mode, then arm).

    Each click is checked on the device first, then on the page. On a loaded machine (CI)
    the mock's telemetry can run seconds behind, so the page shows the previous mode or
    run state for a while after the device changed; waiting on the device keeps the next
    click from landing on a stale button, and the page waits allow for the backlog.
    """
    page_wait = 20000
    sim.cmd({"arm": False})
    sim.cmd({"mode": "ai"})
    sim.wait(lambda st: st["run"] != "armed" and st["mode"] == "ai")
    page, errors = _open(browser, sim.base + "#record")
    page.click("#rec-presets [data-preset=chirp]")
    page.wait_for_function(
        "document.getElementById('rec-fix')?.textContent === 'Switch to Manual'",
        timeout=page_wait,
    )
    page.click("#rec-fix")
    assert sim.wait(lambda st: st["mode"] == "manual", timeout=10)["mode"] == "manual"
    page.wait_for_function(
        "document.getElementById('rec-fix')?.textContent === 'Arm'", timeout=page_wait
    )
    page.click("#rec-fix")
    st = sim.wait(lambda st: st["run"] == "armed", timeout=10)
    assert st["run"] == "armed" and st["mode"] == "manual"
    page.wait_for_function(
        "document.getElementById('rec-warn').hidden", timeout=page_wait
    )
    assert page.locator("#rec-ids code.copy").count() == 1
    sim.cmd({"arm": False})
    sim.cmd({"mode": "ai"})
    sim.wait(lambda st: st["run"] != "armed" and st["mode"] == "ai")
    page.close()
    assert not errors, errors


def test_event_log_survives_a_reboot(sim: Mock) -> None:
    """Entries from before a software reset stay, tagged with their boot (last: it reboots)."""
    before = sim.get("api/events")
    boot0 = before["boot"]
    sim.cmd({"inject_fault": "deadline miss"})
    sim.wait(lambda st: st["run"] == "fault")
    sim.cmd({"arm": False})
    assert sim.post("api/reboot", {})[0] == 200
    deadline = time.monotonic() + 30
    after: dict[str, Any] = {}
    while time.monotonic() < deadline:
        time.sleep(0.5)
        try:
            after = sim.get("api/events")
        except OSError:
            continue
        if after.get("boot", boot0) > boot0:
            break
    assert after["boot"] == boot0 + 1, after
    old = [e for e in after["events"] if e["boot"] == boot0]
    assert any(e["msg"] == "FAULT: deadline miss" for e in old)
    assert after["events"][-1]["boot"] == boot0 + 1
    assert "software restart" in after["events"][-1]["msg"]
