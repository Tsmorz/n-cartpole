# n-cartpole firmware — design

ESP32-S3 firmware that runs a trained n-cartpole TQC actor at the training control
rate (200 Hz for the rig config), on real hardware or on a simulated plant inside
the chip, with a web dashboard served by the device and over-the-air updates of
both the firmware and the model.

Target board: **ESP32-S3 with 16 MB flash and 8 MB octal PSRAM** (e.g.
ESP32-S3-DevKitC-1 N16R8). Framework: **ESP-IDF v5.3** (CMake, `idf.py`).

## Repository layout

```
firmware/
  CMakeLists.txt  sdkconfig.defaults  partitions.csv  version.txt (written by CI from the tag)
  components/cartpole_core/   portable C11, no ESP-IDF headers: model blob, policy, dynamics,
                              EKF, controller tick, simulator. Compiled on the host by
                              tests/test_firmware_core.py against the Python originals.
  main/                       the ESP-IDF app: tasks, hardware, web server, OTA, Wi-Fi
  web/                        the dashboard: index.html, style.css, app.js (vanilla, no build),
                              embedded into the app image
  test/host/ncp_host.c        host harness the parity tests drive over stdin/stdout
  tools/mock_device.py        a stand-in device (same HTTP/WS API, simulated plant) to develop
                              the dashboard without hardware
  docs/design.md              this file
n_cartpole/deploy/firmware_blob.py   checkpoint -> .ncpm blob (and back, for tests)
scripts/export_firmware.py           CLI: task export-firmware
.github/workflows/firmware.yml       build on every push/PR touching firmware/; release on fw-v* tags
```

## The model blob (.ncpm)

Specified byte-for-byte in `components/cartpole_core/include/ncp_model.h`: a
512-byte header (dims, flags, actuator limits, nominal physics, sensor/EKF
parameters, label) followed by float32 (or float16) weights in forward order with the
RunningNorm folded into `in_w/in_b`. The device stores it in its own `model` flash
partition, independent of the firmware image, so firmware and models are updated
separately (and a firmware release never contains a model: checkpoints are not in git).

Goal indices follow `n_cartpole.env.goals.goal_configs`: bit k set = link k down.
For two links: `0 UU, 1 DU, 2 UD, 3 DD` (labels base link first).

## Control pipeline

`components/cartpole_core/include/ncp_ctrl.h` documents the tick; it is the
training env's `step()/_sense()/_make_obs()` seen from the controller. The parity
tests run the C controller and the Python env on the same readings and require the
same observations and commands.

## Cores and tasks

| core | task / interrupt | prio | what |
|---|---|---|---|
| 1 (APP) | `ctrl` | 22 | the control loop; woken by its own gptimer ISR every period |
| 1 (APP) | gptimer ISR | — | allocated from `ctrl` (so it lands on core 1); only `vTaskNotifyGiveFromISR` |
| 1 (APP) | IDLE1 | 0 | must keep running (task watchdog) — the loop budget leaves it ≥ 20 % |
| 0 (PRO) | Wi-Fi, lwIP `tcpip`, `sys_evt`, `esp_timer` | IDF | pinned to core 0 in sdkconfig.defaults |
| 0 (PRO) | `httpd` | 5 | HTTP API, OTA and model uploads, WebSocket |
| 0 (PRO) | `tlm` | 4 | drains the telemetry ring, sends WS frames (~25 msg/s) and status (2 Hz) |
| 0 (PRO) | `main` | 1 | init, then deletes itself |

Rules:

- **Nothing on core 1 but the control loop.** It never logs, never allocates, never
  blocks on a lock held by core 0. Commands arrive through a FreeRTOS queue read with
  zero timeout; telemetry leaves through a single-producer/single-consumer ring.
- **Flash writes stall both cores** (cache disabled). OTA and model uploads are
  refused (HTTP 409) unless the controller is disarmed; the loop is put in `MAINT`
  (drive disabled) for the duration.
- The control task uses the FPU and is pinned (FreeRTOS requires FPU tasks pinned).
- Loop budget: execution ≤ 80 % of the period. The loop runs the estimator and the
  network every tick even when disarmed, so its timing is known before arming. On the
  **rig** backend, arming is refused while the budget is exceeded. On the **sim**
  backend the period stretches instead (slow motion, shown as the real-time factor),
  so oversized models can still be tried.

## Run states and modes

- `IDLE` (disarmed): drive disabled (rig) / zero force (sim); estimation keeps running.
- `ARMED`: commands go to the drive.
- `FAULT`: tripped (off the rail |x| > x_lim, encoder errors, e-stop, deadline misses on
  the rig); drive disabled until re-armed. The simulator also resets itself to DD 1.5 s
  after going off the rail.
- `MAINT`: an upload is writing flash.

Arming the rig is refused (409, with the reason) while: no model, MAINT, the e-stop is
pressed, the cart is off the rail, the model is not 2-link, the loop is over budget,
encoder reads are failing, or a **zero reference is not set** (angles or cart: neither
restored from NVS at boot nor zeroed since). The task counts faults (`fault_count`, the
last one and its tick), so the status keeps a fault's history after it is cleared.

Modes (independent of arming): `ai` (the policy drives to the goal; the cart slider sets
`x_ref`, the point the AI balances around) and `manual` (a PD loop drives the cart to
the slider position, capped at 10 N; to jog the rig or perturb the pendulum).
Goals: UU / UD / DU / DD buttons, or **Random**: a new goal (different from the current)
1 s after each goal is reached.

Backends (Kconfig `NCP_BACKEND_SIM` / `NCP_BACKEND_RIG`, default sim):
- **sim**: `ncp_sim` integrates the plant on the chip; readings use the exported
  encoder model. The dashboard can reset it to any pose.
- **rig**: 2 × AS5048A (SPI, 14-bit) on joint 1 (absolute link-1 angle) and joint 2
  (link 2 *relative to* link 1, through the slip ring; absolute θ₂ = θ₁ + φ), a
  quadrature encoder on the motor (PCNT, ×4) for the cart, a PWM + DIR + EN torque
  command to a FOC driver, an active-high e-stop sense input. Pins, signs, counts/rev,
  pulley radius and PWM full-scale force are Kconfig options.

## HTTP / WebSocket API (port 80; also `http://<hostname>.local`)

All JSON responses carry `Content-Type: application/json`. Errors: HTTP 4xx/5xx with
`{"ok": false, "error": "<reason>"}`.

**Cross-site guard** (`web_post_guard` in `web.c`): every POST must carry
`Content-Type: application/json` (`application/octet-stream` for `/api/model` and
`/api/ota`), and a browser's `Origin` header, when present, must name the `Host` it was
sent to; otherwise 403. Without it, any web page open on a computer on the same network
could make the browser send a "simple" cross-origin POST (`text/plain` or form data, no
preflight) such as `{"arm": true}` to `cartpole.local`. Requiring a non-simple content type
forces a CORS preflight, which this server never answers with permission. Body-less
endpoints (`/api/reboot`, `/api/wifi/forget`) take `{}`. GETs stay open (`/api/info`,
`/api/status` and `/api/events` send `Access-Control-Allow-Origin: *` for tools). Not
covered: DNS rebinding (a page whose own host name resolves to the device) and anyone
who can reach the device directly; there is no password, so keep it on a trusted network.

| method | path | body / response |
|---|---|---|
| GET | `/` `/app.js` `/style.css` | the dashboard |
| GET | `/api/info` | static-ish device info (below) |
| GET | `/api/status` | the status JSON (also pushed over WS) |
| POST | `/api/cmd` | one command object (below) → `{"ok": true}` |
| GET | `/api/model` | the stored `.ncpm` blob (`application/octet-stream`), 404 if none |
| POST | `/api/model` | raw `.ncpm` body; validated (header + CRC), stored, hot-reloaded → `{"ok": true, "label": "..."}`; 409 if armed |
| POST | `/api/ota` | raw app image (`n-cartpole-fw-<ver>.bin`); first byte 0xE9; app descriptor project name must be `n_cartpole_fw` → `{"ok": true, "version": "..."}`, then reboots after 1 s; 409 if armed |
| POST | `/api/wifi` | `{"ssid": "...", "password": "..."}` → `{"ok": true}` at once; the join runs in the background (see below), no reboot; 409 if armed, still starting or a join is running |
| POST | `/api/reboot` | → `{"ok": true}`, reboots after 0.5 s |
| GET | `/api/events` | the event log (below) |
| POST | `/api/wifi/forget` | → `{"ok": true}`, stores an empty SSID (so the Kconfig default is not used either) and reboots into the setup access point; 409 if armed |
| POST | `/api/hostname` | `{"hostname": "bench"}` (1–32 of `a-z0-9-`, not starting/ending with `-`) → `{"ok": true, "hostname": "bench"}`; saved to NVS and applied to mDNS at once; 409 if armed |

Joining a network (`POST /api/wifi`, `net_join_start` in `net.c`): a `wifi_join` task on core 0
sets the station config and connects, waiting up to 20 s for an IP. From the setup access point
it switches to AP+STA so the AP stays up (the AP follows the router's channel, so a client may
drop and re-associate). On success the credentials are written to NVS (only then), `join.state`
is `ok` with the new `join.ip`, and after a grace period (20 s from the AP, 5 s from a station)
the AP is dropped and the station kept: no reboot. On failure nothing is saved, `join.state` is
`failed` with `join.error` ("network not found", "authentication failed (wrong password?)", ...)
until the next attempt, and the device returns to the AP (or reconnects the previous network).
| GET (upgrade) | `/ws` | WebSocket: binary telemetry (ncp_telemetry.h) + text status JSON at 2 Hz |

Commands (`POST /api/cmd`, one key per request):

```
{"arm": true | false}
{"mode": "ai" | "manual"}
{"goal": "UU" | "DU" | "UD" | "DD"}         // also stops random cycling
{"random": true | false}
{"x_ref": 0.12}                             // m, clamped to ±x_lim/2
{"reset": "DD"}                             // sim only: restart the plant at that pose
{"zero": "angles"}                          // rig, disarmed: links hanging at rest -> θ₁ = π, φ = 0
{"zero": "cart"}                            // rig, disarmed: current cart position -> x = 0
```

`/api/info`:

```json
{
  "fw": {"project": "n_cartpole_fw", "version": "0.1.0", "idf": "v5.3.1",
         "built": "Oct 22 2026 12:00:00", "repo": "Tsmorz/n-cartpole"},
  "chip": {"model": "ESP32-S3", "cores": 2, "revision": 2, "mhz": 240,
           "flash_mb": 16, "psram_mb": 8},
  "ota": {"running": "ota_0", "next": "ota_1", "state": "valid"},
  "backend": "sim",
  "hostname": "cartpole",
  "boot": 42, "reset_reason": "power-on" | "crash (panic)" | "brownout (supply dipped)" | ...,
  "telemetry": {"version": 1, "frame_size": 232, "rate_hz": 50},
  "model": null | {
    "label": "...", "step": 2000000, "n_links": 2, "obs_dim": 12, "hidden": 256,
    "blocks": 2, "expand": 4, "params": 1056258, "bytes": 4225548, "crc32": "1a2b3c4d",
    "dtype": "f32", "weights_in": "psram" | "internal",
    "goal_conditioned": true, "action_mode": "delta", "estimator": "ekf",
    "action_history": 2, "delay_steps": 1, "dt": 0.005, "force_max": 15.35,
    "force_slew": 300.0, "x_lim": 0.5, "goals": ["UU", "DU", "UD", "DD"]
  }
}
```

Every JSON shape below is also in **`docs/api-schema.json`** (JSON Schema, written from
`web.c`, `telemetry.c` and `events.c`). `tests/test_api_schema.py` holds the mock to it, and
checks a real device with `NCP_DEVICE=http://<ip> uv run pytest tests/test_api_schema.py -k device`
(GETs only). Change the schema with the firmware.

Status (WS text message and `/api/status`):

```json
{
  "run": "idle" | "armed" | "fault" | "maint", "fault": "" | "off rail" | "e-stop" | ...,
  "mode": "ai" | "manual", "goal": "UU", "random": false, "x_ref": 0.0,
  "settled": true, "backend": "sim",
  "loop": {"period_us": 5000, "exec_avg_us": 900, "exec_max_us": 1400,
           "policy_avg_us": 600, "est_avg_us": 200, "misses": 0, "rt_factor": 1.0,
           "budget_ok": true},
  "heap": {"internal_free": 180000, "internal_min": 150000, "psram_free": 3800000},
  "wifi": {"mode": "sta" | "ap", "ssid": "...", "ip": "192.168.1.40", "rssi": -52,
           "join": {"state": "idle" | "connecting" | "ok" | "failed", "ssid": "...", "ip": "...", "error": "..."}},
  "uptime_s": 1234, "boot": 42, "events_seq": 17, "unix": 1792700000,   // unix only once the clock is set
  "faults": {"count": 2, "last": "e-stop", "ago_s": 81.5},
  "io": {"rig": false}
      | {"rig": true, "estop": false, "drive": false, "duty": 0.0, "dir": 1, "cart_count": -88,
         "read_errors": 0,
         "enc": [{"raw": 8170, "bad": 0, "agc": 72, "ocf": true, "cof": false,
                  "comp_low": false, "comp_high": false}, {...}]},
  "calib": {"angles": true, "cart": true, "off1": 2.81734, "off2": -0.41263, "angles_boot": 40,
            "angles_unix": 1792600000, "x_zero": 0.01275, "cart_boot": 40, "cart_unix": 1792600000},  // rig only
  "cores": [{"id": 0, "load": 12.5}, {"id": 1, "load": 31.0}],
  "tasks": [{"name": "ctrl", "core": 1, "prio": 22, "cpu": 30.9, "stack_free": 2100}, ...]
}
```

`core` is the task's affinity (`-1` = either core); `cpu` is its share of one core
since the previous status message (FreeRTOS run-time stats).

`io` (rig): `enc[0]` is joint 1, `enc[1]` joint 2 (φ, relative to link 1). `raw` is the
last good 14-bit angle and `bad` the bad reads since boot (SPI error, parity, error flag);
`read_errors` counts consecutive ones (the encoder fault trips at
`NCP_ENC_ERROR_LIMIT`). The AS5048A diagnostics register (0x3FFD) is read once a second
per encoder (absent until the first good read): `agc` 0 = strong .. 255 = weak field,
`comp_high` = field too weak (magnet too far), `comp_low` = too strong (too close), `cof` =
CORDIC overflow, `ocf` = offset compensation finished (should be true). `duty` (0–1) and
`dir` are the PWM command, `drive` the EN line. `calib.*_boot` is the boot (NVS boot
counter) on which each zero reference was set, `calib.*_unix` when (if the clock was set).

**Clock:** `app_main` starts SNTP (`pool.ntp.org`, UTC) once the network is up; it retries
quietly without internet (the setup access point). Until it syncs, `unix` is absent from
the status, events carry only `t_ms`, and zeroing stores no date. The page shows local time.

`events_seq` counts the event-log entries; a page fetches `/api/events` when it moves:

```json
{"seq": 17, "boot": 43, "uptime_ms": 81234,
 "events": [{"seq": 1, "boot": 42, "t_ms": 412, "level": "info", "msg": "boot #42: v0.2.0, rig backend, reset: power-on"},
            {"seq": 9, "boot": 42, "t_ms": 60120, "level": "error", "msg": "FAULT: e-stop"},
            {"seq": 10, "boot": 43, "t_ms": 380, "level": "error", "msg": "boot #43: v0.2.0, rig backend, reset: task watchdog"},
  {"seq": 11, "boot": 43, "t_ms": 4120, "unix": 1792700000, "level": "info", "msg": "clock set over SNTP"}, ...]}
```

The log (`main/events.c`, core 0) keeps the latest 40 entries in RTC memory that a reset
does not clear (`RTC_NOINIT_ATTR`, ~3.7 KB), so the entries before a crash, a watchdog
reset, a brownout or a reboot are still there afterwards; `t_ms` counts from the boot that
logged the entry (`boot`). Each entry has its own CRC (one half-written when the chip went
down is dropped, not the log), and a power-on reset starts it empty. It holds the boot line
with the reset reason, arming / disarming / faults / mode and model
changes and loop-budget changes (the `tlm` task turns changes in the control task's status
into entries; deadline misses while armed are summed, one entry per 10 s at most), model
and firmware uploads, Wi-Fi joins, hostname changes, zeroing and reboots.

## Wi-Fi

Station mode with credentials from NVS (or Kconfig defaults). If none are set, or the
network is not found within 15 s, the device starts its own access point
`cartpole-XXXX` (open, 192.168.4.1) so the dashboard is reachable for setup. mDNS
advertises `<hostname>.local` (default `cartpole`; changed from the dashboard with
`POST /api/hostname`, live). `POST /api/wifi/forget` erases the network and reboots into the
access point. Wi-Fi power save is off
(`WIFI_PS_NONE`): modem sleep makes the radio miss multicast queries, so the `.local` name
resolved only intermittently, and it adds latency jitter to the telemetry stream.

## Updates

- **USB flash:** `task fw-flash -- <port>` writes the merged factory image
  (bootloader + partition table + app) from a release or a local build, around the NVS
  partition so the saved Wi-Fi network survives (`--erase-settings` to forget it).
- **Firmware OTA:** drop `n-cartpole-fw-<version>.bin` from the GitHub release onto
  the dashboard. Two app slots (`ota_0`, `ota_1`) with rollback: a new image boots
  `pending`, marks itself valid after 10 s of healthy running (control loop ticking,
  HTTP server up); a crash before that rolls back to the previous slot.
- **Model:** drop a `.ncpm` (from `task export-firmware`) onto the dashboard; no
  reboot. Or flash it over USB with `task fw-flash-model`.
- **Releases:** `task fw-release -- fw-v0.2.0` tags and pushes; the `firmware` workflow
  builds with ESP-IDF v5.3.1 and attaches `n-cartpole-fw-<ver>.bin` (OTA),
  `n-cartpole-fw-<ver>-factory.bin` (USB, offset 0x0) and `SHA256SUMS` to the release.

## Partition table (16 MB)

| name | type | offset | size |
|---|---|---|---|
| nvs | data/nvs | 0x9000 | 24 KB |
| otadata | data/ota | 0xF000 | 8 KB |
| phy_init | data/phy | 0x11000 | 4 KB |
| ota_0 | app | 0x20000 | 3 MB |
| ota_1 | app | 0x320000 | 3 MB |
| model | data/0x40 | 0x620000 | 6 MB |

## Recording and system identification (sim-to-real)

The policy is trained on a *nominal* plant plus randomization. The closer the nominal
model is to the built rig, the better the transfer. The device therefore records
what it sees and does at the full control rate, and an offline tool fits the plant
parameters to the recordings:

```
dashboard: Record (preset) -> device logs every tick -> download .ncpr
task sysid -- recordings/*.ncpr --config config/rig.toml --out config/rig-fitted.toml
task train -- --resume <checkpoint> --config config/rig-fitted.toml ...   (fine-tune on the fitted plant)
task distill / task export-firmware -> drop the new .ncpm on the dashboard
```

**Format:** `components/cartpole_core/include/ncp_record.h` — a 256-byte header (dt,
links, backend, excitation, firmware version, model label + CRC, note) and one
128-byte sample per tick: tick, actual sample time, the calibrated readings, the drive
command and its excitation part, the modelled applied force, the online estimate, the
simulator's true state (sim only), run/mode/goal/flags and the tick period. 200 Hz ×
128 B = 25.6 KB/s; the buffer lives in PSRAM, allocated when a recording starts
(default capacity 120 s ≈ 3 MB, Kconfig `NCP_RECORD_SECONDS`; it shrinks if PSRAM is
short and says so). The control task appends without locks or allocation; the
recording stops itself when full. The last recording stays downloadable until the next
one starts.

**Excitation:** `ncp_excite.h` (chirp, PRBS, steps, sine; deterministic for a seed)
feeds `ncp_ctrl_t.ext_force`. It only reaches the drive while armed. Its amplitude is
capped at the manual force cap (10 N), and the rail fault still trips.

**Presets** (the dashboard's Record card):

| preset | setup | identifies |
|---|---|---|
| Ring-down | disarmed; lift the links by hand and let go; 30 s | joint friction, link COM / inertia (pendulum periods and decay) |
| Cart chirp | armed, Manual (PD holds the cart at the slider), chirp 3 N 0.2→6 Hz, 40 s | cart mass, viscous + Coulomb friction, force gain, link coupling |
| PRBS | armed, Manual, ±3 N, 50 ms bits, 30 s | the same, broadband; delay |
| Steps | armed, Manual, ±4 N, 0.5 s holds, 20 s | delay, slew, force gain |
| Closed loop | armed, AI, no excitation | validation of the fitted model under the policy |

**API additions:**

| method | path | body / response |
|---|---|---|
| POST | `/api/cmd` | `{"record": {"action": "start", "seconds": 60, "note": "ring-down 1", "excite": null \| {"kind": "chirp" \| "prbs" \| "steps" \| "sine", "amp": 3.0, "f0": 0.2, "f1": 6.0, "period": 0.05, "duration": 40, "seed": 1}}}` / `{"record": {"action": "stop"}}` |
| POST | `/api/cmd` | `{"excite": {...} \| null}` — excitation without recording (e.g. to try amplitudes) |
| GET | `/api/record` | the last recording as `application/octet-stream`, `Content-Disposition: attachment; filename="<hostname>-<n>.ncpr"`; 404 if none; 409 while recording |

Status JSON gains:

```json
"record": {"state": "idle" | "recording" | "done", "samples": 6000, "seconds": 30.0,
           "capacity_s": 120.0, "dropped": 0, "note": "ring-down 1"},
"excite": {"kind": "none" | "chirp" | "prbs" | "steps" | "sine", "active": false, "t": 0.0}
```

**Offline fit** (`n_cartpole/sysid/`, `scripts/sysid.py`): velocities come from
non-causal smoothing of the readings (no lag offline). The fit minimizes the error of
short simulated windows (e.g. 0.25 s, multiple shooting) of the dynamics in
`n_cartpole/env/dynamics.py` against the readings, over cart mass, viscous and Coulomb
rail friction, force gain, per-link first/second mass moments (a_k, d_k: COM and
inertia), joint friction, and the actuator delay (grid). It reports the fit, parameter
uncertainties and prediction-vs-measurement plots, and writes a config TOML for
`train.py --config` with the fitted nominal physics and randomization ranges narrowed
to the fit's uncertainty.
