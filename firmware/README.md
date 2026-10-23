# n-cartpole firmware (ESP32-S3)

Runs a trained n-cartpole policy at the training control rate (200 Hz for the rig
config) on an ESP32-S3, either against the real rig or against a plant simulated on
the chip, and serves a dashboard with live telemetry, controls, recordings for
system identification and drag-and-drop updates of the firmware and the model.
Design, API and JSON shapes: [docs/design.md](docs/design.md).

**Board:** ESP32-S3 with 16 MB flash and 8 MB octal PSRAM (e.g. ESP32-S3-DevKitC-1
N16R8). **Framework:** ESP-IDF v5.3.1.

## Quick start

1. **Export a model** (from a checkpoint; models never ship with a firmware release):
   `task export-firmware -- --checkpoint checkpoints/double/tqc-goal/tqc_latest.pt`
   writes an `.ncpm` blob.
2. **Get a firmware image:** build it locally with `task fw-build` (Docker,
   `espressif/idf:v5.3.1`; output in `firmware/build/`), or download a release with
   `task fw-download -- fw-v0.2.0` (into `dist/firmware/`).
3. **Flash over USB** (the merged factory image):
   `task fw-flash -- /dev/cu.usbserial-XXXX [dist/firmware/n-cartpole-fw-0.2.0-factory.bin]`.
   Watch the log with `task fw-monitor -- /dev/cu.usbserial-XXXX`.
   This keeps the saved Wi-Fi network and the rig's zero references: the merged image is written
   in two pieces around the NVS partition (`firmware/tools/flash.py`), and the model partition
   lies beyond it. `--erase-settings` writes the whole image to forget them. Writing
   `factory.bin` yourself with `esptool write_flash 0x0` would erase NVS (the image pads it
   with 0xFF). Updating a board that is already set up is easier still: drag the app image onto
   the dashboard (OTA), or `esptool.py write_flash 0x20000 build/n_cartpole_fw.bin`.
4. **Connect:** with no Wi-Fi credentials (or if the saved network is not found within
   15 s) the device opens an access point `cartpole-XXXX` (open); browse to
   `http://192.168.4.1/` and enter your 2.4 GHz network under Wi-Fi. The device tries it
   with the access point still up and **saves it only if it joins** (no reboot); the page
   shows the result and the new address, or why it failed (wrong password, not found). The
   access point closes about 20 s after a successful join, so put your computer back on
   your network and open `http://cartpole.local/` (mDNS) or the IP shown (also on the
   serial console). The channel switch can drop the page early; the page then tells you
   the device probably joined.
5. **Load the model:** drag the `.ncpm` onto the dashboard (no reboot), or flash it over
   USB with `task fw-flash-model -- <port> model.ncpm` (partition `model` at 0x620000).
6. **Firmware updates:** drag `n-cartpole-fw-<ver>.bin` from a release onto the
   dashboard. The new image boots *pending*, marks itself valid after 10 s of healthy
   running (control loop ticking, HTTP up), and a crash before that rolls back to the
   previous slot.

Releases: `task fw-release -- fw-v0.2.0` tags and pushes; `.github/workflows/firmware.yml`
builds and attaches the OTA image, the factory image and `SHA256SUMS`. Develop the
dashboard without hardware against `task fw-mock`.

The dashboard has four tabs (URL hash `#live`, `#health`, `#record`, `#settings`; the last one
is remembered): **Live** (scene, controller, calibration, state readout: estimate / encoder
readings / true state, network), **Health** (loop timing and deadline misses per minute, cores and
tasks, memory, telemetry link: received fps and lost frames, the rig's sensors and drive: encoder
magnet diagnostics, bad reads, e-stop, drive, zero references; and the device's event log),
**Record** (system identification, with its own Arm and Manual/AI controls and a live trace of
the cart and the force) and **Settings** (Wi-Fi with rename and forget, firmware/model update,
system with the last reset reason, model details and the browser-vs-device check of the model).
The header shows the controller state on every tab (click it, or press Esc, to disarm). When the
connection drops or the device goes quiet, a banner says so and the live values fade. A tab shows
a dot when it needs attention (over budget, missed deadlines, a sensor problem, a new error in
the log, setup mode), and a device in setup mode opens Settings on first load. On a phone the
controls sit between the scene and the plot. Addresses (the IP, `<host>.local`, the ones in the event log
and in Wi-Fi messages) are links, `task …` commands copy on click, a released firmware version
links to its release notes, and hints that ask for something carry the button or link that
does it (Record's "switch to Manual" / "Arm" / "Disarm", "Zero on Live", "upload one").

The device's clock is set over the internet (SNTP) when it is on your Wi-Fi: event-log entries
and zero references then show the local date and time, otherwise time since boot. The Arm button
asks once per page session on the rig (it enables the motor drive), and Health keeps the last
5 minutes of loop timing next to the last 6 seconds.

`tests/test_api_schema.py` checks the mock (and, with `NCP_DEVICE=http://<ip>`, a real device)
against `docs/api-schema.json`. `tests/test_dashboard.py` runs the page against `firmware/tools/mock_device.py` (API checks,
and a Playwright/Chromium pass over every tab at desktop and phone width; `uv run playwright
install chromium` once). The mock pretends to be the rig with `--rig` (`--uncalibrated`: zero
references not set) and takes mock-only commands to show trouble: `{"inject_fault": "e-stop"}`,
`{"inject_io": "weak-magnet" | "bad-reads" | "ok"}`.

## Backends (menuconfig: n-cartpole → Plant backend)

`task fw-build-rig` builds the rig backend into `firmware/build-rig/` (`sdkconfig.rig` over
the defaults; set your pins with menuconfig first). CI compiles it on every push that touches
`firmware/`, next to the sim image it releases.

- **sim** (default): the plant runs on the chip (`ncp_sim`, the training plant with
  the model's encoder model). Reset it to any pose from the dashboard. If a model is
  too slow for its period, the loop stretches its period (slow motion; the status shows
  the real-time factor) instead of overrunning.
- **rig**: 2 × AS5048A (SPI mode 1, 1 MHz; joint 2 through the slip ring, measuring
  link 2 relative to link 1), a quadrature motor encoder (PCNT ×4) for the cart, and a
  PWM + DIR + EN force command to the motor driver. Pins, encoder directions, counts per
  revolution (4000), pulley radius (6.35 mm) and PWM full-scale force are under
  n-cartpole → Rig hardware. Before the first run: hang the links at rest and send
  *Zero angles*, centre the cart and send *Zero cart* (stored in NVS). The drive sits
  behind a small interface in `main/hal_rig.c` so a CAN/UART FOC driver can replace it.

## Recording for system identification

The dashboard's Record card logs every control tick (128 B/sample, PSRAM, up to
`NCP_RECORD_SECONDS` = 120 s) with optional excitation (chirp, PRBS, steps, sine;
capped at 10 N, armed only) on top of the Manual-mode PD or the policy. Download the
`.ncpr` (`GET /api/record`) and fit the plant with
`task sysid -- recordings/*.ncpr --config config/rig.toml --out config/rig-fitted.toml`.
Presets and the format: design.md "Recording and system identification".

## Safety

- Wire a **hardware e-stop that cuts motor power**; the e-stop sense input (active high)
  only tells the firmware. Add a pull-down on the drive EN line so the driver stays
  disabled while the chip resets or boots.
- The drive is enabled only while ARMED. FAULT (cart past the rail limit, e-stop,
  repeated encoder errors, missed deadlines on the rig) disables it until re-armed.
- Uploads (model or firmware) are refused while armed: flash writes stall both cores.
- The dashboard has no password: keep the device on a network you trust. Requests that
  another web site could make your browser send (a plain-text or form POST, or one from a
  different origin) are refused with 403, so a page you happen to have open cannot arm the
  rig or flash it; see design.md "Cross-site guard".
- Arming the rig is refused while the loop is over budget (> 80 % of the period), and until
  both zero references (angles and cart) are set.

## Cores and tasks

Core 1 runs only the control loop: `ctrl` (prio 22), woken by its own gptimer
interrupt (allocated from the task, so it lands on core 1); it never logs, allocates or
blocks on core 0. Core 0 runs Wi-Fi, lwIP, the event loop, esp_timer, `httpd` (prio 5:
API, uploads, WebSocket) and `tlm` (prio 4: telemetry ~25 msg/s, status at 2 Hz, logging
on behalf of core 1). Details: [docs/design.md](docs/design.md#cores-and-tasks).

## Troubleshooting

- **Dashboard says "no model":** upload an `.ncpm`; `/api/info` → `model_error` says why
  a stored one was rejected (CRC, version, shape).
- **Cannot find the device:** watch the serial console (`task fw-monitor`) for the IP;
  `.local` names need mDNS (Bonjour) on the client; fall back to the AP.
- **OTA rejected:** the image must be the app image `n-cartpole-fw-<ver>.bin` (not the
  `-factory.bin`), built as project `n_cartpole_fw`, and the controller must be disarmed.
- **Arm refused "not zeroed":** hang the links at rest and send *Zero angles*, centre the cart
  and send *Zero cart* (Live → Calibrate). Health → Sensors & drive shows both references.
- **Why did it reboot / trip?** Health → Event log (it survives crashes, watchdog resets and
  reboots, not a power-off: the entries before the reset are grouped under "before the
  reset") and Settings → System → last reset (a crash, watchdog or brownout is shown in red).
- **403 from the API (curl, a script):** send `Content-Type: application/json` (or
  `application/octet-stream` for uploads) and no foreign `Origin` header.
- **Encoder trouble:** Health → Sensors & drive: a magnet too far or too close, CORDIC overflow,
  and bad reads that keep rising.
- **`budget_ok: false`:** the network is too large for the period; use a smaller
  hidden size (the status shows policy/estimator µs per tick).
- **Boot loop after an update:** the bootloader rolls back to the previous slot after a
  crash before the image was marked valid; reflash the factory image over USB if both
  slots are bad.
