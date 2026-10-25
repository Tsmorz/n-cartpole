# Hardware build notes — double-pendulum cartpole

Living document. The simulator (`config/rig.toml`) is the source of truth for every
number here; if a sim parameter changes, update the requirement table and the bill of
materials below. Prices are **rough ballpark USD guesses, not quotes** — check current
listings. Part classes are *examples of what meets the spec*, not endorsements; verify
datasheets before buying.

## Status

| item | state |
|---|---|
| Design choice (link length / tips / drive) | **links: keep ref** (0.25 m, tips 0.08 / 0.05 kg; every trained policy uses it). **Drive: pick the motor first**, then the pulley that meets the drive requirements; both candidate drives work in sim (2026-10-25 study below) |
| Sensing (encoder noise, velocity estimate) | **firmware should run a model-based EKF** on the encoder readings (6-state, 200 Hz; fine for an ESP32-S3) and feed the policy its velocity estimates; unmodelled-acceleration setting ~0.5 rad/s^2 tested best. | angle noise ≤ 0.5 mrad. **Total added lag must stay ≤ ~20 ms** (user constraint): the 5 ms action delay (1 step at 200 Hz) uses a quarter, so any velocity filtering may add ≤ ~15 ms; the EKF adds no filter lag — see "Findings" |
| Loop rate | **200 Hz** (user decision, 2026-10-16), 1 control-step (5 ms) action delay assumed; velocity noise from differencing doubles vs 100 Hz, so the noise stage of the training curriculum matters |
| Joint-1 cabling | **decided: slip ring** (user, 2026-10-01) |
| Nothing ordered yet | — |

## Candidate designs (from `scripts/design_study.py`, configs in the session scratchpad)

| | **ref** | **short** | **mid** |
|---|---|---|---|
| Link length (joint-to-joint) | 0.25 m | 0.15 m | 0.20 m |
| Tip masses (link 1 / link 2) | 0.08 / 0.05 kg | 0.08 / 0.08 kg | 0.05 / 0.05 kg |
| Resulting link masses (bar + tip) | 0.131 / 0.101 kg | 0.110 / 0.110 kg | 0.091 / 0.091 kg |
| Motor class | small BLDC (J 1.5e-5, 0.15 N·m, 4000 rpm) | mid BLDC (J 3.0e-5, 0.30 N·m, 3000 rpm) | mid BLDC |
| Pulley | GT2 20T (r 6.35 mm) | GT2 40T (r 12.7 mm) | GT2 40T |
| Effective cart mass (incl. reflected rotor inertia) | 0.92 kg | 0.74 kg | 0.74 kg |
| Peak belt force (derated 0.65) | 15.4 N | 15.3 N | 15.3 N |
| No-load cart speed | 2.7 m/s | 4.0 m/s | 4.0 m/s |
| Swing-up after 150k RL steps, full-realism plant (reached upright / held 5 s, 20 seeds) | 4 / 0 | 4 / 0 | 4 / 0 |
| Fastest open-loop unstable mode UU / DU / UD (s⁻¹) | 10.4 / 7.5 / 7.9 | 13.0 / 9.6 / 10.3 | 11.7 / 8.5 / 8.9 |
| Swing-up work headroom (F · rail / energy range) | 11.1 | 17.3 | 16.9 |
| LQR hold force, full env (UU / UD / DU RMS, 6/6 held each) | 1.53 / 0.75 / 1.21 N | 1.33 / 0.64 / 1.02 N (40T) | ≈ 1.25 / 0.65 / 1.06 N (small motor, 40T) |
| Motor implied at 24 V (Kv, Kt, current for peak torque) | ~167 rpm/V, 57 mN·m/A, 2.6 A | ~125 rpm/V, 76 mN·m/A, 3.9 A | same as short |

The 150k-step row is from 2026-10-01 and was caused by the gated smoothing penalty (found later),
not by the designs. **2026-10-25 re-check** (current `rig.toml`: 200 Hz, ±0.50 m rail, EKF, delta mode):

- **Link geometry barely matters for holding**: all 145 screened designs balance, and LQR hold
  force across the shortlist varies only ~1.2–1.5 N RMS in UU. Shorter links have faster unstable
  modes (short: 13.0 s⁻¹ vs ref 10.4), which is harder with a 5 ms delay and a 1.5 N/step slew
  limit. Every trained policy (the deployed h48 student: 90 % of transitions at full realism)
  is on the ref links, so changing them means retraining for little gain. **Keep ref links.**
- **The speed limit is not what limits the deployed policy**: same student, same seeds, 20 trials
  per pair, full realism, only `v_noload` changed: 2.66 m/s (as trained) 89.7 %, 3.99 m/s
  (mid-drive speed) 88.4 %, no limit at all 89.7 %. Back-EMF cuts the force on 1.3 % of steps
  (−2.0 N when it does); the force limit binds on 0.3 %. The failures are transitions into DU,
  a policy weakness, not drive capability. The sim's force–speed line (full force at stall to 0 at
  no-load) is also pessimistic for a current-limited FOC drive, whose torque stays flat until
  the voltage limit.
- **So the drive is a sourcing choice.** Either candidate drive works; what must hold is
  `M_eff = 0.55 + J/r² ≲ 0.92 kg`, belt force after derating ≥ ~15 N, and `v_noload` ≥ ~2.7 m/s.
  The 40T + 0.3 N·m option has 20 % less effective mass and more speed margin but needs twice the
  torque. Low-Kv gimbal motors (e.g. GM4108, ~540 rpm no-load at 20 V, ~0.15 N·m) fail the speed
  requirement badly (< 1 m/s on a 40T pulley). A 45 mm-class 24 V BLDC datasheet (maxon EC 45 flat:
  181 g·cm² = 1.8e-5 kg·m²) confirms the assumed rotor inertia range is realistic.
- **Next step**: choose a real motor + FOC driver, put its datasheet J / torque / no-load rpm and
  your pulley into `[physics.drive]`, and fine-tune the student's teacher on that plant
  (`train.py --resume <ckpt> --config <rig with your drive>`) before distilling again.

Bar stock is the same for all: 6061-T6 aluminium flat bar **25 × 3 mm**.

## Requirements the build must meet (and why)

| requirement | sim value | why it matters |
|---|---|---|
| Rail usable travel (carriage centre) | **±0.50 m** (user decision 2026-10-19; ~1.2 m rail: 1.0 m travel + 0.10 m carriage + 0.05 m bumper margin each end) | leaving it is a hard failure (end-stop hit) |
| Moving mass of the cart assembly | ≈ 0.55 kg carriage + plate + joint-1 encoder + belt clamp, **plus J/r² of the motor** | cart mass is the strongest lever on control effort; weigh it |
| **Motor + pulley rotor inertia** | ≤ 3e-5 kg·m² | reflects to the cart as J/r²; a NEMA17 stepper (≈ 5.4e-5) on a 20T pulley adds ≈ 1.3 kg and roughly doubles control effort |
| Peak belt force | ≈ 15 N available after derating | swing-up energy; transitions |
| Cart speed capability | torque must still be available at ~2–2.5 m/s (peak cart speed seen in sim); candidates give 2.7–4.0 m/s no-load | back-EMF removes force near no-load speed |
| Force slew | ≈ 300 N/s applied (1.5 N per 5 ms step) | the sim's actuator limit; the controller is trained against it. `--action-mode delta` makes the policy's own command obey it by construction |
| Command-to-force latency | ≤ 5 ms total (1 control step at 200 Hz) | the sim models exactly 1 step of delay |
| **Total loop lag budget** | ≤ ~20 ms = action delay (5 ms at 200 Hz) + any velocity-filter lag (≤ ~15 ms) | more than ~20 ms of lag is considered not controllable on this plant; an EMA with coefficient α adds ≈ α/(1−α) × one control period of lag (at 200 Hz: α = 0.5 → 5 ms, α = 0.8 → 20 ms) |
| Joint-angle encoders | ≥ 14-bit (≥ 16384 counts/rev), **noise ≤ ≈ 0.5 mrad RMS** | at 1.5 mrad the unfiltered UU hold starts failing (2/5); effort scales ~linearly with noise |
| Cart position resolution | ≤ 20 µm (≥ 4000 counts/rev at the pulley) | sim assumes 10 µm on a 20T pulley; 20 µm on a 40T |
| Control loop | 200 Hz, deterministic, ≤ 5 ms jitter | chosen by the user; the earlier LQR screen showed 200 Hz does not by itself improve hold quality with differenced velocities (noise ∝ 1/dt), so encoder noise and velocity estimation matter more |
| Rail sliding friction | ≈ 0.4 N (small MGN12-class, light preload) | sets the floor on how quiet a hold can be; modelled and randomized ×0.5–2 |
| Joint friction | ≈ 0.002 N·m·s/rad (608-class bearings) | randomized ×0.5–2 in training |

## Bill of materials — what to buy

Quantities are for **one rig**. "Depends" marks lines that change with the design choice.

| # | item | spec / requirement | qty | rough cost (USD) | notes |
|---|---|---|---|---|---|
| 1 | Linear rail + carriage | MGN12 (or equivalent) **~1.2 m** rail, 1 × MGN12H carriage; preload light | 1 | 30–60 | friction ≈ 0.4 N is assumed; heavy preload or dirty rail increases it |
| 2 | Rail mounting frame | ~1.4 m aluminium extrusion (e.g. 2020/2040) or flat stiff base, M3/M4 hardware | 1 | 25–50 | rail must be level and straight; stiffness matters |
| 3 | BLDC motor | frameless/servo-style ~42 mm BLDC with **rotor inertia ≤ 3e-5 kg·m²**, peak torque ≥ 0.15 N·m (ref) or ≥ 0.3 N·m (short/mid), ≥ 3000–4000 rpm at 24 V | 1 | 30–80 | **Depends** on candidate. Avoid steppers (rotor inertia too high, bad speed-torque) |
| 4 | Motor encoder | ≥ 4000 counts/rev (quadrature) on the motor shaft, or a separate linear encoder on the rail | 1 | 0–30 | many BLDC kits include one; resolution at the carriage = 2π r / counts |
| 5 | FOC motor driver | torque/current-mode FOC, current-loop ≥ 5 kHz, peak current ≥ torque / Kt with margin, 24 V, accepts commands at 200 Hz with ≤ 2–3 ms latency (SPI/UART/CAN/PWM) | 1 | 25–150 | latency and slew behaviour are part of the sim; measure on a scope |
| 6 | GT2 belt | 6 mm wide, ≈ 2.6–2.8 m loop (≈ 2 × rail + wrap) | 1 | 10–20 | keep tensioned; backlash/compliance limit bandwidth |
| 7 | GT2 drive pulley | **40T** (r = 12.7 mm, short/mid) or 20T (r = 6.35 mm, ref); bore to match motor shaft | 1 | 5–10 | **Depends**. Larger pulley ⇒ less reflected inertia, less belt force per torque |
| 8 | GT2 idler pulley + mount | matching tooth count, bearing-supported, at the far end of the rail | 1 | 5–10 | |
| 9 | Belt clamp / carriage plate | machined or printed plate joining belt and carriage; carries the joint-1 bearing + encoder | 1 | 10–25 | keep mass low; total moving mass target ≈ 0.55 kg |
| 10 | Link bar stock | 6061-T6 aluminium flat bar **25 × 3 mm**, ≈ 0.6 m total length (cut to 2 links) | 1 piece | 10–20 | link length **Depends** (0.15 / 0.20 / 0.25 m joint-to-joint) |
| 11 | Joint bearings | 608 (8 × 22 × 7 mm) deep-groove, light grease | 4 | 5–12 | 2 per joint; stiff bearings add joint friction |
| 12 | Joint axles / clamps | 8 mm shafts or bolts + clamp blocks, low play | 2 sets | 10–25 | play at the joints shows up as unmodelled backlash |
| 13 | Joint-angle encoders | **14-bit (or better) magnetic absolute encoder**, e.g. AS5048A/B class, with diametral magnet, SPI | 2 | 15–35 | noise ≤ 0.5 mrad is the requirement; mounting concentricity matters; verify noise on the bench |
| 14 | Slip ring (**decided**), low-friction, ≥ 6 wires | carries power + SPI (SCLK, MISO, MOSI, CS) + GND to the joint-2 encoder on link 1; choose the lowest-friction, lowest-noise unit that fits the joint-1 shaft/bore, gold contacts | 1 | 15–40 | joint 1 rotates continuously; check starting/friction torque on the datasheet — it adds to joint-1 friction (see Risks) |
| 15 | Tip weights / ballast | brass or steel masses + fasteners to make each tip assembly the design mass (0.05–0.08 kg) | as needed | 5–15 | tip mass **includes** bearing, clamp and encoder board — weigh the whole assembly |
| 16 | Microcontroller | ESP32-S3 dev board (dual-core 240 MHz) | 1 | 10–20 | runs the 200 Hz loop + the distilled h48×2 actor (0.75 ms/inference measured) + EKF (0.45 ms) |
| 17 | Power supply | 24 V, ≥ 5 A | 1 | 20–35 | |
| 18 | Wiring / connectors / fuse | motor phase wire, encoder cable, 5–10 A fuse, connectors | 1 lot | 15–30 | |
| 19 | Emergency stop | physical mushroom e-stop cutting motor power | 1 | 8–15 | **buy this** — a fast cart plus a spinning pendulum is dangerous |
| 20 | End stops / bumpers | mechanical limit switches or rubber bumpers at both rail ends | 2 | 5–15 | the sim terminates at ±0.50 m; hardware needs soft + hard limits |
| 21 | Measurement tools | kitchen scale (±1 g or better), calipers, steel rule, spring/fish scale, stopwatch | 1 lot | 20–40 | needed for the parameter checks below |
| 22 | USB-serial / programming cable, logic analyzer (optional) | | 1 | 10–25 | for latency and noise measurement |

Ballpark total: **≈ $250–600** depending on motor/driver choice (lines 3–5 dominate). Not
included: 3D printing/machining, shipping, tools you already own.

## Checks to run on the built rig (sim parameter ↔ what to measure)

| sim parameter | how to measure |
|---|---|
| Link mass, tip mass | weigh each assembly; compare to `config/rig.toml` (bar ≈ 0.0506 kg per 0.25 m) |
| Link COM | balance the link on a knife-edge; distance from the joint axis |
| Link inertia | small-angle pendulum period about the joint; solve for I |
| Joint friction | free ring-down of each pendulum, fit the exponential decay |
| Cart friction (viscous + Coulomb) | drive at constant slow speed, record force vs speed; or pull with a spring scale |
| Cart effective mass | apply a known force step, measure acceleration |
| Peak force and force-vs-speed | spring scale at stall; log force/current vs cart speed up to ~2.5 m/s |
| Actuator latency and slew | scope on the command pin vs motor current / force |
| Encoder noise and resolution | log both joint encoders and the cart position with everything at rest; compute RMS and step size |
| Loop timing | toggle a pin each iteration; confirm 5 ms with low jitter (the dashboard's loop stats show exec time and misses) |

Then update `config/rig.toml` (`[physics.drive]`, `[physics.links]`, `[hardware.sensors]`,
`cart_coulomb`) with the measured values and re-train — the policy is trained on the
nominal model plus randomization, so the closer the nominal is to the truth, the better.

## Findings from simulation so far (hardware-relevant)

- **Encoder noise is the biggest lever.** Noise-driven hold force scales ≈ linearly with
  angle noise; at 1.5 mrad an unfiltered UU hold starts failing.
- **A heavy velocity filter is NOT an option.** An EMA with α = 0.8 cut force jitter 4–5×
  at 0.5 mrad, but adds ≈ 40 ms of lag, which exceeds the ~20 ms total lag budget. Superseded
  by the lag-budget study below; `rig.toml` keeps `velocity_filter = 0.0`.
- **Lag-budget study (LQR holds, ref design, 100 Hz, 0.5 mrad, 5 seeds).** Within the
  20 ms budget: EMA α = 0.3 (≈ 4 ms lag) cuts force jitter ≈ 20 %, α = 0.5 (≈ 10 ms lag,
  total lag exactly 20 ms) cuts it ≈ 40 % (UU 1.84 → 1.01 N/step) with all holds intact.
- **Model-based observer (Kalman on the linear model, positions only, no filter lag)** is
  promising but unfinished. Without rail friction/randomization it holds UU/UD at 0.14/0.06 N
  RMS (vs 1.45/0.52 N with differenced velocity — ~10× quieter). With realistic rail friction
  it degrades (UU 0.69 N, holds 4/5; UD 3/5), and DU does not hold in my implementation even
  without friction (cause undiagnosed). Treat as a firmware option to develop later, not a
  result to rely on yet.
- **Cart mass is mostly set by the motor**, via reflected rotor inertia — choose a
  low-inertia motor and a larger pulley. Link geometry only moves hold effort ~15–20 %.
- **Loop rate** (superseded): the early LQR screen found 200 Hz did not help with differenced
  velocities (noise ∝ 1/dt); the user chose 200 Hz anyway (2026-10-16) and the EKF removes the
  differencing noise, so the rig runs at 200 Hz.
- **A quiet hold is bounded below** by rail friction (≈ 0.4 N) and encoder noise; expect
  roughly 0.4–1.5 N RMS during holds on real hardware, not ~0 N.
- The old simulator (no delay, no sensors, no friction) gave "0 N holds"; the policy
  learned a ±13 N, 11 Hz dither once a 1-step delay was added. The fix was a realistic
  slew limit plus CAPS smoothness regularization on the actor (`--smooth-temporal/-spatial`).

## Risks and open items

1. **Slip ring on joint 1 (decided).** Link 1 rotates continuously relative to the cart,
   so the joint-2 encoder wiring goes through a slip ring. Open details: (a) its friction
   torque adds to joint-1 friction and is not modelled — measure it by ring-down once built
   and update `joint_friction[0]`; (b) slip-ring contact noise on the SPI lines can corrupt
   encoder reads — keep the SPI clock modest, add error/CRC handling and a bounds check on
   each reading, and keep the wires short; (c) choose a unit whose bore/shaft fits your
   joint-1 axle, or mount it concentric with a short coupling.
2. **Motor choice is the biggest unknown.** Rotor inertia and torque-speed behaviour were
   assumed (generic classes). Check the real datasheet and update `[physics.drive]`.
3. **Cart encoder resolution changes with the pulley.** The sim's 10 µm assumes a 20T
   pulley; with 40T and 4000 counts/rev it is ≈ 20 µm — update `x_resolution`.
4. **Not modelled yet:** belt compliance/backlash, motor cogging, joint backlash, driver
   current-loop dynamics beyond a slew limit, encoder non-linearity/eccentricity, power
   supply sag. Any of these can limit bandwidth.
5. **Safety**: e-stop, hard end stops and a software force/position limit before first
   power-up; start with reduced force limits and the pendulum constrained.
6. **Deployment** (firmware built and running on an ESP32-S3 N16R8 with the sim backend, 2026-10-25):
   `firmware/` — see `firmware/README.md`. The distilled h48×2 student runs at 0.75 ms per inference
   with the weights in internal RAM and the single-precision EKF at 0.45 ms, ~1.3 ms of the 5 ms tick.
   The rig backend (`task fw-build-rig`) compiles but has not run against hardware: encoders, drive
   and e-stop are untested.

## Change log

- 2026-10-25 — Design re-check on the current rig config. Design screen (145 designs, LQR, full-env holds): all balance, hold effort differs ≤ ~20 %. Speed-limit probe on the deployed h48 student (20 trials/pair, full realism, seed 100): v_noload 2.66 / 3.99 / unlimited → 89.7 / 88.4 / 89.7 % — the back-EMF limit is not binding. Decision basis: keep the ref links; the drive is chosen by motor availability against the drive requirements (see Candidate designs). Stale 100 Hz references and the deployment risk updated (firmware runs on the ESP32-S3; ~1.3 ms of the 5 ms tick).
- 2026-10-22 — RL fine-tuning of the h48 student against the teacher's critics (`--actor-from`, lr 1e-4, 2.0M → 2.3M; one crash at 2.05M during a local firmware build, resumed) did not beat plain distillation: full realism, 20 trials/pair, seed 100: 87 % (2.1M) → 73 % (2.2M, UU collapsed to ~20 % while DU rose to 80–90 %) → 92 % (2.3M); re-checked on seed 300: fine-tuned 2.3M 88 % (15 rail exits) vs distilled student 92 % (4). Over both seeds: distilled 91 %, fine-tuned 90 %. **Deploy the distilled h48 student** (`dist/firmware/tqc-rail05-goals-ft-weighted-h48b2-v2_2000000.ncpm`). The oscillation between goals is the familiar fine-tuning instability; if DU needs work, add a behaviour-cloning anchor to the actor loss (stay near the teacher while improving).
- 2026-10-22 — Deployable student found. Distillation recipe v2 (constant LR, last round cosine-annealed, 15k Adam steps/round, 8 DAgger rounds × 80k steps, 400k teacher-buffer observations): full realism, 20 trials per pair — **h48×2 90 %** (38,305 params, 150 KB f32, fits internal SRAM, est. 0.47 ms/inference on the S3 = 9 % of the period), h64×2 88 % (67k params, 270 KB: too big for internal SRAM). Teacher 92 %; the student keeps the teacher's profile (UU 85–100 %, into DU 65–70 %). In the C firmware core's closed-loop sim (nominal plant + encoder noise) the h48 blob succeeds 160/160. Exported as `dist/firmware/tqc-rail05-goals-ft-weighted-h48b2-v2_2000000.ncpm`. Started RL fine-tuning of the h48 student against the teacher's critics (`--actor-from`, lr 1e-4, 2.0M → 2.3M) to work on DU.
- 2026-10-22 — Firmware + deployment tooling (user request). ESP32-S3 (N16R8) firmware under `firmware/`: control loop alone on core 1 (own gptimer ISR), Wi-Fi/httpd/telemetry on core 0; sim and rig backends (2 × AS5048A SPI, PCNT cart encoder, PWM/DIR/EN drive behind an interface since the FOC driver is not chosen); dashboard in the personal site's style with the live network diagram, goal buttons + Random, cart slider (x_ref), task/core table; OTA (two slots + rollback) and model upload by drag and drop; GitHub release on `fw-v*` tags. Recordings (`.ncpr`, 200 Hz, excitation presets: ring-down, chirp, PRBS, steps) + offline sysid (`task sysid`) → fitted config → `train.py --resume --config`. C core parity vs Python: policy 2.4e-5 N, dynamics 1e-9, EKF 1e-8, controller obs 8e-7 / command 1.2e-5 N; C closed loop with the 2M policy (nominal plant + encoder noise) 159/160. The 2M actor (1.06M params) does not fit 200 Hz on the S3 → distillation: first students at full realism (20 trials) h64×2 76 %, h32×2 64 %, h64×1 56 % (teacher 92 %); a longer constant-LR recipe and RL fine-tuning of the student against the teacher's critics (`--actor-from`) are next. Velocity estimator swap on the 2M policy (zero-shot, full realism): EKF 92 % / 0.52 N per-step force change, differencing 93 % / 1.09 N, EMA 0.3 94 % / 0.96 N — the EKF halves the jitter; DU was better with differencing (80–90 % vs 65–70 %), possibly EKF model error under randomization.

- 2026-10-22 — Goal-weighted fine-tune result (`tqc-rail05-goals-ft-weighted`, 1.6M -> 2M at full realism, lr 3e-5): full-realism transition success (20 trials per pair, 10 s, seed 100) 83% -> 88% (1.7M) -> 90% (1.8M) -> 92% (1.9M) -> **92% (2M)**; clean plant 100% (all 16 pairs, 0 rail exits). Into UU from every start 95-100% at 2M (DD->UU swing-up 50% -> 95%); rail exits 8 -> 3 of 320. Still weak: transitions INTO DU (UU->DU 65%, UD->DU 70%, DD->DU 70%), barely moved by the weights because the DU column was already weighted and UD is rarely visited; next lever would be reset-start weights (start at UU/UD/DD, command DU) rather than more goal weight. DD->UU median settle rose 1.8 s (1.7M) -> 3.1 s (2M). Holds at full realism: RMS force ~1.5 N (UU), ~1.1 N (DU), ~0.65 N (UD), ~0.2 N (DD). Hold-start perturbation barely grew (0.05 -> 0.07 rad): hold resets are rare at 30 s episodes. The 2M checkpoint is the best full-realism policy so far.
- 2026-10-22 — Goal sampling was not targeting the weak transitions: the adaptive curriculum's EMA was not saved in checkpoints (every resume restarted it, and the hold perturbations fell back to 0.05 rad), it adapts too slowly at ~75 segments per 100k steps, and it starves swing-ups (it down-weights DD as a goal because DD always succeeds, and the next goal is drawn from the goal just reached: DD->UU fell from 6.2% of segments when uniform to 5.6%). Added fixed `--goal-weights` (replace the curriculum) and curriculum persistence. Restarted the full-realism fine-tune from the 1.6M checkpoint as `tqc-rail05-goals-ft-weighted` (to 2M, lr 3e-5) with weights that keep DD as a launch point and favour the failing transitions: long-run segment mix DD->UU 13.4% (was 6.2%), UU->DU 13.7%, DU->UU 13.0%, DD->DU 8.9%, self-holds ~2% each (the 1.6M policy at full realism: into UU 50-95%, into DU 60-95%, DD and UD 85-100%). Baseline to beat: 83% (1.6M, 20 trials, seed 100).
- 2026-10-22 — Probe fine-tune of the clean 1.5M policy at full realism (`tqc-rail05-goals-ft-full`: resumed with the clean replay buffer, lr 3e-5, 25k actor warm-up, +100k steps) did NOT collapse, unlike every earlier fine-tune, but gained little. Full realism, 20 trials per (start, goal), 10 s, seed 100: clean 1.5M zero-shot 78%; fine-tune 1.6M 83% (~1.5 SE, may be noise), faster settling (DD->UU median 3.4 -> 1.8 s), same 8/320 rail exits; on the clean plant the fine-tune keeps 99% (clean 1.5M: 97%). The full-trained 1.5M policy scores the same overall (77%) with the opposite failure profile: 90-100% into DU/UD but 0-40% into UU, and 16/20 DD->UU trials leave the rail; the clean-trained policies reach UU 50-100% but are weaker into DU (40-70%). CORRECTION to the 2026-10-21 entry: its 5-trial numbers were too optimistic (clean zero-shot at full realism is 78%, not 89%; the fine-tune 5-trial evals of 95-99% were also luck) — use >= 20 trials per pair for these comparisons.
- 2026-10-21 — Rail-±0.50 m goal-conditioned runs finished (1.5M steps each, same nominal rig physics, only `realism` differs). At their own training plant (5 trials, 10 s): `tqc-rail05-goals-clean` 98% at realism 0 (70% @300k -> 99% @950k; holds quieted after 950k: UU hold RMS force ~3 -> 0.3 N); `tqc-rail05-goals-full` 78% (flat since 950k), never learns to hold UU. Realism components one at a time (5 trials; see the 10-22 correction for full realism): the clean policy is barely affected by rail friction (99%) or encoder noise (98%); plant randomization is the component that costs (92%), mostly transitions into DU. So the nominal rig parameters are feasible — the clean policy does every transition within the force/rail/speed limits — and the gap at full realism is about learning/robustness, not wrong physics. Margins are thin: most transitions hit the 15.4 N force limit and peak cart speed reaches ~2.0-2.2 m/s against the 2.66 m/s no-load limit.
- 2026-10-19 — Decision (user): rail travel set to ±0.50 m (~1.2 m rail), reversing the brief ±1.00 m change. `config/rig.toml` x_lim = 0.50; BOM lines 1, 2, 6 and 20 updated (rail ~1.2 m, frame ~1.4 m, belt ~2.7 m). The two goal-conditioned long runs were restarted on this rail: `tqc-rail05-goals-clean` (clean plant) and `tqc-rail05-goals-full` (full realism from scratch).
- 2026-10-19 — Decision (user): rail travel increased to ±1.00 m (~2.2 m rail). `config/rig.toml` x_lim 0.40 -> 1.00; BOM lines 1, 2, 6 and 20 updated (rail ~2.2 m, frame ~2.4 m, belt ~4.6 m). Speed matters more now: the back-EMF limit (2.66 m/s no-load for the shipped small-BLDC/20T drive) is closer to binding with 2 m of run-up; the mid-BLDC/40T drive from the design study gives 4.0 m/s. Not modelled: belt stretch over a 4.6 m belt, deflection of a 2 m rail. The earlier long runs (rail +-0.40 m) were stopped and replaced by two goal-conditioned runs on the new rail (all UU/DU/UD/DD transitions): `tqc-rail1-goals-clean` (clean plant) and `tqc-rail1-goals-full` (full realism from scratch). End-to-end test added for the goal + delta action + EKF + force-history recipe.
- 2026-10-19 — Why the personal-site demo looks effortless: it is the goal-conditioned point-mass `tqc-goal` policy (375k steps) on an idealized plant (20 N, +-0.5 m rail, 800 N/s slew, no delay/noise/friction). Adding rig effects one at a time (cumulative) to that exact policy, DD->UU: idealized 20/20 in 2.6 s; + 1-step delay 20/20; + rig drive (15.4 N, +-0.40 m) 7/20 with 13 rail exits; + speed limit and everything after 0/20. It tolerates delay but not the weaker, shorter-travel drive: it cannot run on the buildable rig. The current rig-trained policy swings up in a similar time on the clean rig plant (median first upright ~2.4-2.7 s) but its reliability under full realism is ~75-80%. NOTE: the rig work so far covers swing-up (DD->UU) and holding only, not the full UU/DU/UD/DD transition set.
- 2026-10-19 — Longer episodes do not rescue the failures (675k + EKF, full realism, 40 episodes x 30 s): 75% caught by 8 s and unchanged to 15 s, 80% by 20 s, then flat; 32/40 hold at the end. 6/40 left the rail (5 of them by 8 s, i.e. in the first swing-up), 2/40 never caught. No catch was ever lost. So the remaining ~20% failure is the initial swing-up/catch under realism, not slowness; a retry window helps little.
- 2026-10-19 — Critic warm-up result: while the actor is frozen the policy keeps its quality (13-17/20 holds, ~0.65 N/step), and when the actor starts updating it degrades within ~25-50k steps (50k warm-up run: 17 -> 13 -> 8 -> 3/20 after unfreezing). So fine-tuning a balanced policy onto a new observation pipeline/plant does not work with this learner (ramp, lower lr, fresh buffer and warm-up all tried). BEST CANDIDATE: the clean-plant delta-mode policy (checkpoints 525k-700k of `tqc-diag-delta-nosmooth`) with the EKF (ekf_accel_std 0.5) in the observation path, unmodified: at full realism 13-17/20 holds across checkpoints (675k: 30/40 = 75%), hold RMS force ~0.9-1.0 N, applied force change 0.37-0.59 N per 5 ms step, commands never above 1.5 N/step, 20/20 at realism 0.5. Failure analysis (675k, 40 episodes): once caught it never lost the balance; the 10 failures are swing-up/catch problems (6 left the rail, 4 never caught). A longer rail does not help (30 -> 32-33/40). If the real rig's friction/noise/model error are within ~half the assumed ranges, expect ~100% holds, so build and measure before more tuning.
- 2026-10-19 — EKF-in-the-loop fine-tunes of the 700k policy at full realism did NOT improve on the zero-shot EKF result (15/20 holds, 0.59 N/step jitter): lr 3e-5 wobbled 12 -> 1 -> 0 -> 6 -> 7 -> 6 /20, lr 1e-4 stayed near 0/20 until the last checkpoint. Same failure pattern as the realism ramp: updating the actor against a critic that has not fit the new observations damages a good policy. Added a critic warm-up (actor frozen for N steps) and started 50k / 100k warm-up runs.
- 2026-10-19 — Stage-1 realism ramp (0 -> 1 over 400k steps) collapsed balancing in both variants: ramp-only held until realism ~0.25 then failed (0/16 at 0.31); the version with a late-phased gated smoothing penalty collapsed immediately (third time that penalty breaks balance). The unmodified 700k clean-plant policy holds 20/20 at realism 0.5 and 15/20 at full realism but jitters the force ~1.2 N per 5 ms step. Encoder noise alone is not the cause (0.5 -> 0.08 mrad only drops jitter 1.23 -> 0.81 N/step; quantization of the differenced velocity and rail friction matter). A light EMA velocity filter (alpha 0.3, 2 ms lag) is a small win (17/20, 1.09 N/step), heavier ones trade holds for jitter. A model-based EKF in the observation path (accel_std 0.5) halves the jitter (0.59 N/step) at the same hold rate, zero-shot. EKF-in-the-loop fine-tunes started (lr 3e-5 / 1e-4, full realism, fresh buffer).
- 2026-10-18 — STAGE 0 DONE (clean plant, 200 Hz, delta action mode, smoothing penalty off, 300 N/s slew, 5 ms delay): 700k-step policy holds upright for a full 5 s in 20/20 episodes (RMS force 0.28 N, applied force change 0.006 N per 5 ms step, commands never above the 1.5 N/step limit, 0 off-rail). Holds first appeared at 525k (19/20). Zero-shot at higher realism the same policy holds 20/20 at 0.25 and 0.5 and 15/20 at full realism, but its force jitters ~1.2 N per step there (noise-driven). Stage 1 started: fine-tune from this policy while the realism dial ramps 0 -> 1 over 400k steps, with and without a late-ramped gated smoothing penalty.
- 2026-10-18 — FIRST BALANCING (clean plant, 200 Hz, delta action mode, smoothing penalty off, 300 N/s slew): at 425k steps 8/30 episodes hold upright the full 5 s (longest stay 13 s). In those holds the applied force changes 0.085 N per 5 ms step, RMS force 1.6 N, links within ~6 deg. Commands never exceed the 1.5 N/step rate limit. Run continues to 700k; then add realism back in stages (rail friction -> encoder noise -> randomization).
- 2026-10-18 — Best recipe so far (clean plant, 200 Hz): delta action mode, smoothing penalty OFF, realistic 300 N/s slew, +100k steps from the stage-0 delta policy -> upright 41% of last 5 s (was 5%), 13/16 episodes with an upright stay > 1 s, hold RMS force 2.2 N, commanded force never changes > 1.48 N per 5 ms step. Still no 5 s hold. 600 N/s was worse; absolute mode catches similarly at 800 N/s but commands 29 N single-step jumps. Continuing this run to 700k steps.
- 2026-10-18 — Stage-0 (clean plant, 200 Hz) 300k-step runs: swing-up solved (20/20) but no balancing in either action mode. Diagnostic fine-tunes: the gated smoothing penalty is what blocks catching (turning it off: 7 upright stays > 1 s vs 0; with slew 800 also: 14, longest 2.8 s); slew 800 alone does nothing. Delta action mode already guarantees smooth commands (worst step 1.5 N), so the penalty is redundant there. Next: delta mode, smoothing off, slew 300 vs 600. NOTE: runs stall when the Mac sleeps (lid closed on battery defeats caffeinate) - keep it plugged in with the lid open.
- 2026-10-16 — Direction change (user): get the controller working on a clean plant first, then add realism back in stages; move to 200 Hz; make controller inputs smooth enough for the actuator. Implemented a `realism` dial (rail friction + encoder noise/quantization + plant randomization, 0..1; delay/slew/drive limits stay), time-consistent discount and windows for 200 Hz, and an optional `delta` action mode (the policy commands force change within the slew limit). Observation extras (past forces/frames) did not help at full realism (final 200k: 0/30 held, upright 1-3% of last 5 s vs 5%). Stage 0 (clean plant, 200 Hz) runs started: absolute vs delta action.
- 2026-10-16 — Added optional observation extras (past commanded forces, past measurement frames): the firmware must keep and feed the same history as the trained policy expects. Fixed a sensing bug that corrupted the differenced velocity on goal-switch steps (goal-conditioned + sensor-model runs only).
- 2026-10-01 — First version. Candidates ref/short/mid; BOM drafted; sensing findings.
- 2026-10-15 — Scratchpad (temp) was cleared: all experiment checkpoints/logs/configs from Oct 1 lost (code + this log survive, committed). Experiments now save under checkpoints/double/. Started a pink- vs white-exploration-noise A/B (200k steps, ref design, full realism, gated+ramped smoothing).
- 2026-10-01 — Gated smoothing (alignment^2) fixes the swing-up stall: after 100k steps all runs swing up (13–19/20); none balance yet (holds still 3.6–5.6 N RMS). Continuing the two best (ramped-in w5, gated w20) to 300k.
- 2026-10-01 — Ablation (60k steps, mid): realism alone still swings up (19/20) but flails (6.9 N RMS holds); smoothing 5 on an ideal plant swings up (18/20); smoothing + noise/friction together idles (even at weight 1). Fix in progress: smoothness penalty gated by goal alignment (+ optional ramp).
- 2026-10-01 — 150k-step swing-up verification of ref/short/mid: all three fail on the full-realism plant (4/20 reached, 0/20 held); ablation started.
- 2026-10-01 — Decision: slip ring for joint 1 (user). BOM line 14 and risk 1 updated.
- 2026-10-01 — Constraint (user): total loop lag must stay ≤ ~20 ms. Dropped the α = 0.8
  velocity-filter recommendation (≈ 40 ms lag). Lag-budget study done: EMA α ≤ 0.5 fits the
  budget (jitter −20 % at α = 0.3, −40 % at α = 0.5); model-based observer ~10× quieter in
  the frictionless ideal but not yet reliable (DU fails, stiction hurts).
- 2026-10-22 — LQR hold study (user request; `n_cartpole/control/`, `scripts/lqr_study.py`, `tests/test_lqr.py`).
  Question: hold the unstable equilibria with LQR and keep the RL policy for swing-up/transitions.
  Design: linearise `dynamics._ode` at each goal (rail Coulomb friction left out: its regularised slope is a
  fictitious 80 N·s/m at rest), ZOH at 5 ms, state augmented with the commands in flight (delay 1) and the
  input = change of commanded force per step (matches `action_mode="delta"`, penalised so holds are smooth).
  All four goals are controllable (rank 7/7) and stable. Open-loop unstable poles: UU 5.4/10.4 s⁻¹, DU 7.5, UD 7.9.
  The default gains saturate the 300 N/s (1.5 N/step) slew limit for any angle error above ~0.01 rad, so the
  catch region is small; a random search over the six weights (45 sets, per goal) widened it only modestly
  (fresh seeds, 40 episodes per cell, perturbation ±s rad / ±2s rad/s: UU 72/55/40 % vs 57/40/32 % at
  s = 0.10/0.15/0.20, DU 98/98/60 vs 95/82/45, UD 100 % either way) at a cost of 0.9 vs 0.75 N RMS hold force in UU.
  Tuned weights: `config/lqr_weights.json`. Catch level (cost-to-go z'Pz, ≥ 95 % caught, randomized plant, EKF,
  delay): 1280 for every goal on a doubling grid (≈ ±6° per link in UU, ±7–12° DU, ±17–34° UD).
  Hybrid (policy drives; LQR from level < 640 back to the policy above 3840; UU/DU/UD only), 20 trials per
  (start, goal), 8 s, the h48×2 RL student, full realism, same seeds: overall 89.7 % -> 92.5 % (+9 of 320 trials),
  no cell worse, 6 better (DU->UU 70 -> 85, DD->DU 85 -> 95, UU->UU 90 -> 95, ...). The remaining failures are the
  policy's swing-up failures (identical off-rail counts). Hold RMS force: UU 2.15 -> 0.88 N, DU 1.28 -> 0.58 N,
  UD 0.61 -> 0.49 N; post-settle falls out of DU (max deviation 3.14 rad in the policy-only run) disappear
  (0.15-0.24 rad). Caveats: one checkpoint and one randomization setting; engage/release fractions not tuned;
  weights tuned on the same plant distribution (other seeds); no rig data yet. Not ported to the firmware.
