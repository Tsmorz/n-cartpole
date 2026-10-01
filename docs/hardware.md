# Hardware build notes — double-pendulum cartpole

Living document. The simulator (`config/rig.toml`) is the source of truth for every
number here; if a sim parameter changes, update the requirement table and the bill of
materials below. Prices are **rough ballpark USD guesses, not quotes** — check current
listings. Part classes are *examples of what meets the spec*, not endorsements; verify
datasheets before buying.

## Status

| item | state |
|---|---|
| Design choice (link length / tips / drive) | **pending** — three candidates under RL swing-up verification (see below) |
| Sensing (encoder noise, velocity estimate) | angle noise ≤ 0.5 mrad. **Total added lag must stay ≤ ~20 ms** (user constraint): the 10 ms action delay uses half, so any velocity filtering may add ≤ ~10 ms (EMA α ≤ 0.5), or use a model-based observer instead — see "Findings" |
| Loop rate | 100 Hz, 1 control-step action delay assumed |
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

**The three designs are not distinguishable yet:** with noise, rail friction, 300 N/s slew and
actor smoothing all on, none of the 150k-step swing-up policies learned to swing up (they
idle at the bottom). The earlier idealised run did. An ablation (smoothing off / weak /
ideal sensing) is in progress to find which factor stalls learning; design choice waits on it.

Bar stock is the same for all: 6061-T6 aluminium flat bar **25 × 3 mm**.

## Requirements the build must meet (and why)

| requirement | sim value | why it matters |
|---|---|---|
| Rail usable travel (carriage centre) | ±0.40 m (1.0 m rail, 0.10 m carriage, 0.05 m bumper margin each end) | leaving it is a hard failure (end-stop hit) |
| Moving mass of the cart assembly | ≈ 0.55 kg carriage + plate + joint-1 encoder + belt clamp, **plus J/r² of the motor** | cart mass is the strongest lever on control effort; weigh it |
| **Motor + pulley rotor inertia** | ≤ 3e-5 kg·m² | reflects to the cart as J/r²; a NEMA17 stepper (≈ 5.4e-5) on a 20T pulley adds ≈ 1.3 kg and roughly doubles control effort |
| Peak belt force | ≈ 15 N available after derating | swing-up energy; transitions |
| Cart speed capability | torque must still be available at ~2–2.5 m/s (peak cart speed seen in sim); candidates give 2.7–4.0 m/s no-load | back-EMF removes force near no-load speed |
| Force slew | ≈ 300 N/s applied (3 N per 10 ms step) | the sim's actuator limit; the controller is trained against it |
| Command-to-force latency | ≤ 10 ms total (1 control step) | the sim models exactly 1 step of delay |
| **Total loop lag budget** | ≤ ~20 ms = action delay (10 ms) + any velocity-filter lag (≤ ~10 ms) | more than ~20 ms of lag is considered not controllable on this plant; an EMA with coefficient α adds ≈ α/(1−α) × 10 ms of lag (α = 0.5 → 10 ms, α = 0.8 → 40 ms) |
| Joint-angle encoders | ≥ 14-bit (≥ 16384 counts/rev), **noise ≤ ≈ 0.5 mrad RMS** | at 1.5 mrad the unfiltered UU hold starts failing (2/5); effort scales ~linearly with noise |
| Cart position resolution | ≤ 20 µm (≥ 4000 counts/rev at the pulley) | sim assumes 10 µm on a 20T pulley; 20 µm on a 40T |
| Control loop | 100 Hz, deterministic | tested: 200 Hz does not help hold quality with differenced velocities |
| Rail sliding friction | ≈ 0.4 N (small MGN12-class, light preload) | sets the floor on how quiet a hold can be; modelled and randomized ×0.5–2 |
| Joint friction | ≈ 0.002 N·m·s/rad (608-class bearings) | randomized ×0.5–2 in training |

## Bill of materials — what to buy

Quantities are for **one rig**. "Depends" marks lines that change with the design choice.

| # | item | spec / requirement | qty | rough cost (USD) | notes |
|---|---|---|---|---|---|
| 1 | Linear rail + carriage | MGN12 (or equivalent) **1.0 m** rail, 1 × MGN12H carriage; preload light | 1 | 25–45 | friction ≈ 0.4 N is assumed; heavy preload or dirty rail increases it |
| 2 | Rail mounting frame | 1.2 m aluminium extrusion (e.g. 2020) or flat stiff base, M3/M4 hardware | 1 | 20–40 | rail must be level and straight; stiffness matters |
| 3 | BLDC motor | frameless/servo-style ~42 mm BLDC with **rotor inertia ≤ 3e-5 kg·m²**, peak torque ≥ 0.15 N·m (ref) or ≥ 0.3 N·m (short/mid), ≥ 3000–4000 rpm at 24 V | 1 | 30–80 | **Depends** on candidate. Avoid steppers (rotor inertia too high, bad speed-torque) |
| 4 | Motor encoder | ≥ 4000 counts/rev (quadrature) on the motor shaft, or a separate linear encoder on the rail | 1 | 0–30 | many BLDC kits include one; resolution at the carriage = 2π r / counts |
| 5 | FOC motor driver | torque/current-mode FOC, current-loop ≥ 5 kHz, peak current ≥ torque / Kt with margin, 24 V, accepts commands at 100 Hz with ≤ 2–3 ms latency (SPI/UART/CAN/PWM) | 1 | 25–150 | latency and slew behaviour are part of the sim; measure on a scope |
| 6 | GT2 belt | 6 mm wide, ≈ 2.1–2.3 m loop (≈ 2 × rail + wrap) | 1 | 8–15 | keep tensioned; backlash/compliance limit bandwidth |
| 7 | GT2 drive pulley | **40T** (r = 12.7 mm, short/mid) or 20T (r = 6.35 mm, ref); bore to match motor shaft | 1 | 5–10 | **Depends**. Larger pulley ⇒ less reflected inertia, less belt force per torque |
| 8 | GT2 idler pulley + mount | matching tooth count, bearing-supported, at the far end of the rail | 1 | 5–10 | |
| 9 | Belt clamp / carriage plate | machined or printed plate joining belt and carriage; carries the joint-1 bearing + encoder | 1 | 10–25 | keep mass low; total moving mass target ≈ 0.55 kg |
| 10 | Link bar stock | 6061-T6 aluminium flat bar **25 × 3 mm**, ≈ 0.6 m total length (cut to 2 links) | 1 piece | 10–20 | link length **Depends** (0.15 / 0.20 / 0.25 m joint-to-joint) |
| 11 | Joint bearings | 608 (8 × 22 × 7 mm) deep-groove, light grease | 4 | 5–12 | 2 per joint; stiff bearings add joint friction |
| 12 | Joint axles / clamps | 8 mm shafts or bolts + clamp blocks, low play | 2 sets | 10–25 | play at the joints shows up as unmodelled backlash |
| 13 | Joint-angle encoders | **14-bit (or better) magnetic absolute encoder**, e.g. AS5048A/B class, with diametral magnet, SPI | 2 | 15–35 | noise ≤ 0.5 mrad is the requirement; mounting concentricity matters; verify noise on the bench |
| 14 | Slip ring (**decided**), low-friction, ≥ 6 wires | carries power + SPI (SCLK, MISO, MOSI, CS) + GND to the joint-2 encoder on link 1; choose the lowest-friction, lowest-noise unit that fits the joint-1 shaft/bore, gold contacts | 1 | 15–40 | joint 1 rotates continuously; check starting/friction torque on the datasheet — it adds to joint-1 friction (see Risks) |
| 15 | Tip weights / ballast | brass or steel masses + fasteners to make each tip assembly the design mass (0.05–0.08 kg) | as needed | 5–15 | tip mass **includes** bearing, clamp and encoder board — weigh the whole assembly |
| 16 | Microcontroller | ESP32-S3 dev board (dual-core 240 MHz) | 1 | 10–20 | runs the 100 Hz loop + the small actor network |
| 17 | Power supply | 24 V, ≥ 5 A | 1 | 20–35 | |
| 18 | Wiring / connectors / fuse | motor phase wire, encoder cable, 5–10 A fuse, connectors | 1 lot | 15–30 | |
| 19 | Emergency stop | physical mushroom e-stop cutting motor power | 1 | 8–15 | **buy this** — a fast cart plus a spinning pendulum is dangerous |
| 20 | End stops / bumpers | mechanical limit switches or rubber bumpers at both rail ends | 2 | 5–15 | the sim terminates at ±0.40 m; hardware needs soft + hard limits |
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
| Loop timing | toggle a pin each iteration; confirm 10 ms with low jitter |

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
- **Loop rate**: 200 Hz does not help (velocity noise ∝ 1/dt); stay at 100 Hz unless
  latency measurement says otherwise.
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
6. **Deployment** (not started): export the actor weights, port the observation encoding
   (including the velocity filter and goal input), validate against the Python outputs
   before closing the loop.

## Change log

- 2026-10-01 — First version. Candidates ref/short/mid; BOM drafted; sensing findings.
- 2026-10-01 — Ablation (60k steps, mid): realism alone still swings up (19/20) but flails (6.9 N RMS holds); smoothing 5 on an ideal plant swings up (18/20); smoothing + noise/friction together idles (even at weight 1). Fix in progress: smoothness penalty gated by goal alignment (+ optional ramp).
- 2026-10-01 — 150k-step swing-up verification of ref/short/mid: all three fail on the full-realism plant (4/20 reached, 0/20 held); ablation started.
- 2026-10-01 — Decision: slip ring for joint 1 (user). BOM line 14 and risk 1 updated.
- 2026-10-01 — Constraint (user): total loop lag must stay ≤ ~20 ms. Dropped the α = 0.8
  velocity-filter recommendation (≈ 40 ms lag). Lag-budget study done: EMA α ≤ 0.5 fits the
  budget (jitter −20 % at α = 0.3, −40 % at α = 0.5); model-based observer ~10× quieter in
  the frictionless ideal but not yet reliable (DU fails, stiction hurts).
