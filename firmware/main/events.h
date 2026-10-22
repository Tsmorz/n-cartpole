/* events.h — a short log of what happened on the device: boot (with the reset reason),
 * arming, faults, uploads, Wi-Fi, calibration. Core 0 only (the control task's changes
 * reach it through the tlm task's watch of the published status). Kept in RTC memory, so
 * it survives a crash, watchdog or software reset (not a power-on); GET /api/events
 * serves it and the status JSON carries "events_seq" so a page fetches it only when
 * something new happened. */
#pragma once

#include <stdint.h>

#include "cJSON.h"

typedef enum {
    EV_INFO = 0,
    EV_WARN,
    EV_ERROR,
} ev_level_t;

/* After settings_init (each entry records the boot it was logged on). Keeps the entries
 * from before the reset unless it was a power-on or the store fails its checks. */
void events_init(uint32_t boot);
void events_add(ev_level_t level, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
uint32_t events_seq(void); /* number of events added since boot */
/* {"seq": n, "boot": b, "uptime_ms": t, "events": [{"seq", "boot", "t_ms", "unix"?, "level", "msg"}, ...]}
 * oldest first; t_ms counts from the boot the entry was logged on; "unix" (s) only when the
 * clock had been set over SNTP. */
cJSON *events_json(void);
/* Unix time (s) once the clock has been set over SNTP, else 0 (the clock starts at 1970). */
uint32_t events_unix_now(void);
/* Why the chip last reset, in words ("power-on", "brownout (supply dipped)", ...). */
const char *events_reset_reason(void);
