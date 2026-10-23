/* heap_watch.h — where internal RAM goes: failed allocations and new low-water marks.
 *
 * heap_watch_init registers a failed-allocation callback that only counts and remembers
 * the last failure (it can run inside the Wi-Fi driver or an ISR, so it must not log or
 * lock). heap_watch_poll, called from the tlm task (core 0, 2 Hz), turns those into event
 * log entries: one per poll that saw failures, plus "internal RAM after start" once, 30 s
 * after boot, and a "new low" entry each time the lowest-since-boot free size falls another
 * 2 KB below the last one logged. The event's time, next to the other events (an upload,
 * a page load, Wi-Fi), says what pushed memory down. */
#pragma once

#include <stdint.h>

#include "cJSON.h"

void heap_watch_init(void);
void heap_watch_poll(void);
/* Adds "internal_largest" (largest free internal block, bytes) and "alloc_failed" (failed
 * allocations since boot, any heap) to the status "heap" object. */
void heap_watch_json(cJSON *heap);
