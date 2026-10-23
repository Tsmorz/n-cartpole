/* heap_watch.c — failed allocations and internal-RAM low-water marks into the event log
 * (see heap_watch.h). */
#include "heap_watch.h"

#include <stdatomic.h>
#include <stddef.h>

#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "events.h"

#define BASELINE_US (30 * 1000000LL) /* boot, Wi-Fi join and the model load are done by then */
#define LOW_STEP 2048                /* log a new low only when it is this much lower */
#define LOW_WARN (16 * 1024)         /* below this, Wi-Fi and HTTP allocations start to fail */

static atomic_uint s_failed;
/* The last failure, written by the callback without a lock: a diagnostic, not a record. */
static volatile size_t s_fail_size;
static volatile uint32_t s_fail_caps;
static const char *volatile s_fail_fn;

static unsigned s_failed_logged;
static size_t s_low_logged; /* 0: the baseline is not logged yet */

static void on_alloc_failed(size_t size, uint32_t caps, const char *function_name) {
    s_fail_size = size;
    s_fail_caps = caps;
    s_fail_fn = function_name;
    atomic_fetch_add_explicit(&s_failed, 1u, memory_order_relaxed);
}

void heap_watch_init(void) { heap_caps_register_failed_alloc_callback(on_alloc_failed); }

static const char *heap_name(uint32_t caps) {
    if (caps & MALLOC_CAP_SPIRAM) return "psram";
    if (caps & MALLOC_CAP_DMA) return "dma";
    if (caps & MALLOC_CAP_INTERNAL) return "internal";
    return "any";
}

void heap_watch_poll(void) {
    unsigned failed = atomic_load_explicit(&s_failed, memory_order_relaxed);
    if (failed != s_failed_logged) {
        const char *fn = s_fail_fn;
        /* One entry per poll: a burst of failures must not flush the 40-entry log. */
        events_add(EV_WARN, "%u allocation(s) failed, last %u B %s in %s", failed - s_failed_logged,
                   (unsigned)s_fail_size, heap_name(s_fail_caps), fn != NULL ? fn : "?");
        s_failed_logged = failed;
    }

    const size_t low = heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL);
    if (s_low_logged == 0) {
        if (esp_timer_get_time() < BASELINE_US) {
            return;
        }
        events_add(EV_INFO, "internal RAM after start: %u KB free, low %u KB, largest block %u KB",
                   (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024),
                   (unsigned)(low / 1024),
                   (unsigned)(heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL) / 1024));
        s_low_logged = low > 0 ? low : 1;
    } else if (low + LOW_STEP <= s_low_logged) {
        events_add(low < LOW_WARN ? EV_WARN : EV_INFO,
                   "internal RAM new low: %u KB (free now %u KB, largest block %u KB)",
                   (unsigned)(low / 1024),
                   (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024),
                   (unsigned)(heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL) / 1024));
        s_low_logged = low;
    }
}

void heap_watch_json(cJSON *heap) {
    cJSON_AddNumberToObject(heap, "internal_largest",
                            (double)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL));
    cJSON_AddNumberToObject(heap, "alloc_failed",
                            (double)atomic_load_explicit(&s_failed, memory_order_relaxed));
}
