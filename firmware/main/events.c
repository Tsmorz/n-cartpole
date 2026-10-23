/* events.c — the device event log (see events.h).
 *
 * The ring lives in RTC memory that a reset does not clear (RTC_NOINIT_ATTR), so the
 * entries that explain a crash, a watchdog reset or a brownout are still there after it.
 * Each entry carries its own CRC: one half-written when the chip went down is dropped,
 * not the whole log. A power-on reset (and a store that fails its checks) starts empty. */
#include "events.h"

#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "esp_attr.h"
#include "esp_log.h"
#include "esp_rom_crc.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

static const char *TAG = "event";

#define EV_N 40 /* entries kept (RTC memory is small: ~3.7 KB here) */
#define EV_MSG 72
/* One entry as JSON at most: fields (~110 B) + the message with every byte escaped (\u00XX). */
#define EVENT_JSON_MAX (110 + 6 * EV_MSG)
#define EV_MAGIC 0x4E435032u /* "NCP2" (v2 entries carry unix time) */
#define UNIX_VALID 1700000000u /* Nov 2023: anything earlier means the clock was never set */

typedef struct {
    uint32_t seq;  /* 1, 2, ... across the boots the store survived */
    uint32_t t_ms; /* since the boot that logged it */
    uint32_t boot; /* NVS boot counter of that boot */
    uint32_t unix; /* wall-clock seconds, 0 if the clock was not set yet */
    uint8_t level;
    char msg[EV_MSG];
    uint32_t crc; /* over the fields above */
} event_t;

typedef struct {
    uint32_t magic;
    uint32_t seq; /* entries added so far; ev[(seq - 1) % EV_N] is the newest */
    event_t ev[EV_N];
} ev_store_t;

static RTC_NOINIT_ATTR ev_store_t s_store;
static uint32_t s_boot;
static SemaphoreHandle_t s_lock;

static uint32_t entry_crc(const event_t *e) {
    return esp_rom_crc32_le(0, (const uint8_t *)e, offsetof(event_t, crc));
}

static bool entry_ok(const event_t *e, uint32_t seq) {
    return e->seq == seq && e->level <= EV_ERROR && memchr(e->msg, '\0', sizeof(e->msg)) != NULL &&
           e->crc == entry_crc(e);
}

void events_init(uint32_t boot) {
    s_boot = boot;
    if (s_lock == NULL) {
        s_lock = xSemaphoreCreateMutex();
    }
    bool keep = esp_reset_reason() != ESP_RST_POWERON && s_store.magic == EV_MAGIC &&
                s_store.seq < 0x7FFFFFFFu;
    if (keep) {
        /* Entries are written before seq moves on, so a valid one past seq was half-done:
         * drop it with any other bad entry by zeroing it. */
        uint32_t first = s_store.seq > EV_N ? s_store.seq - EV_N : 0;
        for (uint32_t i = first; i < s_store.seq; i++) {
            event_t *e = &s_store.ev[i % EV_N];
            if (!entry_ok(e, i + 1)) {
                memset(e, 0, sizeof(*e));
            }
        }
        ESP_LOGI(TAG, "event log kept across the reset (%lu entries so far)", (unsigned long)s_store.seq);
    } else {
        memset(&s_store, 0, sizeof(s_store));
        s_store.magic = EV_MAGIC;
    }
}

void events_add(ev_level_t level, const char *fmt, ...) {
    char msg[EV_MSG];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    if (level == EV_ERROR) {
        ESP_LOGE(TAG, "%s", msg);
    } else if (level == EV_WARN) {
        ESP_LOGW(TAG, "%s", msg);
    } else {
        ESP_LOGI(TAG, "%s", msg);
    }
    if (s_lock == NULL) {
        return;
    }
    xSemaphoreTake(s_lock, portMAX_DELAY);
    uint32_t seq = s_store.seq + 1;
    event_t *e = &s_store.ev[(seq - 1) % EV_N];
    memset(e, 0, sizeof(*e));
    e->seq = seq;
    e->t_ms = (uint32_t)(esp_timer_get_time() / 1000);
    e->boot = s_boot;
    e->unix = events_unix_now();
    e->level = (uint8_t)level;
    strlcpy(e->msg, msg, sizeof(e->msg));
    e->crc = entry_crc(e);
    s_store.seq = seq; /* only now: a reset before this leaves the old entry count */
    xSemaphoreGive(s_lock);
}

uint32_t events_seq(void) { return s_store.seq; }

uint32_t events_unix_now(void) {
    time_t now = time(NULL);
    return now >= (time_t)UNIX_VALID ? (uint32_t)now : 0;
}

/* JSON string body: quote, backslash and control characters escaped. */
static size_t put_escaped(char *out, size_t room, const char *s) {
    size_t n = 0;
    for (; *s != '\0'; s++) {
        unsigned char c = (unsigned char)*s;
        char tmp[8];
        const char *rep = tmp;
        size_t len;
        if (c == '"' || c == '\\') {
            tmp[0] = '\\';
            tmp[1] = (char)c;
            len = 2;
        } else if (c < 0x20) {
            len = (size_t)snprintf(tmp, sizeof(tmp), "\\u%04x", c);
        } else {
            tmp[0] = (char)c;
            len = 1;
        }
        if (n + len >= room) {
            break;
        }
        memcpy(out + n, rep, len);
        n += len;
    }
    return n;
}

/* Written straight into the caller's buffer (no cJSON tree: building one for 40 entries took
 * ~10 KB of internal heap, which is tight next to the Wi-Fi buffers). */
size_t events_json_write(char *buf, size_t size) {
    static const char *const names[] = {"info", "warn", "error"};
    if (s_lock == NULL || size < 128) {
        return 0;
    }
    xSemaphoreTake(s_lock, portMAX_DELAY);
    size_t n = (size_t)snprintf(buf, size, "{\"seq\":%lu,\"boot\":%lu,\"uptime_ms\":%lu,\"events\":[",
                                (unsigned long)s_store.seq, (unsigned long)s_boot,
                                (unsigned long)(esp_timer_get_time() / 1000));
    uint32_t first = s_store.seq > EV_N ? s_store.seq - EV_N : 0;
    bool any = false;
    for (uint32_t i = first; i < s_store.seq && n + EVENT_JSON_MAX + 4 < size; i++) {
        const event_t *e = &s_store.ev[i % EV_N];
        if (e->seq != i + 1) {
            continue; /* dropped at boot */
        }
        n += (size_t)snprintf(buf + n, size - n, "%s{\"seq\":%lu,\"boot\":%lu,\"t_ms\":%lu,", any ? "," : "",
                              (unsigned long)e->seq, (unsigned long)e->boot, (unsigned long)e->t_ms);
        if (e->unix != 0) {
            n += (size_t)snprintf(buf + n, size - n, "\"unix\":%lu,", (unsigned long)e->unix);
        }
        n += (size_t)snprintf(buf + n, size - n, "\"level\":\"%s\",\"msg\":\"",
                              names[e->level <= EV_ERROR ? e->level : 0]);
        n += put_escaped(buf + n, size - n - 3, e->msg);
        n += (size_t)snprintf(buf + n, size - n, "\"}");
        any = true;
    }
    xSemaphoreGive(s_lock);
    n += (size_t)snprintf(buf + n, size - n, "]}");
    return n < size ? n : size - 1;
}

const char *events_reset_reason(void) {
    switch (esp_reset_reason()) {
    case ESP_RST_POWERON:
        return "power-on";
    case ESP_RST_EXT:
        return "external reset";
    case ESP_RST_SW:
        return "software restart";
    case ESP_RST_PANIC:
        return "crash (panic)";
    case ESP_RST_INT_WDT:
        return "interrupt watchdog";
    case ESP_RST_TASK_WDT:
        return "task watchdog";
    case ESP_RST_WDT:
        return "watchdog";
    case ESP_RST_DEEPSLEEP:
        return "deep sleep";
    case ESP_RST_BROWNOUT:
        return "brownout (supply dipped)";
    case ESP_RST_SDIO:
        return "SDIO";
    case ESP_RST_USB:
        return "USB";
    case ESP_RST_JTAG:
        return "JTAG";
    default:
        return "unknown";
    }
}
