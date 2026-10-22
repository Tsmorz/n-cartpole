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
#define EV_MAGIC 0x4E435045u /* "NCPE" */

typedef struct {
    uint32_t seq;  /* 1, 2, ... across the boots the store survived */
    uint32_t t_ms; /* since the boot that logged it */
    uint32_t boot; /* NVS boot counter of that boot */
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
    e->level = (uint8_t)level;
    strlcpy(e->msg, msg, sizeof(e->msg));
    e->crc = entry_crc(e);
    s_store.seq = seq; /* only now: a reset before this leaves the old entry count */
    xSemaphoreGive(s_lock);
}

uint32_t events_seq(void) { return s_store.seq; }

cJSON *events_json(void) {
    static const char *const names[] = {"info", "warn", "error"};
    cJSON *root = cJSON_CreateObject();
    if (root == NULL || s_lock == NULL) {
        return root;
    }
    xSemaphoreTake(s_lock, portMAX_DELAY);
    cJSON_AddNumberToObject(root, "seq", s_store.seq);
    cJSON_AddNumberToObject(root, "boot", s_boot);
    cJSON_AddNumberToObject(root, "uptime_ms", (double)(esp_timer_get_time() / 1000));
    cJSON *arr = cJSON_AddArrayToObject(root, "events");
    uint32_t first = s_store.seq > EV_N ? s_store.seq - EV_N : 0;
    for (uint32_t i = first; i < s_store.seq; i++) {
        const event_t *e = &s_store.ev[i % EV_N];
        if (e->seq != i + 1) {
            continue; /* dropped at boot */
        }
        cJSON *o = cJSON_CreateObject();
        cJSON_AddNumberToObject(o, "seq", e->seq);
        cJSON_AddNumberToObject(o, "boot", e->boot);
        cJSON_AddNumberToObject(o, "t_ms", e->t_ms);
        cJSON_AddStringToObject(o, "level", names[e->level <= EV_ERROR ? e->level : 0]);
        cJSON_AddStringToObject(o, "msg", e->msg);
        cJSON_AddItemToArray(arr, o);
    }
    xSemaphoreGive(s_lock);
    return root;
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
