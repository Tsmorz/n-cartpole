/* led.c — status light on the board's WS2812 RGB LED (see led.h). */
#include "led.h"

#include <math.h>
#include <stdbool.h>

#include "control.h"
#include "driver/rmt_encoder.h"
#include "driver/rmt_tx.h"
#include "esp_log.h"
#include "model_store.h"
#include "ncp_ctrl.h" /* NCP_MODE_* */
#include "net.h"
#include "record.h"

static const char *TAG = "led";

#define RMT_HZ (10 * 1000 * 1000) /* 0.1 us ticks */
#define NET_PERIOD_US 500000      /* net_get_info asks the Wi-Fi driver: 2 Hz is plenty */
#define LEVEL 40                  /* full brightness of a pattern, of 255: a WS2812 is bright */
#define DIM 10

typedef struct {
    uint8_t r, g, b;
} rgb_t;

static rmt_channel_handle_t s_chan;
static rmt_encoder_handle_t s_enc;
static rgb_t s_last = {1, 1, 1}; /* never a real colour: the first update always sends */
static net_info_t s_net;
static int64_t s_net_at = -NET_PERIOD_US;

static rgb_t scale(rgb_t c, float k) {
    return (rgb_t){(uint8_t)(c.r * k + 0.5f), (uint8_t)(c.g * k + 0.5f), (uint8_t)(c.b * k + 0.5f)};
}

static void send(rgb_t c) {
    if (s_chan == NULL || (c.r == s_last.r && c.g == s_last.g && c.b == s_last.b)) {
        return;
    }
    static uint8_t grb[3]; /* the RMT reads it after rmt_transmit returns */
    grb[0] = c.g;
    grb[1] = c.r;
    grb[2] = c.b;
    rmt_transmit_config_t tx = {.loop_count = 0};
    if (rmt_tx_wait_all_done(s_chan, 0) == ESP_OK && rmt_transmit(s_chan, s_enc, grb, 3, &tx) == ESP_OK) {
        s_last = c;
    }
}

void led_init(void) {
#if CONFIG_NCP_LED_GPIO >= 0
    rmt_tx_channel_config_t cfg = {
        .gpio_num = CONFIG_NCP_LED_GPIO,
        .clk_src = RMT_CLK_SRC_DEFAULT,
        .resolution_hz = RMT_HZ,
        .mem_block_symbols = 48,
        .trans_queue_depth = 2,
    };
    /* WS2812: a 0 is 0.3 us high + 0.9 us low, a 1 is 0.9 us high + 0.3 us low, MSB first. */
    rmt_bytes_encoder_config_t enc = {
        .bit0 = {.level0 = 1, .duration0 = 3, .level1 = 0, .duration1 = 9},
        .bit1 = {.level0 = 1, .duration0 = 9, .level1 = 0, .duration1 = 3},
        .flags.msb_first = 1,
    };
    if (rmt_new_tx_channel(&cfg, &s_chan) != ESP_OK || rmt_new_bytes_encoder(&enc, &s_enc) != ESP_OK ||
        rmt_enable(s_chan) != ESP_OK) {
        ESP_LOGW(TAG, "no status LED on GPIO %d", CONFIG_NCP_LED_GPIO);
        s_chan = NULL;
        return;
    }
    send((rgb_t){0, 0, LEVEL}); /* booting */
#endif
}

/* Square wave: on for `duty` of each `period_ms`. */
static bool blink(int64_t ms, int period_ms, float duty) {
    return (float)(ms % period_ms) < duty * (float)period_ms;
}

/* The pattern for the state at time `ms` (led.h has the table). */
static rgb_t pattern(const ctrl_status_t *st, const net_info_t *net, bool recording, int64_t ms) {
    const rgb_t off = {0, 0, 0}, red = {LEVEL, 0, 0}, green = {0, LEVEL, 0}, cyan = {0, LEVEL, LEVEL},
                amber = {LEVEL, LEVEL / 3, 0}, blue = {0, 0, LEVEL}, purple = {LEVEL, 0, LEVEL},
                white = {DIM, DIM, DIM};
    rgb_t c;
    if (st->run == NCP_RUN_MAINT) {
        return blink(ms, 250, 0.5f) ? amber : off;
    }
    if (st->run == NCP_RUN_FAULT) {
        c = blink(ms, 500, 0.5f) ? red : off;
    } else if (st->run == NCP_RUN_ARMED) {
        if (st->mode == NCP_MODE_MANUAL) {
            c = cyan;
        } else if (st->settled) {
            c = green;
        } else { /* breathing, 1 s period, never fully off */
            float k = 0.55f - 0.45f * cosf(2.0f * (float)M_PI * (float)(ms % 1000) / 1000.0f);
            c = scale(green, k);
        }
    } else if (st->model == NULL) {
        c = amber;
    } else if (net->ap) {
        c = blink(ms, 1000, 0.5f) ? purple : off;
    } else if (!net->connected) {
        c = blink(ms, 1000, 0.5f) ? blue : off;
    } else {
        c = blink(ms, 3000, 0.04f) ? white : off;
    }
    if (recording && blink(ms, 1000, 0.12f)) {
        c = red;
    }
    return c;
}

void led_update(int64_t now_us) {
    if (s_chan == NULL) {
        return;
    }
    if (now_us - s_net_at >= NET_PERIOD_US) {
        net_get_info(&s_net);
        s_net_at = now_us;
    }
    ctrl_status_t st;
    control_get_status(&st);
    send(pattern(&st, &s_net, g_rec.state == REC_RECORDING, now_us / 1000));
}
