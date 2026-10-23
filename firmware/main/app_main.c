/* app_main.c — boot on core 0: NVS + settings, model store, plant HAL, the control task
 * (core 1), Wi-Fi + mDNS, the HTTP server and the telemetry task (core 0), then the
 * OTA rollback check. app_main returns and the main task deletes itself. */
#include <stdio.h>

#include "control.h"
#include "esp_app_desc.h"
#include "events.h"
#include "esp_log.h"
#include "esp_netif_sntp.h"
#include "esp_ota_ops.h"
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "hal.h"
#include "heap_watch.h"
#include "model_store.h"
#include "net.h"
#include "ota.h"
#include "settings.h"
#include "telemetry.h"
#include "web.h"

static const char *TAG = "main";

#define WIFI_FALLBACK_MS 15000

/* Wall-clock time for the event log and the zero references; UTC (the page shows local). */
static void on_time_sync(struct timeval *tv) {
    (void)tv;
    events_add(EV_INFO, "clock set over SNTP");
}

static void start_clock(void) {
    esp_sntp_config_t c = ESP_NETIF_SNTP_DEFAULT_CONFIG("pool.ntp.org");
    c.sync_cb = on_time_sync;
    esp_err_t err = esp_netif_sntp_init(&c);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "SNTP not started: %s", esp_err_to_name(err));
    }
}

void app_main(void) {
    heap_watch_init(); /* first: count every failed allocation from here on */
    const esp_app_desc_t *app = esp_app_get_description();
    const esp_partition_t *running = esp_ota_get_running_partition();
    ESP_LOGI(TAG, "%s %s (IDF %s), backend %s, running from %s (%s)", app->project_name,
             app->version, app->idf_ver, HAL_BACKEND_NAME, running != NULL ? running->label : "?",
             ota_running_state());

    ESP_ERROR_CHECK(settings_init());
    const settings_t *cfg = settings_get();
    events_init(cfg->boot_count);
    {
        esp_reset_reason_t r = esp_reset_reason();
        bool bad = r == ESP_RST_PANIC || r == ESP_RST_INT_WDT || r == ESP_RST_TASK_WDT ||
                   r == ESP_RST_WDT || r == ESP_RST_BROWNOUT;
        events_add(bad ? EV_ERROR : EV_INFO, "boot #%lu: v%s, %s backend, reset: %s",
                   (unsigned long)cfg->boot_count, app->version, HAL_BACKEND_NAME, events_reset_reason());
    }

    const model_t *model = NULL;
    if (model_store_init() == ESP_OK) {
        model = model_store_load();
    }
    if (model == NULL) {
        events_add(EV_WARN, "no model (%s): the controller idles until one is uploaded",
                   model_store_error());
    }

    esp_err_t herr = ncp_hal_init();
    if (herr != ESP_OK) {
        events_add(EV_ERROR, "plant hardware init failed (%s): arming will be refused",
                   esp_err_to_name(herr));
    }
#if !HAL_IS_SIM
    if (cfg->have_angle_offsets) {
        ncp_hal_rig_set_angle_offsets(cfg->angle_off1, cfg->angle_off2);
    } else {
        events_add(EV_WARN, "rig angles not zeroed: hang the links at rest and Zero angles");
    }
    if (cfg->have_cart_zero) {
        ncp_hal_rig_set_cart_zero(cfg->x_zero);
    } else {
        events_add(EV_WARN, "cart not zeroed: centre it and Zero cart");
    }
#endif
    control_set_calibration(cfg->have_angle_offsets, cfg->have_cart_zero);

    ESP_ERROR_CHECK(control_start(model));
    ESP_ERROR_CHECK(net_start(cfg->hostname, cfg->ssid, cfg->password));
    ESP_ERROR_CHECK(web_start());
    ESP_ERROR_CHECK(tlm_start());

    ESP_LOGI(TAG, "tasks: ctrl -> core %d (prio %d, gptimer ISR on core %d); httpd -> core 0 "
                  "(prio 5); tlm -> core %d (prio %d); Wi-Fi/lwIP/esp_timer/main -> core 0",
             CTRL_CORE, CTRL_PRIO, CTRL_CORE, TLM_CORE, TLM_PRIO);
    if (model != NULL) {
        ESP_LOGI(TAG, "model '%s': %u links, dt %.4f s, weights in %s", model->hdr.label,
                 (unsigned)model->hdr.n_links, model->hdr.dt,
                 model->in_psram ? "PSRAM" : "internal RAM");
    }

    net_wait_or_fallback(WIFI_FALLBACK_MS);
    net_info_t ni;
    net_get_info(&ni);
    ESP_LOGI(TAG, "dashboard: http://%s/ or http://%s.local/ (%s '%s')", ni.ip, cfg->hostname,
             ni.ap ? "access point" : "Wi-Fi", ni.ssid);
    start_clock(); /* retries quietly until there is internet (not on the setup access point) */

    ota_confirm_or_rollback();
}
