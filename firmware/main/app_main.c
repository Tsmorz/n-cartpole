/* app_main.c — boot on core 0: NVS + settings, model store, plant HAL, the control task
 * (core 1), Wi-Fi + mDNS, the HTTP server and the telemetry task (core 0), then the
 * OTA rollback check. app_main returns and the main task deletes itself. */
#include <stdio.h>

#include "control.h"
#include "esp_app_desc.h"
#include "esp_log.h"
#include "esp_ota_ops.h"
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "hal.h"
#include "model_store.h"
#include "net.h"
#include "ota.h"
#include "settings.h"
#include "telemetry.h"
#include "web.h"

static const char *TAG = "main";

#define WIFI_FALLBACK_MS 15000

void app_main(void) {
    const esp_app_desc_t *app = esp_app_get_description();
    const esp_partition_t *running = esp_ota_get_running_partition();
    ESP_LOGI(TAG, "%s %s (IDF %s), backend %s, running from %s (%s)", app->project_name,
             app->version, app->idf_ver, HAL_BACKEND_NAME, running != NULL ? running->label : "?",
             ota_running_state());

    ESP_ERROR_CHECK(settings_init());
    const settings_t *cfg = settings_get();

    const model_t *model = NULL;
    if (model_store_init() == ESP_OK) {
        model = model_store_load();
    }
    if (model == NULL) {
        ESP_LOGW(TAG, "no model (%s): the controller idles until one is uploaded",
                 model_store_error());
    }

    esp_err_t herr = hal_init();
    if (herr != ESP_OK) {
        ESP_LOGE(TAG, "plant hardware init failed (%s): arming will be refused / fault",
                 esp_err_to_name(herr));
    }
#if !HAL_IS_SIM
    if (cfg->have_angle_offsets) {
        hal_rig_set_angle_offsets(cfg->angle_off1, cfg->angle_off2);
    } else {
        ESP_LOGW(TAG, "rig angles not zeroed yet: hang the links at rest and send {\"zero\":\"angles\"}");
    }
    if (cfg->have_cart_zero) {
        hal_rig_set_cart_zero(cfg->x_zero);
    } else {
        ESP_LOGW(TAG, "cart position not zeroed yet: centre it and send {\"zero\":\"cart\"}");
    }
#endif

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

    ota_confirm_or_rollback();
}
