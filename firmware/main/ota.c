/* ota.c — POST /api/ota (streamed app image) and the rollback check. */
#include "ota.h"

#include <string.h>

#include "cJSON.h"
#include "control.h"
#include "esp_app_desc.h"
#include "esp_app_format.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_ota_ops.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "web.h"

static const char *TAG = "ota";

#define OTA_BUF 4096
/* The app descriptor follows the image header and the first segment header. */
#define APP_DESC_OFFSET (sizeof(esp_image_header_t) + sizeof(esp_image_segment_header_t))
#define FIRST_NEEDED (APP_DESC_OFFSET + sizeof(esp_app_desc_t))
#define RECV_RETRIES 5

static void reboot_cb(void *arg) {
    (void)arg;
    esp_restart();
}

void ota_reboot_later(int delay_ms) {
    static esp_timer_handle_t t;
    if (t == NULL) {
        const esp_timer_create_args_t args = {.callback = reboot_cb, .name = "reboot"};
        if (esp_timer_create(&args, &t) != ESP_OK) {
            esp_restart();
        }
    }
    esp_timer_start_once(t, (uint64_t)delay_ms * 1000);
}

static esp_err_t reply(httpd_req_t *req, const char *status, bool ok, const char *key,
                       const char *value) {
    cJSON *o = cJSON_CreateObject();
    cJSON_AddBoolToObject(o, "ok", ok);
    cJSON_AddStringToObject(o, key, value);
    char *body = cJSON_PrintUnformatted(o);
    cJSON_Delete(o);
    httpd_resp_set_status(req, status);
    httpd_resp_set_type(req, "application/json");
    esp_err_t err = httpd_resp_sendstr(req, body != NULL ? body : "{\"ok\":false}");
    cJSON_free(body);
    return err;
}

/* Receive exactly `len` bytes (retrying socket timeouts). */
static int recv_full(httpd_req_t *req, char *buf, size_t len) {
    size_t got = 0;
    int retries = 0;
    while (got < len) {
        int r = httpd_req_recv(req, buf + got, len - got);
        if (r == HTTPD_SOCK_ERR_TIMEOUT && ++retries <= RECV_RETRIES) {
            continue;
        }
        if (r <= 0) {
            return -1;
        }
        got += (size_t)r;
        retries = 0;
    }
    return (int)got;
}

esp_err_t ota_upload_handler(httpd_req_t *req) {
    ctrl_status_t st;
    control_get_status(&st);
    if (st.run == NCP_RUN_ARMED) {
        return reply(req, "409 Conflict", false, "error", "disarm before updating");
    }
    const esp_partition_t *part = esp_ota_get_next_update_partition(NULL);
    if (part == NULL) {
        return reply(req, "500 Internal Server Error", false, "error", "no OTA slot");
    }
    const size_t total = req->content_len;
    if (total < FIRST_NEEDED || total > part->size) {
        return reply(req, "400 Bad Request", false, "error",
                     total > part->size ? "image larger than the OTA slot" : "image too small");
    }
    char *buf = heap_caps_malloc(OTA_BUF, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (buf == NULL) {
        return reply(req, "500 Internal Server Error", false, "error", "out of memory");
    }

    /* Check the image before touching flash: magic byte and our project name. */
    const char *why = NULL;
    size_t first = total < OTA_BUF ? total : OTA_BUF;
    if (recv_full(req, buf, first) < 0) {
        why = "upload interrupted";
    } else if ((uint8_t)buf[0] != ESP_IMAGE_HEADER_MAGIC) {
        why = "not an ESP32 app image (first byte is not 0xE9)";
    } else {
        esp_app_desc_t desc;
        memcpy(&desc, buf + APP_DESC_OFFSET, sizeof(desc));
        if (desc.magic_word != ESP_APP_DESC_MAGIC_WORD) {
            why = "no app descriptor in the image";
        } else if (strncmp(desc.project_name, OTA_PROJECT_NAME, sizeof(desc.project_name)) != 0) {
            why = "image is not n_cartpole_fw";
        }
    }
    if (why != NULL) {
        heap_caps_free(buf);
        return reply(req, "400 Bad Request", false, "error", why);
    }

    ctrl_cmd_t maint = {.kind = CTRL_CMD_MAINT, .i = 1};
    if (!control_send(&maint) || !control_wait_maint(true, 1000)) {
        heap_caps_free(buf);
        return reply(req, "503 Service Unavailable", false, "error", "control loop busy");
    }
    ESP_LOGI(TAG, "writing %u bytes to %s", (unsigned)total, part->label);

    esp_ota_handle_t h = 0;
    esp_err_t err = esp_ota_begin(part, OTA_WITH_SEQUENTIAL_WRITES, &h);
    size_t written = 0;
    if (err == ESP_OK) {
        err = esp_ota_write(h, buf, first);
        written = first;
    }
    while (err == ESP_OK && written < total) {
        size_t n = total - written < OTA_BUF ? total - written : OTA_BUF;
        if (recv_full(req, buf, n) < 0) {
            err = ESP_FAIL;
            why = "upload interrupted";
            break;
        }
        err = esp_ota_write(h, buf, n);
        written += n;
    }
    heap_caps_free(buf);
    if (err != ESP_OK) {
        if (h != 0) {
            esp_ota_abort(h);
        }
    } else {
        err = esp_ota_end(h);
        if (err == ESP_ERR_OTA_VALIDATE_FAILED) {
            why = "image failed validation";
        }
    }
    esp_app_desc_t desc;
    memset(&desc, 0, sizeof(desc));
    if (err == ESP_OK) {
        err = esp_ota_get_partition_description(part, &desc);
    }
    if (err == ESP_OK && strncmp(desc.project_name, OTA_PROJECT_NAME, sizeof(desc.project_name)) != 0) {
        err = ESP_ERR_INVALID_VERSION;
        why = "image is not n_cartpole_fw";
    }
    if (err == ESP_OK) {
        err = esp_ota_set_boot_partition(part);
    }
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "update failed: %s (%s)", why != NULL ? why : "flash error", esp_err_to_name(err));
        maint.i = 0;
        control_send(&maint);
        return reply(req, "400 Bad Request", false, "error", why != NULL ? why : esp_err_to_name(err));
    }

    char version[sizeof(desc.version) + 1];
    memcpy(version, desc.version, sizeof(desc.version));
    version[sizeof(desc.version)] = '\0';
    ESP_LOGI(TAG, "update %s written to %s; rebooting", version, part->label);
    reply(req, "200 OK", true, "version", version);
    ota_reboot_later(1000); /* stays in MAINT until the reboot */
    return ESP_OK;
}

const char *ota_running_state(void) {
    esp_ota_img_states_t state;
    if (esp_ota_get_state_partition(esp_ota_get_running_partition(), &state) != ESP_OK) {
        return "unknown";
    }
    switch (state) {
    case ESP_OTA_IMG_NEW:
        return "new";
    case ESP_OTA_IMG_PENDING_VERIFY:
        return "pending";
    case ESP_OTA_IMG_VALID:
        return "valid";
    case ESP_OTA_IMG_INVALID:
        return "invalid";
    case ESP_OTA_IMG_ABORTED:
        return "aborted";
    default:
        return "undefined";
    }
}

void ota_confirm_or_rollback(void) {
    esp_ota_img_states_t state;
    if (esp_ota_get_state_partition(esp_ota_get_running_partition(), &state) != ESP_OK ||
        state != ESP_OTA_IMG_PENDING_VERIFY) {
        return;
    }
    ESP_LOGW(TAG, "new firmware pending verification: checking health for 10 s");
    const int64_t until = 10 * 1000000LL;
    while (esp_timer_get_time() < until) {
        vTaskDelay(pdMS_TO_TICKS(100));
    }
    uint32_t t0 = control_ticks();
    vTaskDelay(pdMS_TO_TICKS(1000));
    bool ticking = control_ticks() > t0;
    if (ticking && web_running()) {
        esp_ota_mark_app_valid_cancel_rollback();
        ESP_LOGI(TAG, "firmware marked valid");
    } else {
        ESP_LOGE(TAG, "unhealthy (control loop %s, http %s): rolling back",
                 ticking ? "ok" : "stalled", web_running() ? "ok" : "down");
        esp_ota_mark_app_invalid_rollback_and_reboot();
    }
}
