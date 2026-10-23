/* ota.h — firmware updates over HTTP and the rollback check. */
#pragma once

#include <stdbool.h>

#include "esp_err.h"
#include "esp_http_server.h"

#define OTA_PROJECT_NAME "n_cartpole_fw"

/* POST /api/ota: stream an app image into the next OTA slot. */
esp_err_t ota_upload_handler(httpd_req_t *req);

/* If this image is pending verification: wait until 10 s after boot, then mark it valid
 * when the control loop ticks and the HTTP server runs, else roll back (reboots). */
void ota_confirm_or_rollback(void);

/* "valid", "pending", "new", "invalid", "aborted", "undefined" or "unknown". */
const char *ota_running_state(void);

/* Reboot after delay_ms (from an esp_timer, so the HTTP response goes out first). */
void ota_reboot_later(int delay_ms);
