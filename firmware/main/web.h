/* web.h — HTTP API, dashboard, uploads and the /ws WebSocket (httpd task, core 0, prio 5). */
#pragma once

#include <stdbool.h>
#include <stddef.h>

#include "esp_err.h"
#include "esp_http_server.h"

esp_err_t web_start(void);
bool web_running(void);

/* Queue one WebSocket message to every client (copied; dropped if the send pool is
 * busy). Safe from any core-0 task. */
bool web_ws_broadcast(httpd_ws_type_t type, const void *data, size_t len);
int web_ws_clients(void);
