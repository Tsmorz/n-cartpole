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

/* Cross-site request guard for every POST (call first; NULL: go ahead, else the reason
 * for a 403). A web page elsewhere can make the browser send a "simple" cross-origin POST
 * (text/plain, form data) without asking first, so a command could arm the rig from any
 * site the user has open. Requiring `content_type` (application/json or
 * application/octet-stream) forces a CORS preflight, which this server never approves,
 * and a browser's Origin header, when present, must name this host. */
const char *web_post_guard(httpd_req_t *req, const char *content_type);
esp_err_t web_send_forbidden(httpd_req_t *req, const char *why);
