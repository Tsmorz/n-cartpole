/* web.c — the HTTP server (httpd task: core 0, prio 5): dashboard, JSON API, model and
 * firmware uploads, and the /ws WebSocket. API: docs/design.md. */
#include "web.h"

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "cJSON.h"
#include "control.h"
#include "esp_app_desc.h"
#include "esp_chip_info.h"
#include "esp_flash.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_ota_ops.h"
#include "esp_psram.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "hal.h"
#include "model_store.h"
#include "ota.h"
#include "record.h"
#include "sdkconfig.h"
#include "settings.h"
#include "telemetry.h"

static const char *TAG = "web";

#define MAX_WS_CLIENTS 4
#define WS_SLOTS 3
#define WS_SLOT_SIZE STATUS_JSON_MAX
#define IO_BUF 4096
#define CMD_BODY_MAX 512
#define RECV_RETRIES 5
#define EXCITE_CAP_N 10.0 /* the manual force cap */

extern const uint8_t index_html_start[] asm("_binary_index_html_start");
extern const uint8_t index_html_end[] asm("_binary_index_html_end");
extern const uint8_t app_js_start[] asm("_binary_app_js_start");
extern const uint8_t app_js_end[] asm("_binary_app_js_end");
extern const uint8_t style_css_start[] asm("_binary_style_css_start");
extern const uint8_t style_css_end[] asm("_binary_style_css_end");

static httpd_handle_t s_server;

/* ---------- JSON helpers ---------- */

static esp_err_t send_json(httpd_req_t *req, const char *status, cJSON *o) {
    char *body = cJSON_PrintUnformatted(o);
    cJSON_Delete(o);
    httpd_resp_set_status(req, status);
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store");
    esp_err_t err = httpd_resp_sendstr(req, body != NULL ? body : "{\"ok\":false,\"error\":\"oom\"}");
    cJSON_free(body);
    return err;
}

static esp_err_t send_error(httpd_req_t *req, const char *status, const char *msg) {
    cJSON *o = cJSON_CreateObject();
    cJSON_AddBoolToObject(o, "ok", false);
    cJSON_AddStringToObject(o, "error", msg);
    return send_json(req, status, o);
}

static esp_err_t send_ok(httpd_req_t *req) {
    cJSON *o = cJSON_CreateObject();
    cJSON_AddBoolToObject(o, "ok", true);
    return send_json(req, "200 OK", o);
}

/* Receive exactly len bytes, retrying socket timeouts. */
static bool recv_full(httpd_req_t *req, char *buf, size_t len) {
    size_t got = 0;
    int retries = 0;
    while (got < len) {
        int r = httpd_req_recv(req, buf + got, len - got);
        if (r == HTTPD_SOCK_ERR_TIMEOUT && ++retries <= RECV_RETRIES) {
            continue;
        }
        if (r <= 0) {
            return false;
        }
        got += (size_t)r;
        retries = 0;
    }
    return true;
}

/* Small JSON body -> parsed object (NULL on error; *why set). */
static cJSON *read_json(httpd_req_t *req, const char **why) {
    if (req->content_len == 0 || req->content_len >= CMD_BODY_MAX) {
        *why = "expected a small JSON body";
        return NULL;
    }
    char body[CMD_BODY_MAX];
    if (!recv_full(req, body, req->content_len)) {
        *why = "body not received";
        return NULL;
    }
    body[req->content_len] = '\0';
    cJSON *j = cJSON_Parse(body);
    if (j == NULL || !cJSON_IsObject(j)) {
        cJSON_Delete(j);
        *why = "invalid JSON";
        return NULL;
    }
    return j;
}

/* ---------- static files ---------- */

static esp_err_t send_static(httpd_req_t *req, const char *type, const uint8_t *start,
                             const uint8_t *end) {
    httpd_resp_set_type(req, type);
    httpd_resp_set_hdr(req, "Cache-Control", "no-cache");
    return httpd_resp_send(req, (const char *)start, end - start);
}

static esp_err_t index_get(httpd_req_t *req) {
    return send_static(req, "text/html; charset=utf-8", index_html_start, index_html_end);
}
static esp_err_t app_js_get(httpd_req_t *req) {
    return send_static(req, "application/javascript", app_js_start, app_js_end);
}
static esp_err_t style_css_get(httpd_req_t *req) {
    return send_static(req, "text/css", style_css_start, style_css_end);
}

/* ---------- GET /api/info ---------- */

static cJSON *model_info(const model_t *m) {
    const ncp_header_t *h = &m->hdr;
    cJSON *o = cJSON_CreateObject();
    char crc[12];
    snprintf(crc, sizeof(crc), "%08lx", (unsigned long)h->crc32);
    cJSON_AddStringToObject(o, "label", h->label);
    cJSON_AddNumberToObject(o, "step", (double)h->train_step);
    cJSON_AddNumberToObject(o, "n_links", h->n_links);
    cJSON_AddNumberToObject(o, "obs_dim", h->obs_dim);
    cJSON_AddNumberToObject(o, "hidden", h->hidden);
    cJSON_AddNumberToObject(o, "blocks", h->n_blocks);
    cJSON_AddNumberToObject(o, "expand", h->expand);
    cJSON_AddNumberToObject(o, "params", (double)m->n_weights);
    cJSON_AddNumberToObject(o, "bytes", (double)h->total_size);
    cJSON_AddStringToObject(o, "crc32", crc);
    cJSON_AddStringToObject(o, "dtype", h->dtype == NCP_DTYPE_F16 ? "f16" : "f32");
    cJSON_AddStringToObject(o, "weights_in", m->in_psram ? "psram" : "internal");
    cJSON_AddBoolToObject(o, "goal_conditioned", (h->flags & NCP_FLAG_GOAL) != 0);
    cJSON_AddStringToObject(o, "action_mode", (h->flags & NCP_FLAG_DELTA) ? "delta" : "absolute");
    cJSON_AddStringToObject(o, "estimator", (h->flags & NCP_FLAG_EKF) ? "ekf" : "diff");
    cJSON_AddNumberToObject(o, "action_history", h->action_history);
    cJSON_AddNumberToObject(o, "delay_steps", h->delay_steps);
    cJSON_AddNumberToObject(o, "dt", h->dt);
    cJSON_AddNumberToObject(o, "force_max", h->force_max);
    cJSON_AddNumberToObject(o, "force_slew", h->force_slew);
    cJSON_AddNumberToObject(o, "x_lim", h->x_lim);
    cJSON *goals = cJSON_AddArrayToObject(o, "goals");
    for (int g = 0; g < m->n_goals; g++) {
        cJSON_AddItemToArray(goals, cJSON_CreateString(m->goal_labels[g]));
    }
    return o;
}

static esp_err_t info_get(httpd_req_t *req) {
    const esp_app_desc_t *d = esp_app_get_description();
    cJSON *root = cJSON_CreateObject();

    char built[40];
    snprintf(built, sizeof(built), "%.16s %.16s", d->date, d->time);
    cJSON *fw = cJSON_AddObjectToObject(root, "fw");
    cJSON_AddStringToObject(fw, "project", d->project_name);
    cJSON_AddStringToObject(fw, "version", d->version);
    cJSON_AddStringToObject(fw, "idf", d->idf_ver);
    cJSON_AddStringToObject(fw, "built", built);
    cJSON_AddStringToObject(fw, "repo", "Tsmorz/n-cartpole");

    esp_chip_info_t chip;
    esp_chip_info(&chip);
    uint32_t flash = 0;
    esp_flash_get_size(NULL, &flash);
    cJSON *c = cJSON_AddObjectToObject(root, "chip");
    cJSON_AddStringToObject(c, "model", chip.model == CHIP_ESP32S3 ? "ESP32-S3" : "ESP32");
    cJSON_AddNumberToObject(c, "cores", chip.cores);
    cJSON_AddNumberToObject(c, "revision", chip.revision);
    cJSON_AddNumberToObject(c, "mhz", CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ);
    cJSON_AddNumberToObject(c, "flash_mb", (double)(flash / (1024 * 1024)));
    cJSON_AddNumberToObject(c, "psram_mb",
                            esp_psram_is_initialized() ? (double)(esp_psram_get_size() / (1024 * 1024)) : 0.0);

    const esp_partition_t *run = esp_ota_get_running_partition();
    const esp_partition_t *next = esp_ota_get_next_update_partition(NULL);
    cJSON *ota = cJSON_AddObjectToObject(root, "ota");
    cJSON_AddStringToObject(ota, "running", run != NULL ? run->label : "");
    cJSON_AddStringToObject(ota, "next", next != NULL ? next->label : "");
    cJSON_AddStringToObject(ota, "state", ota_running_state());

    cJSON_AddStringToObject(root, "backend", HAL_BACKEND_NAME);
    cJSON_AddStringToObject(root, "hostname", settings_get()->hostname);

    const model_t *m = model_store_current();
    double dt = m != NULL ? m->hdr.dt : CTRL_DEFAULT_PERIOD_US * 1e-6;
    cJSON *t = cJSON_AddObjectToObject(root, "telemetry");
    cJSON_AddNumberToObject(t, "version", NCP_TLM_VERSION);
    cJSON_AddNumberToObject(t, "frame_size", sizeof(ncp_tlm_frame_t));
    cJSON_AddNumberToObject(t, "rate_hz", (double)lround(1.0 / (dt * CTRL_TLM_DECIMATION)));

    if (m != NULL) {
        cJSON_AddItemToObject(root, "model", model_info(m));
    } else {
        cJSON_AddNullToObject(root, "model");
        cJSON_AddStringToObject(root, "model_error", model_store_error());
    }
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return send_json(req, "200 OK", root);
}

/* ---------- GET /api/status ---------- */

static esp_err_t status_get(httpd_req_t *req) {
    char *buf = heap_caps_malloc(STATUS_JSON_MAX, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (buf == NULL) {
        return send_error(req, "500 Internal Server Error", "out of memory");
    }
    size_t n = tlm_status_json(buf, STATUS_JSON_MAX);
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    esp_err_t err = n > 0 ? httpd_resp_send(req, buf, (ssize_t)n)
                          : httpd_resp_sendstr(req, "{\"run\":\"idle\"}");
    heap_caps_free(buf);
    return err;
}

/* ---------- POST /api/cmd ---------- */

static int pose_index(const model_t *m, const char *label) {
    for (int g = 0; m != NULL && label != NULL && g < (1 << m->hdr.n_links); g++) {
        if (strcmp(m->goal_labels[g], label) == 0) {
            return g;
        }
    }
    return -1;
}

static esp_err_t cmd_post(httpd_req_t *req) {
    const char *why = NULL;
    cJSON *j = read_json(req, &why);
    if (j == NULL) {
        return send_error(req, "400 Bad Request", why);
    }
    const cJSON *item = j->child;
    if (item == NULL || item->string == NULL || item->next != NULL) {
        cJSON_Delete(j);
        return send_error(req, "400 Bad Request", "expected exactly one command");
    }
    ctrl_status_t st;
    control_get_status(&st);
    ctrl_cmd_t c = {0};
    const char *status = "400 Bad Request";
    const char *key = item->string;
    const char *sv = cJSON_IsString(item) ? item->valuestring : NULL;
    bool prepared = false;

    if (strcmp(key, "arm") == 0 && cJSON_IsBool(item)) {
        c.kind = CTRL_CMD_ARM;
        c.i = cJSON_IsTrue(item) ? 1 : 0;
        if (c.i && (why = control_arm_refusal(&st)) != NULL) {
            status = "409 Conflict";
        }
    } else if (strcmp(key, "mode") == 0 && sv != NULL &&
               (strcmp(sv, "ai") == 0 || strcmp(sv, "manual") == 0)) {
        c.kind = CTRL_CMD_MODE;
        c.i = strcmp(sv, "manual") == 0 ? NCP_MODE_MANUAL : NCP_MODE_AI;
        if (st.model == NULL) {
            why = "no model on the device";
            status = "409 Conflict";
        }
    } else if (strcmp(key, "goal") == 0 && sv != NULL) {
        c.kind = CTRL_CMD_GOAL;
        c.i = model_goal_index(st.model, sv);
        if (st.model == NULL) {
            why = "no model on the device";
            status = "409 Conflict";
        } else if (c.i < 0) {
            why = "unknown goal for this model";
        }
    } else if (strcmp(key, "random") == 0 && cJSON_IsBool(item)) {
        c.kind = CTRL_CMD_RANDOM;
        c.i = cJSON_IsTrue(item) ? 1 : 0;
    } else if (strcmp(key, "x_ref") == 0 && cJSON_IsNumber(item)) {
        c.kind = CTRL_CMD_X_REF;
        c.f = (float)item->valuedouble;
    } else if (strcmp(key, "reset") == 0 && sv != NULL) {
        c.kind = CTRL_CMD_RESET_POSE;
        c.i = pose_index(st.model, sv);
        if (!HAL_IS_SIM) {
            why = "reset is simulator-only";
        } else if (st.model == NULL) {
            why = "no model on the device";
            status = "409 Conflict";
        } else if (c.i < 0) {
            why = "unknown pose";
        }
    } else if (strcmp(key, "zero") == 0 && sv != NULL &&
               (strcmp(sv, "angles") == 0 || strcmp(sv, "cart") == 0)) {
        c.kind = strcmp(sv, "angles") == 0 ? CTRL_CMD_ZERO_ANGLES : CTRL_CMD_ZERO_CART;
        if (HAL_IS_SIM) {
            why = "zeroing is rig-only";
        } else if (st.run == NCP_RUN_ARMED) {
            why = "disarm first";
            status = "409 Conflict";
        }
    } else if (strcmp(key, "record") == 0 && cJSON_IsObject(item)) {
        const cJSON *act = cJSON_GetObjectItemCaseSensitive(item, "action");
        const char *a = cJSON_IsString(act) ? act->valuestring : "";
        if (strcmp(a, "stop") == 0) {
            c.kind = CTRL_CMD_RECORD_STOP;
        } else if (strcmp(a, "start") == 0) {
            const cJSON *secs = cJSON_GetObjectItemCaseSensitive(item, "seconds");
            const cJSON *note = cJSON_GetObjectItemCaseSensitive(item, "note");
            double seconds = cJSON_IsNumber(secs) ? secs->valuedouble : 0.0;
            c.kind = CTRL_CMD_RECORD_START;
            why = record_parse_excite(cJSON_GetObjectItemCaseSensitive(item, "excite"),
                                      seconds > 0.0 ? seconds : CONFIG_NCP_RECORD_SECONDS,
                                      EXCITE_CAP_N, &c.exc);
            if (why == NULL && st.run == NCP_RUN_MAINT) {
                why = "update in progress";
                status = "409 Conflict";
            } else if (why == NULL && st.model == NULL && (HAL_IS_SIM || c.exc.kind != NCP_EXC_NONE)) {
                why = HAL_IS_SIM ? "no model on the device (the simulator needs one)"
                                 : "excitation needs a model";
                status = "409 Conflict";
            }
            if (why == NULL) {
                why = record_prepare(seconds, st.dt, cJSON_IsString(note) ? note->valuestring : "",
                                     &c.exc, st.model,
                                     st.model != NULL ? st.model->hdr.n_links : 2);
                prepared = why == NULL;
                status = "409 Conflict";
            }
        } else {
            why = "record.action must be start or stop";
        }
    } else if (strcmp(key, "excite") == 0 && (cJSON_IsObject(item) || cJSON_IsNull(item))) {
        c.kind = CTRL_CMD_EXCITE;
        why = record_parse_excite(item, 30.0, EXCITE_CAP_N, &c.exc);
        if (why == NULL && st.model == NULL && c.exc.kind != NCP_EXC_NONE) {
            why = "excitation needs a model";
            status = "409 Conflict";
        }
    } else {
        why = "unknown command";
    }
    cJSON_Delete(j);
    if (why != NULL) {
        return send_error(req, status, why);
    }
    if (!control_send(&c)) {
        if (prepared) {
            record_cancel();
        }
        return send_error(req, "503 Service Unavailable", "control queue full");
    }
    return send_ok(req);
}

/* ---------- GET/POST /api/model ---------- */

static esp_err_t model_get(httpd_req_t *req) {
    const model_t *m = model_store_current();
    if (m == NULL) {
        return send_error(req, "404 Not Found", "no model on the device");
    }
    size_t len = 0;
    const uint8_t *blob = model_store_mapped(&len);
    httpd_resp_set_type(req, "application/octet-stream");
    httpd_resp_set_hdr(req, "Content-Disposition", "attachment; filename=\"model.ncpm\"");
    if (blob != NULL) {
        for (size_t off = 0; off < len; off += IO_BUF) {
            size_t n = len - off < IO_BUF ? len - off : IO_BUF;
            if (httpd_resp_send_chunk(req, (const char *)blob + off, (ssize_t)n) != ESP_OK) {
                httpd_resp_send_chunk(req, NULL, 0);
                return ESP_FAIL;
            }
        }
        return httpd_resp_send_chunk(req, NULL, 0);
    }
    /* Not mapped: read the partition (flash reads stall core 1, so only while disarmed). */
    ctrl_status_t st;
    control_get_status(&st);
    if (st.run == NCP_RUN_ARMED) {
        return send_error(req, "409 Conflict", "disarm to download the model");
    }
    char *buf = heap_caps_malloc(IO_BUF, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (buf == NULL) {
        return send_error(req, "500 Internal Server Error", "out of memory");
    }
    esp_err_t err = ESP_OK;
    len = m->hdr.total_size;
    for (size_t off = 0; off < len && err == ESP_OK; off += IO_BUF) {
        size_t n = len - off < IO_BUF ? len - off : IO_BUF;
        err = esp_partition_read(model_store_partition(), off, buf, n);
        if (err == ESP_OK) {
            err = httpd_resp_send_chunk(req, buf, (ssize_t)n);
        }
    }
    heap_caps_free(buf);
    httpd_resp_send_chunk(req, NULL, 0);
    return err;
}

static void leave_maint(void) {
    ctrl_cmd_t c = {.kind = CTRL_CMD_MAINT, .i = 0};
    control_send(&c);
}

static esp_err_t model_post(httpd_req_t *req) {
    ctrl_status_t st;
    control_get_status(&st);
    if (st.run == NCP_RUN_ARMED) {
        return send_error(req, "409 Conflict", "disarm before uploading a model");
    }
    const esp_partition_t *part = model_store_partition();
    const size_t total = req->content_len;
    if (part == NULL) {
        return send_error(req, "500 Internal Server Error", "no model partition");
    }
    if (total < NCP_HEADER_SIZE || total > part->size) {
        return send_error(req, "400 Bad Request",
                          total > part->size ? "model larger than the partition" : "not an .ncpm file");
    }
    char *buf = heap_caps_malloc(IO_BUF, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (buf == NULL) {
        return send_error(req, "500 Internal Server Error", "out of memory");
    }
    /* Header first: nothing is erased for a file that is not a model. */
    ncp_header_t h;
    if (!recv_full(req, buf, NCP_HEADER_SIZE)) {
        heap_caps_free(buf);
        return send_error(req, "400 Bad Request", "upload interrupted");
    }
    ncp_err_t perr = ncp_header_parse((const uint8_t *)buf, NCP_HEADER_SIZE, &h);
    if (perr != NCP_OK || h.total_size != total) {
        heap_caps_free(buf);
        return send_error(req, "400 Bad Request",
                          perr != NCP_OK ? ncp_err_str(perr) : "size does not match the header");
    }

    /* MAINT, let go of the old model and free it (two large models may not fit). */
    ctrl_cmd_t c = {.kind = CTRL_CMD_MAINT, .i = 1};
    bool ok = control_send(&c) && control_wait_maint(true, 1000);
    ctrl_cmd_t detach = {.kind = CTRL_CMD_SET_MODEL, .model = NULL};
    ok = ok && control_send(&detach) && control_wait_model(NULL, 1000);
    if (!ok) {
        heap_caps_free(buf);
        leave_maint();
        return send_error(req, "503 Service Unavailable", "control loop busy");
    }
    model_store_unload();
    ESP_LOGI(TAG, "receiving model '%.64s' (%u bytes)", h.label, (unsigned)total);

    model_store_write_begin();
    esp_err_t err = model_store_write(0, buf, NCP_HEADER_SIZE);
    const char *why = NULL;
    for (size_t off = NCP_HEADER_SIZE; err == ESP_OK && off < total;) {
        size_t n = total - off < IO_BUF ? total - off : IO_BUF;
        if (!recv_full(req, buf, n)) {
            why = "upload interrupted";
            break;
        }
        err = model_store_write(off, buf, n);
        off += n;
    }
    heap_caps_free(buf);
    if (why == NULL && err != ESP_OK) {
        why = esp_err_to_name(err);
    }
    const model_t *m = why == NULL ? model_store_load() : NULL;
    if (m == NULL && why == NULL) {
        why = model_store_error();
    }
    ctrl_cmd_t attach = {.kind = CTRL_CMD_SET_MODEL, .model = m};
    control_send(&attach);
    control_wait_model(m, 1000);
    leave_maint();
    if (m == NULL) {
        ESP_LOGE(TAG, "model upload failed: %s", why);
        return send_error(req, "400 Bad Request", why);
    }
    cJSON *o = cJSON_CreateObject();
    cJSON_AddBoolToObject(o, "ok", true);
    cJSON_AddStringToObject(o, "label", m->hdr.label);
    return send_json(req, "200 OK", o);
}

/* ---------- POST /api/wifi, /api/reboot ---------- */

static esp_err_t wifi_post(httpd_req_t *req) {
    ctrl_status_t st;
    control_get_status(&st);
    if (st.run == NCP_RUN_ARMED) {
        return send_error(req, "409 Conflict", "disarm first");
    }
    const char *why = NULL;
    cJSON *j = read_json(req, &why);
    if (j == NULL) {
        return send_error(req, "400 Bad Request", why);
    }
    const cJSON *ssid = cJSON_GetObjectItemCaseSensitive(j, "ssid");
    const cJSON *pass = cJSON_GetObjectItemCaseSensitive(j, "password");
    const char *s = cJSON_IsString(ssid) ? ssid->valuestring : NULL;
    const char *p = cJSON_IsString(pass) ? pass->valuestring : "";
    if (s == NULL || s[0] == '\0' || strlen(s) > 32) {
        why = "ssid must be 1-32 characters";
    } else if (strlen(p) > 64 || (p[0] != '\0' && strlen(p) < 8)) {
        why = "password must be empty or 8-64 characters";
    } else if (settings_save_wifi(s, p) != ESP_OK) {
        why = "could not save to NVS";
    }
    cJSON_Delete(j);
    if (why != NULL) {
        return send_error(req, "400 Bad Request", why);
    }
    ESP_LOGI(TAG, "Wi-Fi credentials saved; rebooting");
    esp_err_t err = send_ok(req);
    ota_reboot_later(1000);
    return err;
}

static esp_err_t reboot_post(httpd_req_t *req) {
    ctrl_cmd_t c = {.kind = CTRL_CMD_ARM, .i = 0};
    control_send(&c); /* drive off before the reset */
    esp_err_t err = send_ok(req);
    ota_reboot_later(500);
    return err;
}

/* ---------- WebSocket ---------- */

typedef struct {
    volatile bool busy;
    httpd_ws_type_t type;
    size_t len;
    uint8_t *data;
} ws_slot_t;

static int s_ws_fds[MAX_WS_CLIENTS];
static int s_ws_n;
static portMUX_TYPE s_ws_mux = portMUX_INITIALIZER_UNLOCKED;
static ws_slot_t s_slots[WS_SLOTS];

static void ws_remove(int fd) {
    taskENTER_CRITICAL(&s_ws_mux);
    for (int i = 0; i < s_ws_n; i++) {
        if (s_ws_fds[i] == fd) {
            s_ws_fds[i] = s_ws_fds[--s_ws_n];
            break;
        }
    }
    taskEXIT_CRITICAL(&s_ws_mux);
}

static void ws_add(int fd) {
    int evict = -1;
    taskENTER_CRITICAL(&s_ws_mux);
    bool present = false;
    for (int i = 0; i < s_ws_n; i++) {
        present = present || s_ws_fds[i] == fd;
    }
    if (!present) {
        if (s_ws_n == MAX_WS_CLIENTS) {
            evict = s_ws_fds[0];
            s_ws_fds[0] = fd;
        } else {
            s_ws_fds[s_ws_n++] = fd;
        }
    }
    taskEXIT_CRITICAL(&s_ws_mux);
    if (evict >= 0) {
        httpd_sess_trigger_close(s_server, evict);
    }
}

int web_ws_clients(void) { return s_ws_n; }

static void ws_send_work(void *arg) {
    ws_slot_t *slot = (ws_slot_t *)arg;
    int fds[MAX_WS_CLIENTS];
    taskENTER_CRITICAL(&s_ws_mux);
    int n = s_ws_n;
    memcpy(fds, s_ws_fds, sizeof(int) * (size_t)n);
    taskEXIT_CRITICAL(&s_ws_mux);
    for (int i = 0; i < n; i++) {
        if (httpd_ws_get_fd_info(s_server, fds[i]) != HTTPD_WS_CLIENT_WEBSOCKET) {
            ws_remove(fds[i]);
            continue;
        }
        httpd_ws_frame_t f = {
            .final = true,
            .fragmented = false,
            .type = slot->type,
            .payload = slot->data,
            .len = slot->len,
        };
        if (httpd_ws_send_frame_async(s_server, fds[i], &f) != ESP_OK) {
            ws_remove(fds[i]);
            httpd_sess_trigger_close(s_server, fds[i]);
        }
    }
    slot->busy = false;
}

bool web_ws_broadcast(httpd_ws_type_t type, const void *data, size_t len) {
    if (s_server == NULL || s_ws_n == 0 || len > WS_SLOT_SIZE) {
        return false;
    }
    ws_slot_t *slot = NULL;
    taskENTER_CRITICAL(&s_ws_mux);
    for (int i = 0; i < WS_SLOTS; i++) {
        if (!s_slots[i].busy && s_slots[i].data != NULL) {
            slot = &s_slots[i];
            slot->busy = true;
            break;
        }
    }
    taskEXIT_CRITICAL(&s_ws_mux);
    if (slot == NULL) {
        return false; /* clients are slow: drop this message */
    }
    memcpy(slot->data, data, len);
    slot->len = len;
    slot->type = type;
    if (httpd_queue_work(s_server, ws_send_work, slot) != ESP_OK) {
        slot->busy = false;
        return false;
    }
    return true;
}

static esp_err_t ws_handler(httpd_req_t *req) {
    if (req->method == HTTP_GET) { /* handshake done */
        ws_add(httpd_req_to_sockfd(req));
        ESP_LOGI(TAG, "WebSocket client %d connected (%d total)", httpd_req_to_sockfd(req), s_ws_n);
        return ESP_OK;
    }
    /* Client messages are not used; read and discard them. */
    httpd_ws_frame_t f;
    memset(&f, 0, sizeof(f));
    esp_err_t err = httpd_ws_recv_frame(req, &f, 0);
    if (err != ESP_OK) {
        return err;
    }
    if (f.len > 0) {
        uint8_t buf[128];
        if (f.len > sizeof(buf)) {
            return ESP_FAIL; /* closes the connection */
        }
        f.payload = buf;
        err = httpd_ws_recv_frame(req, &f, f.len);
    }
    return err;
}

static void on_close(httpd_handle_t hd, int sockfd) {
    (void)hd;
    ws_remove(sockfd);
    close(sockfd);
}

/* ---------- server ---------- */

esp_err_t web_start(void) {
    for (int i = 0; i < WS_SLOTS; i++) {
        s_slots[i].data = heap_caps_malloc(WS_SLOT_SIZE, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        if (s_slots[i].data == NULL) {
            return ESP_ERR_NO_MEM;
        }
    }
    httpd_config_t cfg = HTTPD_DEFAULT_CONFIG();
    cfg.core_id = 0;
    cfg.task_priority = 5;
    cfg.stack_size = 8192;
    cfg.max_uri_handlers = 16;
    cfg.max_open_sockets = 7;
    cfg.lru_purge_enable = true;
    cfg.recv_wait_timeout = 10;
    cfg.send_wait_timeout = 5;
    cfg.close_fn = on_close;

    esp_err_t err = httpd_start(&s_server, &cfg);
    if (err != ESP_OK) {
        s_server = NULL;
        return err;
    }
    const httpd_uri_t uris[] = {
        {.uri = "/", .method = HTTP_GET, .handler = index_get},
        {.uri = "/index.html", .method = HTTP_GET, .handler = index_get},
        {.uri = "/app.js", .method = HTTP_GET, .handler = app_js_get},
        {.uri = "/style.css", .method = HTTP_GET, .handler = style_css_get},
        {.uri = "/api/info", .method = HTTP_GET, .handler = info_get},
        {.uri = "/api/status", .method = HTTP_GET, .handler = status_get},
        {.uri = "/api/cmd", .method = HTTP_POST, .handler = cmd_post},
        {.uri = "/api/model", .method = HTTP_GET, .handler = model_get},
        {.uri = "/api/model", .method = HTTP_POST, .handler = model_post},
        {.uri = "/api/ota", .method = HTTP_POST, .handler = ota_upload_handler},
        {.uri = "/api/record", .method = HTTP_GET, .handler = record_get_handler},
        {.uri = "/api/wifi", .method = HTTP_POST, .handler = wifi_post},
        {.uri = "/api/reboot", .method = HTTP_POST, .handler = reboot_post},
        {.uri = "/ws", .method = HTTP_GET, .handler = ws_handler, .is_websocket = true},
    };
    for (size_t i = 0; i < sizeof(uris) / sizeof(uris[0]); i++) {
        ESP_ERROR_CHECK(httpd_register_uri_handler(s_server, &uris[i]));
    }
    ESP_LOGI(TAG, "HTTP server on port %d (core %d, prio %d)", (int)cfg.server_port,
             (int)cfg.core_id, (int)cfg.task_priority);
    return ESP_OK;
}

bool web_running(void) { return s_server != NULL; }
