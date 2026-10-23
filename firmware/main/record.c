/* record.c — recording buffer, header, download and status (see record.h). */
#include "record.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "esp_app_desc.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "hal.h"
#include "sdkconfig.h"
#include "settings.h"

static const char *TAG = "record";

#define PSRAM_SPARE (256 * 1024) /* PSRAM to leave free after the buffer */
#define IO_CHUNK 4096

rec_t g_rec;

static void set_str(char *dst, size_t size, const char *src) {
    memset(dst, 0, size);
    if (src != NULL) {
        strlcpy(dst, src, size);
    }
}

const char *record_prepare(double seconds, double dt, const char *note,
                           const ncp_excite_params_t *exc, const model_t *model, int n_links) {
    if (g_rec.state == REC_RECORDING || g_rec.state == REC_STARTING) {
        return "already recording";
    }
    if (!(dt > 0.0)) {
        return "no control period";
    }
    double max_s = CONFIG_NCP_RECORD_SECONDS;
    if (!(seconds > 0.0) || seconds > max_s) {
        seconds = max_s;
    }
    /* The previous recording is discarded now. */
    heap_caps_free(g_rec.buf);
    g_rec.buf = NULL;
    g_rec.capacity = 0;
    g_rec.state = REC_IDLE;

    uint32_t want = (uint32_t)ceil(seconds / dt);
    size_t largest = heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    size_t fit = largest > PSRAM_SPARE ? (largest - PSRAM_SPARE) / sizeof(ncp_rec_sample_t) : 0;
    uint32_t cap = want;
    if (cap > fit) {
        cap = (uint32_t)fit;
        ESP_LOGW(TAG, "PSRAM short: recording capped at %.1f s instead of %.1f s", cap * dt,
                 want * dt);
    }
    if (cap < (uint32_t)(1.0 / dt)) {
        return "not enough PSRAM for a recording";
    }
    g_rec.buf = heap_caps_malloc((size_t)cap * sizeof(ncp_rec_sample_t),
                                 MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (g_rec.buf == NULL) {
        return "out of PSRAM";
    }
    g_rec.capacity = cap;
    g_rec.count = 0;
    g_rec.dropped = 0;
    g_rec.number++;

    ncp_rec_header_t *h = &g_rec.hdr;
    memset(h, 0, sizeof(*h));
    memcpy(h->magic, NCP_REC_MAGIC, 4);
    h->version = NCP_REC_VERSION;
    h->header_size = NCP_REC_HEADER_SIZE;
    h->sample_size = sizeof(ncp_rec_sample_t);
    h->model_crc32 = model != NULL ? model->hdr.crc32 : 0;
    h->dt = dt;
    h->n_links = (uint8_t)n_links;
    h->backend = HAL_IS_SIM ? 0 : 1;
    g_rec.excite = exc != NULL && exc->kind != NCP_EXC_NONE;
    if (g_rec.excite) {
        g_rec.exc = *exc;
        h->excite_kind = (uint8_t)exc->kind;
        h->excite[0] = (float)exc->amp;
        h->excite[1] = (float)exc->f0;
        h->excite[2] = (float)exc->f1;
        h->excite[3] = (float)exc->period;
        h->excite[4] = (float)exc->duration;
        h->excite[5] = (float)exc->seed;
    }
    set_str(h->fw_version, sizeof(h->fw_version), esp_app_get_description()->version);
    set_str(h->model_label, sizeof(h->model_label), model != NULL ? model->hdr.label : "");
    set_str(h->note, sizeof(h->note), note);
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    g_rec.state = REC_STARTING;
    ESP_LOGI(TAG, "recording #%lu: up to %.1f s (%lu samples, %lu KB PSRAM), excitation %s, '%s'",
             (unsigned long)g_rec.number, cap * dt, (unsigned long)cap,
             (unsigned long)((cap * sizeof(ncp_rec_sample_t)) / 1024),
             record_excite_kind_str(g_rec.excite ? exc->kind : NCP_EXC_NONE), h->note);
    return NULL;
}

void record_cancel(void) {
    if (g_rec.state == REC_STARTING) {
        g_rec.state = REC_IDLE;
    }
}

esp_err_t record_get_handler(httpd_req_t *req) {
    uint8_t state = g_rec.state;
    if (state == REC_RECORDING || state == REC_STARTING) {
        httpd_resp_set_status(req, "409 Conflict");
        httpd_resp_set_type(req, "application/json");
        return httpd_resp_sendstr(req, "{\"ok\":false,\"error\":\"recording in progress\"}");
    }
    if (state != REC_DONE || g_rec.buf == NULL) {
        httpd_resp_set_status(req, "404 Not Found");
        httpd_resp_set_type(req, "application/json");
        return httpd_resp_sendstr(req, "{\"ok\":false,\"error\":\"no recording\"}");
    }
    char disp[96];
    snprintf(disp, sizeof(disp), "attachment; filename=\"%.32s-%lu.ncpr\"", settings_get()->hostname,
             (unsigned long)g_rec.number);
    httpd_resp_set_type(req, "application/octet-stream");
    httpd_resp_set_hdr(req, "Content-Disposition", disp);
    esp_err_t err = httpd_resp_send_chunk(req, (const char *)&g_rec.hdr, sizeof(g_rec.hdr));
    const uint8_t *p = (const uint8_t *)g_rec.buf;
    const size_t total = (size_t)g_rec.hdr.n_samples * sizeof(ncp_rec_sample_t);
    for (size_t off = 0; err == ESP_OK && off < total; off += IO_CHUNK) {
        size_t n = total - off < IO_CHUNK ? total - off : IO_CHUNK;
        err = httpd_resp_send_chunk(req, (const char *)p + off, (ssize_t)n);
    }
    httpd_resp_send_chunk(req, NULL, 0);
    return err;
}

void record_status_json(cJSON *root, double dt) {
    static const char *names[] = {"idle", "recording", "recording", "done"};
    uint8_t state = g_rec.state;
    uint32_t n = g_rec.count;
    cJSON *o = cJSON_AddObjectToObject(root, "record");
    cJSON_AddStringToObject(o, "state", names[state & 3]);
    cJSON_AddNumberToObject(o, "samples", n);
    double rdt = g_rec.hdr.dt > 0.0 ? g_rec.hdr.dt : dt;
    cJSON_AddNumberToObject(o, "seconds", round(n * rdt * 10.0) / 10.0);
    double cap_s = g_rec.capacity > 0 ? g_rec.capacity * rdt : CONFIG_NCP_RECORD_SECONDS;
    cJSON_AddNumberToObject(o, "capacity_s", round(cap_s * 10.0) / 10.0);
    cJSON_AddNumberToObject(o, "dropped", g_rec.dropped);
    cJSON_AddStringToObject(o, "note", state == REC_IDLE ? "" : g_rec.hdr.note);
}

const char *record_excite_kind_str(int kind) {
    switch (kind) {
    case NCP_EXC_CHIRP:
        return "chirp";
    case NCP_EXC_PRBS:
        return "prbs";
    case NCP_EXC_STEPS:
        return "steps";
    case NCP_EXC_SINE:
        return "sine";
    default:
        return "none";
    }
}

static double num(const cJSON *o, const char *key, double def) {
    const cJSON *v = cJSON_GetObjectItemCaseSensitive(o, key);
    return cJSON_IsNumber(v) ? v->valuedouble : def;
}

const char *record_parse_excite(const cJSON *o, double default_duration, double cap_n,
                                ncp_excite_params_t *out) {
    memset(out, 0, sizeof(*out));
    if (o == NULL || cJSON_IsNull(o)) {
        out->kind = NCP_EXC_NONE;
        return NULL;
    }
    if (!cJSON_IsObject(o)) {
        return "excite must be an object or null";
    }
    const cJSON *k = cJSON_GetObjectItemCaseSensitive(o, "kind");
    const char *kind = cJSON_IsString(k) ? k->valuestring : "";
    if (strcmp(kind, "chirp") == 0) {
        out->kind = NCP_EXC_CHIRP;
    } else if (strcmp(kind, "prbs") == 0) {
        out->kind = NCP_EXC_PRBS;
    } else if (strcmp(kind, "steps") == 0) {
        out->kind = NCP_EXC_STEPS;
    } else if (strcmp(kind, "sine") == 0) {
        out->kind = NCP_EXC_SINE;
    } else if (strcmp(kind, "none") == 0) {
        out->kind = NCP_EXC_NONE;
        return NULL;
    } else {
        return "excite.kind must be chirp, prbs, steps or sine";
    }
    double amp = fabs(num(o, "amp", 3.0));
    out->amp = amp > cap_n ? cap_n : amp;
    out->f0 = num(o, "f0", 0.2);
    out->f1 = num(o, "f1", 6.0);
    out->period = num(o, "period", 0.05);
    out->duration = num(o, "duration", default_duration);
    double seed = num(o, "seed", 1.0);
    out->seed = seed >= 0.0 && seed < 4294967296.0 ? (uint32_t)seed : 1u;
    if ((out->kind == NCP_EXC_CHIRP || out->kind == NCP_EXC_SINE) && !(out->f0 > 0.0)) {
        return "excite.f0 must be > 0";
    }
    if (out->kind == NCP_EXC_CHIRP && !(out->f1 > 0.0)) {
        return "excite.f1 must be > 0";
    }
    if ((out->kind == NCP_EXC_PRBS || out->kind == NCP_EXC_STEPS) && !(out->period > 0.0)) {
        return "excite.period must be > 0";
    }
    if (!(out->duration > 0.0)) {
        return "excite.duration must be > 0";
    }
    return NULL;
}
