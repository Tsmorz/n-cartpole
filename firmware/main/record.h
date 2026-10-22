/* record.h — recordings for offline system identification (.ncpr, ncp_record.h).
 *
 * Core 0 (httpd) allocates the PSRAM sample buffer and fills the header when a
 * recording starts; the control task (core 1) appends one sample per tick without
 * locks or allocation and stops when the buffer is full. Ownership is handed over by
 * `state`: core 0 touches the buffer only in IDLE/DONE, core 1 only in RECORDING.
 * The last recording stays downloadable (GET /api/record) until the next start. */
#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "cJSON.h"
#include "esp_err.h"
#include "esp_http_server.h"
#include "model_store.h"
#include "ncp_excite.h"
#include "ncp_record.h"

typedef enum {
    REC_IDLE = 0,      /* nothing recorded since boot */
    REC_STARTING = 1,  /* buffer ready, start command queued (core 0 -> core 1) */
    REC_RECORDING = 2, /* control task appending */
    REC_DONE = 3,      /* stopped; downloadable */
} rec_state_t;

typedef struct {
    ncp_rec_header_t hdr;
    ncp_rec_sample_t *buf;
    uint32_t capacity;          /* samples */
    volatile uint32_t count;    /* written by the control task only */
    volatile uint32_t dropped;  /* written by the control task only */
    volatile uint8_t state;     /* rec_state_t */
    bool excite;                /* the recording drives the excitation */
    ncp_excite_params_t exc;
    uint32_t number;            /* recordings since boot (file name) */
} rec_t;

extern rec_t g_rec;

/* Core 0: prepare a recording of `seconds` (capped by Kconfig NCP_RECORD_SECONDS and the
 * free PSRAM) at period dt. exc may be NULL. Returns NULL or the reason it cannot start;
 * on success state is REC_STARTING and the caller queues CTRL_CMD_RECORD_START. */
const char *record_prepare(double seconds, double dt, const char *note,
                           const ncp_excite_params_t *exc, const model_t *model, int n_links);

/* Core 0: the start command could not be queued. */
void record_cancel(void);

/* GET /api/record */
esp_err_t record_get_handler(httpd_req_t *req);

/* Status JSON "record" object. */
void record_status_json(cJSON *root, double dt);

/* Parse {"kind": ..., "amp": ...} (amplitude capped at cap_n). NULL or the reason. */
const char *record_parse_excite(const cJSON *o, double default_duration, double cap_n,
                                ncp_excite_params_t *out);
const char *record_excite_kind_str(int kind);
