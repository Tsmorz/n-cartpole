/* telemetry.h — the single-producer (ctrl, core 1) / single-consumer (tlm, core 0)
 * frame ring and the `tlm` task: batched binary frames over the WebSocket
 * (~25 msg/s), status JSON at 2 Hz, and logging on behalf of core 1. */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "esp_err.h"
#include "ncp_telemetry.h"

#define TLM_CORE 0
#define TLM_PRIO 4
#define STATUS_JSON_MAX 6144 /* status JSON buffer (also the WebSocket send slot size) */

/* Producer side (control task). false: ring full, frame dropped. */
bool tlm_push(const ncp_tlm_frame_t *f);
uint32_t tlm_dropped(void);

esp_err_t tlm_start(void);

/* Copy the latest status JSON (NUL-terminated) into buf; returns its length (0: none yet). */
size_t tlm_status_json(char *buf, size_t size);
