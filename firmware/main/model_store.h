/* model_store.h — the .ncpm blob in the `model` flash partition (data, subtype 0x40).
 *
 * Loading validates the blob (header + CRC over the memory-mapped partition) and
 * loads the weights into RAM (f32/f16 decoded to float32, int8 kept as stored):
 * internal RAM when they fit with INTERNAL_SPARE to spare, else PSRAM. The policy
 * scratch is always internal. The mapping stays open
 * so GET /api/model reads through the cache (no flash access that would stall
 * core 1). Everything here runs on core 0 (main, httpd). */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "esp_err.h"
#include "esp_partition.h"
#include "ncp_model.h"

typedef struct {
    ncp_header_t hdr;
    ncp_model_t m;
    void *weights;    /* ncp_weight_ram_bytes: float32, or the int8 image (ncp_model.h) */
    float *scratch;   /* ncp_policy_scratch_floats floats, internal RAM */
    size_t n_weights;
    bool in_psram;
    int n_goals;      /* goals the controller accepts (>= 1) */
    char goal_labels[NCP_MAX_GOALS][NCP_MAX_LINKS + 1];
} model_t;

esp_err_t model_store_init(void);

/* Load the stored model (no-op if one is loaded). NULL: none / invalid, see model_store_error(). */
const model_t *model_store_load(void);

/* Free the loaded model and close the mapping. The control task must have let go of it. */
void model_store_unload(void);

const model_t *model_store_current(void);
const char *model_store_error(void); /* why the last load failed ("no model", "crc", ...) */
const esp_partition_t *model_store_partition(void);

/* The stored blob through the cache (NULL if not mapped). */
const uint8_t *model_store_mapped(size_t *len);

/* Upload: erase-as-you-go sequential writes into the partition (model unloaded first). */
void model_store_write_begin(void);
esp_err_t model_store_write(size_t offset, const void *data, size_t len);

/* Held while the model is freed; other tasks reading model fields (tlm) take it and
 * check that their pointer is still model_store_current(). */
void model_store_lock(void);
void model_store_unlock(void);

/* Goal index of a label such as "DU" for this model, or -1. */
int model_goal_index(const model_t *m, const char *label);
