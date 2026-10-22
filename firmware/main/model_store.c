/* model_store.c — load/validate/decode the stored .ncpm (see model_store.h). */
#include "model_store.h"

#include <stdlib.h>
#include <string.h>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "ncp_policy.h"

static const char *TAG = "model";

#define MODEL_SUBTYPE ((esp_partition_subtype_t)0x40)
#define INTERNAL_SPARE (64 * 1024)  /* internal RAM to leave free after the weights */
#define CHUNK 4096                  /* fallback read size (multiple of 4) */
#define ERASE_BLOCK (64 * 1024)
#define CAPS_INTERNAL (MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)

static const esp_partition_t *s_part;
static model_t *s_cur;
static const void *s_map;
static esp_partition_mmap_handle_t s_map_handle;
static size_t s_map_len;
static char s_err[48] = "no model";
static size_t s_erased;
static SemaphoreHandle_t s_lock;

static void set_err(const char *msg) { strlcpy(s_err, msg, sizeof(s_err)); }

static void unmap(void) {
    if (s_map != NULL) {
        esp_partition_munmap(s_map_handle);
        s_map = NULL;
        s_map_len = 0;
    }
}

void model_store_lock(void) { xSemaphoreTake(s_lock, portMAX_DELAY); }
void model_store_unlock(void) { xSemaphoreGive(s_lock); }

esp_err_t model_store_init(void) {
    s_lock = xSemaphoreCreateMutex();
    if (s_lock == NULL) {
        return ESP_ERR_NO_MEM;
    }
    s_part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, MODEL_SUBTYPE, "model");
    if (s_part == NULL) {
        set_err("no model partition");
        ESP_LOGE(TAG, "partition 'model' (data/0x40) not found: check partitions.csv");
        return ESP_ERR_NOT_FOUND;
    }
    ESP_LOGI(TAG, "partition 'model' at 0x%06lx, %lu KB", (unsigned long)s_part->address,
             (unsigned long)(s_part->size / 1024));
    return ESP_OK;
}

const esp_partition_t *model_store_partition(void) { return s_part; }
const model_t *model_store_current(void) { return s_cur; }
const char *model_store_error(void) { return s_err; }

const uint8_t *model_store_mapped(size_t *len) {
    *len = s_map_len;
    return (const uint8_t *)s_map;
}

void model_store_unload(void) {
    model_store_lock();
    unmap();
    if (s_cur != NULL) {
        heap_caps_free(s_cur->weights);
        heap_caps_free(s_cur->scratch);
        free(s_cur);
        s_cur = NULL;
    }
    set_err("no model");
    model_store_unlock();
}

/* Chunked CRC + decode from the partition, when the blob cannot be mapped. */
static ncp_err_t read_and_decode(const ncp_header_t *h, float *dst) {
    uint8_t *buf = heap_caps_malloc(CHUNK, CAPS_INTERNAL);
    if (buf == NULL) {
        return NCP_E_SIZE;
    }
    ncp_header_t hz = *h;
    hz.crc32 = 0;
    uint32_t crc = ncp_crc32(0, &hz, sizeof(hz));
    const size_t total = h->total_size;
    const size_t wbytes = ncp_weight_bytes(h);
    const size_t elem = h->dtype == NCP_DTYPE_F16 ? 2 : 4;
    size_t off = NCP_HEADER_SIZE, decoded = 0;
    ncp_err_t err = NCP_OK;
    while (off < total) {
        size_t n = total - off < CHUNK ? total - off : CHUNK;
        if (esp_partition_read(s_part, off, buf, n) != ESP_OK) {
            err = NCP_E_SIZE;
            break;
        }
        crc = ncp_crc32(crc, buf, n);
        size_t woff = off - NCP_HEADER_SIZE; /* weight bytes before this chunk */
        if (woff < wbytes) {
            size_t take = wbytes - woff < n ? wbytes - woff : n;
            ncp_weights_decode(h, buf, dst + decoded, take / elem);
            decoded += take / elem;
        }
        off += n;
    }
    heap_caps_free(buf);
    if (err == NCP_OK && crc != h->crc32) {
        err = NCP_E_CRC;
    }
    return err;
}

const model_t *model_store_load(void) {
    if (s_cur != NULL) {
        return s_cur;
    }
    if (s_part == NULL) {
        return NULL;
    }
    uint8_t raw[NCP_HEADER_SIZE];
    if (esp_partition_read(s_part, 0, raw, sizeof(raw)) != ESP_OK) {
        set_err("flash read failed");
        return NULL;
    }
    if (raw[0] == 0xFF && raw[1] == 0xFF && raw[2] == 0xFF && raw[3] == 0xFF) {
        set_err("no model");
        return NULL;
    }
    ncp_header_t h;
    ncp_err_t perr = ncp_header_parse(raw, sizeof(raw), &h);
    if (perr != NCP_OK) {
        set_err(ncp_err_str(perr));
        return NULL;
    }
    if (h.total_size > s_part->size) {
        set_err("model larger than the partition");
        return NULL;
    }

    /* Validate the CRC over the mapped blob; keep the mapping for GET /api/model. */
    const void *map = NULL;
    esp_partition_mmap_handle_t handle = 0;
    bool mapped = esp_partition_mmap(s_part, 0, h.total_size, ESP_PARTITION_MMAP_DATA, &map,
                                     &handle) == ESP_OK;
    if (mapped) {
        perr = ncp_blob_validate((const uint8_t *)map, h.total_size, &h);
        if (perr != NCP_OK) {
            esp_partition_munmap(handle);
            set_err(ncp_err_str(perr));
            return NULL;
        }
    } else {
        ESP_LOGW(TAG, "cannot map %lu bytes; reading the partition in chunks",
                 (unsigned long)h.total_size);
    }

    model_t *m = calloc(1, sizeof(*m));
    const size_t n = ncp_weight_count(&h);
    const size_t bytes = n * sizeof(float);
    const size_t scratch_bytes = ncp_policy_scratch_floats(&h) * sizeof(float);
    if (m == NULL) {
        goto oom;
    }
    m->hdr = h;
    m->hdr.label[sizeof(m->hdr.label) - 1] = '\0';
    m->n_weights = n;
    m->scratch = heap_caps_malloc(scratch_bytes, CAPS_INTERNAL);
    if (m->scratch == NULL) {
        goto oom;
    }
    if (bytes + INTERNAL_SPARE <= heap_caps_get_largest_free_block(CAPS_INTERNAL)) {
        m->weights = heap_caps_malloc(bytes, CAPS_INTERNAL);
    }
    if (m->weights == NULL) {
        m->weights = heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        m->in_psram = m->weights != NULL;
    }
    if (m->weights == NULL) {
        goto oom;
    }

    if (mapped) {
        ncp_weights_decode(&h, (const uint8_t *)map + NCP_HEADER_SIZE, m->weights, n);
    } else {
        perr = read_and_decode(&h, m->weights);
        if (perr != NCP_OK) {
            set_err(ncp_err_str(perr));
            goto fail;
        }
    }
    perr = ncp_model_bind(&m->m, &m->hdr, m->weights);
    if (perr != NCP_OK) {
        set_err(ncp_err_str(perr));
        goto fail;
    }

    m->n_goals = h.n_goals > 0 ? h.n_goals : 1;
    if (m->n_goals > (1 << h.n_links)) {
        m->n_goals = 1 << h.n_links;
    }
    for (int g = 0; g < (1 << h.n_links); g++) {
        ncp_goal_label(h.n_links, g, m->goal_labels[g]);
    }

    if (mapped) {
        s_map = map;
        s_map_handle = handle;
        s_map_len = h.total_size;
    }
    s_cur = m;
    s_err[0] = '\0';
    ESP_LOGI(TAG, "loaded '%s': %u links, hidden %u x %u blocks, %lu params (%s) in %s",
             m->hdr.label, (unsigned)h.n_links, (unsigned)h.hidden, (unsigned)h.n_blocks,
             (unsigned long)n, h.dtype == NCP_DTYPE_F16 ? "f16" : "f32",
             m->in_psram ? "PSRAM" : "internal RAM");
    return s_cur;

oom:
    set_err("out of memory");
fail:
    if (m != NULL) {
        heap_caps_free(m->weights);
        heap_caps_free(m->scratch);
        free(m);
    }
    if (mapped) {
        esp_partition_munmap(handle);
    }
    ESP_LOGE(TAG, "model not loaded: %s", s_err);
    return NULL;
}

void model_store_write_begin(void) {
    unmap();
    s_erased = 0;
}

esp_err_t model_store_write(size_t offset, const void *data, size_t len) {
    if (s_part == NULL) {
        return ESP_ERR_NOT_FOUND;
    }
    if (offset + len > s_part->size) {
        return ESP_ERR_INVALID_SIZE;
    }
    /* Erase just ahead of the writes so no single flash operation stalls for long. */
    while (s_erased < offset + len) {
        size_t blk = s_part->size - s_erased < ERASE_BLOCK ? s_part->size - s_erased : ERASE_BLOCK;
        esp_err_t err = esp_partition_erase_range(s_part, s_erased, blk);
        if (err != ESP_OK) {
            return err;
        }
        s_erased += blk;
    }
    return esp_partition_write(s_part, offset, data, len);
}

int model_goal_index(const model_t *m, const char *label) {
    if (m == NULL || label == NULL) {
        return -1;
    }
    for (int g = 0; g < m->n_goals; g++) {
        if (strcmp(m->goal_labels[g], label) == 0) {
            return g;
        }
    }
    return -1;
}
