/* ncp_model.c — .ncpm blob parsing, validation, CRC, weight decoding, goals. */
#include "ncp_model.h"

#include <math.h>
#include <string.h>

_Static_assert(offsetof(ncp_header_t, dt) == 40, "doubles must start at offset 40");
_Static_assert(offsetof(ncp_header_t, label) == 352, "label offset changed");

const char *ncp_err_str(ncp_err_t err) {
    switch (err) {
    case NCP_OK: return "ok";
    case NCP_E_MAGIC: return "not an .ncpm model (bad magic)";
    case NCP_E_VERSION: return "model written by a newer exporter (version)";
    case NCP_E_SIZE: return "truncated model or sizes disagree with the dims";
    case NCP_E_CRC: return "model corrupted (CRC mismatch)";
    case NCP_E_SHAPE: return "model dims out of range";
    case NCP_E_UNSUPPORTED: return "model needs a feature this firmware lacks";
    }
    return "unknown error";
}

const char *ncp_dtype_name(int dtype) {
    switch (dtype) {
    case NCP_DTYPE_F32: return "f32";
    case NCP_DTYPE_F16: return "f16";
    case NCP_DTYPE_I8: return "i8";
    }
    return "?";
}

/* CRC-32/IEEE, reflected polynomial 0xEDB88320, nibble table (64 bytes). */
static const uint32_t crc_nibble[16] = {
    0x00000000u, 0x1DB71064u, 0x3B6E20C8u, 0x26D930ACu, 0x76DC4190u, 0x6B6B51F4u,
    0x4DB26158u, 0x5005713Cu, 0xEDB88320u, 0xF00F9344u, 0xD6D6A3E8u, 0xCB61B38Cu,
    0x9B64C2B0u, 0x86D3D2D4u, 0xA00AE278u, 0xBDBDF21Cu,
};

uint32_t ncp_crc32(uint32_t crc, const void *data, size_t len) {
    const uint8_t *p = (const uint8_t *)data;
    crc = ~crc;
    for (size_t i = 0; i < len; i++) {
        crc ^= p[i];
        crc = (crc >> 4) ^ crc_nibble[crc & 0x0Fu];
        crc = (crc >> 4) ^ crc_nibble[crc & 0x0Fu];
    }
    return ~crc;
}

uint32_t ncp_blob_crc(const uint8_t *blob, size_t len) {
    const size_t off = offsetof(ncp_header_t, crc32);
    static const uint8_t zero[4] = {0, 0, 0, 0};
    if (len < off + 4) return ncp_crc32(0, blob, len);
    uint32_t crc = ncp_crc32(0, blob, off);
    crc = ncp_crc32(crc, zero, 4);
    return ncp_crc32(crc, blob + off + 4, len - off - 4);
}

size_t ncp_weight_count(const ncp_header_t *h) {
    const size_t H = h->hidden, D = h->obs_dim, E = (size_t)h->hidden * h->expand;
    size_t n = H * D + H;                            /* in_w, in_b */
    n += (size_t)h->n_blocks * (2 * H + 2 * E * H + E + H); /* ln, fc1, fc2 */
    n += 2 * H + H + 1;                              /* final ln, out_w, out_b */
    return n;
}

static size_t pad4(size_t n) { return (n + 3u) & ~(size_t)3u; }

size_t ncp_weight_bytes(const ncp_header_t *h) {
    if (h->dtype == NCP_DTYPE_I8) {
        const size_t H = h->hidden, E = (size_t)h->hidden * h->expand;
        const size_t q = 2 * E * H;                     /* fc1 + fc2 weights per block */
        const size_t per_block = 4 * (E + H) + 2 * pad4(E * H); /* scales + int8 rows */
        return 4 * (ncp_weight_count(h) - (size_t)h->n_blocks * q) + (size_t)h->n_blocks * per_block;
    }
    return ncp_weight_count(h) * (h->dtype == NCP_DTYPE_F16 ? 2u : 4u);
}

size_t ncp_weight_ram_bytes(const ncp_header_t *h) {
    return h->dtype == NCP_DTYPE_I8 ? ncp_weight_bytes(h) : ncp_weight_count(h) * sizeof(float);
}

static int expected_obs_dim(const ncp_header_t *h) {
    int n = h->n_links;
    int d = 2 + 3 * n + h->action_history;
    if (h->flags & NCP_FLAG_GOAL) d += n;
    return d;
}

ncp_err_t ncp_header_parse(const uint8_t *buf, size_t len, ncp_header_t *out) {
    if (len < NCP_HEADER_SIZE) return NCP_E_SIZE;
    ncp_header_t h;
    memcpy(&h, buf, sizeof h); /* little-endian host, packed layout */
    if (memcmp(h.magic, NCP_MAGIC, 4) != 0) return NCP_E_MAGIC;
    if (h.version > NCP_VERSION) return NCP_E_VERSION;
    if (h.version < 1 || h.header_size != NCP_HEADER_SIZE) return NCP_E_UNSUPPORTED;
    if (h.flags & ~(uint8_t)(NCP_FLAG_GOAL | NCP_FLAG_DELTA | NCP_FLAG_EKF))
        return NCP_E_UNSUPPORTED;
    if (h.dtype != NCP_DTYPE_F32 && h.dtype != NCP_DTYPE_F16 && h.dtype != NCP_DTYPE_I8)
        return NCP_E_UNSUPPORTED;
    if (h.n_links < 1 || h.n_links > NCP_MAX_LINKS) return NCP_E_SHAPE;
    if (h.n_blocks > NCP_MAX_BLOCKS || h.expand < 1 || h.hidden < 1) return NCP_E_SHAPE;
    if (h.action_history > NCP_MAX_HIST || h.delay_steps > NCP_MAX_DELAY) return NCP_E_SHAPE;
    if (h.obs_dim != expected_obs_dim(&h) || h.obs_dim > NCP_MAX_OBS) return NCP_E_SHAPE;
    int goals = (h.flags & NCP_FLAG_GOAL) ? (1 << h.n_links) : 1;
    if (h.n_goals != goals) return NCP_E_SHAPE;
    if (!(h.dt > 0.0) || !(h.force_max > 0.0) || !(h.x_lim > 0.0) || !(h.M > 0.0))
        return NCP_E_SHAPE;
    if ((h.flags & NCP_FLAG_DELTA) && (!(h.force_slew > 0.0) || h.action_history < 1))
        return NCP_E_SHAPE;
    for (int k = 0; k < h.n_links; k++)
        if (!(h.masses[k] > 0.0) || !(h.lengths[k] > 0.0)) return NCP_E_SHAPE;
    if (h.cart_coulomb > 0.0 && !(h.cart_coulomb_vscale > 0.0)) return NCP_E_SHAPE;
    if (h.total_size != NCP_HEADER_SIZE + ncp_weight_bytes(&h)) return NCP_E_SIZE;
    h.label[sizeof h.label - 1] = '\0';
    if (out) *out = h;
    return NCP_OK;
}

ncp_err_t ncp_blob_validate(const uint8_t *blob, size_t len, ncp_header_t *out) {
    ncp_header_t h;
    ncp_err_t err = ncp_header_parse(blob, len, &h);
    if (err != NCP_OK) return err;
    if (len != h.total_size) return NCP_E_SIZE;
    if (ncp_blob_crc(blob, len) != h.crc32) return NCP_E_CRC;
    if (out) *out = h;
    return NCP_OK;
}

/* IEEE 754 binary16 -> binary32 (subnormals, inf and NaN included). */
static float half_to_float(uint16_t v) {
    uint32_t sign = (uint32_t)(v & 0x8000u) << 16;
    uint32_t exp = (v >> 10) & 0x1Fu;
    uint32_t man = v & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else { /* subnormal: renormalize */
            int e = -1;
            do {
                man <<= 1;
                e++;
            } while (!(man & 0x400u));
            bits = sign | ((uint32_t)(127 - 15 - e) << 23) | ((man & 0x3FFu) << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

void ncp_weights_decode(const ncp_header_t *h, const uint8_t *src, float *dst, size_t count) {
    if (h->dtype == NCP_DTYPE_F16) {
        for (size_t i = 0; i < count; i++) {
            uint16_t v = (uint16_t)(src[2 * i] | (src[2 * i + 1] << 8));
            dst[i] = half_to_float(v);
        }
    } else {
        memcpy(dst, src, count * sizeof(float));
    }
}

void ncp_weights_load(const ncp_header_t *h, const uint8_t *src, size_t off, size_t n, void *ram) {
    if (h->dtype == NCP_DTYPE_F16) {
        ncp_weights_decode(h, src, (float *)ram + off / 2, n / 2);
    } else {
        memcpy((uint8_t *)ram + off, src, n); /* f32 and int8: the RAM image is the file */
    }
}

/* One float32 tensor of n values at *p; advances *p. */
static const float *take_f32(const uint8_t **p, size_t n) {
    const float *t = (const float *)(const void *)*p;
    *p += n * sizeof(float);
    return t;
}

/* One matrix: float32 [rows x cols], or for int8 the row scales then the padded int8 rows. */
static void take_matrix(const uint8_t **p, int i8, size_t rows, size_t cols, const float **w,
                        const int8_t **q, const float **s) {
    if (i8) {
        *s = take_f32(p, rows);
        *q = (const int8_t *)*p;
        *p += pad4(rows * cols);
    } else {
        *w = take_f32(p, rows * cols);
    }
}

ncp_err_t ncp_model_bind(ncp_model_t *m, const ncp_header_t *h, const void *weights) {
    if (h->n_blocks > NCP_MAX_BLOCKS) return NCP_E_SHAPE;
    memset(m, 0, sizeof *m);
    m->hdr = *h;
    const size_t H = h->hidden, D = h->obs_dim, E = (size_t)h->hidden * h->expand;
    const int i8 = h->dtype == NCP_DTYPE_I8;
    const uint8_t *p = (const uint8_t *)weights;
    m->in_w = take_f32(&p, H * D);
    m->in_b = take_f32(&p, H);
    for (int b = 0; b < h->n_blocks; b++) {
        ncp_block_t *blk = &m->blocks[b];
        blk->ln_w = take_f32(&p, H);
        blk->ln_b = take_f32(&p, H);
        take_matrix(&p, i8, E, H, &blk->fc1_w, &blk->fc1_q, &blk->fc1_s);
        blk->fc1_b = take_f32(&p, E);
        take_matrix(&p, i8, H, E, &blk->fc2_w, &blk->fc2_q, &blk->fc2_s);
        blk->fc2_b = take_f32(&p, H);
    }
    m->final_ln_w = take_f32(&p, H);
    m->final_ln_b = take_f32(&p, H);
    m->out_w = take_f32(&p, H);
    m->out_b = take_f32(&p, 1);
    return ((size_t)(p - (const uint8_t *)weights) == ncp_weight_ram_bytes(h)) ? NCP_OK : NCP_E_SIZE;
}

void ncp_goal_target(int n_links, int goal, double *target) {
    for (int k = 0; k < n_links; k++) target[k] = ((goal >> k) & 1) ? 3.14159265358979323846 : 0.0;
}

void ncp_goal_label(int n_links, int goal, char *out) {
    for (int k = 0; k < n_links; k++) out[k] = ((goal >> k) & 1) ? 'D' : 'U';
    out[n_links] = '\0';
}
