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

size_t ncp_weight_bytes(const ncp_header_t *h) {
    return ncp_weight_count(h) * (h->dtype == NCP_DTYPE_F16 ? 2u : 4u);
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
    if (h.dtype != NCP_DTYPE_F32 && h.dtype != NCP_DTYPE_F16) return NCP_E_UNSUPPORTED;
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

ncp_err_t ncp_model_bind(ncp_model_t *m, const ncp_header_t *h, const float *weights) {
    if (h->n_blocks > NCP_MAX_BLOCKS) return NCP_E_SHAPE;
    memset(m, 0, sizeof *m);
    m->hdr = *h;
    const size_t H = h->hidden, D = h->obs_dim, E = (size_t)h->hidden * h->expand;
    const float *p = weights;
    m->in_w = p; p += H * D;
    m->in_b = p; p += H;
    for (int b = 0; b < h->n_blocks; b++) {
        ncp_block_t *blk = &m->blocks[b];
        blk->ln_w = p; p += H;
        blk->ln_b = p; p += H;
        blk->fc1_w = p; p += E * H;
        blk->fc1_b = p; p += E;
        blk->fc2_w = p; p += H * E;
        blk->fc2_b = p; p += H;
    }
    m->final_ln_w = p; p += H;
    m->final_ln_b = p; p += H;
    m->out_w = p; p += H;
    m->out_b = p; p += 1;
    return ((size_t)(p - weights) == ncp_weight_count(h)) ? NCP_OK : NCP_E_SIZE;
}

void ncp_goal_target(int n_links, int goal, double *target) {
    for (int k = 0; k < n_links; k++) target[k] = ((goal >> k) & 1) ? 3.14159265358979323846 : 0.0;
}

void ncp_goal_label(int n_links, int goal, char *out) {
    for (int k = 0; k < n_links; k++) out[k] = ((goal >> k) & 1) ? 'D' : 'U';
    out[n_links] = '\0';
}
