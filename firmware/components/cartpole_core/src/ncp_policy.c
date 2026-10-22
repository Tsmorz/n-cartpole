/* ncp_policy.c — SimBa actor forward pass (deterministic mean head), float32. */
#include "ncp_policy.h"

#include <math.h>

size_t ncp_policy_scratch_floats(const ncp_header_t *h) {
    return 2u * h->hidden + (size_t)h->hidden * h->expand;
}

void ncp_policy_init(ncp_policy_t *p, const ncp_model_t *m, float *scratch) {
    const size_t H = m->hdr.hidden;
    p->m = m;
    p->h = scratch;
    p->t = scratch + H;
    p->u = scratch + 2 * H;
    p->mean = 0.0f;
}

/* Dot product with four independent accumulators (lets the compiler pipeline
 * the FPU on the Xtensa LX7; the order is fixed, so results are reproducible). */
static inline float dot(const float *w, const float *x, size_t n) {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        s0 += w[i] * x[i];
        s1 += w[i + 1] * x[i + 1];
        s2 += w[i + 2] * x[i + 2];
        s3 += w[i + 3] * x[i + 3];
    }
    for (; i < n; i++) s0 += w[i] * x[i];
    return (s0 + s1) + (s2 + s3);
}

/* out[o] = b[o] + W[o] . x, W row-major [out][in]. */
static void linear(const float *W, const float *b, const float *x, size_t in, size_t out_n,
                   float *out) {
    for (size_t o = 0; o < out_n; o++) out[o] = b[o] + dot(W + o * in, x, in);
}

/* torch.nn.LayerNorm: biased variance, eps inside the sqrt. */
static void layer_norm(const float *x, const float *w, const float *b, size_t n, float *out) {
    float mean = 0.0f;
    for (size_t i = 0; i < n; i++) mean += x[i];
    mean /= (float)n;
    float var = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    float inv = 1.0f / sqrtf(var / (float)n + NCP_LN_EPS);
    for (size_t i = 0; i < n; i++) out[i] = (x[i] - mean) * inv * w[i] + b[i];
}

float ncp_policy_act(ncp_policy_t *p, const float *obs) {
    const ncp_model_t *m = p->m;
    const size_t H = m->hdr.hidden, D = m->hdr.obs_dim, E = H * m->hdr.expand;
    float *h = p->h, *t = p->t, *u = p->u;

    linear(m->in_w, m->in_b, obs, D, H, h);
    for (int bi = 0; bi < m->hdr.n_blocks; bi++) {
        const ncp_block_t *blk = &m->blocks[bi];
        layer_norm(h, blk->ln_w, blk->ln_b, H, t);
        linear(blk->fc1_w, blk->fc1_b, t, H, E, u);
        for (size_t i = 0; i < E; i++)
            if (u[i] < 0.0f) u[i] = 0.0f;
        for (size_t o = 0; o < H; o++) h[o] += blk->fc2_b[o] + dot(blk->fc2_w + o * E, u, E);
    }
    layer_norm(h, m->final_ln_w, m->final_ln_b, H, t);
    p->mean = m->out_b[0] + dot(m->out_w, t, H);
    return tanhf(p->mean) * (float)m->hdr.force_max;
}
