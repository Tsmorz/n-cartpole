/* ncp_excite.c — excitation signals for system identification (ncp_excite.h).
 *
 * Sample k (k = 0, 1, ...) is taken at t = k * dt; the signal is active while
 * k < round(duration / dt) (duration <= 0: until stopped). The chirp's
 * instantaneous frequency is f(t) = f0 * (f1 / f0)^(t / duration) and its phase
 * is integrated sample by sample (phase_0 = 0, phase_{k+1} = phase_k + 2 pi f(t_k) dt),
 * so the value is continuous whatever the sweep. n_cartpole's tests reproduce
 * every kind in Python. */
#include "ncp_excite.h"

#include <math.h>
#include <string.h>

#define NCP_EXC_TWO_PI 6.28318530717958647692

/* xorshift64* seeded through splitmix64 (same generator as ncp_sim). */
static uint64_t seed_mix(uint64_t seed) {
    uint64_t z = seed + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    return z ? z : 0x2545F4914F6CDD1DULL;
}

static uint64_t rng_next(uint64_t *s) {
    uint64_t x = *s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *s = x;
    return x * 0x2545F4914F6CDD1DULL;
}

/* Samples per PRBS bit / step hold (at least 1). */
static long hold_samples(const ncp_excite_t *e) {
    long n = lround(e->p.period / e->dt);
    return n < 1 ? 1 : n;
}

void ncp_excite_start(ncp_excite_t *e, const ncp_excite_params_t *p, double dt) {
    memset(e, 0, sizeof *e);
    e->p = *p;
    e->dt = dt > 0.0 ? dt : 1.0;
    e->rng = seed_mix(p->seed);
    e->active = p->kind != NCP_EXC_NONE;
}

void ncp_excite_stop(ncp_excite_t *e) {
    e->active = false;
    e->value = 0.0;
}

double ncp_excite_next(ncp_excite_t *e) {
    if (!e->active) {
        e->value = 0.0;
        return 0.0;
    }
    const long k = lround(e->t / e->dt); /* t is always k * dt exactly */
    const ncp_excite_params_t *p = &e->p;
    if (p->duration > 0.0 && k >= lround(p->duration / e->dt)) {
        ncp_excite_stop(e);
        return 0.0;
    }
    double v = 0.0;
    switch (p->kind) {
    case NCP_EXC_CHIRP: {
        v = p->amp * sin(e->phase);
        double f = p->f0;
        if (p->duration > 0.0 && p->f0 > 0.0 && p->f1 > 0.0)
            f = p->f0 * pow(p->f1 / p->f0, e->t / p->duration);
        else if (p->duration > 0.0) /* a non-positive end: linear sweep */
            f = p->f0 + (p->f1 - p->f0) * e->t / p->duration;
        e->phase += NCP_EXC_TWO_PI * f * e->dt;
        break;
    }
    case NCP_EXC_PRBS:
        if (k % hold_samples(e) == 0) e->value = (rng_next(&e->rng) >> 63) ? p->amp : -p->amp;
        v = e->value;
        break;
    case NCP_EXC_STEPS: {
        static const double pattern[4] = {1.0, 0.0, -1.0, 0.0};
        v = p->amp * pattern[(k / hold_samples(e)) % 4];
        break;
    }
    case NCP_EXC_SINE:
        v = p->amp * sin(NCP_EXC_TWO_PI * p->f0 * e->t);
        break;
    case NCP_EXC_NONE:
    default:
        v = 0.0;
        break;
    }
    e->value = v;
    e->t = (double)(k + 1) * e->dt;
    return v;
}
