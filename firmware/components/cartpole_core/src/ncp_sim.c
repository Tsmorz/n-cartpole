/* ncp_sim.c — the nominal plant + actuator + encoder model (env.step at realism 1,
 * no per-episode randomization). */
#include "ncp_sim.h"

#include <math.h>
#include <string.h>

#define NCP_SIM_SUBSTEPS 4

/* xorshift64* (Vigna). */
static uint64_t rng_next(uint64_t *s) {
    uint64_t x = *s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *s = x;
    return x * 0x2545F4914F6CDD1DULL;
}

/* Uniform in (0, 1): 53 random bits, never 0 or 1. */
static double rng_uniform(uint64_t *s) {
    return ((double)(rng_next(s) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

/* Standard normal (Box-Muller; the second variate is discarded). */
static double rng_normal(uint64_t *s) {
    const double u1 = rng_uniform(s), u2 = rng_uniform(s);
    return sqrt(-2.0 * log(u1)) * cos(6.28318530717958647692 * u2);
}

void ncp_sim_init(ncp_sim_t *sim, const ncp_header_t *h, uint64_t seed) {
    memset(sim, 0, sizeof *sim);
    ncp_dyn_init(&sim->dyn, h);
    sim->n = h->n_links;
    sim->delay_steps = h->delay_steps;
    sim->slew = h->force_slew;
    sim->x_lim = h->x_lim;
    sim->noise[0] = h->x_noise_std;
    sim->res[0] = h->x_resolution;
    for (int k = 0; k < sim->n; k++) {
        sim->noise[k + 1] = h->angle_noise_std;
        sim->res[k + 1] = h->angle_resolution;
    }
    sim->sensor_model = true;
    /* splitmix64 the seed so nearby seeds give unrelated streams (and never 0). */
    uint64_t z = seed + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    sim->rng = z ? z : 0x2545F4914F6CDD1DULL;
    ncp_sim_reset(sim, (1 << sim->n) - 1, 0.0);
}

void ncp_sim_reset(ncp_sim_t *sim, int goal, double perturb) {
    const int dim = 2 + 2 * sim->n;
    double target[NCP_MAX_LINKS];
    ncp_goal_target(sim->n, goal, target);
    memset(sim->s, 0, sizeof sim->s);
    for (int k = 0; k < sim->n; k++) sim->s[2 + 2 * k] = target[k];
    if (perturb > 0.0)
        for (int i = 0; i < dim; i++) sim->s[i] += (2.0 * rng_uniform(&sim->rng) - 1.0) * perturb;
    memset(sim->delay_buf, 0, sizeof sim->delay_buf);
    sim->delay_head = 0;
    sim->prev_applied = 0.0;
    sim->applied = 0.0;
}

void ncp_sim_read(ncp_sim_t *sim, double *pos) {
    pos[0] = sim->s[0];
    for (int k = 0; k < sim->n; k++) pos[k + 1] = sim->s[2 + 2 * k];
    if (!sim->sensor_model) return;
    for (int i = 0; i <= sim->n; i++) {
        double r = pos[i];
        if (sim->noise[i] > 0.0) r += rng_normal(&sim->rng) * sim->noise[i];
        /* env._read: round(reading / res) * res; rint rounds half to even like numpy */
        if (sim->res[i] > 0.0) r = rint(r / sim->res[i]) * sim->res[i];
        pos[i] = r;
    }
}

void ncp_sim_apply(ncp_sim_t *sim, double cmd) {
    const double fmax = sim->dyn.force_max;
    if (cmd > fmax) cmd = fmax;
    if (cmd < -fmax) cmd = -fmax;
    double applied = cmd;
    if (sim->delay_steps > 0) {
        applied = sim->delay_buf[sim->delay_head];
        sim->delay_buf[sim->delay_head] = cmd;
        sim->delay_head = (sim->delay_head + 1) % sim->delay_steps;
    }
    if (sim->slew > 0.0) {
        const double dF = sim->slew * sim->dyn.dt;
        double d = applied - sim->prev_applied;
        d = d < -dF ? -dF : (d > dF ? dF : d);
        applied = sim->prev_applied + d;
    }
    sim->prev_applied = applied;
    applied = ncp_available_force(&sim->dyn, applied, sim->s[1]);
    sim->applied = applied;
    ncp_dyn_step(&sim->dyn, sim->s, applied, NCP_SIM_SUBSTEPS);
}

bool ncp_sim_off_rail(const ncp_sim_t *sim) { return fabs(sim->s[0]) > sim->x_lim; }
