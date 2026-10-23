/* ncp_sim.h — the plant on the device, for testing without hardware.
 *
 * The training env's plant at realism 1 minus the per-episode randomization:
 * the nominal model with rail friction and the back-EMF limit, the actuator
 * (delay_steps delay, slew limit, available_force at the TRUE speed), and the
 * encoder model (Gaussian noise + quantization). Deterministic for a seed. */
#ifndef NCP_SIM_H
#define NCP_SIM_H

#include <stdbool.h>

#include "ncp_dynamics.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    ncp_dyn_t dyn;
    int n;
    double s[NCP_MAX_STATE]; /* true state */
    double delay_buf[NCP_MAX_DELAY];
    int delay_steps, delay_head;
    double slew, prev_applied, applied; /* applied: force that acted over the last step */
    double noise[NCP_MAX_POS], res[NCP_MAX_POS];
    double x_lim;      /* rail half-length: ncp_sim_off_rail */
    bool sensor_model; /* false: exact readings */
    uint64_t rng;
} ncp_sim_t;

void ncp_sim_init(ncp_sim_t *sim, const ncp_header_t *h, uint64_t seed);

/* Start at equilibrium `goal` (ncp_goal_target), every state entry perturbed
 * uniformly by +-perturb (the env's init_noise is 0.05); actuator idle. */
void ncp_sim_reset(ncp_sim_t *sim, int goal, double perturb);

/* Encoder readings of the true state: [x, th0, ...] with noise and quantization. */
void ncp_sim_read(ncp_sim_t *sim, double *pos);

/* Apply a drive command for one period: delay -> slew -> available_force -> RK4. */
void ncp_sim_apply(ncp_sim_t *sim, double cmd);

/* |x| > x_lim: the env's termination. */
bool ncp_sim_off_rail(const ncp_sim_t *sim);

#ifdef __cplusplus
}
#endif
#endif
