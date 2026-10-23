/* ncp_excite.h — excitation signals for system identification.
 *
 * Added to the controller's force (ncp_ctrl_t.ext_force) while a recording
 * runs, so the logged motion carries information about the plant's parameters.
 * Typical use: MANUAL mode (the PD keeps the cart near x_ref) plus a chirp or
 * PRBS on top; or drive disarmed with no excitation for a pendulum ring-down
 * (released by hand), which identifies joint friction and link inertia.
 * Deterministic for a seed so a recording can be reproduced in simulation. */
#ifndef NCP_EXCITE_H
#define NCP_EXCITE_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    NCP_EXC_NONE = 0,
    NCP_EXC_CHIRP = 1, /* amp * sin(phase), frequency swept f0 -> f1 (log) over duration */
    NCP_EXC_PRBS = 2,  /* +-amp, a new random sign every `period` s */
    NCP_EXC_STEPS = 3, /* +amp, 0, -amp, 0, ... each held `period` s */
    NCP_EXC_SINE = 4,  /* amp * sin(2 pi f0 t) */
} ncp_exc_kind_t;

typedef struct {
    ncp_exc_kind_t kind;
    double amp;      /* N */
    double f0, f1;   /* Hz (chirp, sine) */
    double period;   /* s (PRBS bit, step hold) */
    double duration; /* s; the signal is 0 (and inactive) afterwards */
    uint32_t seed;   /* PRBS */
} ncp_excite_params_t;

typedef struct {
    ncp_excite_params_t p;
    double dt, t, phase, value;
    uint64_t rng;
    bool active;
} ncp_excite_t;

/* Start a signal sampled every dt seconds (kind NONE: inactive, always 0). */
void ncp_excite_start(ncp_excite_t *e, const ncp_excite_params_t *p, double dt);

/* This tick's force offset (N); advances time. 0 once duration has elapsed. */
double ncp_excite_next(ncp_excite_t *e);

void ncp_excite_stop(ncp_excite_t *e);

#ifdef __cplusplus
}
#endif
#endif
