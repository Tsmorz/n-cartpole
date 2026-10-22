/* ncp_ekf.h — n_cartpole/env/estimator.py StateEKF.
 *
 * Working precision is a compile-time choice: double (default; the host tests hold it to
 * the Python filter at 1e-8) or, with NCP_EKF_FLOAT=1 (the firmware build), single, since
 * the ESP32-S3's FPU is single precision and emulated double costs ~10-30x. The float
 * build takes the Jacobian step eps = 1e-3 (1e-6 is below float resolution) and
 * symmetrizes P; the public estimate `x` and the interface stay double either way.
 *
 * Midpoint prediction under the applied force, central-difference Jacobian
 * (eps 1e-6) of the continuous dynamics, F = I + dt A + dt^2/2 A^2, process
 * noise on the velocity states only (force_std / M for the cart, accel_std per
 * link, times dt), position-only update, Joseph-form covariance. */
#ifndef NCP_EKF_H
#define NCP_EKF_H

#include "ncp_dynamics.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef NCP_EKF_FLOAT
#define NCP_EKF_FLOAT 0
#endif
#if NCP_EKF_FLOAT
typedef float ncp_ekf_real;
#else
typedef double ncp_ekf_real;
#endif

typedef struct {
    const ncp_dyn_t *dyn;
    int n, dim;
    double dt;
    double x[NCP_MAX_STATE]; /* the estimate (always double) */
    ncp_ekf_real P[NCP_MAX_STATE][NCP_MAX_STATE];
    ncp_ekf_real R[NCP_MAX_POS]; /* diagonal measurement variance */
    ncp_ekf_real q_vel[NCP_MAX_POS];
} ncp_ekf_t;

/* pos_std: n + 1 reading stds [x, th0, ...] (noise and quantization combined). */
void ncp_ekf_init(ncp_ekf_t *e, const ncp_dyn_t *dyn, const double *pos_std, double force_std,
                  double accel_std);

/* Start from a reading with zero velocities (StateEKF.reset). */
void ncp_ekf_reset(ncp_ekf_t *e, const double *pos);

/* One predict + update; the estimate is left in e->x (StateEKF.update). */
void ncp_ekf_update(ncp_ekf_t *e, double applied_force, const double *pos);

/* StateEKF R as NCPEnv builds it: sqrt(noise^2 + res^2 / 12), floored at 1e-6. */
void ncp_ekf_pos_std(const ncp_header_t *h, double *pos_std /* n + 1 */);

#ifdef __cplusplus
}
#endif
#endif
