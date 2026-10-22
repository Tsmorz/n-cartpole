/* ncp_ekf.h — n_cartpole/env/estimator.py StateEKF, double precision.
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

typedef struct {
    const ncp_dyn_t *dyn;
    int n, dim;
    double dt;
    double x[NCP_MAX_STATE];
    double P[NCP_MAX_STATE][NCP_MAX_STATE];
    double R[NCP_MAX_POS]; /* diagonal measurement variance */
    double q_vel[NCP_MAX_POS];
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
