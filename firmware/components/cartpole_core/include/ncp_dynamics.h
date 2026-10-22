/* ncp_dynamics.h — the n-link rigid-body cart-pendulum, double precision.
 *
 * Line-for-line port of n_cartpole/env/dynamics.py: _coeffs/_model/_ode
 * (manipulator form M(theta) q'' = rhs with a_k, d_k, K[j][k], viscous joint
 * friction on the relative rate, regularized Coulomb rail friction) and
 * available_force (back-EMF speed limit). State [x, xd, th0, th0d, ...],
 * absolute angles, 0 = upright, never wrapped. Used by the EKF (nominal model)
 * and by the on-device simulator. */
#ifndef NCP_DYNAMICS_H
#define NCP_DYNAMICS_H

#include "ncp_model.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Single-precision copy of the coefficients, for ncp_dynf_deriv (the EKF on the chip). */
typedef struct {
    int n;
    float m00;
    float a[NCP_MAX_LINKS];
    float Kd[NCP_MAX_LINKS][NCP_MAX_LINKS];
    float K[NCP_MAX_LINKS][NCP_MAX_LINKS];
    float fric[NCP_MAX_LINKS];
    float b, g, coulomb, coulomb_vscale;
} ncp_dynf_t;

typedef struct {
    int n;           /* links */
    double M;        /* cart mass (PhysicsParams.M) */
    double m00;      /* M + sum(masses) */
    double a[NCP_MAX_LINKS];
    double Kd[NCP_MAX_LINKS][NCP_MAX_LINKS]; /* K with d_k on the diagonal */
    double K[NCP_MAX_LINKS][NCP_MAX_LINKS];
    double fric[NCP_MAX_LINKS];
    double b, g, coulomb, coulomb_vscale;
    double dt, force_max, v_noload; /* v_noload <= 0: no speed limit */
    ncp_dynf_t f;    /* the same model in float, filled by ncp_dyn_init */
} ncp_dyn_t;

/* Nominal model from a blob header (com/inertia as exported). */
void ncp_dyn_init(ncp_dyn_t *d, const ncp_header_t *h);

/* d(state)/dt under cart force F — dynamics._ode. */
void ncp_dyn_deriv(const ncp_dyn_t *d, const double *s, double F, double *out);

/* The same right-hand side in single precision (same source, see ncp_dyn_deriv.inc). */
void ncp_dynf_deriv(const ncp_dynf_t *d, const float *s, float F, float *out);

/* Advance `s` by one control period dt with fixed-step RK4 (`substeps` steps of
 * dt / substeps), F clipped to +-force_max. Python integrates with RK45 at
 * max_step dt/4; at 4 substeps the two agree to ~1e-6, except for a step that
 * starts with the cart inside the stiff Coulomb zone (|xd| of a few
 * cart_coulomb_vscale), where they differ by up to ~1e-4. */
void ncp_dyn_step(const ncp_dyn_t *d, double *s, double F, int substeps);

/* dynamics.available_force: |F| <= force_max, and along the motion
 * F <= force_max * (1 - |xd| / v_noload). */
double ncp_available_force(const ncp_dyn_t *d, double F, double xd);

#ifdef __cplusplus
}
#endif
#endif
