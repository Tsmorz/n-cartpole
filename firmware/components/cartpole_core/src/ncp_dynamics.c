/* ncp_dynamics.c — n_cartpole/env/dynamics.py (_coeffs, _model, _ode, available_force). */
#include "ncp_dynamics.h"

#include <math.h>
#include <string.h>

void ncp_dyn_init(ncp_dyn_t *d, const ncp_header_t *h) {
    memset(d, 0, sizeof *d);
    const int n = h->n_links;
    d->n = n;
    d->M = h->M;
    double msum = 0.0;
    for (int k = 0; k < n; k++) msum += h->masses[k];
    d->m00 = h->M + msum;
    /* beyond_k = mass carried by links further out than k (_mu(m) - m). */
    double beyond[NCP_MAX_LINKS], dd[NCP_MAX_LINKS];
    double acc = 0.0;
    for (int k = n - 1; k >= 0; k--) {
        beyond[k] = acc;
        acc += h->masses[k];
    }
    for (int k = 0; k < n; k++) {
        const double m = h->masses[k], l = h->lengths[k], c = h->com[k];
        d->a[k] = m * c + l * beyond[k];
        dd[k] = h->inertia[k] + m * c * c + l * l * beyond[k];
        d->fric[k] = h->joint_friction[k];
    }
    for (int j = 0; j < n; j++) {
        for (int k = 0; k < n; k++) {
            const int lo = j < k ? j : k, hi = j < k ? k : j;
            d->K[j][k] = h->lengths[lo] * d->a[hi];
            d->Kd[j][k] = (j == k) ? dd[k] : d->K[j][k];
        }
    }
    d->b = h->b;
    d->g = h->g;
    d->coulomb = h->cart_coulomb;
    d->coulomb_vscale = h->cart_coulomb_vscale;
    d->dt = h->dt;
    d->force_max = h->force_max;
    d->v_noload = h->v_noload;

    ncp_dynf_t *f = &d->f;
    f->n = n;
    f->m00 = (float)d->m00;
    f->b = (float)d->b;
    f->g = (float)d->g;
    f->coulomb = (float)d->coulomb;
    f->coulomb_vscale = (float)d->coulomb_vscale;
    for (int j = 0; j < n; j++) {
        f->a[j] = (float)d->a[j];
        f->fric[j] = (float)d->fric[j];
        for (int k = 0; k < n; k++) {
            f->K[j][k] = (float)d->K[j][k];
            f->Kd[j][k] = (float)d->Kd[j][k];
        }
    }
}

/* Double precision: the reference. */
#define REAL double
#define DYN_T ncp_dyn_t
#define SOLVE solve_spd
#define DERIV ncp_dyn_deriv
#define ZERO 0.0
#define SIN sin
#define COS cos
#define TANH tanh
#define SQRT sqrt
#include "ncp_dyn_deriv.inc"
#undef REAL
#undef DYN_T
#undef SOLVE
#undef DERIV
#undef ZERO
#undef SIN
#undef COS
#undef TANH
#undef SQRT

/* Single precision: same equations, for the EKF on the chip. */
#define REAL float
#define DYN_T ncp_dynf_t
#define SOLVE solve_spd_f
#define DERIV ncp_dynf_deriv
#define ZERO 0.0f
#define SIN sinf
#define COS cosf
#define TANH tanhf
#define SQRT sqrtf
#include "ncp_dyn_deriv.inc"
#undef REAL
#undef DYN_T
#undef SOLVE
#undef DERIV
#undef ZERO
#undef SIN
#undef COS
#undef TANH
#undef SQRT

void ncp_dyn_step(const ncp_dyn_t *d, double *s, double F, int substeps) {
    const int dim = 2 + 2 * d->n;
    if (substeps < 1) substeps = 1;
    if (F > d->force_max) F = d->force_max;
    if (F < -d->force_max) F = -d->force_max;
    const double h = d->dt / substeps;
    double k1[NCP_MAX_STATE], k2[NCP_MAX_STATE], k3[NCP_MAX_STATE], k4[NCP_MAX_STATE];
    double tmp[NCP_MAX_STATE];
    for (int it = 0; it < substeps; it++) {
        ncp_dyn_deriv(d, s, F, k1);
        for (int i = 0; i < dim; i++) tmp[i] = s[i] + 0.5 * h * k1[i];
        ncp_dyn_deriv(d, tmp, F, k2);
        for (int i = 0; i < dim; i++) tmp[i] = s[i] + 0.5 * h * k2[i];
        ncp_dyn_deriv(d, tmp, F, k3);
        for (int i = 0; i < dim; i++) tmp[i] = s[i] + h * k3[i];
        ncp_dyn_deriv(d, tmp, F, k4);
        for (int i = 0; i < dim; i++)
            s[i] += h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
}

double ncp_available_force(const ncp_dyn_t *d, double F, double xd) {
    double limit = d->force_max;
    if (d->v_noload > 0.0 && F * xd > 0.0) {
        double f = 1.0 - fabs(xd) / d->v_noload;
        limit *= f > 0.0 ? f : 0.0;
    }
    if (F > limit) return limit;
    if (F < -limit) return -limit;
    return F;
}
