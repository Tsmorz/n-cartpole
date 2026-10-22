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
}

/* Solve A x = r in place (r <- x) for a small symmetric positive-definite A
 * (Cholesky, A destroyed). The mass matrix is SPD for every configuration. */
static void solve_spd(double A[NCP_MAX_POS][NCP_MAX_POS], double *r, int n) {
    for (int j = 0; j < n; j++) {
        double s = A[j][j];
        for (int k = 0; k < j; k++) s -= A[j][k] * A[j][k];
        const double ljj = sqrt(s);
        A[j][j] = ljj;
        for (int i = j + 1; i < n; i++) {
            double t = A[i][j];
            for (int k = 0; k < j; k++) t -= A[i][k] * A[j][k];
            A[i][j] = t / ljj;
        }
    }
    for (int i = 0; i < n; i++) { /* L y = r */
        double t = r[i];
        for (int k = 0; k < i; k++) t -= A[i][k] * r[k];
        r[i] = t / A[i][i];
    }
    for (int i = n - 1; i >= 0; i--) { /* L^T x = y */
        double t = r[i];
        for (int k = i + 1; k < n; k++) t -= A[k][i] * r[k];
        r[i] = t / A[i][i];
    }
}

void ncp_dyn_deriv(const ncp_dyn_t *d, const double *s, double F, double *out) {
    const int n = d->n;
    const double xd = s[1];
    double c[NCP_MAX_LINKS], sn[NCP_MAX_LINKS], thd[NCP_MAX_LINKS], thd2[NCP_MAX_LINKS];
    for (int k = 0; k < n; k++) {
        const double th = s[2 + 2 * k];
        c[k] = cos(th);
        sn[k] = sin(th);
        thd[k] = s[3 + 2 * k];
        thd2[k] = thd[k] * thd[k];
    }

    double A[NCP_MAX_POS][NCP_MAX_POS];
    double r[NCP_MAX_POS];
    A[0][0] = d->m00;
    double r0 = F - d->b * xd;
    double cent = 0.0;
    for (int k = 0; k < n; k++) {
        const double ck = d->a[k] * c[k];
        A[0][k + 1] = ck;
        A[k + 1][0] = ck;
        cent += d->a[k] * sn[k] * thd2[k];
    }
    r0 += cent;
    if (d->coulomb > 0.0) r0 -= d->coulomb * tanh(xd / d->coulomb_vscale);
    r[0] = r0;

    for (int k = 0; k < n; k++) {
        double cor = 0.0;
        for (int j = 0; j < n; j++) {
            /* cos(th_j - th_k), sin(th_j - th_k) by the angle-difference identities */
            const double cjk = c[j] * c[k] + sn[j] * sn[k];
            A[j + 1][k + 1] = d->Kd[j][k] * cjk;
            if (j != k) cor += d->K[j][k] * (sn[j] * c[k] - c[j] * sn[k]) * thd2[j];
        }
        const double omega = thd[k] - (k > 0 ? thd[k - 1] : 0.0);
        double fric = -d->fric[k] * omega;
        if (k + 1 < n) fric += d->fric[k + 1] * (thd[k + 1] - thd[k]);
        r[k + 1] = d->a[k] * d->g * sn[k] + cor + fric;
    }

    solve_spd(A, r, n + 1);
    out[0] = xd;
    out[1] = r[0];
    for (int k = 0; k < n; k++) {
        out[2 + 2 * k] = thd[k];
        out[3 + 2 * k] = r[k + 1];
    }
}

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
