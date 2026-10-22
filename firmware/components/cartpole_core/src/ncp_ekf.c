/* ncp_ekf.c — n_cartpole/env/estimator.py StateEKF, line for line. */
#include "ncp_ekf.h"

#include <math.h>
#include <string.h>

typedef double mat_t[NCP_MAX_STATE][NCP_MAX_STATE];

/* Index of reading i in the state: x -> 0, theta_k -> 2 + 2k. */
static inline int pos_idx(int i) { return i == 0 ? 0 : 2 * i; }

void ncp_ekf_pos_std(const ncp_header_t *h, double *pos_std) {
    const double sx = sqrt(h->x_noise_std * h->x_noise_std + h->x_resolution * h->x_resolution / 12.0);
    const double sa = sqrt(h->angle_noise_std * h->angle_noise_std +
                           h->angle_resolution * h->angle_resolution / 12.0);
    pos_std[0] = sx > 1e-6 ? sx : 1e-6; /* keep R invertible with exact readings */
    for (int k = 0; k < h->n_links; k++) pos_std[k + 1] = sa > 1e-6 ? sa : 1e-6;
}

void ncp_ekf_init(ncp_ekf_t *e, const ncp_dyn_t *dyn, const double *pos_std, double force_std,
                  double accel_std) {
    memset(e, 0, sizeof *e);
    e->dyn = dyn;
    e->n = dyn->n;
    e->dim = 2 + 2 * dyn->n;
    e->dt = dyn->dt;
    for (int i = 0; i <= e->n; i++) e->R[i] = pos_std[i] * pos_std[i];
    const double sc = force_std / dyn->M;
    e->q_vel[0] = sc * sc * e->dt;
    for (int k = 0; k < e->n; k++) e->q_vel[k + 1] = accel_std * accel_std * e->dt;
    for (int i = 0; i < e->dim; i++) e->P[i][i] = 1.0;
}

void ncp_ekf_reset(ncp_ekf_t *e, const double *pos) {
    memset(e->x, 0, sizeof e->x);
    memset(e->P, 0, sizeof e->P);
    e->x[0] = pos[0];
    e->P[0][0] = e->R[0];
    e->P[1][1] = 1.0;
    for (int k = 0; k < e->n; k++) {
        e->x[2 + 2 * k] = pos[k + 1];
        e->P[2 + 2 * k][2 + 2 * k] = e->R[k + 1];
        e->P[3 + 2 * k][3 + 2 * k] = 1.0;
    }
}

/* In-place inverse of a small well-conditioned matrix (Gauss-Jordan, partial pivoting). */
static void invert(double A[NCP_MAX_POS][NCP_MAX_POS], double inv[NCP_MAX_POS][NCP_MAX_POS], int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) inv[i][j] = (i == j) ? 1.0 : 0.0;
    for (int c = 0; c < n; c++) {
        int piv = c;
        for (int r = c + 1; r < n; r++)
            if (fabs(A[r][c]) > fabs(A[piv][c])) piv = r;
        if (piv != c) {
            for (int j = 0; j < n; j++) {
                double t = A[c][j];
                A[c][j] = A[piv][j];
                A[piv][j] = t;
                t = inv[c][j];
                inv[c][j] = inv[piv][j];
                inv[piv][j] = t;
            }
        }
        const double d = 1.0 / A[c][c];
        for (int j = 0; j < n; j++) {
            A[c][j] *= d;
            inv[c][j] *= d;
        }
        for (int r = 0; r < n; r++) {
            if (r == c) continue;
            const double f = A[r][c];
            if (f == 0.0) continue;
            for (int j = 0; j < n; j++) {
                A[r][j] -= f * A[c][j];
                inv[r][j] -= f * inv[c][j];
            }
        }
    }
}

void ncp_ekf_update(ncp_ekf_t *e, double u, const double *pos) {
    const int dim = e->dim, np = e->n + 1;
    const double dt = e->dt;
    const ncp_dyn_t *dyn = e->dyn;
    double k1[NCP_MAX_STATE], tmp[NCP_MAX_STATE], f2[NCP_MAX_STATE], s_pred[NCP_MAX_STATE];
    double *s = e->x;

    /* Midpoint prediction. */
    ncp_dyn_deriv(dyn, s, u, k1);
    for (int i = 0; i < dim; i++) tmp[i] = s[i] + 0.5 * dt * k1[i];
    ncp_dyn_deriv(dyn, tmp, u, f2);
    for (int i = 0; i < dim; i++) s_pred[i] = s[i] + dt * f2[i];

    /* Central-difference Jacobian of the continuous dynamics at the estimate. */
    static const double eps = 1e-6;
    mat_t A, F, T, P;
    double fp[NCP_MAX_STATE], fm[NCP_MAX_STATE];
    for (int r = 0; r < dim; r++) A[r][0] = 0.0; /* the dynamics never read x: exactly 0 */
    for (int i = 1; i < dim; i++) {
        memcpy(tmp, s, sizeof(double) * (size_t)dim);
        tmp[i] = s[i] + eps;
        ncp_dyn_deriv(dyn, tmp, u, fp);
        tmp[i] = s[i] - eps;
        ncp_dyn_deriv(dyn, tmp, u, fm);
        for (int r = 0; r < dim; r++) A[r][i] = (fp[r] - fm[r]) / (2 * eps);
    }
    /* F = I + dt A + dt^2/2 A A */
    const double h2 = 0.5 * dt * dt;
    for (int r = 0; r < dim; r++) {
        for (int c = 0; c < dim; c++) {
            double aa = 0.0;
            for (int k = 0; k < dim; k++) aa += A[r][k] * A[k][c];
            F[r][c] = (r == c ? 1.0 : 0.0) + dt * A[r][c] + h2 * aa;
        }
    }
    /* P = F P F^T + Q (Q on the velocity states) */
    for (int r = 0; r < dim; r++)
        for (int c = 0; c < dim; c++) {
            double t = 0.0;
            for (int k = 0; k < dim; k++) t += F[r][k] * e->P[k][c];
            T[r][c] = t;
        }
    for (int r = 0; r < dim; r++)
        for (int c = 0; c < dim; c++) {
            double t = 0.0;
            for (int k = 0; k < dim; k++) t += T[r][k] * F[c][k];
            P[r][c] = t;
        }
    P[1][1] += e->q_vel[0];
    for (int k = 0; k < e->n; k++) P[3 + 2 * k][3 + 2 * k] += e->q_vel[k + 1];

    /* Position-reading update: S = H P H^T + R, K = P H^T S^-1. */
    double S[NCP_MAX_POS][NCP_MAX_POS], Si[NCP_MAX_POS][NCP_MAX_POS], K[NCP_MAX_STATE][NCP_MAX_POS];
    for (int i = 0; i < np; i++)
        for (int j = 0; j < np; j++) S[i][j] = P[pos_idx(i)][pos_idx(j)] + (i == j ? e->R[i] : 0.0);
    invert(S, Si, np);
    for (int r = 0; r < dim; r++)
        for (int j = 0; j < np; j++) {
            double t = 0.0;
            for (int i = 0; i < np; i++) t += P[r][pos_idx(i)] * Si[i][j];
            K[r][j] = t;
        }
    double y[NCP_MAX_POS];
    for (int i = 0; i < np; i++) y[i] = pos[i] - s_pred[pos_idx(i)];
    for (int r = 0; r < dim; r++) {
        double t = 0.0;
        for (int j = 0; j < np; j++) t += K[r][j] * y[j];
        e->x[r] = s_pred[r] + t;
    }

    /* Joseph form: P = (I - K H) P (I - K H)^T + K R K^T. */
    mat_t IKH;
    for (int r = 0; r < dim; r++)
        for (int c = 0; c < dim; c++) IKH[r][c] = (r == c) ? 1.0 : 0.0;
    for (int r = 0; r < dim; r++)
        for (int j = 0; j < np; j++) IKH[r][pos_idx(j)] -= K[r][j];
    for (int r = 0; r < dim; r++)
        for (int c = 0; c < dim; c++) {
            double t = 0.0;
            for (int k = 0; k < dim; k++) t += IKH[r][k] * P[k][c];
            T[r][c] = t;
        }
    for (int r = 0; r < dim; r++)
        for (int c = 0; c < dim; c++) {
            double t = 0.0;
            for (int k = 0; k < dim; k++) t += T[r][k] * IKH[c][k];
            for (int j = 0; j < np; j++) t += K[r][j] * e->R[j] * K[c][j];
            e->P[r][c] = t;
        }
}
