/* ncp_dyn_deriv.inc — the dynamics._ode right-hand side, written once for two precisions.
 *
 * Included twice by ncp_dynamics.c: with REAL = double (the reference: host tests, the
 * simulator) and with REAL = float (the EKF on the ESP32-S3, whose FPU is single precision:
 * double arithmetic is emulated in software and ~10-30x slower). The includer defines
 * REAL, DYN_T, SOLVE, DERIV, ZERO, SIN, COS, TANH and SQRT. Literals are written through
 * ZERO / casts so the float instance never promotes to double. */

/* Solve A x = r in place (r <- x) for a small symmetric positive-definite A
 * (Cholesky, A destroyed). The mass matrix is SPD for every configuration. */
static void SOLVE(REAL A[NCP_MAX_POS][NCP_MAX_POS], REAL *r, int n) {
    for (int j = 0; j < n; j++) {
        REAL s = A[j][j];
        for (int k = 0; k < j; k++) s -= A[j][k] * A[j][k];
        const REAL ljj = SQRT(s);
        A[j][j] = ljj;
        for (int i = j + 1; i < n; i++) {
            REAL t = A[i][j];
            for (int k = 0; k < j; k++) t -= A[i][k] * A[j][k];
            A[i][j] = t / ljj;
        }
    }
    for (int i = 0; i < n; i++) { /* L y = r */
        REAL t = r[i];
        for (int k = 0; k < i; k++) t -= A[i][k] * r[k];
        r[i] = t / A[i][i];
    }
    for (int i = n - 1; i >= 0; i--) { /* L^T x = y */
        REAL t = r[i];
        for (int k = i + 1; k < n; k++) t -= A[k][i] * r[k];
        r[i] = t / A[i][i];
    }
}

void DERIV(const DYN_T *d, const REAL *s, REAL F, REAL *out) {
    const int n = d->n;
    const REAL xd = s[1];
    REAL c[NCP_MAX_LINKS], sn[NCP_MAX_LINKS], thd[NCP_MAX_LINKS], thd2[NCP_MAX_LINKS];
    for (int k = 0; k < n; k++) {
        const REAL th = s[2 + 2 * k];
        c[k] = COS(th);
        sn[k] = SIN(th);
        thd[k] = s[3 + 2 * k];
        thd2[k] = thd[k] * thd[k];
    }

    REAL A[NCP_MAX_POS][NCP_MAX_POS];
    REAL r[NCP_MAX_POS];
    A[0][0] = d->m00;
    REAL r0 = F - d->b * xd;
    REAL cent = ZERO;
    for (int k = 0; k < n; k++) {
        const REAL ck = d->a[k] * c[k];
        A[0][k + 1] = ck;
        A[k + 1][0] = ck;
        cent += d->a[k] * sn[k] * thd2[k];
    }
    r0 += cent;
    if (d->coulomb > ZERO) r0 -= d->coulomb * TANH(xd / d->coulomb_vscale);
    r[0] = r0;

    for (int k = 0; k < n; k++) {
        REAL cor = ZERO;
        for (int j = 0; j < n; j++) {
            /* cos(th_j - th_k), sin(th_j - th_k) by the angle-difference identities */
            const REAL cjk = c[j] * c[k] + sn[j] * sn[k];
            A[j + 1][k + 1] = d->Kd[j][k] * cjk;
            if (j != k) cor += d->K[j][k] * (sn[j] * c[k] - c[j] * sn[k]) * thd2[j];
        }
        const REAL omega = thd[k] - (k > 0 ? thd[k - 1] : ZERO);
        REAL fric = -d->fric[k] * omega;
        if (k + 1 < n) fric += d->fric[k + 1] * (thd[k + 1] - thd[k]);
        r[k + 1] = d->a[k] * d->g * sn[k] + cor + fric;
    }

    SOLVE(A, r, n + 1);
    out[0] = xd;
    out[1] = r[0];
    for (int k = 0; k < n; k++) {
        out[2 + 2 * k] = thd[k];
        out[3 + 2 * k] = r[k + 1];
    }
}
