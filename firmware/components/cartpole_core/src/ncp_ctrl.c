/* ncp_ctrl.c — one control tick: NPendulumCartpole.step/_sense/_make_obs, device side.
 *
 * The float32 casts mirror numpy's: the kinematic block is computed in double and
 * cast (encode_obs), the action history is float32 / float32(force_max), and the
 * delta-mode rate is float32(action) / float32(force_max) (NEP 50 weak scalars). */
#include "ncp_ctrl.h"

#include <math.h>
#include <string.h>

#define NCP_TWO_PI 6.28318530717958647692

static double clampd(double v, double lo, double hi) { return v < lo ? lo : (v > hi ? hi : v); }
static float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

void ncp_ctrl_init(ncp_ctrl_t *c, const ncp_model_t *m, ncp_policy_t *pol) {
    memset(c, 0, sizeof *c);
    const ncp_header_t *h = &m->hdr;
    c->m = m;
    c->pol = pol;
    c->n = h->n_links;
    c->state_dim = 2 + 2 * h->n_links;
    c->obs_dim = h->obs_dim;
    c->n_goals = h->n_goals;
    ncp_dyn_init(&c->dyn, h);
    double pos_std[NCP_MAX_POS];
    ncp_ekf_pos_std(h, pos_std);
    ncp_ekf_init(&c->ekf, &c->dyn, pos_std, h->ekf_force_std, h->ekf_accel_std);
    c->mode = NCP_MODE_AI;
    c->x_ref = 0.0;
    c->manual_kp = 150.0;
    c->manual_kd = 20.0;
    c->manual_fmax = 10.0;
    ncp_ctrl_set_goal(c, 0);
    ncp_ctrl_reset(c);
}

void ncp_ctrl_reset(ncp_ctrl_t *c) {
    c->est_ready = false;
    memset(c->est, 0, sizeof c->est);
    memset(c->prev_pos, 0, sizeof c->prev_pos);
    memset(c->vel, 0, sizeof c->vel);
    c->prev_cmd = 0.0;
    memset(c->cmd_hist, 0, sizeof c->cmd_hist);
    memset(c->delay_buf, 0, sizeof c->delay_buf);
    c->delay_head = 0;
    c->prev_applied = 0.0;
    c->est_force = 0.0;
    c->action = 0.0f;
    c->settled_steps = 0;
    c->ext_force = 0.0;
    c->cmd_base = 0.0;
}

void ncp_ctrl_set_goal(ncp_ctrl_t *c, int goal) {
    if (goal < 0 || goal >= c->n_goals) goal = 0;
    c->goal = goal;
    ncp_goal_target(c->n, goal, c->target);
    c->settled_steps = 0;
}

void ncp_ctrl_set_x_ref(ncp_ctrl_t *c, double x_ref) {
    const double lim = 0.5 * c->m->hdr.x_lim;
    c->x_ref = clampd(x_ref, -lim, lim);
}

bool ncp_ctrl_settled(const ncp_ctrl_t *c) {
    return c->settled_steps >= (int)c->m->hdr.goal_settle_steps;
}

static void estimate(ncp_ctrl_t *c, const double *pos) {
    const ncp_header_t *h = &c->m->hdr;
    const int n = c->n;
    if (h->flags & NCP_FLAG_EKF) {
        if (!c->est_ready)
            ncp_ekf_reset(&c->ekf, pos);
        else
            ncp_ekf_update(&c->ekf, c->est_force, pos);
        memcpy(c->est, c->ekf.x, sizeof(double) * (size_t)c->state_dim);
    } else {
        if (!c->est_ready) { /* first reading: rates unknown -> 0 */
            for (int i = 0; i <= n; i++) c->vel[i] = 0.0;
        } else {
            const double a = h->velocity_filter;
            for (int i = 0; i <= n; i++) {
                const double diff = (pos[i] - c->prev_pos[i]) / h->dt;
                c->vel[i] = a * c->vel[i] + (1.0 - a) * diff;
            }
        }
        for (int i = 0; i <= n; i++) c->prev_pos[i] = pos[i];
        c->est[0] = pos[0];
        c->est[1] = c->vel[0];
        for (int k = 0; k < n; k++) {
            c->est[2 + 2 * k] = pos[k + 1];
            c->est[3 + 2 * k] = c->vel[k + 1];
        }
    }
    c->est_ready = true;
}

static bool at_goal(const ncp_ctrl_t *c) {
    const ncp_header_t *h = &c->m->hdr;
    for (int k = 0; k < c->n; k++) {
        const double d = remainder(c->est[2 + 2 * k] - c->target[k], NCP_TWO_PI);
        if (!(fabs(d) < h->goal_tol_angle)) return false;
        if (!(fabs(c->est[3 + 2 * k]) < h->goal_tol_vel)) return false;
    }
    return true;
}

static void build_obs(ncp_ctrl_t *c) {
    const ncp_header_t *h = &c->m->hdr;
    const int n = c->n;
    float *o = c->obs;
    int j = 0;
    o[j++] = (float)(c->est[0] - c->x_ref);
    o[j++] = (float)c->est[1];
    for (int k = 0; k < n; k++) {
        const double th = c->est[2 + 2 * k];
        o[j++] = (float)cos(th);
        o[j++] = (float)sin(th);
        o[j++] = (float)c->est[3 + 2 * k];
    }
    if (h->flags & NCP_FLAG_GOAL)
        for (int k = 0; k < n; k++) o[j++] = (float)cos(c->target[k]);
    const float fmax = (float)h->force_max;
    for (int i = 0; i < h->action_history; i++) o[j++] = (float)c->cmd_hist[i] / fmax;
    for (; j < NCP_MAX_OBS; j++) o[j] = 0.0f;
}

double ncp_ctrl_step(ncp_ctrl_t *c, const double *pos) {
    const ncp_header_t *h = &c->m->hdr;
    const double fmax = h->force_max, dt = h->dt;
    const bool delta = (h->flags & NCP_FLAG_DELTA) != 0;

    /* 1. estimate */
    estimate(c, pos);
    c->settled_steps = at_goal(c) ? c->settled_steps + 1 : 0;

    /* 2. observation */
    build_obs(c);

    /* 3-4. action -> command */
    /* ext_force (excitation) is superposed on the target before the rule and clip. */
    const double ext = c->ext_force;
    double raw;
    if (c->mode == NCP_MODE_AI) {
        const float a = ncp_policy_act(c->pol, c->obs);
        c->action = a;
        if (delta) {
            const float rate = clampf(a / (float)fmax, -1.0f, 1.0f);
            c->cmd_base = clampd(c->cmd_base + (double)rate * h->force_slew * dt, -fmax, fmax);
            raw = ext != 0.0 ? clampd(c->cmd_base + ext, -fmax, fmax) : c->cmd_base;
        } else {
            raw = ext != 0.0 ? clampd((double)a + ext, -fmax, fmax)
                             : (double)clampf(a, -(float)fmax, (float)fmax);
        }
    } else {
        const double cap = c->manual_fmax < fmax ? c->manual_fmax : fmax;
        const double f = clampd(c->manual_kp * (c->x_ref - c->est[0]) - c->manual_kd * c->est[1],
                                -cap, cap) + ext;
        c->action = NAN;
        if (delta) { /* the command still moves at most slew * dt per tick */
            const double step = h->force_slew * dt;
            raw = clampd(c->prev_cmd + clampd(f - c->prev_cmd, -step, step), -fmax, fmax);
        } else {
            raw = clampd(f, -fmax, fmax);
        }
        c->cmd_base = raw; /* switching to AI continues from the current command */
    }
    for (int i = NCP_MAX_HIST - 1; i > 0; i--) c->cmd_hist[i] = c->cmd_hist[i - 1];
    c->cmd_hist[0] = raw;
    c->prev_cmd = raw;

    /* 5. actuator model for the estimator: delay -> slew -> available_force */
    double applied = raw;
    if (h->delay_steps > 0) {
        applied = c->delay_buf[c->delay_head];
        c->delay_buf[c->delay_head] = raw;
        c->delay_head = (c->delay_head + 1) % h->delay_steps;
    }
    if (h->force_slew > 0.0) {
        const double dF = h->force_slew * dt;
        applied = c->prev_applied + clampd(applied - c->prev_applied, -dF, dF);
    }
    c->prev_applied = applied;
    c->est_force = ncp_available_force(&c->dyn, applied, c->est[1]);
    return raw;
}
