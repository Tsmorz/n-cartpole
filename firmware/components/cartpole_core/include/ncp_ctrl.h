/* ncp_ctrl.h — one control tick, exactly as the policy was trained.
 *
 * Reproduces NPendulumCartpole.step/_sense/_make_obs from the controller's
 * side (n_cartpole/env/cartpole.py), so the network sees the observations it
 * was trained on:
 *
 *   1. estimate: first tick after ncp_ctrl_reset -> EKF reset (or diff: zero
 *      velocities); afterwards EKF update with est_force, the force the model
 *      says acted over the last period (or finite differences + EMA).
 *   2. obs = [x - x_ref, xd, cos th_k, sin th_k, thd_k ...]
 *            [+ cos(target_k) per link if goal-conditioned]
 *            [+ cmd_hist[0..action_history) / force_max, newest first].
 *   3. action: AI mode -> policy; MANUAL mode -> PD on the cart to x_ref.
 *   4. command: delta mode -> clip(prev_cmd + clip(a / Fmax, -1, 1) * slew * dt, +-Fmax);
 *      absolute -> clip(a, +-Fmax). Pushed onto cmd_hist; becomes prev_cmd.
 *   5. actuator model (for the EKF only — the drive itself provides the real
 *      delay): delay_steps-step delay line -> slew limit vs the last applied
 *      force -> available_force at the estimated cart speed = est_force.
 *
 * The returned command goes to the drive. x_ref shifts the cart setpoint: the
 * plant is translation-invariant, so the AI balances around x_ref when it sees
 * x - x_ref (|x_ref| is clamped to half the rail). */
#ifndef NCP_CTRL_H
#define NCP_CTRL_H

#include <stdbool.h>

#include "ncp_ekf.h"
#include "ncp_policy.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { NCP_MODE_AI = 0, NCP_MODE_MANUAL = 1 } ncp_mode_t;

typedef struct {
    const ncp_model_t *m;
    ncp_policy_t *pol;
    ncp_dyn_t dyn; /* nominal plant (EKF + available_force) */
    ncp_ekf_t ekf;
    int n, state_dim, obs_dim, n_goals;

    bool est_ready;
    double est[NCP_MAX_STATE]; /* current state estimate */
    double prev_pos[NCP_MAX_POS], vel[NCP_MAX_POS]; /* diff estimator */

    double prev_cmd;               /* last command (N) */
    double cmd_hist[NCP_MAX_HIST]; /* newest first */
    double delay_buf[NCP_MAX_DELAY];
    int delay_head;
    double prev_applied; /* modelled force after the slew limit */
    double est_force;    /* modelled force over the coming period: next EKF input */

    float obs[NCP_MAX_OBS]; /* observation of the last tick */
    float action;           /* policy output of the last tick (N); NAN in MANUAL */

    int goal;
    double target[NCP_MAX_LINKS];
    double x_ref;
    ncp_mode_t mode;
    double manual_kp, manual_kd, manual_fmax; /* PD gains (N/m, N s/m) and force cap (N) */

    int settled_steps; /* consecutive ticks at the goal (goal_tol_*) */

    /* Excitation for system identification (ncp_excite.h), set by the caller
     * before each ncp_ctrl_step: added to this tick's target force (MANUAL: on
     * top of the PD force; AI: on top of the policy's command, as a disturbance)
     * before the slew/rate rule and the force clip, so prev_cmd, cmd_hist and
     * the returned command all include it. 0 = none.
     * AI delta mode: the policy's rate rule integrates cmd_base (without the
     * excitation) and the command is clip(cmd_base + ext_force, +-force_max),
     * so the excitation is superposed on the policy's command instead of being
     * integrated by it. With ext_force = 0 the command is exactly the old one. */
    double ext_force;
    double cmd_base; /* AI delta: the policy's integrated command, ext_force excluded */
} ncp_ctrl_t;

/* Bind to a model and its policy; mode AI, goal 0, x_ref 0, PD 150 / 20, cap 10 N.
 * c->ekf keeps a pointer to c->dyn: do not move or copy *c after init (keep it
 * static or heap-allocated). In MANUAL delta mode the command moves toward the PD
 * force by at most force_slew * dt per tick. */
void ncp_ctrl_init(ncp_ctrl_t *c, const ncp_model_t *m, ncp_policy_t *pol);

/* Forget the estimator and actuator history (on arm, after a sim reset, after a
 * reading glitch). Like env.reset(): cmd_hist, prev_cmd, prev_applied, est_force = 0. */
void ncp_ctrl_reset(ncp_ctrl_t *c);

void ncp_ctrl_set_goal(ncp_ctrl_t *c, int goal); /* resets settled_steps */
void ncp_ctrl_set_x_ref(ncp_ctrl_t *c, double x_ref);

/* One tick. pos: n + 1 readings [x, th0, th1, ...] (absolute angles, unwrapped).
 * Returns the force command for the drive (N). */
double ncp_ctrl_step(ncp_ctrl_t *c, const double *pos);

/* true once settled_steps >= goal_settle_steps. */
bool ncp_ctrl_settled(const ncp_ctrl_t *c);

#ifdef __cplusplus
}
#endif
#endif
