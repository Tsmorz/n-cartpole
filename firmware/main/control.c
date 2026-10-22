/* control.c — the control task, pinned to core 1 at priority 22.
 *
 * It creates its own gptimer and registers the alarm callback from inside the task,
 * so the timer interrupt is allocated on core 1; the ISR only notifies the task.
 * Each tick: drain commands (zero timeout) -> read the plant -> ncp_ctrl_step ->
 * safety checks -> actuate (drive when ARMED; the simulator always advances) ->
 * random-goal sequencer -> timing -> telemetry frame (every 4th tick) + status.
 * Nothing here logs, allocates or blocks on core 0: the tlm task logs on our behalf
 * (it watches the published status) and persists zeroing results to NVS. */
#include "control.h"

#include <math.h>
#include <string.h>

#include "driver/gptimer.h"
#include "esp_attr.h"
#include "esp_random.h"
#include "esp_task_wdt.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "hal.h"
#include "ncp_ctrl.h"
#include "ncp_excite.h"
#include "ncp_record.h"
#include "record.h"
#include "ncp_telemetry.h"
#include "telemetry.h"

#define CMD_QUEUE_LEN 16
#define BUDGET 0.8           /* execution <= 80 % of the period */
#define MAX_STRETCH 20       /* sim slow motion: period <= 20 x nominal */
#define WDT_EVERY 20         /* feed the task watchdog every N ticks */
#define RANDOM_DELAY_S 1.0   /* random goal: next goal 1 s after settling */
#define OFFRAIL_RESET_S 1.5  /* simulator: back to DD 1.5 s after leaving the rail */
#ifdef CONFIG_NCP_ENC_ERROR_LIMIT
#define ENC_ERROR_LIMIT CONFIG_NCP_ENC_ERROR_LIMIT
#else
#define ENC_ERROR_LIMIT 3
#endif

static QueueHandle_t s_queue;
static TaskHandle_t s_task;
static gptimer_handle_t s_timer;
static const model_t *s_initial;

/* ---- state owned by the control task ---- */
static const model_t *s_model;
static ncp_policy_t s_pol;
static ncp_ctrl_t s_ctrl;
static uint8_t s_run = NCP_RUN_IDLE;
static uint8_t s_fault = CTRL_FAULT_NONE;
static bool s_random;
static int s_random_wait = -1; /* ticks until the next random goal, -1: not counting */
static int s_offrail_ticks;
static uint32_t s_tick;
static uint32_t s_nominal_us = CTRL_DEFAULT_PERIOD_US, s_period_us = CTRL_DEFAULT_PERIOD_US;
static int s_window;      /* ticks per statistics window (~1 s) */
static double s_pos[NCP_MAX_POS];
static float s_cmd;
static bool s_overrun; /* the previous tick ran past its period */
static bool s_timed;   /* a full timing window has been measured for this model */
static ncp_excite_t s_exc;
static bool s_exc_on, s_exc_from_rec;
static double s_ext; /* excitation force of this tick (N) */
static uint32_t s_fault_count, s_last_fault_tick;
static uint8_t s_last_fault = CTRL_FAULT_NONE;
static bool s_cal_angles = HAL_IS_SIM, s_cal_cart = HAL_IS_SIM; /* set before the task starts */

/* timing window */
static uint32_t s_w_n, s_w_max, s_misses;
static uint64_t s_w_exec, s_w_pol, s_w_est;
static uint32_t s_exec_avg, s_exec_max, s_pol_avg, s_est_avg;

/* Filled by the --wrap'ed core functions during ncp_ctrl_step (control task only). */
static uint32_t s_pol_us, s_est_us;

/* ---- shared with core 0 ---- */
static volatile uint32_t s_seq;
static ctrl_status_t s_pub;
static const model_t *volatile s_ack_model;
static volatile bool s_ack_maint;
static volatile bool s_pending_angles, s_pending_cart;
static double s_zero_off1, s_zero_off2, s_zero_x;

/* ---------- timing hooks (main/CMakeLists.txt links with --wrap) ---------- */

float __real_ncp_policy_act(ncp_policy_t *p, const float *obs);
float __wrap_ncp_policy_act(ncp_policy_t *p, const float *obs);
void __real_ncp_ekf_update(ncp_ekf_t *e, double applied_force, const double *pos);
void __wrap_ncp_ekf_update(ncp_ekf_t *e, double applied_force, const double *pos);

float __wrap_ncp_policy_act(ncp_policy_t *p, const float *obs) {
    int64_t t0 = esp_timer_get_time();
    float a = __real_ncp_policy_act(p, obs);
    s_pol_us += (uint32_t)(esp_timer_get_time() - t0);
    return a;
}

void __wrap_ncp_ekf_update(ncp_ekf_t *e, double applied_force, const double *pos) {
    int64_t t0 = esp_timer_get_time();
    __real_ncp_ekf_update(e, applied_force, pos);
    s_est_us += (uint32_t)(esp_timer_get_time() - t0);
}

/* ---------- helpers ---------- */

static bool IRAM_ATTR on_alarm(gptimer_handle_t timer, const gptimer_alarm_event_data_t *edata,
                               void *ctx) {
    (void)timer;
    (void)edata;
    BaseType_t woken = pdFALSE;
    vTaskNotifyGiveFromISR((TaskHandle_t)ctx, &woken);
    return woken == pdTRUE;
}

static void set_period(uint32_t us) {
    if (us == s_period_us) {
        return;
    }
    static gptimer_alarm_config_t alarm; /* static: may be read by an IRAM driver path */
    alarm.alarm_count = us;
    alarm.reload_count = 0;
    alarm.flags.auto_reload_on_alarm = 1;
    if (gptimer_set_alarm_action(s_timer, &alarm) == ESP_OK) {
        s_period_us = us;
    }
}

static double x_lim(void) { return s_model != NULL ? s_model->hdr.x_lim : 0.0; }
static double period_s(void) { return s_nominal_us * 1e-6; }
static int all_down(void) { return (1 << s_model->hdr.n_links) - 1; }

static void publish(void) {
    uint32_t seq = s_seq + 1;
    s_seq = seq; /* odd: writing */
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    ctrl_status_t *p = &s_pub;
    p->ticks = s_tick;
    p->run = s_run;
    p->fault = s_fault;
    p->mode = s_model != NULL ? (uint8_t)s_ctrl.mode : NCP_MODE_AI;
    p->goal = s_model != NULL ? (uint8_t)s_ctrl.goal : 0;
    p->random = s_random;
    p->settled = s_model != NULL && ncp_ctrl_settled(&s_ctrl);
    p->estop = ncp_hal_estop();
    p->off_rail = s_model != NULL && ncp_hal_off_rail(s_pos, x_lim());
    p->x_ref = s_model != NULL ? (float)s_ctrl.x_ref : 0.0f;
    p->period_us = s_period_us;
    p->nominal_period_us = s_nominal_us;
    p->exec_avg_us = s_exec_avg;
    p->exec_max_us = s_exec_max;
    p->policy_avg_us = s_pol_avg;
    p->est_avg_us = s_est_avg;
    p->misses = s_misses;
    p->read_errors = ncp_hal_read_errors();
    p->tlm_dropped = tlm_dropped();
    p->rt_factor = (float)s_nominal_us / (float)s_period_us;
    p->budget_ok = s_timed && s_exec_avg <= (uint32_t)(BUDGET * s_nominal_us);
    p->model = s_model;
    p->excite_kind = s_exc_on ? (uint8_t)s_exc.p.kind : NCP_EXC_NONE;
    p->excite_active = s_exc_on && s_exc.active;
    p->excite_t = s_exc_on ? (float)s_exc.t : 0.0f;
    p->dt = (float)period_s();
    p->fault_count = s_fault_count;
    p->last_fault = s_last_fault;
    p->last_fault_tick = s_last_fault_tick;
    p->cal_angles = s_cal_angles;
    p->cal_cart = s_cal_cart;
    ncp_hal_io(&p->io);
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    s_seq = seq + 1; /* even: stable */
}

static void rec_stop(void);

static void excite_stop(void) {
    if (s_exc_on) {
        ncp_excite_stop(&s_exc);
    }
    s_exc_on = false;
    s_exc_from_rec = false;
    s_ext = 0.0;
}

static void attach_model(const model_t *m) {
    rec_stop(); /* the period may change */
    excite_stop();
    s_model = m;
    if (s_run != NCP_RUN_MAINT) {
        s_run = NCP_RUN_IDLE;
    }
    s_fault = CTRL_FAULT_NONE;
    s_random = false;
    s_random_wait = -1;
    s_offrail_ticks = 0;
    memset(s_pos, 0, sizeof(s_pos));
    ncp_hal_set_model(m != NULL ? &m->hdr : NULL);
    if (m != NULL) {
        ncp_policy_init(&s_pol, &m->m, m->scratch);
        ncp_ctrl_init(&s_ctrl, &m->m, &s_pol);
        ncp_ctrl_set_goal(&s_ctrl, all_down() < m->n_goals ? all_down() : 0);
        s_nominal_us = (uint32_t)lround(m->hdr.dt * 1e6);
        if (s_nominal_us < 500) {
            s_nominal_us = 500;
        }
    } else {
        s_nominal_us = CTRL_DEFAULT_PERIOD_US;
    }
    s_window = (int)(1000000u / s_nominal_us);
    if (s_window < 1) {
        s_window = 1;
    }
    s_overrun = false;
    s_timed = false;
    s_w_n = 0;
    s_w_exec = s_w_pol = s_w_est = 0;
    s_w_max = 0;
    s_exec_avg = s_exec_max = s_pol_avg = s_est_avg = 0;
    set_period(s_nominal_us);
    s_ack_model = m;
}

static bool can_arm(void) {
    if (s_model == NULL || s_run == NCP_RUN_MAINT || ncp_hal_estop() ||
        ncp_hal_off_rail(s_pos, x_lim())) {
        return false;
    }
#if !HAL_IS_SIM
    if (s_model->hdr.n_links != 2 || !s_timed || s_exec_avg > (uint32_t)(BUDGET * s_nominal_us) ||
        ncp_hal_read_errors() > 0 || !s_cal_angles || !s_cal_cart) {
        return false;
    }
#endif
    return true;
}

static void trip(ctrl_fault_t f) {
    s_run = NCP_RUN_FAULT;
    s_fault = (uint8_t)f;
    s_fault_count++;
    s_last_fault = (uint8_t)f;
    s_last_fault_tick = s_tick;
}

static void rec_stop(void) {
    if (g_rec.state != REC_RECORDING) {
        return;
    }
    g_rec.hdr.n_samples = g_rec.count;
    g_rec.hdr.dropped = g_rec.dropped;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    g_rec.state = REC_DONE;
    if (s_exc_from_rec) {
        excite_stop();
    }
}

static void rec_start(void) {
    if (g_rec.state != REC_STARTING) {
        return;
    }
    g_rec.count = 0;
    g_rec.dropped = 0;
    g_rec.hdr.dt = period_s();
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    g_rec.state = REC_RECORDING;
    if (g_rec.excite && s_model != NULL) {
        ncp_excite_start(&s_exc, &g_rec.exc, period_s());
        s_exc_on = true;
        s_exc_from_rec = true;
    }
}

/* One sample per tick while recording (PSRAM; no locks, no allocation). */
static void rec_append(int64_t t0, uint32_t period_meas, uint32_t missed, bool read_ok,
                       double out) {
    if (g_rec.state != REC_RECORDING) {
        return;
    }
    uint32_t i = g_rec.count;
    if (i == 0) {
        g_rec.hdr.start_us = (uint64_t)t0;
    } else {
        g_rec.dropped = g_rec.dropped + missed;
    }
    if (i >= g_rec.capacity) {
        rec_stop();
        return;
    }
    ncp_rec_sample_t r;
    memset(&r, 0, sizeof(r));
    r.tick = s_tick;
    r.t_us = (uint32_t)((uint64_t)t0 - g_rec.hdr.start_us);
    const int np = s_model != NULL ? s_ctrl.n + 1 : (HAL_IS_SIM ? 0 : 3);
    for (int k = 0; k < np && k < NCP_MAX_POS; k++) {
        r.pos[k] = (float)s_pos[k];
    }
    r.cmd = (float)out;
    r.applied = (float)(s_run == NCP_RUN_ARMED && s_model != NULL ? s_ctrl.est_force : 0.0);
    r.excite = (float)s_ext;
    uint8_t flags = 0;
    const double *truth = ncp_hal_sim_truth();
    for (int k = 0; k < NCP_MAX_STATE; k++) {
        r.truth[k] = NAN;
    }
    if (s_model != NULL) {
        for (int k = 0; k < s_ctrl.state_dim && k < NCP_MAX_STATE; k++) {
            r.est[k] = (float)s_ctrl.est[k];
            if (truth != NULL) {
                r.truth[k] = (float)truth[k];
            }
        }
        if (truth != NULL) {
            for (int k = s_ctrl.state_dim; k < NCP_MAX_STATE; k++) {
                r.truth[k] = 0.0f;
            }
            flags |= NCP_REC_TRUTH;
        }
        r.mode = (uint8_t)s_ctrl.mode;
        r.goal = (uint8_t)s_ctrl.goal;
    }
    r.run = s_run;
    if (s_run == NCP_RUN_ARMED) {
        flags |= NCP_REC_ARMED;
    }
    if (s_exc_on && s_exc.active && s_run == NCP_RUN_ARMED) {
        flags |= NCP_REC_EXCITE;
    }
    if (!read_ok) {
        flags |= NCP_REC_ENC_ERR;
    }
    r.flags = flags;
    r.period_us = (uint16_t)(period_meas > 0xFFFF ? 0xFFFF : period_meas);
    memcpy(&g_rec.buf[i], &r, sizeof(r));
    __atomic_store_n(&g_rec.count, i + 1, __ATOMIC_RELEASE);
    if (i + 1 >= g_rec.capacity) {
        rec_stop(); /* full */
    }
}

static void handle(const ctrl_cmd_t *c) {
    switch (c->kind) {
    case CTRL_CMD_SET_MODEL:
        attach_model(c->model);
        return;
    case CTRL_CMD_MAINT:
        if (c->i) {
            rec_stop();
            excite_stop();
            s_run = NCP_RUN_MAINT;
        } else if (s_run == NCP_RUN_MAINT) {
            s_run = NCP_RUN_IDLE;
        }
        s_ack_maint = c->i != 0;
        return;
    case CTRL_CMD_RECORD_START:
        rec_start();
        return;
    case CTRL_CMD_RECORD_STOP:
        rec_stop();
        return;
    case CTRL_CMD_ZERO_ANGLES:
        if (!HAL_IS_SIM && s_run != NCP_RUN_ARMED) {
            ncp_hal_rig_zero_angles(&s_zero_off1, &s_zero_off2);
            s_cal_angles = true;
            if (s_model != NULL) {
                ncp_ctrl_reset(&s_ctrl);
            }
            __atomic_thread_fence(__ATOMIC_SEQ_CST);
            s_pending_angles = true;
        }
        return;
    case CTRL_CMD_ZERO_CART:
        if (!HAL_IS_SIM && s_run != NCP_RUN_ARMED) {
            ncp_hal_rig_zero_cart(&s_zero_x);
            s_cal_cart = true;
            if (s_model != NULL) {
                ncp_ctrl_reset(&s_ctrl);
            }
            __atomic_thread_fence(__ATOMIC_SEQ_CST);
            s_pending_cart = true;
        }
        return;
    default:
        break;
    }
    if (s_model == NULL) {
        return;
    }
    switch (c->kind) {
    case CTRL_CMD_ARM:
        if (c->i) {
            if (s_run != NCP_RUN_ARMED && can_arm()) {
                ncp_ctrl_reset(&s_ctrl);
                s_run = NCP_RUN_ARMED;
                s_fault = CTRL_FAULT_NONE;
            }
        } else if (s_run == NCP_RUN_ARMED || s_run == NCP_RUN_FAULT) {
            s_run = NCP_RUN_IDLE;
            s_fault = CTRL_FAULT_NONE;
            s_ctrl.ext_force = 0.0;
        }
        break;
    case CTRL_CMD_MODE:
        s_ctrl.mode = c->i == NCP_MODE_MANUAL ? NCP_MODE_MANUAL : NCP_MODE_AI;
        break;
    case CTRL_CMD_GOAL:
        if (c->i >= 0 && c->i < s_model->n_goals) {
            s_random = false;
            s_random_wait = -1;
            ncp_ctrl_set_goal(&s_ctrl, (int)c->i);
        }
        break;
    case CTRL_CMD_RANDOM:
        s_random = c->i != 0 && s_model->n_goals > 1;
        s_random_wait = -1;
        break;
    case CTRL_CMD_X_REF: {
        double lim = 0.5 * s_model->hdr.x_lim;
        double x = c->f;
        ncp_ctrl_set_x_ref(&s_ctrl, x > lim ? lim : (x < -lim ? -lim : x));
        break;
    }
    case CTRL_CMD_RESET_POSE:
        if (HAL_IS_SIM && c->i >= 0 && c->i < (1 << s_model->hdr.n_links)) {
            ncp_hal_sim_reset((int)c->i);
            ncp_ctrl_reset(&s_ctrl);
            s_offrail_ticks = 0;
            if (s_run == NCP_RUN_FAULT) {
                s_run = NCP_RUN_IDLE;
                s_fault = CTRL_FAULT_NONE;
            }
        }
        break;
    case CTRL_CMD_EXCITE:
        excite_stop();
        if (c->exc.kind != NCP_EXC_NONE) {
            ncp_excite_start(&s_exc, &c->exc, period_s());
            s_exc_on = true;
        }
        break;
    default:
        break;
    }
}

static void random_goal_step(void) {
    if (!s_random) {
        return;
    }
    if (s_random_wait < 0) {
        if (ncp_ctrl_settled(&s_ctrl)) {
            s_random_wait = (int)lround(RANDOM_DELAY_S / s_model->hdr.dt);
        }
        return;
    }
    if (--s_random_wait > 0) {
        return;
    }
    int n = s_model->n_goals;
    int g = (int)(esp_random() % (uint32_t)(n - 1));
    if (g >= s_ctrl.goal) {
        g++; /* uniformly among the other goals */
    }
    ncp_ctrl_set_goal(&s_ctrl, g);
    s_random_wait = -1;
}

static void fill_frame(ncp_tlm_frame_t *f, uint32_t exec_us, uint32_t period_meas) {
    memset(f, 0, sizeof(*f));
    f->tick = s_tick;
    f->run = s_run;
    uint8_t flags = HAL_IS_SIM ? NCP_TLM_SIM : 0;
    if (s_model != NULL) {
        const int sd = s_ctrl.state_dim, np = s_ctrl.n + 1, od = s_ctrl.obs_dim;
        f->mode = (uint8_t)s_ctrl.mode;
        f->goal = (uint8_t)s_ctrl.goal;
        for (int i = 0; i < sd && i < NCP_MAX_STATE; i++) {
            f->est[i] = (float)s_ctrl.est[i];
        }
        for (int i = 0; i < np && i < NCP_MAX_POS; i++) {
            f->meas[i] = (float)s_pos[i];
        }
        const double *truth = ncp_hal_sim_truth();
        if (truth != NULL) {
            for (int i = 0; i < sd && i < NCP_MAX_STATE; i++) {
                f->truth[i] = (float)truth[i];
            }
            flags |= NCP_TLM_TRUTH;
        }
        for (int i = 0; i < od && i < NCP_MAX_OBS; i++) {
            f->obs[i] = s_ctrl.obs[i];
        }
        f->cmd = s_cmd;
        /* sim: the force that acted; rig: the actuator model's estimate when driving */
        f->applied = (float)(HAL_IS_SIM ? ncp_hal_sim_applied()
                                        : (s_run == NCP_RUN_ARMED ? s_ctrl.est_force : 0.0));
        f->x_ref = (float)s_ctrl.x_ref;
        if (ncp_ctrl_settled(&s_ctrl)) {
            flags |= NCP_TLM_SETTLED;
        }
    }
#if !HAL_IS_SIM
    if (s_model == NULL) {
        for (int i = 0; i < 3; i++) {
            f->meas[i] = (float)s_pos[i];
        }
    }
#endif
    if (s_random) {
        flags |= NCP_TLM_RANDOM;
    }
    f->flags = flags;
    f->loop_us = (uint16_t)(exec_us > 0xFFFF ? 0xFFFF : exec_us);
    f->policy_us = (uint16_t)(s_pol_us > 0xFFFF ? 0xFFFF : s_pol_us);
    f->est_us = (uint16_t)(s_est_us > 0xFFFF ? 0xFFFF : s_est_us);
    f->period_us = (uint16_t)(period_meas > 0xFFFF ? 0xFFFF : period_meas);
}

static void window_stats(uint32_t exec_us) {
    s_w_n++;
    s_w_exec += exec_us;
    s_w_pol += s_pol_us;
    s_w_est += s_est_us;
    if (exec_us > s_w_max) {
        s_w_max = exec_us;
    }
    if ((int)s_w_n < s_window) {
        return;
    }
    s_exec_avg = (uint32_t)(s_w_exec / s_w_n);
    s_exec_max = s_w_max;
    s_pol_avg = (uint32_t)(s_w_pol / s_w_n);
    s_est_avg = (uint32_t)(s_w_est / s_w_n);
    s_timed = true;
    s_w_n = 0;
    s_w_exec = s_w_pol = s_w_est = 0;
    s_w_max = 0;
#if HAL_IS_SIM
    /* Slow motion instead of overruns: stretch the period so the worst tick of the
     * window fits the budget; shrink back (with hysteresis) when it fits again. */
    uint32_t want = (uint32_t)ceil(s_exec_max / BUDGET / 100.0) * 100u;
    if (want < s_nominal_us) {
        want = s_nominal_us;
    }
    if (want > MAX_STRETCH * s_nominal_us) {
        want = MAX_STRETCH * s_nominal_us;
    }
    if (want > s_period_us || want < (s_period_us * 9u) / 10u) {
        set_period(want);
    }
#endif
}

/* One control period. t0: wake-up time; missed: periods lost since the last wake. */
static void tick(int64_t t0, uint32_t period_meas, uint32_t missed) {
    ctrl_cmd_t cmd;
    while (xQueueReceive(s_queue, &cmd, 0) == pdTRUE) {
        handle(&cmd);
    }
    s_tick++;
    s_pol_us = 0;
    s_est_us = 0;

    bool read_ok = true;
    double out = 0.0;
    if (s_model == NULL) {
#if !HAL_IS_SIM
        read_ok = ncp_hal_read(s_pos); /* readings (and recordings) work without a model */
#endif
        ncp_hal_apply(false, 0.0);
    } else {
        read_ok = ncp_hal_read(s_pos);
        /* Excitation only while armed (it advances only then), capped at the manual cap. */
        s_ext = 0.0;
        if (s_exc_on && s_run == NCP_RUN_ARMED) {
            double e = ncp_excite_next(&s_exc);
            double cap = s_ctrl.manual_fmax;
            s_ext = e > cap ? cap : (e < -cap ? -cap : e);
        }
        s_ctrl.ext_force = s_ext;
        double cmd_n = ncp_ctrl_step(&s_ctrl, s_pos);
        s_cmd = (float)cmd_n;

        /* Safety, before anything reaches the drive. */
        const bool off_rail = ncp_hal_off_rail(s_pos, x_lim());
        if (s_run == NCP_RUN_ARMED) {
            if (ncp_hal_estop()) {
                trip(CTRL_FAULT_ESTOP);
            } else if (off_rail) {
                trip(CTRL_FAULT_OFF_RAIL);
            } else if (!read_ok && ncp_hal_read_errors() >= ENC_ERROR_LIMIT) {
                trip(CTRL_FAULT_ENCODER);
            }
#if !HAL_IS_SIM
            else if (missed > 0 || s_overrun) {
                trip(CTRL_FAULT_DEADLINE);
            }
#endif
        }
        if (!read_ok && s_run != NCP_RUN_ARMED) {
            ncp_ctrl_reset(&s_ctrl); /* glitch while idle: restart the estimator */
        }
        const bool armed = s_run == NCP_RUN_ARMED;
        out = armed ? cmd_n : 0.0;
        ncp_hal_apply(armed, out);

#if HAL_IS_SIM
        if (off_rail) {
            if (++s_offrail_ticks >= (int)lround(OFFRAIL_RESET_S / s_model->hdr.dt)) {
                ncp_hal_sim_reset(all_down());
                ncp_ctrl_reset(&s_ctrl);
                s_offrail_ticks = 0;
            }
        } else {
            s_offrail_ticks = 0;
        }
#endif
        random_goal_step();
    }
    rec_append(t0, period_meas, missed, read_ok, out);

    if (s_run != NCP_RUN_MAINT) {
        s_misses += missed;
    }
    uint32_t exec_us = (uint32_t)(esp_timer_get_time() - t0);
    s_overrun = exec_us > s_period_us && s_run != NCP_RUN_MAINT;
    if (s_overrun) {
        s_misses++;
    }
    window_stats(exec_us);

    if (s_tick % CTRL_TLM_DECIMATION == 0) {
        ncp_tlm_frame_t f;
        fill_frame(&f, exec_us, period_meas);
        tlm_push(&f);
        publish();
    }
    if (s_tick % WDT_EVERY == 0) {
        esp_task_wdt_reset();
    }
}

static void control_task(void *arg) {
    (void)arg;
    esp_task_wdt_add(NULL);

    gptimer_config_t tc = {
        .clk_src = GPTIMER_CLK_SRC_DEFAULT,
        .direction = GPTIMER_COUNT_UP,
        .resolution_hz = 1000000, /* 1 tick = 1 us */
    };
    ESP_ERROR_CHECK(gptimer_new_timer(&tc, &s_timer));
    gptimer_event_callbacks_t cbs = {.on_alarm = on_alarm};
    /* Registering here (core 1) allocates the timer interrupt on core 1. */
    ESP_ERROR_CHECK(gptimer_register_event_callbacks(s_timer, &cbs, xTaskGetCurrentTaskHandle()));
    s_period_us = 0; /* force set_period */
    attach_model(s_initial);
    publish();
    ESP_ERROR_CHECK(gptimer_enable(s_timer));
    ESP_ERROR_CHECK(gptimer_start(s_timer));

    int64_t last = esp_timer_get_time();
    for (;;) {
        uint32_t n = ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(500));
        int64_t t0 = esp_timer_get_time();
        if (n == 0) { /* timer stalled (e.g. flash writes in MAINT) */
            esp_task_wdt_reset();
            last = t0;
            continue;
        }
        uint32_t period_meas = (uint32_t)(t0 - last);
        last = t0;
        tick(t0, period_meas, n - 1);
    }
}

/* ---------- core 0 interface ---------- */

void control_set_calibration(bool angles, bool cart) {
    s_cal_angles = HAL_IS_SIM || angles;
    s_cal_cart = HAL_IS_SIM || cart;
}

esp_err_t control_start(const model_t *model) {
    s_initial = model;
    s_ack_model = NULL;
    s_queue = xQueueCreate(CMD_QUEUE_LEN, sizeof(ctrl_cmd_t));
    if (s_queue == NULL) {
        return ESP_ERR_NO_MEM;
    }
    BaseType_t ok = xTaskCreatePinnedToCore(control_task, "ctrl", 12288, NULL, CTRL_PRIO, &s_task,
                                            CTRL_CORE);
    return ok == pdPASS ? ESP_OK : ESP_ERR_NO_MEM;
}

bool control_send(const ctrl_cmd_t *cmd) {
    return s_queue != NULL && xQueueSend(s_queue, cmd, 0) == pdTRUE;
}

void control_get_status(ctrl_status_t *out) {
    for (int tries = 0; tries < 1000; tries++) {
        uint32_t a = s_seq;
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
        if (a & 1u) {
            continue;
        }
        memcpy(out, &s_pub, sizeof(*out));
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
        if (s_seq == a) {
            return;
        }
    }
    memcpy(out, &s_pub, sizeof(*out)); /* writer stuck mid-update: best effort */
}

uint32_t control_ticks(void) {
    ctrl_status_t st;
    control_get_status(&st);
    return st.ticks;
}

const char *control_arm_refusal(const ctrl_status_t *st) {
    if (st->model == NULL) {
        return "no model on the device";
    }
    if (st->run == NCP_RUN_MAINT) {
        return "update in progress";
    }
    if (st->estop) {
        return "e-stop active";
    }
    if (st->off_rail) {
        return "cart is off the rail";
    }
#if !HAL_IS_SIM
    if (st->model->hdr.n_links != 2) {
        return "the rig has 2 links; this model has a different count";
    }
    if (!st->budget_ok) {
        return "loop over budget: the model is too slow for this period";
    }
    if (st->read_errors > 0) {
        return "encoder read errors";
    }
    if (!st->cal_angles || !st->cal_cart) {
        return !st->cal_angles ? "angles not zeroed: hang the links at rest and Zero angles"
                               : "cart not zeroed: centre it and Zero cart";
    }
#endif
    return NULL;
}

static bool wait_until(bool (*done)(const void *), const void *arg, int timeout_ms) {
    for (int t = 0; t <= timeout_ms; t += 2) {
        if (done(arg)) {
            return true;
        }
        vTaskDelay(pdMS_TO_TICKS(2));
    }
    return done(arg);
}

static bool model_acked(const void *m) { return s_ack_model == (const model_t *)m; }
static bool maint_acked(const void *on) { return s_ack_maint == *(const bool *)on; }

bool control_wait_model(const model_t *model, int timeout_ms) {
    return wait_until(model_acked, model, timeout_ms);
}

bool control_wait_maint(bool on, int timeout_ms) { return wait_until(maint_acked, &on, timeout_ms); }

bool control_take_angle_offsets(double *off1, double *off2) {
    if (!s_pending_angles) {
        return false;
    }
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    *off1 = s_zero_off1;
    *off2 = s_zero_off2;
    s_pending_angles = false;
    return true;
}

bool control_take_cart_zero(double *x_zero) {
    if (!s_pending_cart) {
        return false;
    }
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    *x_zero = s_zero_x;
    s_pending_cart = false;
    return true;
}

const char *ctrl_fault_str(uint8_t fault) {
    switch (fault) {
    case CTRL_FAULT_OFF_RAIL:
        return "off rail";
    case CTRL_FAULT_ESTOP:
        return "e-stop";
    case CTRL_FAULT_ENCODER:
        return "encoder errors";
    case CTRL_FAULT_DEADLINE:
        return "deadline miss";
    default:
        return "";
    }
}

const char *ctrl_run_str(uint8_t run) {
    switch (run) {
    case NCP_RUN_ARMED:
        return "armed";
    case NCP_RUN_FAULT:
        return "fault";
    case NCP_RUN_MAINT:
        return "maint";
    default:
        return "idle";
    }
}
