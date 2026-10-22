/* control.h — the control task (core 1, prio 22) and its interface to core 0.
 *
 * Core 0 talks to it only through a command queue (read with zero timeout each tick),
 * a seqlock-protected status snapshot, a few acknowledgement words, and the telemetry
 * ring (telemetry.h). See docs/design.md "Cores and tasks". */
#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "esp_err.h"
#include "model_store.h"
#include "ncp_ctrl.h"      /* ncp_mode_t */
#include "ncp_excite.h"
#include "ncp_telemetry.h" /* NCP_RUN_* */

#define CTRL_CORE 1
#define CTRL_PRIO 22
#define CTRL_DEFAULT_PERIOD_US 5000 /* without a model */
#define CTRL_TLM_DECIMATION 4      /* telemetry frame every 4th tick: 50 Hz at 200 Hz */

typedef enum {
    CTRL_CMD_ARM,         /* i: 1 arm, 0 disarm */
    CTRL_CMD_MODE,        /* i: ncp_mode_t */
    CTRL_CMD_GOAL,        /* i: goal index (stops random cycling) */
    CTRL_CMD_RANDOM,      /* i: 0/1 */
    CTRL_CMD_X_REF,       /* f: m */
    CTRL_CMD_RESET_POSE,  /* i: goal index (sim) */
    CTRL_CMD_ZERO_ANGLES, /* rig, disarmed */
    CTRL_CMD_ZERO_CART,   /* rig, disarmed */
    CTRL_CMD_SET_MODEL,   /* model (NULL: let go of the current one) */
    CTRL_CMD_MAINT,       /* i: 1 enter, 0 leave (entering stops a recording) */
    CTRL_CMD_RECORD_START, /* g_rec prepared by record_prepare (state REC_STARTING) */
    CTRL_CMD_RECORD_STOP,
    CTRL_CMD_EXCITE,      /* exc (kind NONE: stop); not recorded unless a recording runs */
} ctrl_cmd_kind_t;

typedef struct {
    ctrl_cmd_kind_t kind;
    int32_t i;
    float f;
    const model_t *model;
    ncp_excite_params_t exc;
} ctrl_cmd_t;

typedef enum {
    CTRL_FAULT_NONE = 0,
    CTRL_FAULT_OFF_RAIL,
    CTRL_FAULT_ESTOP,
    CTRL_FAULT_ENCODER,
    CTRL_FAULT_DEADLINE,
} ctrl_fault_t;

typedef struct {
    uint32_t ticks;
    uint8_t run;   /* NCP_RUN_* */
    uint8_t fault; /* ctrl_fault_t */
    uint8_t mode;  /* ncp_mode_t */
    uint8_t goal;
    bool random, settled, estop, off_rail;
    float x_ref;
    uint32_t period_us, nominal_period_us;
    uint32_t exec_avg_us, exec_max_us, policy_avg_us, est_avg_us;
    uint32_t misses, read_errors, tlm_dropped;
    float rt_factor;
    bool budget_ok;
    const model_t *model; /* the model the loop is running (NULL: none) */
    uint8_t excite_kind;  /* ncp_exc_kind_t */
    bool excite_active;
    float excite_t;       /* s since the excitation started */
    float dt;             /* control period of the model (s), default without one */
} ctrl_status_t;

/* Create the control task pinned to core 1; it starts with `model` (may be NULL). */
esp_err_t control_start(const model_t *model);

/* Queue a command (never blocks). false: queue full. */
bool control_send(const ctrl_cmd_t *cmd);

void control_get_status(ctrl_status_t *out);
uint32_t control_ticks(void);

/* NULL if arming is possible now, else the reason (checked again by the task). */
const char *control_arm_refusal(const ctrl_status_t *st);

/* Block (core 0) until the task runs `model` / is in or out of MAINT. */
bool control_wait_model(const model_t *model, int timeout_ms);
bool control_wait_maint(bool on, int timeout_ms);

/* Rig zeroing results waiting to be written to NVS by core 0. */
bool control_take_angle_offsets(double *off1, double *off2);
bool control_take_cart_zero(double *x_zero);

const char *ctrl_fault_str(uint8_t fault);
const char *ctrl_run_str(uint8_t run);
