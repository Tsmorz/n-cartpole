/* hal.h — the plant behind the control loop: the on-chip simulator (hal_sim.c) or the
 * real rig (hal_rig.c), chosen by Kconfig NCP_BACKEND_*. hal_init runs on core 0 at
 * boot (so peripheral interrupts land there); everything else is called only from the
 * control task on core 1 and must not block, log or allocate. */
#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "esp_err.h"
#include "ncp_model.h"
#include "sdkconfig.h"

#if CONFIG_NCP_BACKEND_RIG
#define HAL_BACKEND_NAME "rig"
#define HAL_IS_SIM 0
#else
#define HAL_BACKEND_NAME "sim"
#define HAL_IS_SIM 1
#endif

esp_err_t hal_init(void);

/* A new model (or NULL): the simulator rebuilds its plant from the header. */
void hal_set_model(const ncp_header_t *h);

/* n + 1 readings [x, th0, th1, ...] (absolute, unwrapped angles). false: a bad read;
 * pos then holds the last good values. */
bool hal_read(double *pos);

/* Consecutive bad reads so far (rig), 0 on the simulator. */
uint32_t hal_read_errors(void);

/* One period of actuation: rig -> drive enabled and commanded only when `armed`;
 * sim -> the plant advances one model dt under `armed ? force : 0` (it always runs). */
void hal_apply(bool armed, double force);

bool hal_estop(void);
bool hal_off_rail(const double *pos, double x_lim);

/* Simulator only (no-ops on the rig). */
void hal_sim_reset(int goal);
const double *hal_sim_truth(void); /* true state, NULL on the rig */
double hal_sim_applied(void);      /* force that acted over the last period */

/* Rig only (no-ops on the simulator): re-reference from the current readings
 * (links hanging at rest -> th1 = pi, phi = 0; current cart position -> x = 0),
 * returning the values to persist; and restore persisted ones at boot. */
void hal_rig_zero_angles(double *off1, double *off2);
void hal_rig_zero_cart(double *x_zero);
void hal_rig_set_angle_offsets(double off1, double off2);
void hal_rig_set_cart_zero(double x_zero);
