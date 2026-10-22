/* hal.h — the plant behind the control loop: the on-chip simulator (hal_sim.c) or the
 * real rig (hal_rig.c), chosen by Kconfig NCP_BACKEND_*. ncp_hal_init runs on core 0 at
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

esp_err_t ncp_hal_init(void);

/* A new model (or NULL): the simulator rebuilds its plant from the header. */
void ncp_hal_set_model(const ncp_header_t *h);

/* n + 1 readings [x, th0, th1, ...] (absolute, unwrapped angles). false: a bad read;
 * pos then holds the last good values. */
bool ncp_hal_read(double *pos);

/* Consecutive bad reads so far (rig), 0 on the simulator. */
uint32_t ncp_hal_read_errors(void);

/* One period of actuation: rig -> drive enabled and commanded only when `armed`;
 * sim -> the plant advances one model dt under `armed ? force : 0` (it always runs). */
void ncp_hal_apply(bool armed, double force);

bool ncp_hal_estop(void);

/* What the sensors and the drive are doing, for the dashboard (control task only: it
 * copies this into the published status). AS5048A diagnostics are read once a second
 * per encoder (register 0x3FFD): AGC 0 = strong field .. 255 = weak field, COMP low /
 * high = magnet too close / too far, COF = CORDIC overflow, OCF = offset compensation
 * finished (should be set). */
typedef struct {
    bool rig;          /* false: the simulator (no sensors, no drive) */
    bool estop;
    bool drive_on;     /* EN asserted */
    float duty;        /* PWM duty 0..1 */
    int8_t dir;        /* +1 / -1: the commanded direction */
    int32_t cart_count;
    uint32_t read_errors; /* consecutive bad joint reads */
    struct {
        uint16_t raw;  /* last good raw angle, 0..16383 */
        uint32_t bad;  /* bad reads since boot (SPI, parity, error flag) */
        bool diag;     /* the fields below have been read */
        uint8_t agc;
        bool ocf, cof, comp_low, comp_high;
    } enc[2];
} ncp_hal_io_t;

void ncp_hal_io(ncp_hal_io_t *out);
bool ncp_hal_off_rail(const double *pos, double x_lim);

/* Simulator only (no-ops on the rig). */
void ncp_hal_sim_reset(int goal);
const double *ncp_hal_sim_truth(void); /* true state, NULL on the rig */
double ncp_hal_sim_applied(void);      /* force that acted over the last period */

/* Rig only (no-ops on the simulator): re-reference from the current readings
 * (links hanging at rest -> th1 = pi, phi = 0; current cart position -> x = 0),
 * returning the values to persist; and restore persisted ones at boot. */
void ncp_hal_rig_zero_angles(double *off1, double *off2);
void ncp_hal_rig_zero_cart(double *x_zero);
void ncp_hal_rig_set_angle_offsets(double off1, double off2);
void ncp_hal_rig_set_cart_zero(double x_zero);
