/* hal_sim.c — the simulator backend: ncp_sim integrates the plant on the chip and
 * produces readings through the model's encoder model. Compiled only for
 * CONFIG_NCP_BACKEND_SIM. Called from the control task except ncp_hal_init. */
#include "sdkconfig.h"

#if !CONFIG_NCP_BACKEND_RIG

#include <math.h>
#include <string.h>

#include "esp_random.h"
#include "hal.h"
#include "ncp_sim.h"

static ncp_sim_t s_sim;
static bool s_ready;

esp_err_t ncp_hal_init(void) { return ESP_OK; }

void ncp_hal_set_model(const ncp_header_t *h) {
    if (h == NULL) {
        s_ready = false;
        return;
    }
    uint64_t seed = ((uint64_t)esp_random() << 32) | esp_random();
    ncp_sim_init(&s_sim, h, seed);
    ncp_sim_reset(&s_sim, (1 << h->n_links) - 1, 0.0); /* hanging at rest: DD */
    s_ready = true;
}

bool ncp_hal_read(double *pos) {
    if (!s_ready) {
        return false;
    }
    ncp_sim_read(&s_sim, pos);
    return true;
}

uint32_t ncp_hal_read_errors(void) { return 0; }

void ncp_hal_apply(bool armed, double force) {
    if (s_ready) {
        ncp_sim_apply(&s_sim, armed ? force : 0.0);
    }
}

bool ncp_hal_estop(void) { return false; }

void ncp_hal_io(ncp_hal_io_t *out) { memset(out, 0, sizeof(*out)); }

bool ncp_hal_off_rail(const double *pos, double x_lim) {
    (void)pos;
    (void)x_lim;
    return s_ready && ncp_sim_off_rail(&s_sim);
}

void ncp_hal_sim_reset(int goal) {
    if (s_ready) {
        ncp_sim_reset(&s_sim, goal, 0.01);
    }
}

const double *ncp_hal_sim_truth(void) { return s_ready ? s_sim.s : NULL; }
double ncp_hal_sim_applied(void) { return s_ready ? s_sim.applied : 0.0; }

void ncp_hal_rig_zero_angles(double *off1, double *off2) {
    *off1 = 0.0;
    *off2 = 0.0;
}
void ncp_hal_rig_zero_cart(double *x_zero) { *x_zero = 0.0; }
void ncp_hal_rig_set_angle_offsets(double off1, double off2) {
    (void)off1;
    (void)off2;
}
void ncp_hal_rig_set_cart_zero(double x_zero) { (void)x_zero; }

#endif
