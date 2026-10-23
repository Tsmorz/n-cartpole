/* hal_rig.c — the real rig: 2 x AS5048A (SPI, 14-bit) on the joints, a quadrature
 * motor encoder (PCNT, x4) for the cart, a PWM + DIR + EN force command to the motor
 * driver and an active-high e-stop sense input. Compiled only for CONFIG_NCP_BACKEND_RIG.
 *
 * Joint 1 measures link 1 absolutely; joint 2 sits on link 1 (wired through the slip
 * ring) and measures link 2 relative to link 1:
 *   th1 = off1 + sign1 * unwrap1,  phi = off2 + sign2 * unwrap2,  th2 = th1 + phi.
 * {"zero": "angles"} (links hanging at rest) sets th1 = pi, phi = 0.
 * Cart: x = sign * count * 2 pi r / cpr - x_zero; {"zero": "cart"} sets x = 0 here.
 *
 * ncp_hal_init runs on core 0 (PCNT's overflow interrupt lands there); the SPI reads are
 * polling transactions issued from the control task on core 1 (no interrupt, no DMA). */
#include "sdkconfig.h"

#if CONFIG_NCP_BACKEND_RIG

#include <math.h>
#include <string.h>

#include "driver/gpio.h"
#include "driver/ledc.h"
#include "driver/pulse_cnt.h"
#include "driver/spi_master.h"
#include "esp_check.h"
#include "esp_log.h"
#include "hal.h"

static const char *TAG = "rig";

#define ENC_COUNTS 16384
#define ENC_HALF (ENC_COUNTS / 2)
#define ENC_SPI_HOST SPI2_HOST
#define PCNT_LIMIT 30000

#define AS5048_CMD_READ_ANGLE 0xFFFF /* read 0x3FFF, R = 1, even parity */
#define AS5048_CMD_CLEAR_ERROR 0x4001 /* read 0x0001 (clears the error flag) */
#define AS5048_CMD_NOP 0x0000
#define AS5048_CMD_READ_DIAG 0x7FFD /* read 0x3FFD (diagnostics + AGC), R = 1, even parity */
#define DIAG_EVERY 100 /* reads between diagnostics (alternating encoders): 1 s each at 200 Hz */
#define AS5048_EF 0x4000

#if CONFIG_NCP_ENC1_REVERSED
#define ENC1_SIGN (-1.0)
#else
#define ENC1_SIGN (1.0)
#endif
#if CONFIG_NCP_ENC2_REVERSED
#define ENC2_SIGN (-1.0)
#else
#define ENC2_SIGN (1.0)
#endif
#if CONFIG_NCP_CART_REVERSED
#define CART_SIGN (-1.0)
#else
#define CART_SIGN (1.0)
#endif

#define CART_M_PER_COUNT (2.0 * M_PI * (CONFIG_NCP_PULLEY_RADIUS_UM * 1e-6) / CONFIG_NCP_MOTOR_CPR)
#define PWM_FULL_SCALE_N (CONFIG_NCP_PWM_FULL_SCALE_MN * 1e-3)

typedef struct {
    spi_device_handle_t dev;
    bool have;      /* first good reading taken */
    uint16_t last;  /* last raw angle */
    int32_t acc;    /* unwrapped counts */
    uint32_t bad;   /* bad reads since boot */
    bool diag;      /* diagnostics read at least once */
    uint16_t diag_word; /* the last good diagnostics register (14 bits) */
} enc_t;

static enc_t s_enc[2];
static pcnt_unit_handle_t s_pcnt;
static double s_off1, s_off2 = 0.0, s_x_zero;
static bool s_normalized; /* first reading wrapped into [-pi, pi) */
static uint32_t s_errors;
static bool s_hw_ok; /* ncp_hal_init succeeded: never touch half-initialized drivers from core 1 */
static double s_last[3]; /* last good [x, th1, th2] */
static int32_t s_cart_count;
static int s_diag_wait, s_diag_next;

/* ---------- force drive ----------
 * The actuator sits behind this small interface so a CAN/UART FOC driver (torque
 * commands over a bus) can replace the PWM + DIR + EN implementation without touching
 * the control loop. Calls come from the control task: no blocking, no logging. */
typedef struct {
    esp_err_t (*init)(void);
    void (*enable)(bool on);
    void (*set_force)(double newtons); /* + = +x */
} hal_drive_t;

#define PWM_MODE LEDC_LOW_SPEED_MODE
#define PWM_TIMER LEDC_TIMER_0
#define PWM_CHANNEL LEDC_CHANNEL_0
#define PWM_RES LEDC_TIMER_11_BIT
#define PWM_MAX_DUTY ((1u << 11) - 1u)

static bool s_drive_on;
static uint32_t s_duty = UINT32_MAX;
static int s_dir = -1;

static esp_err_t pwm_init(void) {
    gpio_config_t out = {
        .pin_bit_mask = (1ULL << CONFIG_NCP_DIR_GPIO) | (1ULL << CONFIG_NCP_EN_GPIO),
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    ESP_RETURN_ON_ERROR(gpio_set_level(CONFIG_NCP_EN_GPIO, 0), TAG, "EN");
    ESP_RETURN_ON_ERROR(gpio_config(&out), TAG, "drive GPIOs");
    ESP_RETURN_ON_ERROR(gpio_set_level(CONFIG_NCP_EN_GPIO, 0), TAG, "EN");
    ESP_RETURN_ON_ERROR(gpio_set_level(CONFIG_NCP_DIR_GPIO, 0), TAG, "DIR");

    ledc_timer_config_t tc = {
        .speed_mode = PWM_MODE,
        .duty_resolution = PWM_RES,
        .timer_num = PWM_TIMER,
        .freq_hz = CONFIG_NCP_PWM_FREQ_HZ,
        .clk_cfg = LEDC_AUTO_CLK,
    };
    ESP_RETURN_ON_ERROR(ledc_timer_config(&tc), TAG, "LEDC timer");
    ledc_channel_config_t cc = {
        .gpio_num = CONFIG_NCP_PWM_GPIO,
        .speed_mode = PWM_MODE,
        .channel = PWM_CHANNEL,
        .intr_type = LEDC_INTR_DISABLE,
        .timer_sel = PWM_TIMER,
        .duty = 0,
        .hpoint = 0,
    };
    ESP_RETURN_ON_ERROR(ledc_channel_config(&cc), TAG, "LEDC channel");
    s_duty = 0;
    return ESP_OK;
}

static void pwm_set_force(double f) {
    double frac = fabs(f) / PWM_FULL_SCALE_N;
    if (frac > 1.0) {
        frac = 1.0;
    }
    uint32_t duty = (uint32_t)lround(frac * PWM_MAX_DUTY);
#if CONFIG_NCP_DIR_REVERSED
    int dir = f >= 0.0 ? 0 : 1;
#else
    int dir = f >= 0.0 ? 1 : 0;
#endif
    if (dir != s_dir) {
        gpio_set_level(CONFIG_NCP_DIR_GPIO, (uint32_t)dir);
        s_dir = dir;
    }
    if (duty != s_duty) {
        ledc_set_duty(PWM_MODE, PWM_CHANNEL, duty);
        ledc_update_duty(PWM_MODE, PWM_CHANNEL);
        s_duty = duty;
    }
}

static void pwm_enable(bool on) {
    if (!on) {
        pwm_set_force(0.0);
    }
    if (on != s_drive_on) {
        gpio_set_level(CONFIG_NCP_EN_GPIO, on ? 1 : 0);
        s_drive_on = on;
    }
}

static const hal_drive_t s_drive_pwm = {
    .init = pwm_init,
    .enable = pwm_enable,
    .set_force = pwm_set_force,
};
static const hal_drive_t *s_drive = &s_drive_pwm;

/* ---------- joint encoders (AS5048A) ---------- */

static esp_err_t enc_xfer(spi_device_handle_t dev, uint16_t cmd, uint16_t *resp) {
    spi_transaction_t t;
    memset(&t, 0, sizeof(t));
    t.flags = SPI_TRANS_USE_TXDATA | SPI_TRANS_USE_RXDATA;
    t.length = 16;
    t.tx_data[0] = (uint8_t)(cmd >> 8);
    t.tx_data[1] = (uint8_t)(cmd & 0xFF);
    esp_err_t err = spi_device_polling_transmit(dev, &t);
    *resp = (uint16_t)((t.rx_data[0] << 8) | t.rx_data[1]);
    return err;
}

/* A fresh angle: the response to a command arrives in the next frame, so send
 * READ ANGLE then NOP. Checks even parity and the error flag. */
static bool enc_read(enc_t *e, uint16_t *angle) {
    uint16_t r;
    if (enc_xfer(e->dev, AS5048_CMD_READ_ANGLE, &r) != ESP_OK ||
        enc_xfer(e->dev, AS5048_CMD_NOP, &r) != ESP_OK) {
        e->bad++;
        return false;
    }
    if (__builtin_parity(r) != 0 || (r & AS5048_EF) != 0) {
        uint16_t dummy;
        enc_xfer(e->dev, AS5048_CMD_CLEAR_ERROR, &dummy);
        enc_xfer(e->dev, AS5048_CMD_NOP, &dummy);
        e->bad++;
        return false;
    }
    *angle = (uint16_t)(r & 0x3FFF);
    return true;
}

/* The diagnostics register; a bad frame just leaves the previous value. */
static void enc_read_diag(enc_t *e) {
    uint16_t r;
    if (enc_xfer(e->dev, AS5048_CMD_READ_DIAG, &r) != ESP_OK ||
        enc_xfer(e->dev, AS5048_CMD_NOP, &r) != ESP_OK || __builtin_parity(r) != 0 ||
        (r & AS5048_EF) != 0) {
        return;
    }
    e->diag_word = (uint16_t)(r & 0x3FFF);
    e->diag = true;
}

static void enc_unwrap(enc_t *e, uint16_t raw) {
    if (!e->have) {
        e->acc = raw;
        e->have = true;
    } else {
        int32_t d = (int32_t)raw - (int32_t)e->last;
        if (d > ENC_HALF) {
            d -= ENC_COUNTS;
        } else if (d < -ENC_HALF) {
            d += ENC_COUNTS;
        }
        e->acc += d;
    }
    e->last = raw;
}

static double enc_rad(const enc_t *e) { return e->acc * (2.0 * M_PI / ENC_COUNTS); }

/* Shift whole turns so the first reading lands in [-pi, pi). */
static void normalize_turns(enc_t *e, double sign, double angle) {
    double k = floor((angle + M_PI) / (2.0 * M_PI));
    e->acc -= (int32_t)(sign * k * ENC_COUNTS);
}

static esp_err_t enc_init(void) {
    spi_bus_config_t bus = {
        .mosi_io_num = CONFIG_NCP_SPI_MOSI_GPIO,
        .miso_io_num = CONFIG_NCP_SPI_MISO_GPIO,
        .sclk_io_num = CONFIG_NCP_SPI_SCLK_GPIO,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = 4,
    };
    ESP_RETURN_ON_ERROR(spi_bus_initialize(ENC_SPI_HOST, &bus, SPI_DMA_DISABLED), TAG, "SPI bus");
    const int cs[2] = {CONFIG_NCP_ENC1_CS_GPIO, CONFIG_NCP_ENC2_CS_GPIO};
    for (int i = 0; i < 2; i++) {
        spi_device_interface_config_t dev = {
            .mode = 1, /* CPOL 0, CPHA 1 */
            .clock_speed_hz = CONFIG_NCP_SPI_CLOCK_HZ,
            .spics_io_num = cs[i],
            .cs_ena_pretrans = 0,
            .cs_ena_posttrans = 1,
            .queue_size = 1,
        };
        ESP_RETURN_ON_ERROR(spi_bus_add_device(ENC_SPI_HOST, &dev, &s_enc[i].dev), TAG,
                            "encoder %d", i + 1);
    }
    return ESP_OK;
}

/* ---------- cart encoder (PCNT x4) ---------- */

static esp_err_t cart_init(void) {
    pcnt_unit_config_t uc = {
        .low_limit = -PCNT_LIMIT,
        .high_limit = PCNT_LIMIT,
        .flags.accum_count = 1,
    };
    ESP_RETURN_ON_ERROR(pcnt_new_unit(&uc, &s_pcnt), TAG, "PCNT unit");
    pcnt_glitch_filter_config_t gf = {.max_glitch_ns = 1000};
    ESP_RETURN_ON_ERROR(pcnt_unit_set_glitch_filter(s_pcnt, &gf), TAG, "PCNT filter");

    pcnt_chan_config_t ca = {.edge_gpio_num = CONFIG_NCP_QUAD_A_GPIO,
                             .level_gpio_num = CONFIG_NCP_QUAD_B_GPIO};
    pcnt_chan_config_t cb = {.edge_gpio_num = CONFIG_NCP_QUAD_B_GPIO,
                             .level_gpio_num = CONFIG_NCP_QUAD_A_GPIO};
    pcnt_channel_handle_t cha, chb;
    ESP_RETURN_ON_ERROR(pcnt_new_channel(s_pcnt, &ca, &cha), TAG, "PCNT A");
    ESP_RETURN_ON_ERROR(pcnt_new_channel(s_pcnt, &cb, &chb), TAG, "PCNT B");
    ESP_RETURN_ON_ERROR(pcnt_channel_set_edge_action(cha, PCNT_CHANNEL_EDGE_ACTION_DECREASE,
                                                     PCNT_CHANNEL_EDGE_ACTION_INCREASE),
                        TAG, "PCNT A edge");
    ESP_RETURN_ON_ERROR(pcnt_channel_set_level_action(cha, PCNT_CHANNEL_LEVEL_ACTION_KEEP,
                                                      PCNT_CHANNEL_LEVEL_ACTION_INVERSE),
                        TAG, "PCNT A level");
    ESP_RETURN_ON_ERROR(pcnt_channel_set_edge_action(chb, PCNT_CHANNEL_EDGE_ACTION_INCREASE,
                                                     PCNT_CHANNEL_EDGE_ACTION_DECREASE),
                        TAG, "PCNT B edge");
    ESP_RETURN_ON_ERROR(pcnt_channel_set_level_action(chb, PCNT_CHANNEL_LEVEL_ACTION_KEEP,
                                                      PCNT_CHANNEL_LEVEL_ACTION_INVERSE),
                        TAG, "PCNT B level");
    /* Watch points at the limits: the driver folds each overflow into the accumulated count. */
    ESP_RETURN_ON_ERROR(pcnt_unit_add_watch_point(s_pcnt, PCNT_LIMIT), TAG, "PCNT high");
    ESP_RETURN_ON_ERROR(pcnt_unit_add_watch_point(s_pcnt, -PCNT_LIMIT), TAG, "PCNT low");
    ESP_RETURN_ON_ERROR(pcnt_unit_enable(s_pcnt), TAG, "PCNT enable");
    ESP_RETURN_ON_ERROR(pcnt_unit_clear_count(s_pcnt), TAG, "PCNT clear");
    ESP_RETURN_ON_ERROR(pcnt_unit_start(s_pcnt), TAG, "PCNT start");
    return ESP_OK;
}

static double cart_x_raw(void) {
    int count = 0;
    pcnt_unit_get_count(s_pcnt, &count);
    s_cart_count = count;
    return CART_SIGN * count * CART_M_PER_COUNT;
}

/* ---------- HAL ---------- */

esp_err_t ncp_hal_init(void) {
    ESP_RETURN_ON_ERROR(s_drive->init(), TAG, "drive");
    ESP_RETURN_ON_ERROR(enc_init(), TAG, "joint encoders");
    ESP_RETURN_ON_ERROR(cart_init(), TAG, "cart encoder");
#if CONFIG_NCP_ESTOP_GPIO >= 0
    gpio_config_t in = {
        .pin_bit_mask = 1ULL << CONFIG_NCP_ESTOP_GPIO,
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_ENABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    ESP_RETURN_ON_ERROR(gpio_config(&in), TAG, "e-stop GPIO");
#endif
    ESP_LOGI(TAG, "encoders on SPI2 @ %d Hz (CS %d, %d), cart PCNT A%d/B%d, %.3f um/count, "
                  "drive PWM %d / DIR %d / EN %d @ %d Hz, %.2f N full scale, e-stop %d",
             CONFIG_NCP_SPI_CLOCK_HZ, CONFIG_NCP_ENC1_CS_GPIO, CONFIG_NCP_ENC2_CS_GPIO,
             CONFIG_NCP_QUAD_A_GPIO, CONFIG_NCP_QUAD_B_GPIO, CART_M_PER_COUNT * 1e6,
             CONFIG_NCP_PWM_GPIO, CONFIG_NCP_DIR_GPIO, CONFIG_NCP_EN_GPIO,
             CONFIG_NCP_PWM_FREQ_HZ, PWM_FULL_SCALE_N, CONFIG_NCP_ESTOP_GPIO);
    s_hw_ok = true;
    return ESP_OK;
}

void ncp_hal_set_model(const ncp_header_t *h) { (void)h; }

bool ncp_hal_read(double *pos) {
    if (!s_hw_ok) {
        s_errors++;
        memcpy(pos, s_last, sizeof(s_last));
        return false;
    }
    uint16_t raw1 = 0, raw2 = 0;
    bool ok1 = enc_read(&s_enc[0], &raw1);
    bool ok2 = enc_read(&s_enc[1], &raw2);
    double x = cart_x_raw() - s_x_zero;
    if (++s_diag_wait >= DIAG_EVERY) {
        s_diag_wait = 0;
        enc_read_diag(&s_enc[s_diag_next]);
        s_diag_next ^= 1;
    }
    if (!(ok1 && ok2)) {
        s_errors++;
        s_last[0] = x;
        memcpy(pos, s_last, sizeof(s_last));
        return false;
    }
    s_errors = 0;
    enc_unwrap(&s_enc[0], raw1);
    enc_unwrap(&s_enc[1], raw2);
    double th1 = s_off1 + ENC1_SIGN * enc_rad(&s_enc[0]);
    double phi = s_off2 + ENC2_SIGN * enc_rad(&s_enc[1]);
    if (!s_normalized) {
        normalize_turns(&s_enc[0], ENC1_SIGN, th1);
        normalize_turns(&s_enc[1], ENC2_SIGN, phi);
        th1 = s_off1 + ENC1_SIGN * enc_rad(&s_enc[0]);
        phi = s_off2 + ENC2_SIGN * enc_rad(&s_enc[1]);
        s_normalized = true;
    }
    s_last[0] = x;
    s_last[1] = th1;
    s_last[2] = th1 + phi;
    memcpy(pos, s_last, sizeof(s_last));
    return true;
}

uint32_t ncp_hal_read_errors(void) { return s_errors; }

void ncp_hal_apply(bool armed, double force) {
    if (!s_hw_ok) {
        return;
    }
    s_drive->enable(armed);
    if (armed) {
        s_drive->set_force(force);
    }
}

bool ncp_hal_estop(void) {
#if CONFIG_NCP_ESTOP_GPIO >= 0
    return gpio_get_level(CONFIG_NCP_ESTOP_GPIO) == 1;
#else
    return false;
#endif
}

void ncp_hal_io(ncp_hal_io_t *out) {
    memset(out, 0, sizeof(*out));
    out->rig = true;
    out->estop = ncp_hal_estop();
    out->drive_on = s_drive_on;
    out->duty = s_duty == UINT32_MAX ? 0.0f : (float)s_duty / (float)PWM_MAX_DUTY;
#if CONFIG_NCP_DIR_REVERSED
    out->dir = s_dir == 0 ? 1 : -1;
#else
    out->dir = s_dir == 1 ? 1 : -1;
#endif
    out->cart_count = s_cart_count;
    out->read_errors = s_errors;
    for (int i = 0; i < 2; i++) {
        const enc_t *e = &s_enc[i];
        out->enc[i].raw = e->last;
        out->enc[i].bad = e->bad;
        out->enc[i].diag = e->diag;
        out->enc[i].agc = (uint8_t)(e->diag_word & 0xFF);
        out->enc[i].ocf = (e->diag_word >> 8) & 1;
        out->enc[i].cof = (e->diag_word >> 9) & 1;
        out->enc[i].comp_low = (e->diag_word >> 10) & 1;
        out->enc[i].comp_high = (e->diag_word >> 11) & 1;
    }
}

bool ncp_hal_off_rail(const double *pos, double x_lim) { return fabs(pos[0]) > x_lim; }

void ncp_hal_sim_reset(int goal) { (void)goal; }
const double *ncp_hal_sim_truth(void) { return NULL; }
double ncp_hal_sim_applied(void) { return 0.0; }

void ncp_hal_rig_zero_angles(double *off1, double *off2) {
    s_off1 = M_PI - ENC1_SIGN * enc_rad(&s_enc[0]);
    s_off2 = -ENC2_SIGN * enc_rad(&s_enc[1]);
    *off1 = s_off1;
    *off2 = s_off2;
}

void ncp_hal_rig_zero_cart(double *x_zero) {
    s_x_zero = s_hw_ok ? cart_x_raw() : 0.0;
    *x_zero = s_x_zero;
}

void ncp_hal_rig_set_angle_offsets(double off1, double off2) {
    s_off1 = off1;
    s_off2 = off2;
}

void ncp_hal_rig_set_cart_zero(double x_zero) { s_x_zero = x_zero; }

#endif
