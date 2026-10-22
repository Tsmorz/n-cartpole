/* settings.h — persistent settings in NVS (namespace "ncp"), with Kconfig defaults.
 * Written from core 0 only: an NVS write stalls both cores while flash is busy. */
#pragma once

#include <stdbool.h>

#include "esp_err.h"

typedef struct {
    char ssid[33];
    char password[65];
    char hostname[33];
    /* Rig zero references ({"zero": ...}); offsets in rad, x_zero in m. */
    bool have_angle_offsets, have_cart_zero;
    double angle_off1, angle_off2, x_zero;
} settings_t;

/* nvs_flash_init (erasing an incompatible NVS partition) and load. */
esp_err_t settings_init(void);
const settings_t *settings_get(void);

esp_err_t settings_save_wifi(const char *ssid, const char *password);
esp_err_t settings_save_angle_offsets(double off1, double off2);
esp_err_t settings_save_cart_zero(double x_zero);
