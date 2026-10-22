/* settings.h — persistent settings in NVS (namespace "ncp"), with Kconfig defaults.
 * Written from core 0 only: an NVS write stalls both cores while flash is busy. */
#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "esp_err.h"

typedef struct {
    char ssid[33];
    char password[65];
    char hostname[33];
    /* Rig zero references ({"zero": ...}); offsets in rad, x_zero in m. */
    bool have_angle_offsets, have_cart_zero;
    double angle_off1, angle_off2, x_zero;
    /* Boots since the NVS partition was formatted (this one included), and the boot on
     * which each zero reference was set (0: unknown / before the counter existed). */
    uint32_t boot_count, angles_boot, cart_boot;
    /* When each zero reference was set (unix s; 0: the clock was not set then). */
    uint32_t angles_unix, cart_unix;
} settings_t;

/* nvs_flash_init (erasing an incompatible NVS partition) and load. */
esp_err_t settings_init(void);
const settings_t *settings_get(void);

esp_err_t settings_save_wifi(const char *ssid, const char *password);
/* Forget the network: stored as an empty SSID, so the Kconfig default is not used either
 * and the next boot opens the setup access point. */
esp_err_t settings_forget_wifi(void);
/* 1-32 of [a-z0-9-], not starting or ending with '-' (the mDNS name). */
bool settings_hostname_valid(const char *hostname);
esp_err_t settings_save_hostname(const char *hostname);
esp_err_t settings_save_angle_offsets(double off1, double off2);
esp_err_t settings_save_cart_zero(double x_zero);
