/* net.h — Wi-Fi station from NVS/Kconfig with a fallback open access point
 * "cartpole-XXXX", and mDNS <hostname>.local + _http._tcp. */
#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "esp_err.h"

typedef struct {
    bool ap;       /* true: our own access point */
    bool connected;
    char ssid[33];
    char ip[16];
    int rssi;      /* station only */
} net_info_t;

esp_err_t net_start(const char *hostname, const char *ssid, const char *password);

/* Wait up to timeout_ms for a station IP; otherwise (or with no credentials) start the AP. */
void net_wait_or_fallback(int timeout_ms);

void net_get_info(net_info_t *out);
