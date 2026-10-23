/* net.h — Wi-Fi station from NVS/Kconfig with a fallback open access point
 * "cartpole-XXXX", and mDNS <hostname>.local + _http._tcp. A new network can be joined at
 * run time (net_join_start) without a reboot: the setup AP stays up while the station
 * tries, and the credentials are stored only if the join works. */
#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "esp_err.h"

typedef enum {
    NET_JOIN_IDLE = 0,
    NET_JOIN_CONNECTING,
    NET_JOIN_OK,     /* joined; sticky for a few seconds, then back to IDLE */
    NET_JOIN_FAILED, /* sticky until the next attempt; join_error says why */
} net_join_state_t;

typedef struct {
    bool ap;       /* true: our own access point */
    bool connected;
    char ssid[33];
    char ip[16];
    int rssi;      /* station only */
    net_join_state_t join;
    char join_ssid[33]; /* network being / just joined */
    char join_ip[16];   /* its address once joined (the AP address is still `ip` until the AP drops) */
    char join_error[80];
} net_info_t;

esp_err_t net_start(const char *hostname, const char *ssid, const char *password);

/* Wait up to timeout_ms for a station IP; otherwise (or with no credentials) start the AP. */
void net_wait_or_fallback(int timeout_ms);

void net_get_info(net_info_t *out);

/* Start joining ssid/password in the background (core 0) and return at once; poll
 * net_get_info().join. ESP_ERR_INVALID_STATE while the network is still starting or a
 * join is already running. On success the credentials are saved to NVS and, if we were an
 * access point, the AP is dropped after a grace period so a client can read the result;
 * on failure nothing is saved and the previous network (or the AP) stays. */
esp_err_t net_join_start(const char *ssid, const char *password);

/* A new mDNS / DHCP hostname, live (the DHCP name reaches the router on the next lease). */
void net_set_hostname(const char *hostname);
