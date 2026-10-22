/* net.c — Wi-Fi (station, fallback open AP) and mDNS. Runs on core 0: the Wi-Fi, lwIP
 * and event tasks are pinned there by sdkconfig.defaults. */
#include "net.h"

#include <stdio.h>
#include <string.h>

#include "esp_event.h"
#include "esp_log.h"
#include "esp_mac.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "mdns.h"

static const char *TAG = "net";

#define GOT_IP_BIT BIT0

static esp_netif_t *s_sta, *s_ap;
static EventGroupHandle_t s_events;
static volatile bool s_ap_mode;
static bool s_have_creds;
static char s_ssid[33];
static char s_ip[16] = "0.0.0.0";

static void on_event(void *arg, esp_event_base_t base, int32_t id, void *data) {
    (void)arg;
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        xEventGroupClearBits(s_events, GOT_IP_BIT);
        if (!s_ap_mode) {
            esp_wifi_connect(); /* keep trying (the AP fallback decides after 15 s) */
        }
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        const ip_event_got_ip_t *e = (const ip_event_got_ip_t *)data;
        esp_ip4addr_ntoa(&e->ip_info.ip, s_ip, sizeof(s_ip));
        ESP_LOGI(TAG, "connected to '%s', IP %s", s_ssid, s_ip);
        xEventGroupSetBits(s_events, GOT_IP_BIT);
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_AP_STACONNECTED) {
        ESP_LOGI(TAG, "a client joined the access point");
    }
}

static void start_mdns(const char *hostname) {
    esp_err_t err = mdns_init();
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "mDNS init failed: %s", esp_err_to_name(err));
        return;
    }
    mdns_hostname_set(hostname);
    mdns_instance_name_set("n-cartpole");
    mdns_service_add(NULL, "_http", "_tcp", 80, NULL, 0);
    ESP_LOGI(TAG, "mDNS: http://%s.local", hostname);
}

esp_err_t net_start(const char *hostname, const char *ssid, const char *password) {
    s_events = xEventGroupCreate();
    if (s_events == NULL) {
        return ESP_ERR_NO_MEM;
    }
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    s_sta = esp_netif_create_default_wifi_sta();
    s_ap = esp_netif_create_default_wifi_ap();
    esp_netif_set_hostname(s_sta, hostname);

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, on_event, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, on_event, NULL));

    s_have_creds = ssid != NULL && ssid[0] != '\0';
    if (s_have_creds) {
        wifi_config_t wc;
        memset(&wc, 0, sizeof(wc));
        strlcpy((char *)wc.sta.ssid, ssid, sizeof(wc.sta.ssid));
        strlcpy((char *)wc.sta.password, password != NULL ? password : "", sizeof(wc.sta.password));
        wc.sta.threshold.authmode = (password != NULL && password[0] != '\0') ? WIFI_AUTH_WEP
                                                                               : WIFI_AUTH_OPEN;
        strlcpy(s_ssid, ssid, sizeof(s_ssid));
        ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
        ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wc));
        ESP_ERROR_CHECK(esp_wifi_start());
        ESP_LOGI(TAG, "connecting to '%s'...", ssid);
    }
    start_mdns(hostname);
    return ESP_OK;
}

static void start_ap(void) {
    s_ap_mode = true;
    if (s_have_creds) {
        esp_wifi_disconnect();
        esp_wifi_stop();
    }
    uint8_t mac[6] = {0};
    esp_read_mac(mac, ESP_MAC_WIFI_SOFTAP);
    wifi_config_t ap;
    memset(&ap, 0, sizeof(ap));
    snprintf(s_ssid, sizeof(s_ssid), "cartpole-%02X%02X", mac[4], mac[5]);
    memcpy(ap.ap.ssid, s_ssid, strlen(s_ssid));
    ap.ap.ssid_len = (uint8_t)strlen(s_ssid);
    ap.ap.channel = 1;
    ap.ap.authmode = WIFI_AUTH_OPEN;
    ap.ap.max_connection = 4;
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &ap));
    ESP_ERROR_CHECK(esp_wifi_start());
    esp_netif_ip_info_t ip;
    if (esp_netif_get_ip_info(s_ap, &ip) == ESP_OK) {
        esp_ip4addr_ntoa(&ip.ip, s_ip, sizeof(s_ip));
    }
    ESP_LOGW(TAG, "access point '%s' (open): http://%s/", s_ssid, s_ip);
}

void net_wait_or_fallback(int timeout_ms) {
    if (s_have_creds) {
        EventBits_t bits = xEventGroupWaitBits(s_events, GOT_IP_BIT, pdFALSE, pdTRUE,
                                               pdMS_TO_TICKS(timeout_ms));
        if (bits & GOT_IP_BIT) {
            return;
        }
        ESP_LOGW(TAG, "'%s' not reached within %d s: starting the setup access point", s_ssid,
                 timeout_ms / 1000);
    } else {
        ESP_LOGW(TAG, "no Wi-Fi credentials: starting the setup access point");
    }
    start_ap();
}

void net_get_info(net_info_t *out) {
    memset(out, 0, sizeof(*out));
    out->ap = s_ap_mode;
    strlcpy(out->ssid, s_ssid, sizeof(out->ssid));
    strlcpy(out->ip, s_ip, sizeof(out->ip));
    if (s_ap_mode) {
        out->connected = true;
        return;
    }
    out->connected = s_events != NULL && (xEventGroupGetBits(s_events) & GOT_IP_BIT) != 0;
    wifi_ap_record_t rec;
    if (out->connected && esp_wifi_sta_get_ap_info(&rec) == ESP_OK) {
        out->rssi = rec.rssi;
    }
    if (!out->connected) {
        strlcpy(out->ip, "", sizeof(out->ip));
    }
}
