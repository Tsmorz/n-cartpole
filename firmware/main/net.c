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
#include "freertos/task.h"
#include "mdns.h"
#include "settings.h"

static const char *TAG = "net";

#define GOT_IP_BIT BIT0

#define JOIN_TIMEOUT_MS 20000
#define JOIN_GRACE_AP_MS 20000 /* keep the setup AP this long after a join so the page can show the result */
#define JOIN_GRACE_STA_MS 5000

static esp_netif_t *s_sta, *s_ap;
static EventGroupHandle_t s_events;
static volatile bool s_ap_mode;
static volatile bool s_ready;   /* net_wait_or_fallback has decided STA or AP */
static volatile bool s_joining; /* a join attempt owns the station */
static volatile uint8_t s_last_reason;
static bool s_have_creds;
static char s_ssid[33];                /* joined network (station) or our AP name */
static char s_ap_ip[16] = "0.0.0.0";
static char s_sta_ip[16] = "0.0.0.0";

static volatile net_join_state_t s_join;
static char s_join_ssid[33], s_join_pass[65], s_join_ip[16], s_join_error[80];
static volatile bool s_join_busy;

static void on_event(void *arg, esp_event_base_t base, int32_t id, void *data) {
    (void)arg;
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        if (!s_joining) { /* a join sets its own config first, then connects */
            esp_wifi_connect();
        }
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        const wifi_event_sta_disconnected_t *d = (const wifi_event_sta_disconnected_t *)data;
        s_last_reason = d->reason;
        xEventGroupClearBits(s_events, GOT_IP_BIT);
        if (s_joining || !s_ap_mode) {
            esp_wifi_connect(); /* keep trying (the AP fallback / the join timeout decides) */
        }
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        const ip_event_got_ip_t *e = (const ip_event_got_ip_t *)data;
        esp_ip4addr_ntoa(&e->ip_info.ip, s_sta_ip, sizeof(s_sta_ip));
        ESP_LOGI(TAG, "station got IP %s", s_sta_ip);
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

static void fill_sta_config(wifi_config_t *wc, const char *ssid, const char *password) {
    memset(wc, 0, sizeof(*wc));
    strlcpy((char *)wc->sta.ssid, ssid, sizeof(wc->sta.ssid));
    strlcpy((char *)wc->sta.password, password != NULL ? password : "", sizeof(wc->sta.password));
    wc->sta.threshold.authmode = (password != NULL && password[0] != '\0') ? WIFI_AUTH_WEP
                                                                           : WIFI_AUTH_OPEN;
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
        fill_sta_config(&wc, ssid, password);
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
        esp_ip4addr_ntoa(&ip.ip, s_ap_ip, sizeof(s_ap_ip));
    }
    ESP_LOGW(TAG, "access point '%s' (open): http://%s/", s_ssid, s_ap_ip);
}

void net_wait_or_fallback(int timeout_ms) {
    if (s_have_creds) {
        EventBits_t bits = xEventGroupWaitBits(s_events, GOT_IP_BIT, pdFALSE, pdTRUE,
                                               pdMS_TO_TICKS(timeout_ms));
        if (bits & GOT_IP_BIT) {
            s_ready = true;
            return;
        }
        ESP_LOGW(TAG, "'%s' not reached within %d s: starting the setup access point", s_ssid,
                 timeout_ms / 1000);
    } else {
        ESP_LOGW(TAG, "no Wi-Fi credentials: starting the setup access point");
    }
    start_ap();
    s_ready = true;
}

/* ---------- joining a network at run time ---------- */

static void join_error_text(char *out, size_t n) {
    switch (s_last_reason) {
    case WIFI_REASON_NO_AP_FOUND:
    case WIFI_REASON_NO_AP_FOUND_W_COMPATIBLE_SECURITY:
    case WIFI_REASON_NO_AP_FOUND_IN_AUTHMODE_THRESHOLD:
    case WIFI_REASON_NO_AP_FOUND_IN_RSSI_THRESHOLD:
        snprintf(out, n, "network not found (out of range, or 5 GHz only?)");
        break;
    case WIFI_REASON_AUTH_FAIL:
    case WIFI_REASON_AUTH_EXPIRE:
    case WIFI_REASON_HANDSHAKE_TIMEOUT:
    case WIFI_REASON_4WAY_HANDSHAKE_TIMEOUT:
        snprintf(out, n, "authentication failed (wrong password?)");
        break;
    case 0:
        snprintf(out, n, "no response from the network");
        break;
    default:
        snprintf(out, n, "could not connect (reason %d)", (int)s_last_reason);
        break;
    }
}

static void join_task(void *arg) {
    (void)arg;
    const bool was_ap = s_ap_mode;
    /* The stored credentials are only replaced on success, so they are the way back. */
    char old_ssid[33], old_pass[65];
    strlcpy(old_ssid, settings_get()->ssid, sizeof(old_ssid));
    strlcpy(old_pass, settings_get()->password, sizeof(old_pass));

    wifi_config_t wc;
    fill_sta_config(&wc, s_join_ssid, s_join_pass);
    s_last_reason = 0;
    s_join_error[0] = '\0';
    s_join_ip[0] = '\0';
    s_joining = true;
    xEventGroupClearBits(s_events, GOT_IP_BIT);
    ESP_LOGI(TAG, "joining '%s'%s...", s_join_ssid, was_ap ? " (setup AP stays up)" : "");
    if (was_ap) {
        esp_wifi_set_mode(WIFI_MODE_APSTA); /* the AP follows the station's channel */
    }
    esp_wifi_set_config(WIFI_IF_STA, &wc);
    esp_wifi_disconnect();
    esp_wifi_connect();

    EventBits_t bits = xEventGroupWaitBits(s_events, GOT_IP_BIT, pdFALSE, pdTRUE,
                                           pdMS_TO_TICKS(JOIN_TIMEOUT_MS));
    uint32_t grace_ms;
    if (bits & GOT_IP_BIT) {
        esp_err_t err = settings_save_wifi(s_join_ssid, s_join_pass);
        if (err != ESP_OK) {
            ESP_LOGW(TAG, "joined, but could not save the credentials: %s", esp_err_to_name(err));
        }
        strlcpy(s_join_ip, s_sta_ip, sizeof(s_join_ip));
        s_have_creds = true;
        s_joining = false;
        ESP_LOGI(TAG, "joined '%s', IP %s: http://%s.local/", s_join_ssid, s_join_ip,
                 settings_get()->hostname);
        if (!was_ap) {
            strlcpy(s_ssid, s_join_ssid, sizeof(s_ssid));
        }
        s_join = NET_JOIN_OK;
        grace_ms = was_ap ? JOIN_GRACE_AP_MS : JOIN_GRACE_STA_MS;
        vTaskDelay(pdMS_TO_TICKS(grace_ms));
        if (was_ap) {
            strlcpy(s_ssid, s_join_ssid, sizeof(s_ssid));
            s_ap_mode = false;
            esp_wifi_set_mode(WIFI_MODE_STA); /* drops the AP, keeps the station */
            ESP_LOGW(TAG, "setup access point closed");
        }
        s_join = NET_JOIN_IDLE;
    } else {
        join_error_text(s_join_error, sizeof(s_join_error));
        ESP_LOGW(TAG, "could not join '%s': %s", s_join_ssid, s_join_error);
        s_joining = false;
        if (was_ap) {
            esp_wifi_disconnect();
            esp_wifi_set_mode(WIFI_MODE_AP);
        } else {
            wifi_config_t back;
            fill_sta_config(&back, old_ssid, old_pass);
            esp_wifi_set_config(WIFI_IF_STA, &back);
            esp_wifi_disconnect();
            esp_wifi_connect();
        }
        s_join = NET_JOIN_FAILED;
    }
    s_join_busy = false;
    vTaskDelete(NULL);
}

esp_err_t net_join_start(const char *ssid, const char *password) {
    if (!s_ready || s_join_busy || s_events == NULL) {
        return ESP_ERR_INVALID_STATE;
    }
    s_join_busy = true;
    strlcpy(s_join_ssid, ssid, sizeof(s_join_ssid));
    strlcpy(s_join_pass, password != NULL ? password : "", sizeof(s_join_pass));
    s_join = NET_JOIN_CONNECTING;
    /* Core 0, never the control core. */
    if (xTaskCreatePinnedToCore(join_task, "wifi_join", 4096, NULL, 3, NULL, 0) != pdPASS) {
        s_join_busy = false;
        s_join = NET_JOIN_IDLE;
        return ESP_ERR_NO_MEM;
    }
    return ESP_OK;
}

void net_get_info(net_info_t *out) {
    memset(out, 0, sizeof(*out));
    out->ap = s_ap_mode;
    out->join = s_join;
    strlcpy(out->join_ssid, s_join_ssid, sizeof(out->join_ssid));
    strlcpy(out->join_ip, s_join_ip, sizeof(out->join_ip));
    strlcpy(out->join_error, s_join_error, sizeof(out->join_error));
    strlcpy(out->ssid, s_ssid, sizeof(out->ssid));
    if (s_ap_mode) {
        strlcpy(out->ip, s_ap_ip, sizeof(out->ip));
        out->connected = true;
        return;
    }
    strlcpy(out->ip, s_sta_ip, sizeof(out->ip));
    out->connected = s_events != NULL && (xEventGroupGetBits(s_events) & GOT_IP_BIT) != 0;
    wifi_ap_record_t rec;
    if (out->connected && esp_wifi_sta_get_ap_info(&rec) == ESP_OK) {
        out->rssi = rec.rssi;
    }
    if (!out->connected) {
        strlcpy(out->ip, "", sizeof(out->ip));
    }
}
