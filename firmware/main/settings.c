/* settings.c — NVS-backed settings (see settings.h). */
#include "settings.h"

#include <string.h>

#include "esp_log.h"
#include "nvs.h"
#include "nvs_flash.h"
#include "sdkconfig.h"

static const char *TAG = "settings";
static const char *NS = "ncp";
static settings_t s_settings;

static void load_str(nvs_handle_t h, const char *key, char *dst, size_t size) {
    size_t len = size;
    if (nvs_get_str(h, key, dst, &len) != ESP_OK) {
        dst[0] = '\0';
    }
}

static bool load_double(nvs_handle_t h, const char *key, double *out) {
    double v = 0.0;
    size_t len = sizeof(v);
    if (nvs_get_blob(h, key, &v, &len) == ESP_OK && len == sizeof(v)) {
        *out = v;
        return true;
    }
    return false;
}

esp_err_t settings_init(void) {
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_LOGW(TAG, "NVS partition reformatted (%s)", esp_err_to_name(err));
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    if (err != ESP_OK) {
        return err;
    }

    memset(&s_settings, 0, sizeof(s_settings));
    nvs_handle_t h;
    if (nvs_open(NS, NVS_READONLY, &h) == ESP_OK) {
        load_str(h, "ssid", s_settings.ssid, sizeof(s_settings.ssid));
        load_str(h, "pass", s_settings.password, sizeof(s_settings.password));
        load_str(h, "host", s_settings.hostname, sizeof(s_settings.hostname));
        bool a = load_double(h, "off1", &s_settings.angle_off1);
        bool b = load_double(h, "off2", &s_settings.angle_off2);
        s_settings.have_angle_offsets = a && b;
        s_settings.have_cart_zero = load_double(h, "xzero", &s_settings.x_zero);
        nvs_close(h);
    }
    if (s_settings.ssid[0] == '\0' && CONFIG_NCP_WIFI_SSID[0] != '\0') {
        strlcpy(s_settings.ssid, CONFIG_NCP_WIFI_SSID, sizeof(s_settings.ssid));
        strlcpy(s_settings.password, CONFIG_NCP_WIFI_PASSWORD, sizeof(s_settings.password));
    }
    if (s_settings.hostname[0] == '\0') {
        strlcpy(s_settings.hostname, CONFIG_NCP_HOSTNAME, sizeof(s_settings.hostname));
    }
    return ESP_OK;
}

const settings_t *settings_get(void) { return &s_settings; }

esp_err_t settings_save_wifi(const char *ssid, const char *password) {
    nvs_handle_t h;
    esp_err_t err = nvs_open(NS, NVS_READWRITE, &h);
    if (err != ESP_OK) {
        return err;
    }
    err = nvs_set_str(h, "ssid", ssid);
    if (err == ESP_OK) {
        err = nvs_set_str(h, "pass", password);
    }
    if (err == ESP_OK) {
        err = nvs_commit(h);
    }
    nvs_close(h);
    if (err == ESP_OK) {
        strlcpy(s_settings.ssid, ssid, sizeof(s_settings.ssid));
        strlcpy(s_settings.password, password, sizeof(s_settings.password));
    }
    return err;
}

static esp_err_t save_doubles(const char *k1, double v1, const char *k2, double v2) {
    nvs_handle_t h;
    esp_err_t err = nvs_open(NS, NVS_READWRITE, &h);
    if (err != ESP_OK) {
        return err;
    }
    err = nvs_set_blob(h, k1, &v1, sizeof(v1));
    if (err == ESP_OK && k2 != NULL) {
        err = nvs_set_blob(h, k2, &v2, sizeof(v2));
    }
    if (err == ESP_OK) {
        err = nvs_commit(h);
    }
    nvs_close(h);
    return err;
}

esp_err_t settings_save_angle_offsets(double off1, double off2) {
    esp_err_t err = save_doubles("off1", off1, "off2", off2);
    if (err == ESP_OK) {
        s_settings.angle_off1 = off1;
        s_settings.angle_off2 = off2;
        s_settings.have_angle_offsets = true;
    }
    return err;
}

esp_err_t settings_save_cart_zero(double x_zero) {
    esp_err_t err = save_doubles("xzero", x_zero, NULL, 0.0);
    if (err == ESP_OK) {
        s_settings.x_zero = x_zero;
        s_settings.have_cart_zero = true;
    }
    return err;
}
