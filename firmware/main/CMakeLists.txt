# The ESP-IDF application: tasks, hardware, web server, OTA, Wi-Fi.
idf_component_register(
    SRCS
        "app_main.c"
        "control.c"
        "hal_sim.c"
        "hal_rig.c"
        "model_store.c"
        "telemetry.c"
        "web.c"
        "net.c"
        "settings.c"
        "ota.c"
        "record.c"
        "events.c"
    INCLUDE_DIRS "."
    # The dashboard, served from the app image (symbols _binary_<name>_start/_end).
    EMBED_FILES
        "../web/index.html"
        "../web/app.js"
        "../web/style.css"
    REQUIRES
        cartpole_core
        esp_driver_gptimer
        esp_driver_pcnt
        esp_driver_ledc
        esp_driver_spi
        esp_driver_gpio
        esp_timer
        esp_event
        esp_netif
        esp_wifi
        esp_http_server
        esp_partition
        esp_psram
        spi_flash
        app_update
        esp_app_format
        bootloader_support
        nvs_flash
        json
)

# Time the network and the estimator inside ncp_ctrl_step without touching the
# portable core: calls from ncp_ctrl.c to these go through __wrap_* in control.c.
target_link_libraries(${COMPONENT_LIB} INTERFACE
    "-Wl,--wrap=ncp_policy_act"
    "-Wl,--wrap=ncp_ekf_update")
