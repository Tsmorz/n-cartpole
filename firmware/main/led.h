/* led.h — the dev board's addressable RGB LED (WS2812 on CONFIG_NCP_LED_GPIO) as a status
 * light, so the device's state is visible without the dashboard. Highest priority first:
 *
 *   amber, fast blink      writing flash (firmware/model upload, settings): do not unplug
 *   red, blinking          fault (off rail, e-stop, encoder, deadline): disarm to clear
 *   green, solid           armed, AI, settled at the goal
 *   green, breathing       armed, AI, on the way to the goal
 *   cyan, solid            armed, manual (PD to the slider)
 *   + a red blip each 1 s  a recording is running (over any of the above)
 *   amber, solid           no model: the controller idles until one is uploaded
 *   purple, slow blink     setup access point (cartpole-XXXX): join it to set the Wi-Fi
 *   blue, slow blink       joining the Wi-Fi network
 *   white, short blip/3 s  idle and connected: alive, ready to arm
 *   blue, solid            booting
 *
 * Core 0 only (app_main, then the tlm task). The RMT interrupt is allocated on core 0. */
#pragma once

#include <stdint.h>

/* Configure the RMT channel and show "booting". No-op when CONFIG_NCP_LED_GPIO < 0. */
void led_init(void);
/* Recompute the pattern from the device's state and send it if it changed; call at ~25 Hz. */
void led_update(int64_t now_us);
