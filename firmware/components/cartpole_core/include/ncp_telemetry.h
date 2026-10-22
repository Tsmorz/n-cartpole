/* ncp_telemetry.h — what the control task streams to the dashboard.
 *
 * WebSocket /ws binary message = ncp_tlm_msg_hdr_t followed by `count` frames.
 * Little-endian, packed; firmware/web/app.js decodes these exact offsets (and
 * firmware/tools/mock_device.py writes them), so change all three together and
 * bump NCP_TLM_VERSION. Unused array entries (n_links < 4, shorter obs) are 0;
 * truth[] is only valid with NCP_TLM_TRUTH (the simulator backend). */
#ifndef NCP_TELEMETRY_H
#define NCP_TELEMETRY_H

#include <stdint.h>

#include "ncp_model.h"

#define NCP_TLM_VERSION 1

enum { NCP_RUN_IDLE = 0, NCP_RUN_ARMED = 1, NCP_RUN_FAULT = 2, NCP_RUN_MAINT = 3 };

enum {
    NCP_TLM_SETTLED = 1u << 0, /* at the goal for goal_settle_steps */
    NCP_TLM_RANDOM = 1u << 1,  /* random goal cycling on */
    NCP_TLM_SIM = 1u << 2,     /* simulator backend */
    NCP_TLM_TRUTH = 1u << 3,   /* truth[] holds the simulated true state */
};

typedef struct __attribute__((packed)) {
    uint32_t tick;   /* control ticks since boot */
    uint8_t run;     /* NCP_RUN_* */
    uint8_t mode;    /* ncp_mode_t */
    uint8_t goal;    /* goal index */
    uint8_t flags;   /* NCP_TLM_* */
    float est[NCP_MAX_STATE];  /* estimated state (EKF / diff) */
    float meas[NCP_MAX_POS];   /* encoder readings [x, th0, ...] */
    float truth[NCP_MAX_STATE];/* simulator true state */
    float obs[NCP_MAX_OBS];    /* network input */
    float cmd;      /* force command sent to the drive (N) */
    float applied;  /* modelled applied force (N) */
    float x_ref;    /* cart setpoint (m) */
    uint16_t loop_us;   /* execution time of this tick */
    uint16_t policy_us; /* of which the network */
    uint16_t est_us;    /* of which the estimator */
    uint16_t period_us; /* time since the previous tick */
} ncp_tlm_frame_t;

_Static_assert(sizeof(ncp_tlm_frame_t) == 232, "telemetry frame layout changed: update app.js");

typedef struct __attribute__((packed)) {
    uint8_t version;      /* NCP_TLM_VERSION */
    uint8_t count;        /* frames that follow */
    uint16_t frame_size;  /* sizeof(ncp_tlm_frame_t) */
} ncp_tlm_msg_hdr_t;

#endif
