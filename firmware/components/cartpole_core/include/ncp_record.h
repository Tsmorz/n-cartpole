/* ncp_record.h — the recording file (.ncpr) for offline system identification.
 *
 * The control task appends one sample per control tick (200 Hz) to a PSRAM
 * buffer while a recording runs; GET /api/record downloads header + samples.
 * n_cartpole/sysid/recording.py reads it and n_cartpole/sysid/fit.py fits the
 * plant parameters (scripts/sysid.py). Little-endian, packed. Change this, the
 * reader, the dashboard and firmware/tools/mock_device.py together and bump
 * NCP_REC_VERSION. */
#ifndef NCP_RECORD_H
#define NCP_RECORD_H

#include <stdint.h>

#include "ncp_model.h"

#ifdef __cplusplus
extern "C" {
#endif

#define NCP_REC_MAGIC "NCPR"
#define NCP_REC_VERSION 1
#define NCP_REC_HEADER_SIZE 256

typedef struct __attribute__((packed)) {
    char magic[4];         /* "NCPR" */
    uint16_t version;      /* NCP_REC_VERSION */
    uint16_t header_size;  /* NCP_REC_HEADER_SIZE */
    uint32_t sample_size;  /* sizeof(ncp_rec_sample_t) */
    uint32_t n_samples;
    uint32_t dropped;      /* ticks not recorded (buffer full / overrun) */
    uint32_t model_crc32;  /* crc32 of the model that was running (0: none) */
    double dt;             /* control period the samples were taken at (s) */
    uint8_t n_links;
    uint8_t backend;       /* 0 sim, 1 rig */
    uint8_t excite_kind;   /* ncp_exc_kind_t */
    uint8_t reserved0;
    float excite[6];       /* amp, f0, f1, period, duration, seed (as float) */
    uint64_t start_us;     /* esp_timer time of the first sample */
    char fw_version[32];
    char model_label[64];
    char note[64];         /* free text from the dashboard */
    uint8_t reserved1[28];
} ncp_rec_header_t;

_Static_assert(sizeof(ncp_rec_header_t) == NCP_REC_HEADER_SIZE, "ncp_rec_header_t must be 256 bytes");

enum {
    NCP_REC_ARMED = 1u << 0,   /* the drive was enabled */
    NCP_REC_EXCITE = 1u << 1,  /* excitation active this tick */
    NCP_REC_TRUTH = 1u << 2,   /* truth[] valid (sim backend) */
    NCP_REC_ENC_ERR = 1u << 3, /* an encoder read failed this tick (pos[] is the last good value) */
};

typedef struct __attribute__((packed)) {
    uint32_t tick;              /* control tick counter (gaps = dropped) */
    uint32_t t_us;              /* microseconds since start_us (actual sample time) */
    float pos[NCP_MAX_POS];     /* readings [x, th0, th1, ...]: calibrated, absolute, unwrapped */
    float cmd;                  /* force command sent to the drive this tick (N) */
    float applied;              /* the controller's model of the applied force (N) */
    float excite;               /* excitation part of cmd (N) */
    float est[NCP_MAX_STATE];   /* online estimate (EKF / diff) */
    float truth[NCP_MAX_STATE]; /* simulator true state (NaN on the rig) */
    uint8_t run, mode, goal, flags; /* NCP_RUN_*, ncp_mode_t, goal index, NCP_REC_* */
    uint16_t period_us;         /* time since the previous tick */
    uint16_t reserved;
} ncp_rec_sample_t;

_Static_assert(sizeof(ncp_rec_sample_t) == 128, "recording sample layout changed: update the readers");

#ifdef __cplusplus
}
#endif
#endif
