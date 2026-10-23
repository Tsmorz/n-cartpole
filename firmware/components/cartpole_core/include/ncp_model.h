/* ncp_model.h — the model blob (.ncpm) the firmware runs.
 *
 * One file carries everything the controller needs: the actor weights (the
 * RunningNorm already folded into in_w/in_b, only the mean row of the output
 * head), the nominal physics the EKF and the on-device simulator integrate,
 * the actuator limits, the sensor model and the observation layout. It is
 * written by n_cartpole/deploy/firmware_blob.py (`task export-firmware`), read
 * here, and parsed again by the dashboard (firmware/web/app.js) to run the same
 * network in the browser for the activation diagram.
 *
 * Layout (little-endian): a 512-byte ncp_header_t, then the weights as
 * float32 (dtype 0), IEEE float16 (dtype 1) or int8 (dtype 2), tensors back to
 * back in this order (H = hidden, E = hidden * expand, D = obs_dim):
 *   in_w [H x D], in_b [H],
 *   per block b: ln_w [H], ln_b [H], fc1_w [E x H], fc1_b [E], fc2_w [H x E], fc2_b [H],
 *   final_ln_w [H], final_ln_b [H], out_w [H], out_b [1].
 * Matrices are row-major [out][in], as in torch.nn.Linear.weight.
 * int8 (n_cartpole/deploy/quantize.py): only fc1_w and fc2_w are quantized, each
 * stored as float32 scale [rows], then int8 q [rows x cols], then zero bytes up to a
 * multiple of 4; w = q * scale[row]. Every other tensor stays float32. The device
 * keeps this image as it is (1 byte per matrix weight in RAM); f32 and f16 are
 * decoded to float32.
 * crc32 is CRC-32/IEEE (zlib.crc32) of the whole file with the crc32 field
 * itself read as zero.
 *
 * Portable C11: no ESP-IDF headers, so tests/test_firmware_core.py compiles it
 * on the host. */
#ifndef NCP_MODEL_H
#define NCP_MODEL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NCP_MAGIC "NCPM"
#define NCP_VERSION 1
#define NCP_HEADER_SIZE 512
#define NCP_MAX_LINKS 4
#define NCP_MAX_STATE (2 + 2 * NCP_MAX_LINKS) /* [x, xd, th0, th0d, ...] */
#define NCP_MAX_POS (1 + NCP_MAX_LINKS)       /* [x, th0, th1, ...] readings */
#define NCP_MAX_HIST 8                        /* action_history */
#define NCP_MAX_DELAY 8                       /* delay_steps */
#define NCP_MAX_GOALS (1 << NCP_MAX_LINKS)
/* kinematics (2 + 3n) + goal (n) + action history */
#define NCP_MAX_OBS (2 + 3 * NCP_MAX_LINKS + NCP_MAX_LINKS + NCP_MAX_HIST)

enum { NCP_DTYPE_F32 = 0, NCP_DTYPE_F16 = 1, NCP_DTYPE_I8 = 2 };

enum {
    NCP_FLAG_GOAL = 1u << 0,  /* goal-conditioned: cos(target) per link after the kinematics */
    NCP_FLAG_DELTA = 1u << 1, /* action_mode "delta": action is a fraction of the slew per step */
    NCP_FLAG_EKF = 1u << 2,   /* velocities from the EKF; otherwise differenced (+ EMA) */
};

typedef struct __attribute__((packed)) {
    char magic[4];        /* "NCPM" */
    uint16_t version;     /* NCP_VERSION */
    uint16_t header_size; /* NCP_HEADER_SIZE */
    uint32_t total_size;  /* header + weights, bytes */
    uint32_t crc32;       /* see above */
    uint32_t train_step;  /* checkpoint step, for display */
    uint8_t n_links, n_blocks, expand, dtype;
    uint16_t hidden, obs_dim;
    uint8_t flags, action_history, delay_steps, n_goals;
    uint16_t goal_settle_steps, reserved0;
    uint32_t reserved1; /* offset 40: the doubles start 8-aligned */
    /* Timing and actuator. force_slew <= 0: no slew limit. */
    double dt, force_max, force_slew, x_lim;
    /* Nominal plant (n_cartpole.env.dynamics.PhysicsParams). v_noload <= 0: none. */
    double M, g, b, cart_coulomb, cart_coulomb_vscale, v_noload;
    double masses[NCP_MAX_LINKS], lengths[NCP_MAX_LINKS], com[NCP_MAX_LINKS];
    double inertia[NCP_MAX_LINKS], joint_friction[NCP_MAX_LINKS];
    /* Velocity estimator (SensorModel) and the encoder model it was trained with. */
    double ekf_force_std, ekf_accel_std, velocity_filter;
    double x_noise_std, angle_noise_std, x_resolution, angle_resolution;
    /* "At goal": every link within tol_angle of its target and slower than tol_vel. */
    double goal_tol_angle, goal_tol_vel;
    char label[64]; /* NUL-terminated, e.g. "tqc-rail05-goals-ft-weighted@2000000" */
    uint8_t reserved2[96];
} ncp_header_t;

_Static_assert(sizeof(ncp_header_t) == NCP_HEADER_SIZE, "ncp_header_t must be 512 bytes");

typedef enum {
    NCP_OK = 0,
    NCP_E_MAGIC,       /* not an .ncpm file */
    NCP_E_VERSION,     /* written by a newer exporter */
    NCP_E_SIZE,        /* truncated, or sizes disagree with the dims */
    NCP_E_CRC,         /* corrupted */
    NCP_E_SHAPE,       /* dims out of range (n_links, obs_dim vs flags, ...) */
    NCP_E_UNSUPPORTED, /* valid but needs a feature this firmware lacks */
} ncp_err_t;

/* Pointers to each tensor inside the weight image (owned by the caller). An int8 model
 * has fc1_w/fc2_w NULL and fc1_q/fc1_s, fc2_q/fc2_s (int8 rows, float32 row scales) set. */
typedef struct {
    const float *ln_w, *ln_b, *fc1_w, *fc1_b, *fc2_w, *fc2_b;
    const int8_t *fc1_q, *fc2_q;
    const float *fc1_s, *fc2_s;
} ncp_block_t;

#define NCP_MAX_BLOCKS 4

typedef struct {
    ncp_header_t hdr;
    const float *in_w, *in_b;
    ncp_block_t blocks[NCP_MAX_BLOCKS];
    const float *final_ln_w, *final_ln_b, *out_w, *out_b;
} ncp_model_t;

const char *ncp_err_str(ncp_err_t err);

/* "f32", "f16", "i8" ("?" for anything else). */
const char *ncp_dtype_name(int dtype);

/* Streaming CRC-32/IEEE (zlib.crc32): crc = ncp_crc32(0, a, na); crc = ncp_crc32(crc, b, nb); */
uint32_t ncp_crc32(uint32_t crc, const void *data, size_t len);

/* CRC of a whole blob with its crc32 field read as zero. */
uint32_t ncp_blob_crc(const uint8_t *blob, size_t len);

/* Validate a header (magic, version, dims, total_size vs the dims). Does not check the CRC.
 * `len` is how many header bytes are available (>= NCP_HEADER_SIZE). */
ncp_err_t ncp_header_parse(const uint8_t *buf, size_t len, ncp_header_t *out);

/* Header + CRC over a complete in-memory blob of `len` bytes. */
ncp_err_t ncp_blob_validate(const uint8_t *blob, size_t len, ncp_header_t *out);

/* Number of weights (parameters) after the header (in_w .. out_b). */
size_t ncp_weight_count(const ncp_header_t *h);

/* Bytes the weights take in the file (count * 4 or * 2; int8: see the layout above). */
size_t ncp_weight_bytes(const ncp_header_t *h);

/* Bytes of the in-RAM weight image ncp_model_bind takes: count * 4 for f32 and f16 (decoded),
 * the file bytes for int8 (kept as is). */
size_t ncp_weight_ram_bytes(const ncp_header_t *h);

/* Decode `count` weights from an f32 or f16 file into float32. */
void ncp_weights_decode(const ncp_header_t *h, const uint8_t *src, float *dst, size_t count);

/* Load `n` file weight bytes that start `off` bytes into the weights (off and n multiples of 4,
 * the last chunk may end at ncp_weight_bytes) into their place in the RAM image `ram`
 * (ncp_weight_ram_bytes). Any dtype; lets a loader stream the file in chunks. */
void ncp_weights_load(const ncp_header_t *h, const uint8_t *src, size_t off, size_t n, void *ram);

/* Point m's tensors into the RAM image `weights` (ncp_weight_ram_bytes, 4-byte aligned). */
ncp_err_t ncp_model_bind(ncp_model_t *m, const ncp_header_t *h, const void *weights);

/* Target angles of goal `goal` (bit k set: link k down, pi; else up, 0) —
 * n_cartpole.env.goals.goal_configs. Labels: 'U'/'D' per link, base link first. */
void ncp_goal_target(int n_links, int goal, double *target);
void ncp_goal_label(int n_links, int goal, char *out /* n_links + 1 bytes */);

#ifdef __cplusplus
}
#endif
#endif
