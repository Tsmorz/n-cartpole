/* ncp_policy.h — the SimBa actor's deterministic forward pass, float32 activations (int8 or
 * float32 block matrices, see ncp_model.h).
 *
 * in_proj -> blocks x [h += fc2(relu(fc1(LayerNorm(h))))] -> LayerNorm -> mean;
 * force = tanh(mean) * force_max  (n_cartpole/policy/simba.py, tqc.py). The
 * same function as personal-site/assets/js/swingup.js createPolicy(). */
#ifndef NCP_POLICY_H
#define NCP_POLICY_H

#include "ncp_model.h"

#ifdef __cplusplus
extern "C" {
#endif

#define NCP_LN_EPS 1e-5f

typedef struct {
    const ncp_model_t *m;
    float *h, *t, *u; /* scratch: H, H, E floats */
    float mean;       /* pre-tanh output of the last call */
} ncp_policy_t;

/* Floats of scratch ncp_policy_init needs: 2 * hidden + hidden * expand. */
size_t ncp_policy_scratch_floats(const ncp_header_t *h);

void ncp_policy_init(ncp_policy_t *p, const ncp_model_t *m, float *scratch);

/* obs: obs_dim floats. Returns the actor's force in newtons, tanh(mean) * force_max. */
float ncp_policy_act(ncp_policy_t *p, const float *obs);

#ifdef __cplusplus
}
#endif
#endif
