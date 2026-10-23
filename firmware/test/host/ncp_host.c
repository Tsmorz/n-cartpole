/* ncp_host.c — host harness for the cartpole_core parity tests (tests/test_firmware_core.py).
 *
 *   ncp_host <model.ncpm> <command> [args] [key=value ...]
 *
 * Commands (stdin is read line by line; every number is printed with %.17g):
 *   info                 header fields, one "key=value" per line
 *   weights              the decoded float32 weights, one per line
 *   policy               obs_dim floats -> "force mean"
 *   deriv                state F -> d(state)/dt                (ncp_dyn_deriv)
 *   step                 state F -> next state                 (ncp_dyn_step, substeps=4)
 *   avail                F xd -> available force               (ncp_available_force)
 *   ekf                  first line: pos -> estimate after reset; then F pos -> estimate
 *   ctrl                 one tick per line of n + 1 readings (optionally + the true cart
 *                        speed, which then replaces the estimate in available_force: the
 *                        env uses the true speed there). Prints
 *                        obs[obs_dim] cmd est_force action settled_steps est[state_dim].
 *                        Control lines: "reset", "goal G", "xref X", "mode ai|manual".
 *   simcmd G P SEED      reset the sim at goal G perturbed by P; prints the state, then per
 *                        stdin command: "applied state[state_dim] reading[n+1]"
 *   sim FROM TO STEPS SEED       one closed-loop episode (sim + ctrl):
 *                        "success settle_time off_rail steps est_settled"
 *   matrix TRIALS SECONDS SEED   every FROM -> TO pair, TRIALS episodes each
 *   bench [ITERS]        microseconds per ncp_ctrl_step / policy / EKF update
 *   excite KIND AMP F0 F1 PERIOD DURATION SEED N
 *                        N samples of ncp_excite (KIND none|chirp|prbs|steps|sine):
 *                        "value active" per line
 *   record OUT SECONDS   closed loop sim + ctrl + excitation, written to OUT as .ncpr
 *                        (ncp_record.h). Options: arm=1 mode=manual|ai start=G goal=G
 *                        xref=X lift0=RAD lift1=RAD seed=S excite=KIND amp= f0= f1=
 *                        period= edur= eseed= (plus noise= perturb= realism=)
 *   ctrl also accepts "ext F" lines (ncp_ctrl_t.ext_force for the following ticks).
 * Options: realism=R scales the encoder noise and quantization (ctrl/sim/matrix/simcmd),
 *          like EnvConfig.realism; noise=0 turns the sim's encoder model off;
 *          perturb=P is the closed-loop start perturbation (default 0.05, the env's
 *          init_noise); substeps=N for step.
 */
#define _POSIX_C_SOURCE 200809L
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "ncp_ctrl.h"
#include "ncp_dynamics.h"
#include "ncp_ekf.h"
#include "ncp_excite.h"
#include "ncp_model.h"
#include "ncp_policy.h"
#include "ncp_record.h"
#include "ncp_sim.h"

#define MAX_TOK 64

static ncp_model_t g_model;
static ncp_policy_t g_pol;
static float *g_weights, *g_scratch;
static int g_header_crc_ok;

/* key=value options */
static double opt_realism = -1.0, opt_perturb = 0.05;
static int opt_noise = 1, opt_substeps = 4;
#define MAX_OPTS 32
static const char *opt_keys[MAX_OPTS], *opt_vals[MAX_OPTS];
static int n_opts;

static const char *opt_str(const char *key, const char *def) {
    for (int i = 0; i < n_opts; i++)
        if (!strcmp(opt_keys[i], key)) return opt_vals[i];
    return def;
}

static double opt_num(const char *key, double def) {
    const char *v = opt_str(key, NULL);
    return v ? atof(v) : def;
}

static void die(const char *msg) {
    fprintf(stderr, "error: %s\n", msg);
    exit(2);
}

static unsigned char *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open model file");
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 0) die("cannot size model file");
    unsigned char *buf = malloc((size_t)n + 1);
    if (!buf || fread(buf, 1, (size_t)n, f) != (size_t)n) die("cannot read model file");
    fclose(f);
    *len = (size_t)n;
    return buf;
}

static void load_model(const char *path) {
    size_t len;
    unsigned char *blob = read_file(path, &len);
    ncp_header_t h;
    ncp_err_t err = ncp_blob_validate(blob, len, &h);
    if (err != NCP_OK) die(ncp_err_str(err));
    g_header_crc_ok = ncp_blob_crc(blob, len) == h.crc32;
    size_t count = ncp_weight_count(&h);
    g_weights = malloc(count * sizeof(float));
    g_scratch = malloc(ncp_policy_scratch_floats(&h) * sizeof(float));
    if (!g_weights || !g_scratch) die("out of memory");
    ncp_weights_decode(&h, blob + NCP_HEADER_SIZE, g_weights, count);
    if (ncp_model_bind(&g_model, &h, g_weights) != NCP_OK) die("bind failed");
    ncp_policy_init(&g_pol, &g_model, g_scratch);
    free(blob);
    if (opt_realism >= 0.0) { /* EnvConfig.set_realism on the sensor model */
        ncp_header_t *m = &g_model.hdr;
        m->x_noise_std *= opt_realism;
        m->angle_noise_std *= opt_realism;
        m->x_resolution *= opt_realism;
        m->angle_resolution *= opt_realism;
    }
}

/* Split a line into numbers; returns the count (-1 if a token is not a number). */
static int parse_nums(char *line, double *out, int max) {
    int n = 0;
    char *save = NULL;
    for (char *tok = strtok_r(line, " \t\r\n,", &save); tok; tok = strtok_r(NULL, " \t\r\n,", &save)) {
        char *end;
        double v = strtod(tok, &end);
        if (end == tok || *end) return -1;
        if (n < max) out[n] = v;
        n++;
    }
    return n;
}

static void print_vec(const double *v, int n) {
    for (int i = 0; i < n; i++) printf(i ? " %.17g" : "%.17g", v[i]);
}

static void print_arr(const char *key, const double *v, int n) {
    printf("%s=", key);
    print_vec(v, n);
    printf("\n");
}

/* The blob header is packed, so its double arrays may be unaligned: copy before printing (gcc
 * rejects `-Waddress-of-packed-member` under -Werror; the void * parameter is exempt). */
static void print_packed(const char *key, const void *packed, int n) {
    double v[NCP_MAX_LINKS];
    memcpy(v, packed, sizeof(double) * (size_t)n);
    print_arr(key, v, n);
}

static void cmd_info(void) {
    const ncp_header_t *h = &g_model.hdr;
    const int n = h->n_links;
    printf("version=%u\nheader_size=%u\ntotal_size=%u\ncrc32=%u\ncrc_ok=%d\ntrain_step=%u\n",
           h->version, h->header_size, h->total_size, h->crc32, g_header_crc_ok, h->train_step);
    printf("n_links=%u\nn_blocks=%u\nexpand=%u\ndtype=%u\nhidden=%u\nobs_dim=%u\n", h->n_links,
           h->n_blocks, h->expand, h->dtype, h->hidden, h->obs_dim);
    printf("flags=%u\naction_history=%u\ndelay_steps=%u\nn_goals=%u\ngoal_settle_steps=%u\n",
           h->flags, h->action_history, h->delay_steps, h->n_goals, h->goal_settle_steps);
    printf("dt=%.17g\nforce_max=%.17g\nforce_slew=%.17g\nx_lim=%.17g\n", h->dt, h->force_max,
           h->force_slew, h->x_lim);
    printf("M=%.17g\ng=%.17g\nb=%.17g\ncart_coulomb=%.17g\ncart_coulomb_vscale=%.17g\n"
           "v_noload=%.17g\n",
           h->M, h->g, h->b, h->cart_coulomb, h->cart_coulomb_vscale, h->v_noload);
    print_packed("masses", h->masses, n);
    print_packed("lengths", h->lengths, n);
    print_packed("com", h->com, n);
    print_packed("inertia", h->inertia, n);
    print_packed("joint_friction", h->joint_friction, n);
    printf("ekf_force_std=%.17g\nekf_accel_std=%.17g\nvelocity_filter=%.17g\n", h->ekf_force_std,
           h->ekf_accel_std, h->velocity_filter);
    printf("x_noise_std=%.17g\nangle_noise_std=%.17g\nx_resolution=%.17g\nangle_resolution=%.17g\n",
           h->x_noise_std, h->angle_noise_std, h->x_resolution, h->angle_resolution);
    printf("goal_tol_angle=%.17g\ngoal_tol_vel=%.17g\nlabel=%s\n", h->goal_tol_angle,
           h->goal_tol_vel, h->label);
    printf("weight_count=%zu\nweight_bytes=%zu\n", ncp_weight_count(h), ncp_weight_bytes(h));
    double pos_std[NCP_MAX_POS];
    ncp_ekf_pos_std(h, pos_std);
    print_arr("pos_std", pos_std, n + 1);
    printf("goal_labels=");
    for (int g = 0; g < (1 << n); g++) {
        char lab[NCP_MAX_LINKS + 1];
        ncp_goal_label(n, g, lab);
        printf(g ? " %s" : "%s", lab);
    }
    printf("\ngoal_targets=");
    for (int g = 0; g < (1 << n); g++) {
        double t[NCP_MAX_LINKS];
        ncp_goal_target(n, g, t);
        for (int k = 0; k < n; k++) printf((g || k) ? " %.17g" : "%.17g", t[k]);
    }
    printf("\n");
}

static void cmd_weights(void) {
    size_t count = ncp_weight_count(&g_model.hdr);
    for (size_t i = 0; i < count; i++) printf("%.9g\n", (double)g_weights[i]);
}

static void cmd_policy(void) {
    const int D = g_model.hdr.obs_dim;
    char line[8192];
    double v[NCP_MAX_OBS];
    float obs[NCP_MAX_OBS];
    while (fgets(line, sizeof line, stdin)) {
        int k = parse_nums(line, v, NCP_MAX_OBS);
        if (k == 0) continue;
        if (k != D) die("policy: expected obs_dim numbers per line");
        for (int i = 0; i < D; i++) obs[i] = (float)v[i];
        float f = ncp_policy_act(&g_pol, obs);
        printf("%.9g %.9g\n", (double)f, (double)g_pol.mean);
    }
}

static void cmd_dyn(int which) {
    ncp_dyn_t d;
    ncp_dyn_init(&d, &g_model.hdr);
    const int dim = 2 + 2 * d.n;
    char line[8192];
    double v[MAX_TOK], out[NCP_MAX_STATE];
    while (fgets(line, sizeof line, stdin)) {
        int k = parse_nums(line, v, MAX_TOK);
        if (k == 0) continue;
        if (which == 2) {
            if (k != 2) die("avail: expected F xd");
            printf("%.17g\n", ncp_available_force(&d, v[0], v[1]));
            continue;
        }
        if (k != dim + 1) die("deriv/step: expected state F");
        if (which == 0) {
            ncp_dyn_deriv(&d, v, v[dim], out);
        } else {
            memcpy(out, v, sizeof(double) * (size_t)dim);
            ncp_dyn_step(&d, out, v[dim], opt_substeps);
        }
        print_vec(out, dim);
        printf("\n");
    }
}

static void cmd_ekf(void) {
    ncp_dyn_t d;
    ncp_ekf_t e;
    ncp_dyn_init(&d, &g_model.hdr);
    double pos_std[NCP_MAX_POS];
    ncp_ekf_pos_std(&g_model.hdr, pos_std);
    ncp_ekf_init(&e, &d, pos_std, g_model.hdr.ekf_force_std, g_model.hdr.ekf_accel_std);
    const int np = d.n + 1;
    int first = 1;
    char line[8192];
    double v[MAX_TOK];
    while (fgets(line, sizeof line, stdin)) {
        int k = parse_nums(line, v, MAX_TOK);
        if (k == 0) continue;
        if (first) {
            if (k != np) die("ekf: first line must be the n + 1 readings");
            ncp_ekf_reset(&e, v);
            first = 0;
        } else {
            if (k != np + 1) die("ekf: expected F pos...");
            ncp_ekf_update(&e, v[0], v + 1);
        }
        print_vec(e.x, e.dim);
        printf("\n");
    }
}

static void cmd_ctrl(void) {
    static ncp_ctrl_t c; /* the EKF keeps a pointer into it: never moved */
    ncp_ctrl_init(&c, &g_model, &g_pol);
    const int np = c.n + 1;
    char line[8192];
    double v[MAX_TOK];
    while (fgets(line, sizeof line, stdin)) {
        char *p = line;
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p) continue;
        if (isalpha((unsigned char)*p)) {
            char word[32] = {0}, arg[64] = {0};
            if (sscanf(p, "%31s %63s", word, arg) < 1) continue;
            if (!strcmp(word, "reset")) {
                ncp_ctrl_reset(&c);
            } else if (!strcmp(word, "goal")) {
                ncp_ctrl_set_goal(&c, atoi(arg));
            } else if (!strcmp(word, "xref")) {
                ncp_ctrl_set_x_ref(&c, atof(arg));
            } else if (!strcmp(word, "ext")) {
                c.ext_force = atof(arg);
            } else if (!strcmp(word, "mode")) {
                c.mode = !strcmp(arg, "manual") ? NCP_MODE_MANUAL : NCP_MODE_AI;
            } else {
                die("ctrl: unknown control line");
            }
            continue;
        }
        int k = parse_nums(p, v, MAX_TOK);
        if (k != np && k != np + 1) die("ctrl: expected n + 1 readings [+ true cart speed]");
        double cmd = ncp_ctrl_step(&c, v);
        if (k == np + 1) c.est_force = ncp_available_force(&c.dyn, c.prev_applied, v[np]);
        for (int i = 0; i < c.obs_dim; i++) printf(i ? " %.9g" : "%.9g", (double)c.obs[i]);
        printf(" %.17g %.17g %.9g %d ", cmd, c.est_force, (double)c.action, c.settled_steps);
        print_vec(c.est, c.state_dim);
        printf("\n");
    }
}

static void sim_init(ncp_sim_t *sim, uint64_t seed) {
    ncp_sim_init(sim, &g_model.hdr, seed);
    sim->sensor_model = opt_noise != 0;
}

static void cmd_simcmd(int goal, double perturb, uint64_t seed) {
    ncp_sim_t sim;
    sim_init(&sim, seed);
    ncp_sim_reset(&sim, goal, perturb);
    const int dim = 2 + 2 * sim.n;
    print_vec(sim.s, dim);
    printf("\n");
    char line[8192];
    double v[MAX_TOK], pos[NCP_MAX_POS];
    while (fgets(line, sizeof line, stdin)) {
        int k = parse_nums(line, v, MAX_TOK);
        if (k == 0) continue;
        if (k != 1) die("simcmd: one command per line");
        ncp_sim_apply(&sim, v[0]);
        ncp_sim_read(&sim, pos);
        printf("%.17g ", sim.applied);
        print_vec(sim.s, dim);
        printf(" ");
        print_vec(pos, sim.n + 1);
        printf(" %d\n", ncp_sim_off_rail(&sim));
    }
}

/* Every link within tol of its target and slower than tol_vel (env._at_goal, true state). */
static int true_at_goal(const ncp_sim_t *sim, const double *target) {
    const ncp_header_t *h = &g_model.hdr;
    for (int k = 0; k < sim->n; k++) {
        double d = remainder(sim->s[2 + 2 * k] - target[k], 6.28318530717958647692);
        if (!(fabs(d) < h->goal_tol_angle) || !(fabs(sim->s[3 + 2 * k]) < h->goal_tol_vel)) return 0;
    }
    return 1;
}

typedef struct {
    int success, off_rail, steps, est_settled;
    double settle_time;
} episode_t;

static episode_t run_episode(ncp_ctrl_t *c, ncp_sim_t *sim, int from, int to, int steps) {
    const ncp_header_t *h = &g_model.hdr;
    double target[NCP_MAX_LINKS], pos[NCP_MAX_POS];
    ncp_goal_target(sim->n, to, target);
    ncp_sim_reset(sim, from, opt_perturb);
    ncp_ctrl_reset(c);
    ncp_ctrl_set_goal(c, to);
    episode_t ep = {0, 0, 0, 0, NAN};
    int settled = 0;
    for (int i = 0; i < steps; i++) {
        ncp_sim_read(sim, pos);
        double cmd = ncp_ctrl_step(c, pos);
        ncp_sim_apply(sim, cmd);
        ep.steps = i + 1;
        if (ncp_sim_off_rail(sim)) {
            ep.off_rail = 1;
            break;
        }
        settled = true_at_goal(sim, target) ? settled + 1 : 0;
        if (settled == h->goal_settle_steps && isnan(ep.settle_time))
            ep.settle_time = (ep.steps - h->goal_settle_steps) * h->dt;
    }
    ep.success = !ep.off_rail && settled >= h->goal_settle_steps;
    ep.est_settled = ncp_ctrl_settled(c);
    return ep;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static void cmd_sim(int from, int to, int steps, uint64_t seed) {
    static ncp_ctrl_t c;
    ncp_sim_t sim;
    ncp_ctrl_init(&c, &g_model, &g_pol);
    sim_init(&sim, seed);
    episode_t ep = run_episode(&c, &sim, from, to, steps);
    printf("%d %.17g %d %d %d\n", ep.success, ep.settle_time, ep.off_rail, ep.steps, ep.est_settled);
}

static void cmd_matrix(int trials, double seconds, uint64_t seed) {
    static ncp_ctrl_t c;
    ncp_ctrl_init(&c, &g_model, &g_pol);
    const ncp_header_t *h = &g_model.hdr;
    const int G = h->n_goals, n = h->n_links;
    const int steps = (int)lround(seconds / h->dt);
    int total = 0, ok_total = 0;
    double *times = malloc(sizeof(double) * (size_t)(trials > 0 ? trials : 1));
    char lab_f[NCP_MAX_LINKS + 1], lab_t[NCP_MAX_LINKS + 1];
    printf("from to success trials off_rail median_settle_s est_settled\n");
    for (int f = 0; f < G; f++) {
        for (int t = 0; t < G; t++) {
            int ok = 0, off = 0, est = 0, nt = 0;
            for (int i = 0; i < trials; i++) {
                ncp_sim_t sim;
                sim_init(&sim, seed + (uint64_t)(1000 * (f * G + t) + i));
                episode_t ep = run_episode(&c, &sim, f, t, steps);
                ok += ep.success;
                off += ep.off_rail;
                est += ep.est_settled;
                if (!isnan(ep.settle_time)) times[nt++] = ep.settle_time;
            }
            qsort(times, (size_t)nt, sizeof(double), cmp_double);
            double med = nt ? (nt % 2 ? times[nt / 2] : 0.5 * (times[nt / 2 - 1] + times[nt / 2])) : NAN;
            ncp_goal_label(n, f, lab_f);
            ncp_goal_label(n, t, lab_t);
            printf("%s %s %d %d %d %.3f %d\n", lab_f, lab_t, ok, trials, off, med, est);
            total += trials;
            ok_total += ok;
            fflush(stdout);
        }
    }
    printf("overall %d/%d %.1f%%\n", ok_total, total, total ? 100.0 * ok_total / total : 0.0);
    free(times);
}

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec * 1e-3;
}

static void cmd_bench(int iters) {
    static ncp_ctrl_t c;
    ncp_sim_t sim;
    ncp_ctrl_init(&c, &g_model, &g_pol);
    sim_init(&sim, 1);
    const int G = g_model.hdr.n_goals;
    ncp_ctrl_set_goal(&c, 0);
    ncp_sim_reset(&sim, G - 1, 0.05);
    double pos[NCP_MAX_POS];
    double t_ctrl = 0.0, t_pol = 0.0, t_ekf = 0.0;
    volatile float sink = 0.0f;
    for (int i = 0; i < iters; i++) {
        ncp_sim_read(&sim, pos);
        double t0 = now_us();
        double cmd = ncp_ctrl_step(&c, pos);
        double t1 = now_us();
        sink += ncp_policy_act(&g_pol, c.obs);
        double t2 = now_us();
        ncp_ekf_t e = c.ekf; /* a copy, so the controller's filter is undisturbed */
        ncp_ekf_update(&e, c.est_force, pos);
        double t3 = now_us();
        t_ctrl += t1 - t0;
        t_pol += t2 - t1;
        t_ekf += t3 - t2;
        ncp_sim_apply(&sim, cmd);
        if (ncp_sim_off_rail(&sim)) {
            ncp_sim_reset(&sim, G - 1, 0.05);
            ncp_ctrl_reset(&c);
        }
    }
    (void)sink;
    printf("ctrl_us=%.3f\npolicy_us=%.3f\nekf_us=%.3f\niters=%d\n", t_ctrl / iters, t_pol / iters,
           t_ekf / iters, iters);
}

static ncp_exc_kind_t exc_kind(const char *name) {
    if (!strcmp(name, "chirp")) return NCP_EXC_CHIRP;
    if (!strcmp(name, "prbs")) return NCP_EXC_PRBS;
    if (!strcmp(name, "steps")) return NCP_EXC_STEPS;
    if (!strcmp(name, "sine")) return NCP_EXC_SINE;
    if (!strcmp(name, "none")) return NCP_EXC_NONE;
    die("unknown excitation kind");
    return NCP_EXC_NONE;
}

static void cmd_excite(char **a) {
    ncp_excite_params_t p = {exc_kind(a[0]), atof(a[1]), atof(a[2]), atof(a[3]),
                             atof(a[4]), atof(a[5]), (uint32_t)strtoul(a[6], NULL, 10)};
    ncp_excite_t e;
    ncp_excite_start(&e, &p, g_model.hdr.dt);
    const int n = atoi(a[7]);
    for (int i = 0; i < n; i++) {
        double v = ncp_excite_next(&e);
        printf("%.17g %d\n", v, e.active);
    }
}

static void put_str(char *dst, size_t size, const char *src) {
    memset(dst, 0, size);
    if (src) memcpy(dst, src, strnlen(src, size - 1));
}

static void cmd_record(const char *out, double seconds) {
    static ncp_ctrl_t c;
    ncp_sim_t sim;
    const ncp_header_t *h = &g_model.hdr;
    const int n = h->n_links, G = h->n_goals;
    ncp_ctrl_init(&c, &g_model, &g_pol);
    sim_init(&sim, (uint64_t)opt_num("seed", 1));
    const int start = (int)opt_num("start", (1 << n) - 1);
    ncp_sim_reset(&sim, start, opt_perturb);
    sim.s[2] += opt_num("lift0", 0.0);
    if (n > 1) sim.s[4] += opt_num("lift1", 0.0);
    ncp_ctrl_reset(&c);
    ncp_ctrl_set_goal(&c, (int)opt_num("goal", start < G ? start : 0));
    ncp_ctrl_set_x_ref(&c, opt_num("xref", 0.0));
    c.mode = !strcmp(opt_str("mode", "manual"), "ai") ? NCP_MODE_AI : NCP_MODE_MANUAL;
    const int armed = (int)opt_num("arm", 1);

    ncp_excite_params_t ep = {exc_kind(opt_str("excite", "none")), opt_num("amp", 3.0),
                              opt_num("f0", 0.2), opt_num("f1", 6.0), opt_num("period", 0.05),
                              opt_num("edur", seconds), (uint32_t)opt_num("eseed", 1)};
    ncp_excite_t ex;
    ncp_excite_start(&ex, &ep, h->dt);

    const long N = lround(seconds / h->dt);
    ncp_rec_sample_t *buf = calloc((size_t)(N > 0 ? N : 1), sizeof *buf);
    if (!buf) die("out of memory");
    long count = 0;
    double pos[NCP_MAX_POS];
    for (long i = 0; i < N; i++) {
        ncp_rec_sample_t *r = &buf[count];
        double truth[NCP_MAX_STATE];
        memcpy(truth, sim.s, sizeof truth);
        ncp_sim_read(&sim, pos);
        const double ext = armed ? ncp_excite_next(&ex) : 0.0;
        c.ext_force = ext;
        const double cmd = ncp_ctrl_step(&c, pos);
        ncp_sim_apply(&sim, armed ? cmd : 0.0);
        r->tick = (uint32_t)i;
        r->t_us = (uint32_t)lround((double)i * h->dt * 1e6);
        for (int k = 0; k < NCP_MAX_POS; k++) r->pos[k] = k <= n ? (float)pos[k] : 0.0f;
        r->cmd = (float)cmd;
        r->applied = (float)c.est_force;
        r->excite = (float)ext;
        for (int k = 0; k < NCP_MAX_STATE; k++) {
            r->est[k] = k < c.state_dim ? (float)c.est[k] : 0.0f;
            r->truth[k] = k < c.state_dim ? (float)truth[k] : 0.0f;
        }
        r->run = armed ? 1 : 0;
        r->mode = (uint8_t)c.mode;
        r->goal = (uint8_t)c.goal;
        r->flags = (uint8_t)((armed ? NCP_REC_ARMED : 0) | (ex.active || ext != 0.0 ? NCP_REC_EXCITE : 0) |
                             NCP_REC_TRUTH);
        r->period_us = (uint16_t)lround(h->dt * 1e6);
        count++;
        if (ncp_sim_off_rail(&sim)) break; /* the device would fault and stop */
    }

    ncp_rec_header_t hd;
    memset(&hd, 0, sizeof hd);
    memcpy(hd.magic, NCP_REC_MAGIC, 4);
    hd.version = NCP_REC_VERSION;
    hd.header_size = NCP_REC_HEADER_SIZE;
    hd.sample_size = sizeof(ncp_rec_sample_t);
    hd.n_samples = (uint32_t)count;
    hd.model_crc32 = h->crc32;
    hd.dt = h->dt;
    hd.n_links = (uint8_t)n;
    hd.backend = 0;
    hd.excite_kind = (uint8_t)ep.kind;
    const float exc[6] = {(float)ep.amp, (float)ep.f0, (float)ep.f1, (float)ep.period,
                          (float)ep.duration, (float)ep.seed};
    memcpy(hd.excite, exc, sizeof exc);
    hd.start_us = 1000000u;
    put_str(hd.fw_version, sizeof hd.fw_version, "host");
    put_str(hd.model_label, sizeof hd.model_label, h->label);
    put_str(hd.note, sizeof hd.note, opt_str("note", ""));
    FILE *f = fopen(out, "wb");
    if (!f) die("cannot write recording");
    if (fwrite(&hd, sizeof hd, 1, f) != 1 ||
        (count && fwrite(buf, sizeof *buf, (size_t)count, f) != (size_t)count))
        die("short write");
    fclose(f);
    free(buf);
    printf("samples=%ld\n", count);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.ncpm> <command> [args] [key=value ...]\n", argv[0]);
        return 1;
    }
    /* Pull key=value options out of the argument list. */
    char *pos_args[16];
    int npos = 0;
    for (int i = 3; i < argc; i++) {
        char *eq = strchr(argv[i], '=');
        if (eq) {
            *eq = '\0';
            const char *k = argv[i], *v = eq + 1;
            if (!strcmp(k, "realism")) opt_realism = atof(v);
            else if (!strcmp(k, "noise")) opt_noise = atoi(v);
            else if (!strcmp(k, "perturb")) opt_perturb = atof(v);
            else if (!strcmp(k, "substeps")) opt_substeps = atoi(v);
            else if (n_opts < MAX_OPTS) {
                opt_keys[n_opts] = k;
                opt_vals[n_opts++] = v;
            }
        } else if (npos < 16) {
            pos_args[npos++] = argv[i];
        }
    }
    load_model(argv[1]);
    const char *cmd = argv[2];
    if (!strcmp(cmd, "info")) cmd_info();
    else if (!strcmp(cmd, "weights")) cmd_weights();
    else if (!strcmp(cmd, "policy")) cmd_policy();
    else if (!strcmp(cmd, "deriv")) cmd_dyn(0);
    else if (!strcmp(cmd, "step")) cmd_dyn(1);
    else if (!strcmp(cmd, "avail")) cmd_dyn(2);
    else if (!strcmp(cmd, "ekf")) cmd_ekf();
    else if (!strcmp(cmd, "ctrl")) cmd_ctrl();
    else if (!strcmp(cmd, "simcmd") && npos >= 3)
        cmd_simcmd(atoi(pos_args[0]), atof(pos_args[1]), strtoull(pos_args[2], NULL, 10));
    else if (!strcmp(cmd, "sim") && npos >= 4)
        cmd_sim(atoi(pos_args[0]), atoi(pos_args[1]), atoi(pos_args[2]), strtoull(pos_args[3], NULL, 10));
    else if (!strcmp(cmd, "matrix") && npos >= 3)
        cmd_matrix(atoi(pos_args[0]), atof(pos_args[1]), strtoull(pos_args[2], NULL, 10));
    else if (!strcmp(cmd, "bench")) cmd_bench(npos >= 1 ? atoi(pos_args[0]) : 2000);
    else if (!strcmp(cmd, "excite") && npos >= 8) cmd_excite(pos_args);
    else if (!strcmp(cmd, "record") && npos >= 2) cmd_record(pos_args[0], atof(pos_args[1]));
    else die("unknown command or missing arguments");
    return 0;
}
