/* telemetry.c — frame ring (ctrl -> tlm) and the tlm task (core 0, prio 4). */
#include "telemetry.h"

#include <math.h>
#include <string.h>

#include "cJSON.h"
#include "control.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "events.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "freertos/idf_additions.h"
#include "hal.h"
#include "heap_watch.h"
#include "led.h"
#include "model_store.h"
#include "net.h"
#include "record.h"
#include "settings.h"
#include "web.h"

static const char *TAG = "tlm";

#define RING_N 64              /* frames (power of two): 1.3 s at 50 Hz */
#define SEND_PERIOD_MS 40      /* ~25 WebSocket messages per second */
#define STATUS_PERIOD_MS 500   /* status JSON at 2 Hz */
#define FRAMES_PER_MSG 16
#define STATUS_SIZE STATUS_JSON_MAX
#define MAX_TASKS 40

/* ---------- SPSC ring ---------- */

static ncp_tlm_frame_t s_ring[RING_N];
static volatile uint32_t s_head; /* written by the producer only */
static volatile uint32_t s_tail; /* written by the consumer only */
static volatile uint32_t s_dropped;

bool tlm_push(const ncp_tlm_frame_t *f) {
    uint32_t head = s_head;
    uint32_t tail = __atomic_load_n(&s_tail, __ATOMIC_ACQUIRE);
    if (head - tail >= RING_N) {
        s_dropped = s_dropped + 1;
        return false;
    }
    memcpy(&s_ring[head % RING_N], f, sizeof(*f));
    __atomic_store_n(&s_head, head + 1, __ATOMIC_RELEASE);
    return true;
}

uint32_t tlm_dropped(void) { return s_dropped; }

static size_t tlm_pop(ncp_tlm_frame_t *out, size_t max) {
    uint32_t tail = s_tail;
    uint32_t head = __atomic_load_n(&s_head, __ATOMIC_ACQUIRE);
    size_t n = head - tail;
    if (n > max) {
        n = max;
    }
    for (size_t i = 0; i < n; i++) {
        memcpy(&out[i], &s_ring[(tail + i) % RING_N], sizeof(out[i]));
    }
    __atomic_store_n(&s_tail, tail + (uint32_t)n, __ATOMIC_RELEASE);
    return n;
}

/* ---------- status JSON ---------- */

typedef struct {
    TaskHandle_t handle;
    configRUN_TIME_COUNTER_TYPE run_time;
} prev_task_t;

static TaskStatus_t s_tasks[MAX_TASKS];
static prev_task_t s_prev[MAX_TASKS];
static int s_n_prev;
static configRUN_TIME_COUNTER_TYPE s_prev_total;

static char *s_status;     /* latest status JSON */
static size_t s_status_len;
static SemaphoreHandle_t s_status_lock;

static double round1(double v) { return round(v * 10.0) / 10.0; }

static void add_tasks(cJSON *root) {
    configRUN_TIME_COUNTER_TYPE total = 0;
    UBaseType_t n = uxTaskGetSystemState(s_tasks, MAX_TASKS, &total);
    const double span = (double)(total - s_prev_total);
    TaskHandle_t idle[2] = {xTaskGetIdleTaskHandleForCore(0), xTaskGetIdleTaskHandleForCore(1)};
    double idle_cpu[2] = {100.0, 100.0};

    cJSON *arr = cJSON_AddArrayToObject(root, "tasks");
    for (UBaseType_t i = 0; i < n; i++) {
        const TaskStatus_t *t = &s_tasks[i];
        double cpu = 0.0;
        for (int j = 0; j < s_n_prev; j++) {
            if (s_prev[j].handle == t->xHandle) {
                if (span > 0.0) {
                    cpu = 100.0 * (double)(t->ulRunTimeCounter - s_prev[j].run_time) / span;
                }
                break;
            }
        }
        if (cpu < 0.0) {
            cpu = 0.0;
        }
        for (int c = 0; c < 2; c++) {
            if (t->xHandle == idle[c]) {
                idle_cpu[c] = cpu;
            }
        }
#if defined(configTASKLIST_INCLUDE_COREID) && configTASKLIST_INCLUDE_COREID == 1
        BaseType_t core = t->xCoreID;
#else
        BaseType_t core = xTaskGetCoreID(t->xHandle);
#endif
        cJSON *o = cJSON_CreateObject();
        cJSON_AddStringToObject(o, "name", t->pcTaskName);
        cJSON_AddNumberToObject(o, "core", core == tskNO_AFFINITY ? -1 : (double)core);
        cJSON_AddNumberToObject(o, "prio", (double)t->uxCurrentPriority);
        cJSON_AddNumberToObject(o, "cpu", round1(cpu));
        cJSON_AddNumberToObject(o, "stack_free", (double)t->usStackHighWaterMark);
        cJSON_AddItemToArray(arr, o);
    }
    s_n_prev = 0;
    for (UBaseType_t i = 0; i < n && i < MAX_TASKS; i++) {
        s_prev[s_n_prev].handle = s_tasks[i].xHandle;
        s_prev[s_n_prev].run_time = s_tasks[i].ulRunTimeCounter;
        s_n_prev++;
    }
    s_prev_total = total;

    cJSON *cores = cJSON_AddArrayToObject(root, "cores");
    for (int c = 0; c < 2; c++) {
        double load = 100.0 - idle_cpu[c];
        cJSON *o = cJSON_CreateObject();
        cJSON_AddNumberToObject(o, "id", c);
        cJSON_AddNumberToObject(o, "load", round1(load < 0.0 ? 0.0 : (load > 100.0 ? 100.0 : load)));
        cJSON_AddItemToArray(cores, o);
    }
}

/* Model strings for one status cycle, copied under the model lock (an upload may free
 * the model the snapshot points to). */
typedef struct {
    bool have;
    char label[sizeof(((ncp_header_t *)0)->label)];
    char goal[NCP_MAX_LINKS + 1];
} model_strs_t;

static void model_strings(const ctrl_status_t *st, model_strs_t *out) {
    memset(out, 0, sizeof(*out));
    model_store_lock();
    const model_t *m = model_store_current();
    if (st->model != NULL && st->model == m) {
        out->have = true;
        strlcpy(out->label, m->hdr.label, sizeof(out->label));
        if (st->goal < NCP_MAX_GOALS) {
            strlcpy(out->goal, m->goal_labels[st->goal], sizeof(out->goal));
        }
    }
    model_store_unlock();
}

/* Sensors, drive and zero references (rig); {"rig": false} on the simulator. */
static void add_io(cJSON *root, const ctrl_status_t *st) {
    const ncp_hal_io_t *io = &st->io;
    cJSON *o = cJSON_AddObjectToObject(root, "io");
    cJSON_AddBoolToObject(o, "rig", io->rig);
    if (!io->rig) {
        return;
    }
    cJSON_AddBoolToObject(o, "estop", io->estop);
    cJSON_AddBoolToObject(o, "drive", io->drive_on);
    cJSON_AddNumberToObject(o, "duty", round(io->duty * 1000.0) / 1000.0);
    cJSON_AddNumberToObject(o, "dir", io->dir);
    cJSON_AddNumberToObject(o, "cart_count", io->cart_count);
    cJSON_AddNumberToObject(o, "read_errors", io->read_errors);
    cJSON *enc = cJSON_AddArrayToObject(o, "enc");
    for (int i = 0; i < 2; i++) {
        cJSON *e = cJSON_CreateObject();
        cJSON_AddNumberToObject(e, "raw", io->enc[i].raw);
        cJSON_AddNumberToObject(e, "bad", io->enc[i].bad);
        if (io->enc[i].diag) {
            cJSON_AddNumberToObject(e, "agc", io->enc[i].agc);
            cJSON_AddBoolToObject(e, "ocf", io->enc[i].ocf);
            cJSON_AddBoolToObject(e, "cof", io->enc[i].cof);
            cJSON_AddBoolToObject(e, "comp_low", io->enc[i].comp_low);
            cJSON_AddBoolToObject(e, "comp_high", io->enc[i].comp_high);
        }
        cJSON_AddItemToArray(enc, e);
    }

    const settings_t *cfg = settings_get();
    cJSON *cal = cJSON_AddObjectToObject(root, "calib");
    cJSON_AddBoolToObject(cal, "angles", st->cal_angles);
    cJSON_AddBoolToObject(cal, "cart", st->cal_cart);
    if (cfg->have_angle_offsets) {
        cJSON_AddNumberToObject(cal, "off1", round(cfg->angle_off1 * 1e5) / 1e5);
        cJSON_AddNumberToObject(cal, "off2", round(cfg->angle_off2 * 1e5) / 1e5);
        cJSON_AddNumberToObject(cal, "angles_boot", cfg->angles_boot);
        if (cfg->angles_unix != 0) {
            cJSON_AddNumberToObject(cal, "angles_unix", cfg->angles_unix);
        }
    }
    if (cfg->have_cart_zero) {
        cJSON_AddNumberToObject(cal, "x_zero", round(cfg->x_zero * 1e5) / 1e5);
        cJSON_AddNumberToObject(cal, "cart_boot", cfg->cart_boot);
        if (cfg->cart_unix != 0) {
            cJSON_AddNumberToObject(cal, "cart_unix", cfg->cart_unix);
        }
    }
}

static void build_status(const ctrl_status_t *st, const model_strs_t *ms) {
    cJSON *root = cJSON_CreateObject();
    if (root == NULL) {
        return;
    }
    cJSON_AddStringToObject(root, "run", ctrl_run_str(st->run));
    cJSON_AddStringToObject(root, "fault", st->run == NCP_RUN_FAULT ? ctrl_fault_str(st->fault) : "");
    cJSON_AddStringToObject(root, "mode", st->mode == NCP_MODE_MANUAL ? "manual" : "ai");
    cJSON_AddStringToObject(root, "goal", ms->goal);
    cJSON_AddBoolToObject(root, "random", st->random);
    cJSON_AddNumberToObject(root, "x_ref", round(st->x_ref * 1e4) / 1e4);
    cJSON_AddBoolToObject(root, "settled", st->settled);
    cJSON_AddStringToObject(root, "backend", HAL_BACKEND_NAME);
    if (ms->have) {
        cJSON_AddStringToObject(root, "model", ms->label);
    } else {
        cJSON_AddNullToObject(root, "model");
    }

    cJSON *loop = cJSON_AddObjectToObject(root, "loop");
    cJSON_AddNumberToObject(loop, "period_us", st->period_us);
    cJSON_AddNumberToObject(loop, "exec_avg_us", st->exec_avg_us);
    cJSON_AddNumberToObject(loop, "exec_max_us", st->exec_max_us);
    cJSON_AddNumberToObject(loop, "policy_avg_us", st->policy_avg_us);
    cJSON_AddNumberToObject(loop, "est_avg_us", st->est_avg_us);
    cJSON_AddNumberToObject(loop, "misses", st->misses);
    cJSON_AddNumberToObject(loop, "rt_factor", round(st->rt_factor * 1000.0) / 1000.0);
    cJSON_AddBoolToObject(loop, "budget_ok", st->budget_ok);

    record_status_json(root, st->dt);
    cJSON *exc = cJSON_AddObjectToObject(root, "excite");
    cJSON_AddStringToObject(exc, "kind", record_excite_kind_str(st->excite_kind));
    cJSON_AddBoolToObject(exc, "active", st->excite_active);
    cJSON_AddNumberToObject(exc, "t", round(st->excite_t * 100.0) / 100.0);

    cJSON *heap = cJSON_AddObjectToObject(root, "heap");
    cJSON_AddNumberToObject(heap, "internal_free", (double)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
    cJSON_AddNumberToObject(heap, "internal_min",
                            (double)heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL));
    cJSON_AddNumberToObject(heap, "psram_free", (double)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    heap_watch_json(heap);

    net_info_t ni;
    net_get_info(&ni);
    cJSON *wifi = cJSON_AddObjectToObject(root, "wifi");
    cJSON_AddStringToObject(wifi, "mode", ni.ap ? "ap" : "sta");
    cJSON_AddStringToObject(wifi, "ssid", ni.ssid);
    cJSON_AddStringToObject(wifi, "ip", ni.ip);
    cJSON_AddNumberToObject(wifi, "rssi", ni.rssi);
    static const char *const join_names[] = {"idle", "connecting", "ok", "failed"};
    cJSON *join = cJSON_AddObjectToObject(wifi, "join");
    cJSON_AddStringToObject(join, "state", join_names[ni.join]);
    cJSON_AddStringToObject(join, "ssid", ni.join_ssid);
    cJSON_AddStringToObject(join, "ip", ni.join_ip);
    cJSON_AddStringToObject(join, "error", ni.join_error);

    cJSON_AddNumberToObject(root, "uptime_s", (double)(esp_timer_get_time() / 1000000));
    cJSON_AddNumberToObject(root, "boot", settings_get()->boot_count);
    cJSON_AddNumberToObject(root, "events_seq", events_seq());
    if (events_unix_now() != 0) {
        cJSON_AddNumberToObject(root, "unix", events_unix_now()); /* the clock is set (SNTP) */
    }

    cJSON *faults = cJSON_AddObjectToObject(root, "faults");
    cJSON_AddNumberToObject(faults, "count", st->fault_count);
    cJSON_AddStringToObject(faults, "last", ctrl_fault_str(st->last_fault));
    if (st->fault_count > 0) {
        cJSON_AddNumberToObject(faults, "ago_s",
                                round((double)(st->ticks - st->last_fault_tick) * st->dt * 10.0) / 10.0);
    }

    add_io(root, st);
    add_tasks(root);

    xSemaphoreTake(s_status_lock, portMAX_DELAY);
    if (cJSON_PrintPreallocated(root, s_status, STATUS_SIZE, false)) {
        s_status_len = strlen(s_status);
    } else {
        s_status_len = 0;
    }
    xSemaphoreGive(s_status_lock);
    cJSON_Delete(root);
}

size_t tlm_status_json(char *buf, size_t size) {
    if (s_status_lock == NULL || size == 0) {
        return 0;
    }
    xSemaphoreTake(s_status_lock, portMAX_DELAY);
    size_t n = s_status_len < size - 1 ? s_status_len : size - 1;
    memcpy(buf, s_status, n);
    buf[n] = '\0';
    xSemaphoreGive(s_status_lock);
    return n;
}

/* ---------- logging and persistence on behalf of core 1 ---------- */

/* Every 0.5 s: turn changes in the control task's status into events (the event log
 * also logs them). Faults are counted by the task, so one between two looks still shows. */
static void watch(const ctrl_status_t *st, const model_strs_t *ms) {
    static ctrl_status_t prev;
    static bool have;
    static uint32_t miss_acc;
    static int64_t miss_at;
    static bool budget_state = true, budget_logged = true;
    static int64_t budget_since;
    if (!have) {
        prev = *st;
        have = true;
        return;
    }
    if (st->fault_count > prev.fault_count) {
        uint32_t n = st->fault_count - prev.fault_count;
        events_add(EV_ERROR, "FAULT: %s%s", ctrl_fault_str(st->last_fault),
                   n > 1 ? " (and earlier ones since the last look)" : "");
    }
    if (st->run != prev.run) {
        if (st->run == NCP_RUN_ARMED) {
            events_add(EV_INFO, "armed: %s%s%s", st->mode == NCP_MODE_MANUAL ? "manual" : "AI",
                       st->mode == NCP_MODE_MANUAL ? "" : " -> ", st->mode == NCP_MODE_MANUAL ? "" : ms->goal);
        } else if (st->run == NCP_RUN_IDLE && (prev.run == NCP_RUN_ARMED || prev.run == NCP_RUN_FAULT)) {
            events_add(EV_INFO, prev.run == NCP_RUN_FAULT ? "fault cleared (disarmed)" : "disarmed");
        }
    }
    if (st->mode != prev.mode && st->model != NULL) {
        events_add(EV_INFO, "mode: %s", st->mode == NCP_MODE_MANUAL ? "manual" : "AI");
    }
    if (st->model != prev.model) {
        events_add(EV_INFO, "model: %s", ms->have ? ms->label : "none");
    }
    /* Budget: logged only once a new state has held for 10 s. A loop averaging right at the
     * 80 % line flips budget_ok every second or two, which would flood the 40-entry log. */
    int64_t t_now = esp_timer_get_time();
    if (st->model == NULL) {
        budget_state = budget_logged = true;
    } else if (st->budget_ok != budget_state) {
        budget_state = st->budget_ok;
        budget_since = t_now;
    } else if (budget_state != budget_logged && t_now - budget_since >= 10 * 1000000LL) {
        budget_logged = budget_state;
        events_add(budget_state ? EV_INFO : EV_WARN,
                   "loop budget %s for 10 s: exec avg %lu us, max %lu us, period %lu us",
                   budget_state ? "ok" : "exceeded", (unsigned long)st->exec_avg_us,
                   (unsigned long)st->exec_max_us, (unsigned long)st->nominal_period_us);
    }
    if (st->period_us != prev.period_us && HAL_IS_SIM) {
        ESP_LOGI(TAG, "sim period %lu us (real-time factor %.2f)", (unsigned long)st->period_us,
                 (double)st->rt_factor);
    }
    /* Deadline misses while armed: summed, at most one event per 10 s. */
    if (st->misses > prev.misses && st->run == NCP_RUN_ARMED) {
        miss_acc += st->misses - prev.misses;
    }
    int64_t now = esp_timer_get_time();
    if (miss_acc > 0 && now - miss_at > 10 * 1000000LL) {
        events_add(EV_WARN, "%lu deadline miss(es) while armed", (unsigned long)miss_acc);
        miss_acc = 0;
        miss_at = now;
    }
    prev = *st;

    double a, b;
    if (control_take_angle_offsets(&a, &b)) {
        esp_err_t err = settings_save_angle_offsets(a, b);
        events_add(err == ESP_OK ? EV_INFO : EV_ERROR, "angles zeroed: off1 %.4f, off2 %.4f rad%s%s", a,
                   b, err == ESP_OK ? "" : ", not saved: ", err == ESP_OK ? "" : esp_err_to_name(err));
    }
    if (control_take_cart_zero(&a)) {
        esp_err_t err = settings_save_cart_zero(a);
        events_add(err == ESP_OK ? EV_INFO : EV_ERROR, "cart zeroed: x_zero %.4f m%s%s", a,
                   err == ESP_OK ? "" : ", not saved: ", err == ESP_OK ? "" : esp_err_to_name(err));
    }
}

/* ---------- the task ---------- */

static void tlm_task(void *arg) {
    (void)arg;
    const size_t msg_size = sizeof(ncp_tlm_msg_hdr_t) + FRAMES_PER_MSG * sizeof(ncp_tlm_frame_t);
    /* PSRAM: only this task touches them, a few times a second (internal RAM is for the weights). */
    uint8_t *msg = heap_caps_malloc_prefer(msg_size, 2, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT,
                                           MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    char *text = heap_caps_malloc_prefer(STATUS_SIZE, 2, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT,
                                         MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    configASSERT(msg != NULL && text != NULL);
    ncp_tlm_frame_t *frames = (ncp_tlm_frame_t *)(msg + sizeof(ncp_tlm_msg_hdr_t));

    TickType_t wake = xTaskGetTickCount();
    int64_t next_status = 0;
    for (;;) {
        /* Up to 4 messages per wake-up so a backlog (after an upload) clears quickly. */
        for (int k = 0; k < 4; k++) {
            size_t n = tlm_pop(frames, FRAMES_PER_MSG);
            if (n == 0) {
                break;
            }
            if (web_ws_clients() > 0) {
                ncp_tlm_msg_hdr_t hdr = {
                    .version = NCP_TLM_VERSION,
                    .count = (uint8_t)n,
                    .frame_size = (uint16_t)sizeof(ncp_tlm_frame_t),
                };
                memcpy(msg, &hdr, sizeof(hdr));
                web_ws_broadcast(HTTPD_WS_TYPE_BINARY, msg,
                                 sizeof(hdr) + n * sizeof(ncp_tlm_frame_t));
            }
        }

        int64_t now = esp_timer_get_time();
        led_update(now);
        if (now >= next_status) {
            next_status = now + STATUS_PERIOD_MS * 1000LL;
            ctrl_status_t st;
            control_get_status(&st);
            model_strs_t ms;
            model_strings(&st, &ms);
            watch(&st, &ms);
            build_status(&st, &ms);
            heap_watch_poll();
            if (web_ws_clients() > 0) {
                size_t len = tlm_status_json(text, STATUS_SIZE);
                if (len > 0) {
                    web_ws_broadcast(HTTPD_WS_TYPE_TEXT, text, len);
                }
            }
        }
        vTaskDelayUntil(&wake, pdMS_TO_TICKS(SEND_PERIOD_MS));
    }
}

esp_err_t tlm_start(void) {
    s_status = heap_caps_calloc_prefer(1, STATUS_SIZE, 2, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT,
                                       MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    s_status_lock = xSemaphoreCreateMutex();
    if (s_status == NULL || s_status_lock == NULL) {
        return ESP_ERR_NO_MEM;
    }
    BaseType_t ok = xTaskCreatePinnedToCore(tlm_task, "tlm", 6144, NULL, TLM_PRIO, NULL, TLM_CORE);
    return ok == pdPASS ? ESP_OK : ESP_ERR_NO_MEM;
}
