/* dwa_cpu.c -- serial CPU port of eval/dwa/planner.py (closed-form DWA).
 *
 * Purpose: measure the wall-clock cost of one DWA planning step as a function
 * of the candidate-grid density NC = v_samples * omega_samples ("scan count"),
 * scanning the NC candidates SERIALLY (the torch planner does them as one
 * vectorized batch). The numerical result is meant to match the torch planner
 * closely (double precision here vs float32 there; tiny ties may diverge).
 *
 * Inputs:
 *   - a binary snapshot file (header + per-step DWA inputs); see README
 *     ("snapshots.bin" fixture). The header carries the env-derived geometry
 *     (dt, v_min, v_max, omega_max, v_acc_max, omega_acc_max, dist_clip_m, N).
 *   - eval/dwa_config.json for the pure DWA knobs (predict_time, v_samples,
 *     omega_samples, alpha/beta/gamma, robot_radius_m, brakes, rotate_away).
 *     This mirrors DWAConfig.from_configs: geometry from the env, knobs from
 *     dwa_config.json. --grid NVxNW overrides v_samples/omega_samples.
 *
 * Output:
 *   - prints a "RESULT ..." line with NC and timing (total + per-plan).
 *   - with --out PATH, writes the chosen (v, omega) per snapshot as a flat
 *     little-endian float64 array [n_snapshots, 2] for offline verification.
 *
 * Build: see build.bat (MSVC). Single translation unit, C99, libm only.
 */

#define _CRT_SECURE_NO_WARNINGS  /* fopen/strncpy/strtod are fine here */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#ifdef _WIN32
#include <windows.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---------------------------------------------------------------------------
 * Minimal flat-JSON reader for "key": number / true / false. Good enough for
 * the controlled eval/dwa_config.json (a flat object of scalars).
 * ------------------------------------------------------------------------- */

static const char *json_find(const char *s, const char *key) {
    char pat[128];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(s, pat);
    if (!p) return NULL;
    p += strlen(pat);
    while (*p && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n' || *p == ':'))
        p++;
    return p;
}

static double json_num(const char *s, const char *key, double dflt, int *found) {
    const char *p = json_find(s, key);
    if (!p) { if (found) *found = 0; return dflt; }
    char *end = NULL;
    double v = strtod(p, &end);
    if (end == p) { if (found) *found = 0; return dflt; }
    if (found) *found = 1;
    return v;
}

static int json_bool(const char *s, const char *key, int dflt) {
    const char *p = json_find(s, key);
    if (!p) return dflt;
    if (strncmp(p, "true", 4) == 0) return 1;
    if (strncmp(p, "false", 5) == 0) return 0;
    return strtod(p, NULL) != 0.0;
}

static char *read_text_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 0) { fclose(f); return NULL; }
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t got = fread(buf, 1, (size_t)n, f);
    buf[got] = '\0';
    fclose(f);
    return buf;
}

/* ---------------------------------------------------------------------------
 * Resolved config (geometry from snapshot header, knobs from dwa_config.json).
 * ------------------------------------------------------------------------- */

typedef struct {
    /* env-derived geometry (from snapshot header) */
    double dt, v_min, v_max, omega_max, v_acc_max, omega_acc_max, dist_clip_m;
    /* pure DWA knobs (from dwa_config.json) */
    double predict_time, robot_radius_m, alpha, beta, gamma, v_brake, w_brake;
    int v_samples, omega_samples, rotate_away;
} Cfg;

/* ---------------------------------------------------------------------------
 * One serial DWA plan for a single env. Mirrors DWAPlanner.plan_batch for one
 * (env). Scratch buffers (ox/oy [3N], dist/head/vel/adm [NC]) are caller-owned
 * and reused across snapshots so this allocates nothing.
 * ------------------------------------------------------------------------- */

static void plan_one(const Cfg *c, int N,
                     const double *cosA, const double *sinA, double half,
                     int NV, int NW, const double *vcand, const double *wcand,
                     const double *rays, double tx, double ty, double vx, double omega,
                     double *ox, double *oy,
                     double *dist_arr, double *head_arr, double *vel_arr, char *adm_arr,
                     double *out_v, double *out_w) {
    const int NC = NV * NW;
    const double dist_clip = c->dist_clip_m;
    const double far_pt = 10.0 * dist_clip; /* phantom location for invalid rays */
    const double BIG = 10.0 * dist_clip;    /* "no collision in horizon" arc */
    const double T = c->predict_time;
    const double rr = c->robot_radius_m;
    const double rr2 = rr * rr;
    const double eps_w = 1e-6;
    const double twopi = 2.0 * M_PI;

    /* 1) Obstacle LINE FIELD: each ray -> centre + 2 perpendicular endpoints
     *    (3N points). Invalid rays (<=0 or >=dist_clip) move to "far". */
    for (int i = 0; i < N; i++) {
        double r = rays[i];
        int invalid = (r <= 0.0) || (r >= dist_clip);
        if (invalid) {
            ox[i] = far_pt;          oy[i] = far_pt;
            ox[N + i] = far_pt;      oy[N + i] = far_pt;
            ox[2 * N + i] = far_pt;  oy[2 * N + i] = far_pt;
        } else {
            double ca = cosA[i], sa = sinA[i];
            double cx0 = r * ca, cy0 = r * sa;
            double off = r * half;
            ox[i] = cx0;             oy[i] = cy0;
            ox[N + i] = cx0 - off * sa;  oy[N + i] = cy0 + off * ca;
            ox[2 * N + i] = cx0 + off * sa; oy[2 * N + i] = cy0 - off * ca;
        }
    }
    const int M = 3 * N;

    /* 2) Dynamic window V_d. */
    double dv = c->v_acc_max * c->dt;
    double dw = c->omega_acc_max * c->dt;
    double vd_lo = vx - dv; if (vd_lo < c->v_min) vd_lo = c->v_min;
    double vd_hi = vx + dv; if (vd_hi > c->v_max) vd_hi = c->v_max;
    double wd_lo = omega - dw; if (wd_lo < -c->omega_max) wd_lo = -c->omega_max;
    double wd_hi = omega + dw; if (wd_hi > c->omega_max) wd_hi = c->omega_max;

    double best_h = -1.0, best_d = -1.0, best_v = -1.0;
    int any_adm = 0;

    /* 3+4) Serial scan over the NC candidates (k over v, j over omega). The
     * flat index k*NW+j matches the torch candidate order so argmax ties break
     * identically (first occurrence). */
    for (int k = 0; k < NV; k++) {
        for (int j = 0; j < NW; j++) {
            int idx = k * NW + j;
            double v = vcand[idx];
            double w = wcand[idx];

            int in_vd = (v >= vd_lo) && (v <= vd_hi);
            int in_wd = (w >= wd_lo) && (w <= wd_hi);
            int in_dyn = in_vd && in_wd;

            /* arc-to-collision: min over the 3N obstacle points. */
            double arc_first = BIG;
            if (fabs(w) < eps_w) {
                /* straight-line branch */
                for (int m = 0; m < M; m++) {
                    double oxx = ox[m], oyy = oy[m];
                    double delta = rr2 - oyy * oyy;
                    int same_sign = (v * oxx) > 0.0;
                    if (delta > 0.0 && same_sign) {
                        double arc = fabs(oxx) - sqrt(delta);
                        if (arc < 0.0) arc = 0.0;
                        if (arc < arc_first) arc_first = arc;
                    }
                }
            } else {
                /* circular-arc branch */
                double r_signed = v / w;
                double R = fabs(r_signed);
                double sign_w = (w > 0.0) ? 1.0 : -1.0;
                double theta_0 = atan2(-r_signed, 0.0);
                double swept_total = fabs(w) * T;
                for (int m = 0; m < M; m++) {
                    double oxx = ox[m], oyy = oy[m];
                    double dy = oyy - r_signed;
                    double dCP2 = oxx * oxx + dy * dy;
                    double dCP = sqrt(dCP2);
                    double min_dist = fabs(dCP - R);
                    if (min_dist < rr) {
                        double denom = 2.0 * R * dCP;
                        if (denom < 1e-12) denom = 1e-12;
                        double cos_half = (R * R + dCP2 - rr2) / denom;
                        if (cos_half > 1.0) cos_half = 1.0;
                        else if (cos_half < -1.0) cos_half = -1.0;
                        double half_angle = acos(cos_half);
                        double theta_Q = atan2(dy, oxx);
                        double a = (theta_Q - theta_0) * sign_w;
                        double swept_to_Q = a - floor(a / twopi) * twopi;  /* [0,2pi) */
                        double swept_first = swept_to_Q - half_angle;
                        if (swept_first >= 0.0 && swept_first <= swept_total) {
                            double arc = R * swept_first;
                            if (arc < arc_first) arc_first = arc;
                        }
                    }
                }
            }

            double max_arc_traj = fabs(v) * T;
            double dist;
            if (arc_first >= max_arc_traj) dist = dist_clip;          /* no hit in horizon */
            else dist = arc_first;                                    /* = min(arc_first, max_arc_traj) */

            /* end-of-horizon pose (robot frame) for the heading term */
            double wT = w * T;
            double x_end, y_end;
            if (fabs(w) >= eps_w) {
                double r2 = v / w;
                x_end = r2 * sin(wT);
                y_end = r2 * (1.0 - cos(wT));
            } else {
                x_end = v * T;
                y_end = 0.0;
            }
            double th_end = wT;
            double target_dir = atan2(ty - y_end, tx - x_end);
            double diff = target_dir - th_end;
            double dwrap = diff + M_PI;
            dwrap = dwrap - floor(dwrap / twopi) * twopi;  /* [0,2pi) */
            diff = dwrap - M_PI;                           /* [-pi,pi) */
            double heading_raw = M_PI - fabs(diff);
            double vel_raw = fabs(v);

            /* admissibility V_a (paper brake-distance test) AND V_d */
            double v_lim = sqrt(2.0 * dist * c->v_brake);
            double w_lim = sqrt(2.0 * dist * c->w_brake);
            int in_va = (fabs(v) <= v_lim) && (fabs(w) <= w_lim);
            int adm = in_va && in_dyn;

            dist_arr[idx] = dist;
            head_arr[idx] = heading_raw;
            vel_arr[idx] = vel_raw;
            adm_arr[idx] = (char)adm;
            if (adm) {
                any_adm = 1;
                if (heading_raw > best_h) best_h = heading_raw;
                if (dist > best_d) best_d = dist;
                if (vel_raw > best_v) best_v = vel_raw;
            }
        }
    }

    /* per-env normalization floors (clamp_min 1.0); -1 stays if none admissible */
    if (best_h < 1.0) best_h = 1.0;
    if (best_d < 1.0) best_d = 1.0;
    if (best_v < 1.0) best_v = 1.0;

    /* 5) argmax (strict > => first max wins, like torch argmax) */
    double best_score = -1e300;
    int best_idx = 0;
    for (int idx = 0; idx < NC; idx++) {
        double score;
        if (adm_arr[idx]) {
            double h_n = head_arr[idx] / best_h;
            double d_n = dist_arr[idx] / best_d;
            double v_n = vel_arr[idx] / best_v;
            score = c->alpha * h_n + c->beta * d_n + c->gamma * v_n;
        } else {
            score = -1e10;
        }
        if (score > best_score) { best_score = score; best_idx = idx; }
    }

    double cv = vcand[best_idx];
    double cw = wcand[best_idx];
    if (!any_adm) {
        if (c->rotate_away) {
            double target_angle = atan2(ty, tx);
            double wsign = (target_angle >= 0.0) ? 1.0 : -1.0;
            cv = 0.0;
            cw = wsign * c->omega_max;
        } else {
            cv = 0.0;
            cw = 0.0;
        }
    }
    *out_v = cv;
    *out_w = cw;
}

/* ---------------------------------------------------------------------------
 * Snapshot file I/O.  Layout (little-endian, packed):
 *   char[4]  "DWA1"
 *   int32    N             (rays per snapshot)
 *   int32    n_snapshots
 *   int32    reserved0
 *   int32    reserved1
 *   double   dt, v_min, v_max, omega_max, v_acc_max, omega_acc_max, dist_clip_m
 *   then n_snapshots records, each (4 + N) doubles: tx, ty, vx, omega, rays[N]
 * ------------------------------------------------------------------------- */

typedef struct {
    int N, n;
    double dt, v_min, v_max, omega_max, v_acc_max, omega_acc_max, dist_clip_m;
    double *recs;       /* [n * (4 + N)] */
} Snapshots;

static int load_snapshots(const char *path, Snapshots *s) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open snapshot file: %s\n", path); return 1; }
    char magic[4];
    int32_t N = 0, n = 0, r0 = 0, r1 = 0;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "DWA1", 4) != 0) {
        fprintf(stderr, "bad snapshot magic\n"); fclose(f); return 1;
    }
    fread(&N, sizeof(int32_t), 1, f);
    fread(&n, sizeof(int32_t), 1, f);
    fread(&r0, sizeof(int32_t), 1, f);
    fread(&r1, sizeof(int32_t), 1, f);
    double geo[7];
    if (fread(geo, sizeof(double), 7, f) != 7) {
        fprintf(stderr, "truncated snapshot header\n"); fclose(f); return 1;
    }
    s->N = (int)N; s->n = (int)n;
    s->dt = geo[0]; s->v_min = geo[1]; s->v_max = geo[2]; s->omega_max = geo[3];
    s->v_acc_max = geo[4]; s->omega_acc_max = geo[5]; s->dist_clip_m = geo[6];

    size_t per = (size_t)(4 + s->N);
    size_t total = (size_t)s->n * per;
    s->recs = (double *)malloc(total * sizeof(double));
    if (!s->recs) { fprintf(stderr, "oom snapshots\n"); fclose(f); return 1; }
    size_t got = fread(s->recs, sizeof(double), total, f);
    fclose(f);
    if (got != total) {
        fprintf(stderr, "truncated snapshot records: got %zu want %zu\n", got, total);
        return 1;
    }
    return 0;
}

/* ---------------------------------------------------------------------------
 * High-resolution timer.
 * ------------------------------------------------------------------------- */

#ifdef _WIN32
static double now_sec(void) {
    static LARGE_INTEGER freq;
    static int init = 0;
    if (!init) { QueryPerformanceFrequency(&freq); init = 1; }
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)freq.QuadPart;
}
#else
#include <time.h>
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
#endif

/* ---------------------------------------------------------------------------
 * One full serial pass over all snapshots for a fixed grid. Returns a checksum
 * (to keep the optimizer honest) and stores the chosen actions into out[2n].
 * ------------------------------------------------------------------------- */

static double run_pass(const Cfg *c, const Snapshots *S, int N, int per,
                       const double *cosA, const double *sinA, double half,
                       int NV, int NW, const double *vcand, const double *wcand,
                       double *ox, double *oy, double *dist_arr, double *head_arr,
                       double *vel_arr, char *adm_arr, double *out) {
    double acc = 0.0;
    for (int s = 0; s < S->n; s++) {
        const double *rec = S->recs + (size_t)s * per;
        double ov, ow;
        plan_one(c, N, cosA, sinA, half, NV, NW, vcand, wcand,
                 rec + 4, rec[0], rec[1], rec[2], rec[3],
                 ox, oy, dist_arr, head_arr, vel_arr, adm_arr, &ov, &ow);
        out[2 * s] = ov;
        out[2 * s + 1] = ow;
        acc += ov + ow;
    }
    return acc;
}

/* Build the candidate grid (NV,NW), run `warmup` untimed passes then `repeat`
 * timed passes over all snapshots, and report the MIN per-plan time (min is the
 * most stable estimator under OS/CPU jitter). With out_path != NULL, writes the
 * chosen actions [n,2] float64. Prints one RESULT line. Returns 0 on success. */
static int eval_grid(const Cfg *base, const Snapshots *S,
                     const double *cosA, const double *sinA, double half,
                     int NV, int NW, int repeat, int warmup,
                     const char *out_path) {
    Cfg c = *base;
    c.v_samples = NV; c.omega_samples = NW;
    int NC = NV * NW;
    int N = S->N;
    const int per = 4 + N;

    double *vcand = (double *)malloc((size_t)NC * sizeof(double));
    double *wcand = (double *)malloc((size_t)NC * sizeof(double));
    double *ox = (double *)malloc((size_t)(3 * N) * sizeof(double));
    double *oy = (double *)malloc((size_t)(3 * N) * sizeof(double));
    double *dist_arr = (double *)malloc((size_t)NC * sizeof(double));
    double *head_arr = (double *)malloc((size_t)NC * sizeof(double));
    double *vel_arr  = (double *)malloc((size_t)NC * sizeof(double));
    char   *adm_arr  = (char *)  malloc((size_t)NC * sizeof(char));
    double *out = (double *)malloc((size_t)S->n * 2 * sizeof(double));
    if (!vcand || !wcand || !ox || !oy || !dist_arr || !head_arr || !vel_arr || !adm_arr || !out) {
        fprintf(stderr, "oom (grid %dx%d)\n", NV, NW);
        free(vcand); free(wcand); free(ox); free(oy);
        free(dist_arr); free(head_arr); free(vel_arr); free(adm_arr); free(out);
        return 1;
    }
    for (int k = 0; k < NV; k++) {
        double vg = c.v_min + (double)k * (c.v_max - c.v_min) / (double)(NV - 1);
        for (int j = 0; j < NW; j++) {
            double wg = -c.omega_max + (double)j * (2.0 * c.omega_max) / (double)(NW - 1);
            vcand[k * NW + j] = vg;
            wcand[k * NW + j] = wg;
        }
    }

    volatile double sink = 0.0;
    for (int w = 0; w < warmup; w++)
        sink += run_pass(&c, S, N, per, cosA, sinA, half, NV, NW, vcand, wcand,
                         ox, oy, dist_arr, head_arr, vel_arr, adm_arr, out);

    double best = 1e300;
    for (int rep = 0; rep < repeat; rep++) {
        double t0 = now_sec();
        sink += run_pass(&c, S, N, per, cosA, sinA, half, NV, NW, vcand, wcand,
                         ox, oy, dist_arr, head_arr, vel_arr, adm_arr, out);
        double t1 = now_sec();
        double per_plan = (S->n > 0) ? (t1 - t0) / (double)S->n : 0.0;
        if (per_plan < best) best = per_plan;
    }
    (void)sink;

    if (out_path) {
        FILE *f = fopen(out_path, "wb");
        if (!f) { fprintf(stderr, "cannot write --out %s\n", out_path); }
        else { fwrite(out, sizeof(double), (size_t)S->n * 2, f); fclose(f); }
    }

    printf("RESULT grid=%dx%d NC=%d N=%d n=%d repeat=%d warmup=%d per_plan_us=%.4f\n",
           NV, NW, NC, N, S->n, repeat, warmup, best * 1e6);

    free(vcand); free(wcand); free(ox); free(oy);
    free(dist_arr); free(head_arr); free(vel_arr); free(adm_arr); free(out);
    return 0;
}

/* ------------------------------------------------------------------------- */

int main(int argc, char **argv) {
    const char *snap_path = NULL;
    const char *cfg_path = NULL;
    const char *out_path = NULL;
    const char *sweep = NULL;
    int grid_nv = -1, grid_nw = -1;
    int repeat = 5;
    int warmup = 1;
    int quiet = 0;

    /* positional: <snapshots> <dwa_config.json>; then flags */
    int pos = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--grid") == 0 && i + 1 < argc) {
            char tmp[64];
            strncpy(tmp, argv[++i], sizeof(tmp) - 1);
            tmp[sizeof(tmp) - 1] = '\0';
            char *x = strpbrk(tmp, "xX,");
            if (!x) { fprintf(stderr, "bad --grid (want NVxNW)\n"); return 2; }
            *x = '\0';
            grid_nv = atoi(tmp);
            grid_nw = atoi(x + 1);
        } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            out_path = argv[++i];
        } else if (strcmp(argv[i], "--sweep") == 0 && i + 1 < argc) {
            sweep = argv[++i];
        } else if (strcmp(argv[i], "--repeat") == 0 && i + 1 < argc) {
            repeat = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--quiet") == 0) {
            quiet = 1;
        } else if (pos == 0) {
            snap_path = argv[i]; pos++;
        } else if (pos == 1) {
            cfg_path = argv[i]; pos++;
        }
    }
    if (!snap_path || !cfg_path) {
        fprintf(stderr,
                "usage: %s <snapshots.bin> <dwa_config.json>\n"
                "  [--grid NVxNW]          single grid (also dumps --out)\n"
                "  [--sweep NVxNW,...]     time several grids in one process\n"
                "  [--out actions.bin]     write chosen (v,omega) [n,2] f64\n"
                "  [--repeat R] [--warmup W] [--quiet]\n",
                argv[0]);
        return 2;
    }
    if (repeat < 1) repeat = 1;

    Snapshots S;
    if (load_snapshots(snap_path, &S)) return 1;

    char *cfg_text = read_text_file(cfg_path);
    if (!cfg_text) { fprintf(stderr, "cannot read dwa_config: %s\n", cfg_path); return 1; }

    Cfg c;
    /* geometry from snapshot header */
    c.dt = S.dt; c.v_min = S.v_min; c.v_max = S.v_max; c.omega_max = S.omega_max;
    c.v_acc_max = S.v_acc_max; c.omega_acc_max = S.omega_acc_max; c.dist_clip_m = S.dist_clip_m;
    /* knobs from dwa_config.json */
    c.predict_time   = json_num(cfg_text, "predict_time", 1.0, NULL);
    c.robot_radius_m = json_num(cfg_text, "robot_radius_m", 0.1, NULL);
    c.alpha          = json_num(cfg_text, "alpha_heading", 0.8, NULL);
    c.beta           = json_num(cfg_text, "beta_clearance", 0.1, NULL);
    c.gamma          = json_num(cfg_text, "gamma_velocity", 1.0, NULL);
    c.v_brake        = json_num(cfg_text, "v_brake_acc", 0.5, NULL);
    c.w_brake        = json_num(cfg_text, "omega_brake_acc", 1.0, NULL);
    c.v_samples      = (int)json_num(cfg_text, "v_samples", 21, NULL);
    c.omega_samples  = (int)json_num(cfg_text, "omega_samples", 41, NULL);
    c.rotate_away    = json_bool(cfg_text, "rotate_away_mode", 1);
    free(cfg_text);

    if (grid_nv > 0 && grid_nw > 0) { c.v_samples = grid_nv; c.omega_samples = grid_nw; }
    if (warmup < 0) warmup = 0;
    (void)quiet;
    int N = S.N;

    /* ray angle table (independent of the env: arange(N) * 2pi/N), shared
     * across all grids in a sweep. */
    double *cosA = (double *)malloc((size_t)N * sizeof(double));
    double *sinA = (double *)malloc((size_t)N * sizeof(double));
    if (!cosA || !sinA) { fprintf(stderr, "oom ray table\n"); return 1; }
    double dtheta = 2.0 * M_PI / (double)N;
    double half = 0.5 * dtheta;
    for (int i = 0; i < N; i++) {
        double a = i * dtheta;
        cosA[i] = cos(a);
        sinA[i] = sin(a);
    }

    int rc = 0;
    if (sweep) {
        /* Time several grids in ONE process: the CPU stays in a consistent
         * boost/cache state, so per-plan cost is monotonic and comparable
         * across grids. --out is ignored here -- use single --grid mode to
         * dump actions for the correctness diff. */
        char buf[1024];
        strncpy(buf, sweep, sizeof(buf) - 1);
        buf[sizeof(buf) - 1] = '\0';
        char *tok = strtok(buf, ", ");
        while (tok) {
            char *x = strpbrk(tok, "xX");
            if (x) {
                *x = '\0';
                int nv = atoi(tok), nw = atoi(x + 1);
                if (nv >= 2 && nw >= 2)
                    rc |= eval_grid(&c, &S, cosA, sinA, half, nv, nw, repeat, warmup, NULL);
                else
                    fprintf(stderr, "skip bad grid in --sweep: %sx%s\n", tok, x + 1);
            }
            tok = strtok(NULL, ", ");
        }
    } else {
        int NV = c.v_samples, NW = c.omega_samples;
        if (NV < 2 || NW < 2) {
            fprintf(stderr, "v_samples/omega_samples must be >= 2\n");
            free(cosA); free(sinA); free(S.recs);
            return 2;
        }
        rc = eval_grid(&c, &S, cosA, sinA, half, NV, NW, repeat, warmup, out_path);
    }

    free(cosA); free(sinA); free(S.recs);
    return rc;
}
