/* DWA implementation: see dwa.h for the contract.
 *
 * Trajectories are evaluated by forward-integrating (vx, omega) with a
 * piecewise-constant assumption over predict_steps sub-steps. With omega = 0
 * this is a straight line; with omega != 0 it is a circular arc with radius
 * v/omega (paper Eq. (12) / Sec. 3.2).
 *
 * dist(v,w) is the arc length traversed before any obstacle line gets closer
 * than robot_radius_m. If the arc reaches the horizon without contact, dist is
 * dist_clip_m (paper's large-constant fallback).
 *
 * heading(v,w) uses the predicted pose at the end of predict_time (a simpler
 * stand-in for the paper's "pose after maximal deceleration"; in practice for
 * a one-step-look-ahead it gives equivalent ranking and avoids re-introducing
 * the brake model into the heading term).
 *
 * Output is the raw weighted sum without spatial smoothing (paper's sigma);
 * the smoothing kernel mainly improves side clearance for trajectory plots
 * and does not change the argmax in a dense grid.
 */

#include "dwa.h"

#include <math.h>
#include <stddef.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


static inline double wrap_pi(double a) {
    while (a > M_PI)  a -= 2.0 * M_PI;
    while (a < -M_PI) a += 2.0 * M_PI;
    return a;
}

static inline double clamp(double x, double lo, double hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

/* Squared distance from a point (px, py) to the line segment (ax, ay)-(bx, by). */
static double point_seg_dist2(double px, double py,
                              double ax, double ay,
                              double bx, double by)
{
    double dx = bx - ax;
    double dy = by - ay;
    double len2 = dx * dx + dy * dy;
    double t;
    if (len2 <= 1e-12) {
        /* Degenerate to point distance. */
        double ex = px - ax, ey = py - ay;
        return ex * ex + ey * ey;
    }
    t = ((px - ax) * dx + (py - ay) * dy) / len2;
    if (t < 0.0) t = 0.0;
    else if (t > 1.0) t = 1.0;
    double qx = ax + t * dx;
    double qy = ay + t * dy;
    double ex = px - qx;
    double ey = py - qy;
    return ex * ex + ey * ey;
}


/* Build obstacle line field from rays. Each ray endpoint gets a short segment
 * perpendicular to the ray direction, with length = dist * angular_step (the
 * tangential width of one ray "tile" at that distance). Lines are stored as
 * (xa, ya, xb, yb) packed in `lines`.
 *
 * Returns the number of valid lines (skips rays that hit the patch boundary,
 * i.e. dist >= clip_m, since those represent free space rather than obstacles).
 */
static int build_line_field(const double *ray_dists,
                            const double *ray_angles,
                            int n_rays,
                            double dist_clip_m,
                            double *lines)
{
    int k = 0;
    double dtheta;
    if (ray_angles != NULL) {
        dtheta = 0.0; /* will be computed per-pair; here we keep length = patch_resolution */
        /* When custom angles are supplied we can still pick a reasonable tile
         * width from the average spacing. */
        dtheta = (n_rays > 0) ? (2.0 * M_PI / (double)n_rays) : 0.0;
    } else {
        dtheta = (n_rays > 0) ? (2.0 * M_PI / (double)n_rays) : 0.0;
    }
    for (int i = 0; i < n_rays; i++) {
        double d = ray_dists[i];
        if (!(d > 0.0)) continue;
        if (d >= dist_clip_m) continue; /* no obstacle within view: skip */
        double a = (ray_angles != NULL) ? ray_angles[i]
                                         : (double)i * dtheta;
        double cx = d * cos(a);
        double cy = d * sin(a);
        double tx = -sin(a);
        double ty = cos(a);
        double half_len = 0.5 * d * dtheta;
        lines[4 * k + 0] = cx - half_len * tx;
        lines[4 * k + 1] = cy - half_len * ty;
        lines[4 * k + 2] = cx + half_len * tx;
        lines[4 * k + 3] = cy + half_len * ty;
        k++;
    }
    return k;
}


/* Simulate one (v, w) candidate forward and report:
 *   *out_dist     = arc length traversed before robot_radius collision (or full horizon)
 *   *out_end_x,y,th = pose at end of predict_time (regardless of collision)
 * Returns 1 if the trajectory ever collided (i.e. dist < dist_clip), else 0.
 */
static int simulate_trajectory(const DWAConfig *cfg,
                               double v, double w,
                               const double *lines, int n_lines,
                               double *out_dist,
                               double *out_end_x,
                               double *out_end_y,
                               double *out_end_th)
{
    int steps = (cfg->predict_steps > 0) ? cfg->predict_steps : 20;
    double dt_sim = cfg->predict_time / (double)steps;
    double x = 0.0, y = 0.0, th = 0.0;
    double arc = 0.0;
    int collided = 0;
    double r2 = cfg->robot_radius_m * cfg->robot_radius_m;
    double traveled_total = 0.0;

    for (int s = 0; s < steps; s++) {
        /* Piecewise-constant velocity over [t, t+dt_sim]: yaw advances
         * linearly, position advances along the resulting arc. Discretize
         * with midpoint heading for second-order accuracy. */
        double th_mid = th + 0.5 * w * dt_sim;
        double dx = v * cos(th_mid) * dt_sim;
        double dy = v * sin(th_mid) * dt_sim;
        x += dx;
        y += dy;
        th += w * dt_sim;
        traveled_total += fabs(v) * dt_sim;

        if (!collided) {
            double min_d2 = 1e30;
            for (int j = 0; j < n_lines; j++) {
                double d2 = point_seg_dist2(x, y,
                                            lines[4 * j + 0], lines[4 * j + 1],
                                            lines[4 * j + 2], lines[4 * j + 3]);
                if (d2 < min_d2) min_d2 = d2;
                if (d2 <= r2) {
                    /* Collision at this sub-step; freeze the dist measurement. */
                    arc = traveled_total;
                    collided = 1;
                    break;
                }
            }
            (void)min_d2;
        }
    }
    if (!collided) arc = cfg->dist_clip_m;

    *out_dist = arc;
    *out_end_x = x;
    *out_end_y = y;
    *out_end_th = th;
    return collided;
}


DWA_API int dwa_plan(const DWAConfig *cfg,
                     double vx_cur, double omega_cur,
                     double target_x_local, double target_y_local,
                     const double *ray_dists,
                     const double *ray_angles,
                     int n_rays,
                     double *scratch,
                     int scratch_n,
                     DWAOutput *out)
{
    if (cfg == NULL || ray_dists == NULL || scratch == NULL || out == NULL) return -1;
    if (n_rays < 0 || cfg->v_samples < 2 || cfg->omega_samples < 2) return -1;
    if (scratch_n < 4 * n_rays) return -1;

    double *lines = scratch;
    int n_lines = build_line_field(ray_dists, ray_angles, n_rays,
                                   cfg->dist_clip_m, lines);

    /* V_s ∩ V_d : center-clamped intervals. */
    double v_lo = cfg->v_min;
    double v_hi = cfg->v_max;
    double w_lo = -cfg->omega_max;
    double w_hi =  cfg->omega_max;

    double dv = cfg->v_acc_max * cfg->dt;
    double dw = cfg->omega_acc_max * cfg->dt;
    double vd_lo = vx_cur - dv;     if (vd_lo > v_lo) v_lo = vd_lo;
    double vd_hi = vx_cur + dv;     if (vd_hi < v_hi) v_hi = vd_hi;
    double wd_lo = omega_cur - dw;  if (wd_lo > w_lo) w_lo = wd_lo;
    double wd_hi = omega_cur + dw;  if (wd_hi < w_hi) w_hi = wd_hi;
    if (v_lo > v_hi || w_lo > w_hi) {
        /* Degenerate window: degrade to current values. */
        v_lo = v_hi = clamp(vx_cur, cfg->v_min, cfg->v_max);
        w_lo = w_hi = clamp(omega_cur, -cfg->omega_max, cfg->omega_max);
    }

    int NV = cfg->v_samples;
    int NW = cfg->omega_samples;

    /* First pass: collect raw scores; track max of each component for
     * normalization (paper normalizes each term to [0,1] over the grid). */
    double best_h = -1.0, best_d = -1.0, best_v = -1.0;

    /* Stage 1: pre-compute per-candidate (heading_raw, dist_raw, vel_raw, admissible). */
    /* We store these inline in scratch *after* the line field. Layout:
     *   lines  [0 .. 4*n_lines)
     *   per_cand [4*n_lines .. 4*n_lines + 4*NV*NW)   columns: h, d, v, adm
     */
    int per_offset = 4 * n_lines;
    int per_n = 4 * NV * NW;
    if (scratch_n < per_offset + per_n) return -1;
    double *per = scratch + per_offset;

    for (int iv = 0; iv < NV; iv++) {
        double v = (NV == 1) ? v_lo : v_lo + (v_hi - v_lo) * (double)iv / (double)(NV - 1);
        for (int iw = 0; iw < NW; iw++) {
            double w = (NW == 1) ? w_lo : w_lo + (w_hi - w_lo) * (double)iw / (double)(NW - 1);

            double dist, ex, ey, eth;
            simulate_trajectory(cfg, v, w, lines, n_lines, &dist, &ex, &ey, &eth);

            /* Admissibility (V_a): must be able to brake before the obstacle. */
            int admissible = 1;
            double v_lim = sqrt(2.0 * dist * cfg->v_brake_acc);
            double w_lim = sqrt(2.0 * dist * cfg->omega_brake_acc);
            if (fabs(v) > v_lim || fabs(w) > w_lim) admissible = 0;

            /* heading: alignment of predicted pose with target direction. */
            double tx = target_x_local - ex;
            double ty = target_y_local - ey;
            double target_dir = atan2(ty, tx);
            double diff = wrap_pi(target_dir - eth);
            /* heading_raw in [0, pi]; larger = better aligned. Use pi - |diff|. */
            double heading_raw = M_PI - fabs(diff);
            if (heading_raw < 0.0) heading_raw = 0.0;

            double vel_raw = fabs(v);

            int idx = 4 * (iv * NW + iw);
            per[idx + 0] = heading_raw;
            per[idx + 1] = dist;
            per[idx + 2] = vel_raw;
            per[idx + 3] = admissible ? 1.0 : 0.0;

            if (admissible) {
                if (heading_raw > best_h) best_h = heading_raw;
                if (dist > best_d) best_d = dist;
                if (vel_raw > best_v) best_v = vel_raw;
            }
        }
    }

    /* Floors to keep division stable. */
    if (best_h <= 0.0) best_h = 1.0;
    if (best_d <= 0.0) best_d = 1.0;
    if (best_v <= 0.0) best_v = 1.0;

    /* Second pass: weighted normalized sum, argmax over admissible. */
    double best_score = -1.0;
    int best_iv = -1, best_iw = -1;

    for (int iv = 0; iv < NV; iv++) {
        for (int iw = 0; iw < NW; iw++) {
            int idx = 4 * (iv * NW + iw);
            if (per[idx + 3] < 0.5) continue; /* not admissible */
            double h = per[idx + 0] / best_h;
            double d = per[idx + 1] / best_d;
            double vv = per[idx + 2] / best_v;
            double s = cfg->alpha_heading * h
                     + cfg->beta_clearance * d
                     + cfg->gamma_velocity * vv;
            if (s > best_score) {
                best_score = s;
                best_iv = iv;
                best_iw = iw;
            }
        }
    }

    if (best_iv < 0) {
        /* V_r is empty: rotate-away fallback. Turn toward target side. */
        if (cfg->rotate_away_mode) {
            double target_dir = atan2(target_y_local, target_x_local);
            double w_sign = (target_dir >= 0.0) ? 1.0 : -1.0;
            out->vx_cmd = 0.0;
            out->omega_cmd = w_sign * cfg->omega_max;
            out->score = 0.0;
            out->found = 0;
        } else {
            out->vx_cmd = 0.0;
            out->omega_cmd = 0.0;
            out->score = 0.0;
            out->found = 0;
        }
        return 0;
    }

    double v_chosen = (NV == 1) ? v_lo
                    : v_lo + (v_hi - v_lo) * (double)best_iv / (double)(NV - 1);
    double w_chosen = (NW == 1) ? w_lo
                    : w_lo + (w_hi - w_lo) * (double)best_iw / (double)(NW - 1);

    out->vx_cmd = v_chosen;
    out->omega_cmd = w_chosen;
    out->score = best_score;
    out->found = 1;
    return 0;
}
