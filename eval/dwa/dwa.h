/* DWA local planner — Fox/Burgard/Thrun 1997.
 *
 * Single-step planner: given current state, ray observation, and goal in the
 * robot frame, return the chosen (vx, omega) by maximizing
 *     G(v,w) = alpha * heading(v,w) + beta * dist(v,w) + gamma * velocity(v,w)
 * over the intersection of V_s, V_a, V_d (see paper Sec. 4).
 *
 * Conventions used by this module:
 *   - All distances are in meters, angles in radians.
 *   - Robot frame: +x forward (heading), +y left; yaw is 0 inside the planner
 *     (callers must convert global -> robot frame before invoking dwa_plan).
 *   - Rays are produced uniformly around the robot at angles
 *         ray_angle[i] = i * (2*pi / n_rays)
 *     measured counter-clockwise from +x. dwa.c internally turns each ray
 *     endpoint into a short obstacle line segment perpendicular to the ray.
 */

#ifndef GRALP_EVAL_DWA_H
#define GRALP_EVAL_DWA_H

#if defined(_WIN32) || defined(__CYGWIN__)
#  define DWA_API __declspec(dllexport)
#else
#  if defined(__GNUC__) || defined(__clang__)
#    define DWA_API __attribute__((visibility("default")))
#  else
#    define DWA_API
#  endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DWAConfig {
    /* Physical limits and sampling. */
    double dt;                /* control period (s); must match env. */
    double predict_time;      /* forward simulation horizon (s) */

    double v_min;             /* translational vel min (m/s); 0 disables backing */
    double v_max;             /* translational vel max */
    double omega_max;         /* rotational vel bound (rad/s); range is [-omega_max, omega_max] */

    double v_acc_max;         /* translational accel for dynamic window (m/s^2) */
    double omega_acc_max;     /* rotational accel for dynamic window (rad/s^2) */
    double v_brake_acc;       /* translational brake accel for admissibility (m/s^2) */
    double omega_brake_acc;   /* rotational brake accel for admissibility (rad/s^2) */

    int v_samples;            /* grid resolution along v axis */
    int omega_samples;        /* grid resolution along omega axis */

    /* Objective weights (paper Eq. (13)). */
    double alpha_heading;
    double beta_clearance;
    double gamma_velocity;

    /* Geometry / clipping. */
    double robot_radius_m;    /* used as collision threshold along trajectory */
    double dist_clip_m;       /* cap when no obstacle is hit (paper's "large constant") */

    /* Behavior flags. */
    int rotate_away_mode;     /* when V_r is empty, command (0, +omega_max) toward target side */

    /* Forward-sim step count (predict_time is divided into this many sub-steps). */
    int predict_steps;
} DWAConfig;


typedef struct DWAOutput {
    double vx_cmd;            /* chosen translational velocity (m/s) */
    double omega_cmd;         /* chosen rotational velocity (rad/s) */
    double score;             /* G(v,w) for the chosen pair, 0..(alpha+beta+gamma) */
    int found;                /* 1 if V_r non-empty; 0 if we fell back to rotate-away */
} DWAOutput;


/* Plan one DWA step.
 *
 * vx_cur, omega_cur : actual current velocity (m/s, rad/s)
 * target_x_local, target_y_local : goal in robot frame (m)
 * ray_dists : n_rays distances (m). ray angles are derived from index uniformly
 *             over [0, 2*pi); pass NULL ray_angles to use that default.
 * ray_angles : optional explicit ray angles (rad), length n_rays. May be NULL.
 * n_rays : number of rays
 *
 * Returns the chosen action via *out. dwa_plan is reentrant and allocates no
 * heap memory at runtime (uses a small caller-supplied scratch buffer).
 *
 * scratch / scratch_n : working buffer of doubles. Need at least 4*n_rays
 *   entries (xa, ya, xb, yb for each obstacle line). If too small, returns -1.
 *   Returns 0 on success, -1 on bad inputs.
 */
DWA_API int dwa_plan(
    const DWAConfig *cfg,
    double vx_cur, double omega_cur,
    double target_x_local, double target_y_local,
    const double *ray_dists,
    const double *ray_angles,
    int n_rays,
    double *scratch,
    int scratch_n,
    DWAOutput *out);


#ifdef __cplusplus
}
#endif

#endif /* GRALP_EVAL_DWA_H */
