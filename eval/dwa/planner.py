from __future__ import annotations

"""Vectorized DWA local planner (Fox/Burgard/Thrun 1997), pure torch.

Plans for ``B`` environments and ``NC = v_samples * omega_samples`` candidate
(v, omega) pairs in a single batched forward pass on the same device as the
caller. No CPU/GPU sync: inputs and outputs are device tensors, the planner
state (candidate grid) is allocated once at construction time.

Algorithm follows the paper:
  - Search space V_s n V_a n V_d on a (v, omega) grid.
  - Trajectories approximated by piecewise-constant velocity for predict_time
    seconds (midpoint Euler with predict_steps sub-steps).
  - Obstacle line field: each ray endpoint becomes a short segment of width
    ``ray_dist * (2*pi / n_rays)`` perpendicular to the ray direction.
  - dist(v, w) = arc length traveled before any point on the predicted
    trajectory comes within robot_radius_m of any obstacle line segment.
  - admissibility: |v| <= sqrt(2*dist*v_brake_acc) and same for omega.
  - dynamic window: v in [vx_cur +/- v_acc*dt], omega in [omega_cur +/- omega_acc*dt].
  - Objective G = alpha*heading + beta*dist + gamma*velocity, each term
    normalized to [0,1] within the env's admissible set, argmax over candidates.
  - rotate_away fallback when the admissible set is empty for an env.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple, Union

import torch


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class DWAConfig:
    """DWA hyperparameters. Fields that overlap the env are auto-derived by
    ``from_configs``; pure DWA knobs come from ``eval/dwa_config.json``."""

    dt: float
    predict_time: float
    v_min: float
    v_max: float
    omega_max: float
    v_acc_max: float
    omega_acc_max: float
    v_brake_acc: float
    omega_brake_acc: float
    v_samples: int
    omega_samples: int
    alpha_heading: float
    beta_clearance: float
    gamma_velocity: float
    robot_radius_m: float
    dist_clip_m: float
    rotate_away_mode: bool
    predict_steps: int = 20

    @classmethod
    def from_configs(cls, env_cfg: Mapping[str, Any], dwa_cfg: Mapping[str, Any]) -> "DWAConfig":
        lim = env_cfg.get("limits", {}) or {}
        sim = env_cfg.get("sim", {}) or {}
        obs = env_cfg.get("obs", {}) or {}
        rew = env_cfg.get("reward", {}) or {}

        vx_max = float(lim.get("vx_max", 0.6))
        omega_max = float(lim.get("omega_max", 1.5))
        dt = float(sim.get("dt", 0.1))
        v_min = 0.0 if bool(rew.get("orientation_verify", False)) else -vx_max

        # Dynamic-window width: env semantics allow a single control step to
        # reach the full velocity range, so V_d effectively degenerates to V_s.
        v_acc = (vx_max - v_min) / max(dt, 1e-9)
        w_acc = (2.0 * omega_max) / max(dt, 1e-9)

        # Brake decelerations stay independent and paper-scale: too high here
        # makes V_a vacuous (any non-immediate collision becomes stoppable).
        v_brake = float(dwa_cfg.get("v_brake_acc", 0.5))
        w_brake = float(dwa_cfg.get("omega_brake_acc", 1.0))

        return cls(
            dt=dt,
            predict_time=float(dwa_cfg.get("predict_time", 1.0)),
            v_min=v_min,
            v_max=vx_max,
            omega_max=omega_max,
            v_acc_max=v_acc,
            omega_acc_max=w_acc,
            v_brake_acc=v_brake,
            omega_brake_acc=w_brake,
            v_samples=int(dwa_cfg.get("v_samples", 21)),
            omega_samples=int(dwa_cfg.get("omega_samples", 41)),
            alpha_heading=float(dwa_cfg.get("alpha_heading", 0.8)),
            beta_clearance=float(dwa_cfg.get("beta_clearance", 0.1)),
            gamma_velocity=float(dwa_cfg.get("gamma_velocity", 1.0)),
            robot_radius_m=float(dwa_cfg.get("robot_radius_m", 0.1)),
            dist_clip_m=float(obs.get("patch_meters", 10.0)),
            rotate_away_mode=bool(dwa_cfg.get("rotate_away_mode", True)),
            predict_steps=int(dwa_cfg.get("predict_steps", 20)),
        )


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class DWAPlanner:
    """Stateless DWA planner on a fixed torch device.

    Candidate (v, w) grid and a few constant tensors are allocated once at
    construction. ``plan_batch`` takes per-env state, returns per-env action,
    all on the same device. No `.item()` / `.cpu()` inside the planner.
    """

    def __init__(self, cfg: DWAConfig, device: Union[torch.device, str] = "cpu",
                 dtype: torch.dtype = torch.float32) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.dtype = dtype

        NV = int(cfg.v_samples)
        NW = int(cfg.omega_samples)
        if NV < 2 or NW < 2:
            raise ValueError(f"v_samples/omega_samples must be >= 2 (got {NV}, {NW})")
        v_grid = torch.linspace(cfg.v_min, cfg.v_max, NV, device=self.device, dtype=dtype)
        w_grid = torch.linspace(-cfg.omega_max, cfg.omega_max, NW, device=self.device, dtype=dtype)
        # Cartesian product flattened so we iterate over NC candidates as a single axis.
        self._v_cand = v_grid.repeat_interleave(NW)        # [NC] = [NV*NW]
        self._w_cand = w_grid.repeat(NV)                   # [NC]
        self.NC = NV * NW

        # ray angle table — populated lazily on first plan_batch (depends on N).
        self._n_rays_cache: int = -1
        self._cos_a: torch.Tensor = torch.empty(0, device=self.device, dtype=dtype)
        self._sin_a: torch.Tensor = torch.empty(0, device=self.device, dtype=dtype)
        self._half_tile: float = 0.0  # 0.5 * (2pi / N) once N is known

        # Cumulative diagnostic counter, in env-steps.
        #   fallback : admissible set empty -> rotate_away / stand-still triggered
        #              (hard failure -- DWA cannot pick any candidate).
        # Stays on GPU; sync to host only via get_stats().
        self._fallback_envsteps: torch.Tensor = torch.zeros((), device=self.device, dtype=torch.int64)
        self._total_envsteps: torch.Tensor = torch.zeros((), device=self.device, dtype=torch.int64)

    # ------------------------------------------------------------------
    # Fallback-rate stats. ``reset_stats`` before a rollout, ``get_stats``
    # after to read the cumulative rate. The single .item() inside get_stats
    # is the only host sync introduced by these counters.
    # ------------------------------------------------------------------

    def reset_stats(self) -> None:
        self._fallback_envsteps.zero_()
        self._total_envsteps.zero_()

    def get_stats(self) -> Dict[str, float]:
        n_fb = int(self._fallback_envsteps.item())
        n_total = int(self._total_envsteps.item())
        denom = float(n_total) if n_total > 0 else 1.0
        return {
            "fallback_envsteps": n_fb,
            "total_envsteps": n_total,
            "fallback_rate": n_fb / denom,
        }

    # ------------------------------------------------------------------
    # Lazily cache the per-ray angle table once we know N.
    # ------------------------------------------------------------------

    def _ensure_ray_tables(self, n_rays: int) -> None:
        if n_rays == self._n_rays_cache:
            return
        dtheta = 2.0 * math.pi / float(max(n_rays, 1))
        ang = torch.arange(n_rays, device=self.device, dtype=self.dtype) * dtheta
        self._cos_a = torch.cos(ang)
        self._sin_a = torch.sin(ang)
        self._half_tile = 0.5 * dtheta
        self._n_rays_cache = int(n_rays)

    # ------------------------------------------------------------------
    # Main entry: one DWA step for B independent envs.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def plan_batch(self,
                   vx_cur: torch.Tensor,
                   omega_cur: torch.Tensor,
                   target_x_local: torch.Tensor,
                   target_y_local: torch.Tensor,
                   rays_m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Plan one DWA step for B envs.

        All inputs are torch tensors on the same device as the planner.
        Shapes: ``vx_cur, omega_cur, target_x_local, target_y_local`` are ``[B]``;
        ``rays_m`` is ``[B, N]``.
        Returns ``(vx_out, omega_out)`` as ``[B]`` tensors on the same device.
        """
        cfg = self.cfg
        device = self.device
        dtype = self.dtype

        vx_cur = vx_cur.to(device=device, dtype=dtype)
        omega_cur = omega_cur.to(device=device, dtype=dtype)
        tx = target_x_local.to(device=device, dtype=dtype)
        ty = target_y_local.to(device=device, dtype=dtype)
        rays_m = rays_m.to(device=device, dtype=dtype)
        if rays_m.dim() != 2:
            raise ValueError(f"rays_m must be [B, N], got shape {tuple(rays_m.shape)}")
        B, N = rays_m.shape
        NC = self.NC
        self._ensure_ray_tables(N)

        # ------------------------------------------------------------------
        # 1) Obstacle LINE FIELD (Fox/Burgard/Thrun 1997, sec 5.2): each sensor
        # reading is a short line segment PERPENDICULAR to the beam at the
        # measured range, with length = beam breadth (range * dtheta).  Each
        # segment is represented by 3 points -- its centre and two endpoints --
        # so the inflated segments TILE the angular bins with no inter-ray gap.
        # (A single centre point per ray leaves blind spots between adjacent
        # rays that the robot body can clip, which over-collides unless the
        # footprint is grossly over-inflated.)  The obstacle axis becomes 3*N.
        # Rays at the patch boundary (free space) are shifted to a far "phantom"
        # location so they never trigger a collision.
        # ------------------------------------------------------------------
        cos_a = self._cos_a            # [N]
        sin_a = self._sin_a            # [N]
        half = self._half_tile         # 0.5 * (2*pi / N) = half beam width (rad)
        cx0 = rays_m * cos_a           # [B, N] segment centre (robot frame)
        cy0 = rays_m * sin_a
        off = rays_m * half            # [B, N] half beam-breadth = perp offset
        # perpendicular to beam i is (-sin_a, cos_a); +/- segment endpoints
        cx_p = cx0 - off * sin_a; cy_p = cy0 + off * cos_a
        cx_m = cx0 + off * sin_a; cy_m = cy0 - off * cos_a
        cx = torch.cat([cx0, cx_p, cx_m], dim=1)   # [B, 3N]
        cy = torch.cat([cy0, cy_p, cy_m], dim=1)
        far = 10.0 * cfg.dist_clip_m
        invalid_ray = (rays_m <= 0.0) | (rays_m >= cfg.dist_clip_m)  # [B, N]
        invalid = invalid_ray.repeat(1, 3)         # [B, 3N], blocks share ray order
        cx = torch.where(invalid, torch.full_like(cx, far), cx)
        cy = torch.where(invalid, torch.full_like(cy, far), cy)

        # ------------------------------------------------------------------
        # 2) Dynamic-window mask V_d (per env, [B, NC]).
        # ------------------------------------------------------------------
        v_cand = self._v_cand                                # [NC]
        w_cand = self._w_cand                                # [NC]
        dv = cfg.v_acc_max * cfg.dt
        dw = cfg.omega_acc_max * cfg.dt
        vd_lo = (vx_cur - dv).clamp_min(cfg.v_min)            # [B]
        vd_hi = (vx_cur + dv).clamp_max(cfg.v_max)            # [B]
        wd_lo = (omega_cur - dw).clamp_min(-cfg.omega_max)    # [B]
        wd_hi = (omega_cur + dw).clamp_max(cfg.omega_max)     # [B]
        in_vd = (v_cand.unsqueeze(0) >= vd_lo.unsqueeze(1)) & (v_cand.unsqueeze(0) <= vd_hi.unsqueeze(1))  # [B, NC]
        in_wd = (w_cand.unsqueeze(0) >= wd_lo.unsqueeze(1)) & (w_cand.unsqueeze(0) <= wd_hi.unsqueeze(1))  # [B, NC]
        in_dyn = in_vd & in_wd

        # ------------------------------------------------------------------
        # 3) Closed-form arc-to-collision per (env, candidate, obstacle).
        # Each candidate trajectory is either a straight line (omega = 0) or a
        # circular arc (omega != 0), both parameterized in the robot frame
        # with yaw 0 at t=0. For each obstacle point P we solve analytically
        # for the arc length at which the robot center first enters a
        # robot_radius circle around P. Total tensor: [B, NC, N] obstacles
        # (vs the old [B, NC, S, N] sample tensor -- 20x smaller).
        # ------------------------------------------------------------------
        v_b = v_cand.unsqueeze(0).expand(B, NC)               # [B, NC]
        w_b = w_cand.unsqueeze(0).expand(B, NC)
        T_pred = float(cfg.predict_time)
        rr = float(cfg.robot_radius_m)
        rr2 = rr * rr
        eps_w = 1e-6
        BIG = float(cfg.dist_clip_m) * 10.0

        # Broadcast to [B, NC, N]
        v_e = v_b.unsqueeze(-1)                               # [B, NC, 1]
        w_e = w_b.unsqueeze(-1)                               # [B, NC, 1]
        ox = cx.unsqueeze(1)                                  # [B, 1, N]
        oy = cy.unsqueeze(1)                                  # [B, 1, N]

        # --- Straight-line branch ------------------------------------------
        # D^2(s) = (s*sign(v) - ox)^2 + oy^2, parameterized by arc s = |v|*t.
        # First hit at s = |ox| - sqrt(rr^2 - oy^2), valid when oy^2 < rr^2
        # AND sign(ox) == sign(v) (obstacle ahead in the direction of motion).
        # arc_line is set to BIG when the formula doesn't apply.
        max_arc_line = v_e.abs() * T_pred                     # [B, NC, 1]
        delta_line = rr2 - oy * oy                            # [B, 1, N]
        # Same-sign mask: obstacle in front of motion direction. v_e == 0
        # gives same_sign=False so robot-at-rest never collides.
        same_sign = (v_e * ox) > 0                            # [B, NC, N]
        valid_line = (delta_line > 0.0) & same_sign
        sqrt_delta = delta_line.clamp_min(0.0).sqrt()
        arc_line = ox.abs() - sqrt_delta                      # [B, NC, N]
        arc_line = arc_line.clamp(min=0.0)
        arc_line = torch.where(valid_line, arc_line, torch.full_like(arc_line, BIG))

        # --- Circular-arc branch -------------------------------------------
        # C = (0, r) where r = v/omega; R = |r|. Robot starts at (0,0) at
        # angle theta_0 = atan2(-r, 0). At angle (theta_0 + omega*t) the robot
        # is at distance R from C; the closest obstacle approach is when the
        # arc passes through the point Q on the circle nearest to P.
        w_safe = torch.where(w_e.abs() >= eps_w, w_e, torch.ones_like(w_e))
        r_signed = v_e / w_safe                               # [B, NC, 1]
        R = r_signed.abs()                                    # [B, NC, 1]
        # |CP|^2 = ox^2 + (oy - r)^2
        dCP2 = ox * ox + (oy - r_signed).square()             # [B, NC, N]
        dCP = dCP2.sqrt()
        # min distance from circle to P = ||CP| - R|; collision if < rr.
        min_dist_circle = (dCP - R).abs()
        approaches = min_dist_circle < rr                     # [B, NC, N]
        # Triangle: R, dCP, rr; the chord at distance rr from P subtends an
        # angular half-width acos((R^2 + dCP^2 - rr^2) / (2 R dCP)) about Q.
        denom_circ = (2.0 * R * dCP).clamp_min(1e-12)
        cos_half = ((R * R + dCP2 - rr2) / denom_circ).clamp(-1.0, 1.0)
        half_angle = torch.acos(cos_half)                     # [B, NC, N]
        # theta_Q (angle of closest point Q around C): atan2(oy - r, ox).
        theta_Q = torch.atan2(oy - r_signed, ox)              # [B, NC, N]
        # theta_0 (robot's start angle around C): atan2(-r, 0) ; sign of -r.
        theta_0 = torch.atan2(-r_signed, torch.zeros_like(r_signed))  # [B, NC, 1]
        # Swept angle in direction of omega; map to [0, 2pi).
        sign_w = torch.sign(w_e)
        swept_to_Q = ((theta_Q - theta_0) * sign_w).remainder(2.0 * math.pi)
        # First-hit swept angle (before reaching Q).
        swept_first = swept_to_Q - half_angle                 # [B, NC, N]
        swept_total = w_e.abs() * T_pred                      # [B, NC, 1]
        in_horizon = (swept_first >= 0.0) & (swept_first <= swept_total)
        arc_circ = R * swept_first.clamp(min=0.0)             # [B, NC, N]
        valid_circ = approaches & in_horizon
        arc_circ = torch.where(valid_circ, arc_circ, torch.full_like(arc_circ, BIG))

        # --- Combine line / circle branches by |omega| threshold -----------
        use_line = (w_e.abs() < eps_w).expand_as(arc_line)
        arc_per_obs = torch.where(use_line, arc_line, arc_circ)  # [B, NC, N]

        # First (smallest) arc length where the trajectory collides with any
        # obstacle; BIG when nothing is hit within the predict horizon.
        arc_first = arc_per_obs.amin(dim=-1)                  # [B, NC]
        # Clamp to predict horizon distance; cap by dist_clip_m on no-hit so
        # admissibility / normalization stays consistent with the prior
        # sample-based version.
        max_arc_traj = v_b.abs() * T_pred
        dist = torch.minimum(arc_first, max_arc_traj)
        dist_clip_t = torch.tensor(cfg.dist_clip_m, device=device, dtype=dtype)
        # arc_first >= max_arc => no collision in horizon => use dist_clip.
        no_hit = arc_first >= max_arc_traj
        dist = torch.where(no_hit, dist_clip_t.expand_as(dist), dist)

        # ------------------------------------------------------------------
        # 4) Score components (raw, then per-env normalized to [0,1]).
        # ------------------------------------------------------------------
        # End-of-horizon pose (robot frame), needed for the heading term.
        # Reuse closed-form positions evaluated at t = T_pred.
        wT = w_b * T_pred                                     # [B, NC]
        w_safe_2d = torch.where(w_b.abs() >= eps_w, w_b, torch.ones_like(w_b))
        r_2d = v_b / w_safe_2d                                # [B, NC]
        is_circ_2d = w_b.abs() >= eps_w
        x_end_circ = r_2d * torch.sin(wT)
        y_end_circ = r_2d * (1.0 - torch.cos(wT))
        x_end_line = v_b * T_pred
        x_end = torch.where(is_circ_2d, x_end_circ, x_end_line)
        y_end = torch.where(is_circ_2d, y_end_circ, torch.zeros_like(x_end_line))
        th_end = wT                                            # [B, NC]

        target_dir = torch.atan2(ty.unsqueeze(-1) - y_end, tx.unsqueeze(-1) - x_end)
        diff = target_dir - th_end
        diff = (diff + math.pi).remainder(2.0 * math.pi) - math.pi
        heading_raw = math.pi - diff.abs()                    # [B, NC]; max = pi
        vel_raw = v_b.abs()                                   # [B, NC]

        # Admissibility V_a (using paper's brake-distance test).
        v_lim = torch.sqrt(2.0 * dist * cfg.v_brake_acc)
        w_lim = torch.sqrt(2.0 * dist * cfg.omega_brake_acc)
        in_va = (v_b.abs() <= v_lim) & (w_b.abs() <= w_lim)
        admissible = in_va & in_dyn                           # [B, NC]

        # Per-env max for normalization; floor by 1.0 to keep the division stable.
        masked_h = torch.where(admissible, heading_raw, torch.full_like(heading_raw, -1.0))
        masked_d = torch.where(admissible, dist, torch.full_like(dist, -1.0))
        masked_v = torch.where(admissible, vel_raw, torch.full_like(vel_raw, -1.0))
        best_h = masked_h.amax(dim=-1, keepdim=True).clamp_min(1.0)
        best_d = masked_d.amax(dim=-1, keepdim=True).clamp_min(1.0)
        best_v = masked_v.amax(dim=-1, keepdim=True).clamp_min(1.0)

        h_n = heading_raw / best_h
        d_n = dist / best_d
        v_n = vel_raw / best_v
        score = (cfg.alpha_heading * h_n
                 + cfg.beta_clearance * d_n
                 + cfg.gamma_velocity * v_n)
        score = torch.where(admissible, score, torch.full_like(score, -1e10))

        # ------------------------------------------------------------------
        # 5) Argmax + rotate-away fallback.
        # ------------------------------------------------------------------
        best_idx = score.argmax(dim=-1)                       # [B]
        chosen_v = v_b.gather(1, best_idx.unsqueeze(-1)).squeeze(-1)
        chosen_w = w_b.gather(1, best_idx.unsqueeze(-1)).squeeze(-1)

        no_admissible = ~admissible.any(dim=-1)

        # Cumulative fallback stats: stays on GPU.
        self._fallback_envsteps += no_admissible.sum().to(torch.int64)
        self._total_envsteps += B

        if cfg.rotate_away_mode:
            # Turn toward the half-plane the goal is in; stand still otherwise.
            target_angle = torch.atan2(ty, tx)
            w_sign = torch.where(target_angle >= 0.0,
                                 torch.ones_like(target_angle),
                                 -torch.ones_like(target_angle))
            chosen_v = torch.where(no_admissible, torch.zeros_like(chosen_v), chosen_v)
            chosen_w = torch.where(no_admissible, w_sign * cfg.omega_max, chosen_w)
        else:
            chosen_v = torch.where(no_admissible, torch.zeros_like(chosen_v), chosen_v)
            chosen_w = torch.where(no_admissible, torch.zeros_like(chosen_w), chosen_w)

        return chosen_v, chosen_w
