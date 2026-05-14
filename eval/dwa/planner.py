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
from typing import Any, Mapping, Tuple, Union

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
        # 1) Obstacle point set: each ray endpoint becomes one obstacle point.
        # Rays that hit the patch boundary (free space) get shifted to a far
        # "phantom" location so they never trigger a collision -- this lets us
        # drop a torch.where on the hot 4-D distance tensor below.
        # The paper-style line-segment representation only matters when the
        # tile width (ray_dist * 2*pi/N) is comparable to robot_radius; at our
        # N=105 / patch=10m / robot_radius=0.1m the tile is ~0.06m and a single
        # point is dense enough.
        # ------------------------------------------------------------------
        cos_a = self._cos_a            # [N]
        sin_a = self._sin_a            # [N]
        cx = rays_m * cos_a            # [B, N] obstacle point x (robot frame)
        cy = rays_m * sin_a            # [B, N]
        far = 10.0 * cfg.dist_clip_m
        invalid_ray = (rays_m <= 0.0) | (rays_m >= cfg.dist_clip_m)
        cx = torch.where(invalid_ray, torch.full_like(cx, far), cx)
        cy = torch.where(invalid_ray, torch.full_like(cy, far), cy)

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
        # 3) Forward simulate every (env, candidate) closed-form for all S
        # sub-steps at once, then compute distance to obstacles as a single
        # 4-D tensor reduction. This replaces a Python loop of S small GPU
        # kernels with a few large ones, removing launch-latency overhead.
        # ------------------------------------------------------------------
        v_b = v_cand.unsqueeze(0).expand(B, NC)               # [B, NC]
        w_b = w_cand.unsqueeze(0).expand(B, NC)
        steps = int(cfg.predict_steps)
        dt_sim = cfg.predict_time / float(steps)
        r_safe2 = cfg.robot_radius_m * cfg.robot_radius_m

        t_seq = torch.arange(1, steps + 1, device=device, dtype=dtype) * dt_sim  # [S]
        wt = w_b.unsqueeze(-1) * t_seq.view(1, 1, -1)         # [B, NC, S]
        eps_w = 1e-9
        # Safe division: replace |w|<eps with 1.0 to avoid NaN; we pick the
        # straight-line branch for those positions via torch.where below.
        w_safe = torch.where(w_b.abs() < eps_w, torch.ones_like(w_b), w_b)
        r_arc = (v_b / w_safe).unsqueeze(-1)                  # [B, NC, 1]
        is_circ = (w_b.abs() >= eps_w).unsqueeze(-1)          # [B, NC, 1]

        # Robot starts at (0,0,0) in its own frame each step.
        x_circ = r_arc * torch.sin(wt)
        y_circ = r_arc * (1.0 - torch.cos(wt))
        x_line = v_b.unsqueeze(-1) * t_seq.view(1, 1, -1)
        x_sub = torch.where(is_circ, x_circ, x_line)          # [B, NC, S]
        y_sub = torch.where(is_circ, y_circ, torch.zeros_like(x_line))
        th_sub = wt                                            # [B, NC, S]

        # Point-to-point distance squared on the hot 4-D tensor:
        #   [B, NC, S, N] = (px - cx_e)^2 + (py - cy_e)^2
        # This is the memory-bound bottleneck of plan_batch (a single ~166 MB
        # tensor at the default grid size); keeping it to two squared diffs +
        # one amin is what makes it survive on a laptop GPU.
        cx_e = cx.view(B, 1, 1, N)
        cy_e = cy.view(B, 1, 1, N)
        px = x_sub.unsqueeze(-1)                              # [B, NC, S, 1]
        py = y_sub.unsqueeze(-1)
        d2 = (px - cx_e).square() + (py - cy_e).square()      # [B, NC, S, N]
        min_d2_per_step = d2.amin(dim=-1)                     # [B, NC, S]

        # First sub-step at which the trajectory enters the safety circle. If
        # none, dist falls back to dist_clip_m. We do this with a single amin:
        # at non-collision steps put dist_clip, at collision steps put arc len.
        arc_per_step = v_b.abs().unsqueeze(-1) * t_seq.view(1, 1, -1)  # [B, NC, S]
        collide_step = min_d2_per_step <= r_safe2
        arc_or_clip = torch.where(
            collide_step, arc_per_step,
            torch.tensor(cfg.dist_clip_m, device=device, dtype=dtype),
        )
        dist = arc_or_clip.amin(dim=-1)                       # [B, NC]

        # ------------------------------------------------------------------
        # 4) Score components (raw, then per-env normalized to [0,1]).
        # ------------------------------------------------------------------
        x_end = x_sub[..., -1]
        y_end = y_sub[..., -1]
        th_end = th_sub[..., -1]
        target_dir = torch.atan2(ty.unsqueeze(-1) - y_end, tx.unsqueeze(-1) - x_end)  # [B, NC]
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
