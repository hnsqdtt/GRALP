from __future__ import annotations

"""
Key points:
- No global map or goal; each step samples per-ray FOV distances according to the empty/obstacle ratio.
- Only yaw, world-frame linear velocity, and two-step command history are tracked for rewards; position is not tracked.
- The reference direction is randomly chosen from sectors satisfying the safety distance; if none exist, use the farthest sector and uniformly sample within its width.
- Episodes never terminate (done is always False) to simplify continuous control training.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import math
import torch

from .ray import compute_ray_defaults


def _wrap_angle_pi(yaw: torch.Tensor) -> torch.Tensor:
    return ((yaw + math.pi) % (2.0 * math.pi)) - math.pi


@dataclass
class SimGPUEnvConfig:
    dt: float = 0.1
    n_envs: int = 256
    patch_meters: float = 10.0  # view radius (meters)
    ray_step_m: float = 0.025
    n_rays: int = 0  # derive from compute_ray_defaults when zero
    ray_max_gap: float = 0.25
    safe_distance_m: float = 0.75
    vx_max: float = 1.5
    omega_max: float = 2.0
    w_collision: float = 1.0
    w_progress: float = 0.01
    w_limits: float = 0.1
    orientation_verify: bool = False
    w_jerk: float = 0.0
    w_jerk_omega: float = 0.0
    reward_time: float = 0.0  # per-step deduction
    collision_done: bool = False
    blank_ratio_base: float = 40.0        # baseline empty proportion (%)
    blank_ratio_randmax: float = 40.0     # extra random added in [0, randmax] (%)
    blank_ratio_std_ratio: float = 0.33
    narrow_passage_gaussian: bool = False
    narrow_passage_std_ratio: float = 0.3
    device: Optional[str] = None  # e.g., "cuda" or "cpu"; None => auto
    task_point_max_dist_m: float = 8.0
    task_point_success_radius_m: float = 0.25
    task_point_random_interval_max: int = 0


class SimRandomGPUBatchEnv:
    """Batch randomized ray GPU environment with PPO-compatible observations and rewards.
    Keeps only command history (prev/prev_prev) and yaw; FOV is resampled every step.
    """

    def __init__(self, cfg: SimGPUEnvConfig) -> None:
        self.cfg = cfg
        dev = torch.device(cfg.device) if cfg.device is not None else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.device = dev
        n_rays = int(cfg.n_rays)
        if n_rays <= 0:
            n_rays, _, _ = compute_ray_defaults(
                {
                    "ray_max_gap": float(cfg.ray_max_gap),
                },
                float(cfg.patch_meters),
            )
        self.n_rays = int(max(0, n_rays))
        self.view_radius_m = float(cfg.patch_meters)
        if self.view_radius_m <= 0.0:
            raise ValueError(f"SimGPUEnvConfig.patch_meters must be positive, got {self.view_radius_m}")
        if cfg.vx_max <= 0.0 or cfg.omega_max <= 0.0:
            raise ValueError(
                f"SimGPUEnvConfig velocity limits must be positive; "
                f"got vx_max={cfg.vx_max}, omega_max={cfg.omega_max}"
            )
        if cfg.task_point_max_dist_m < cfg.task_point_success_radius_m:
            raise ValueError(
                "SimGPUEnvConfig.task_point_max_dist_m must be >= task_point_success_radius_m; "
                f"got max_dist={cfg.task_point_max_dist_m}, success_radius={cfg.task_point_success_radius_m}"
            )

        B = int(cfg.n_envs)
        self.B = B
        self.t = torch.zeros((B,), dtype=torch.int32, device=self.device)
        self.yaw = torch.zeros((B,), dtype=torch.float32, device=self.device)
        self.vel_xy = torch.zeros((B, 2), dtype=torch.float32, device=self.device)
        self.pos_xy = torch.zeros((B, 2), dtype=torch.float32, device=self.device)
        self.prev_cmd = torch.zeros((B, 3), dtype=torch.float32, device=self.device)
        self.prev_prev_cmd = torch.zeros((B, 3), dtype=torch.float32, device=self.device)
        if self.n_rays > 0:
            ang = (
                torch.arange(self.n_rays, device=self.device, dtype=torch.float32)
                * (2.0 * math.pi / float(self.n_rays))
            )
            self._ray_ang = ang
        else:
            self._ray_ang = torch.zeros((0,), device=self.device, dtype=torch.float32)
        self._rays_m = torch.zeros((B, self.n_rays), dtype=torch.float32, device=self.device)
        self._ref_vec = torch.zeros((B, 2), dtype=torch.float32, device=self.device)
        self._ref_feat = torch.zeros((B, 2), dtype=torch.float32, device=self.device)
        self._global_task_xy = torch.zeros((B, 2), dtype=torch.float32, device=self.device)
        self._local_task_xy = torch.zeros((B, 2), dtype=torch.float32, device=self.device)
        interval_max = int(getattr(self.cfg, "task_point_random_interval_max", 0))
        self._task_redraw_interval_max = interval_max
        if interval_max > 0:
            self._task_redraw_counter = torch.zeros((B,), dtype=torch.int32, device=self.device)
            self._task_redraw_target = torch.randint(
                1,
                interval_max + 1,
                (B,),
                device=self.device,
                dtype=torch.int32,
            )
        else:
            self._task_redraw_counter = torch.zeros((B,), dtype=torch.int32, device=self.device)
            self._task_redraw_target = torch.zeros((B,), dtype=torch.int32, device=self.device)
        self._resample_fov_and_ref()
        self._sample_new_global_task_points(mask=torch.ones((B,), dtype=torch.bool, device=self.device))

    def get_limits(self) -> torch.Tensor:
        return torch.tensor([self.cfg.vx_max, self.cfg.omega_max], device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def reset(self) -> torch.Tensor:
        self.t.zero_()
        self.yaw.zero_()
        self.vel_xy.zero_()
        self.pos_xy.zero_()
        self.prev_cmd.zero_()
        self.prev_prev_cmd.zero_()
        self._resample_fov_and_ref()
        self._sample_new_global_task_points(mask=torch.ones((self.B,), dtype=torch.bool, device=self.device))
        interval_max = getattr(self, "_task_redraw_interval_max", 0)
        if interval_max > 0:
            self._task_redraw_counter.zero_()
            self._task_redraw_target = torch.randint(
                1,
                interval_max + 1,
                (self.B,),
                device=self.device,
                dtype=torch.int32,
            )
        else:
            self._task_redraw_counter.zero_()
            self._task_redraw_target.zero_()
        return self.observe()

    @torch.no_grad()
    def observe(self) -> torch.Tensor:
        return self._build_obs(self._rays_m, self._ref_feat)

    @torch.no_grad()
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Action [B,3] (vx, vy, omega in SI units) clipped by limits.
        Returns (obs_next, reward, terminated, info).
        """
        if action.shape[0] != self.B or action.dim() != 2 or action.shape[1] not in (2, 3):
            raise AssertionError(f"expected action shape [{self.B},2] or [{self.B},3], got {tuple(action.shape)}")
        if action.device != self.device:
            action = action.to(self.device, non_blocking=True)

        vx_max = float(self.cfg.vx_max)
        om_max = float(self.cfg.omega_max)
        dt = float(self.cfg.dt)
        interval_max = getattr(self, "_task_redraw_interval_max", 0)
        vx_cmd = action[:, 0].clamp(-vx_max, vx_max)
        if action.shape[1] == 2:
            vy_cmd = torch.zeros_like(vx_cmd)
            om_cmd = action[:, 1].clamp(-om_max, om_max)
        else:
            om_cmd = action[:, 2].clamp(-om_max, om_max)
        vy_cmd = torch.zeros_like(vx_cmd)

        pos_prev = self.pos_xy.clone()
        dx_local = vx_cmd * dt
        dy_local = vy_cmd * dt
        yaw_end = _wrap_angle_pi(self.yaw + om_cmd * dt)
        c1 = torch.cos(yaw_end)
        s1 = torch.sin(yaw_end)
        vx_w_end = c1 * vx_cmd - s1 * vy_cmd
        vy_w_end = s1 * vx_cmd + c1 * vy_cmd
        self.vel_xy[:, 0] = vx_w_end
        self.vel_xy[:, 1] = vy_w_end
        self.yaw = yaw_end
        self.t += 1
        if interval_max > 0:
            self._task_redraw_counter += 1
        self.pos_xy[:, 0] = self.pos_xy[:, 0] + vx_w_end * dt
        self.pos_xy[:, 1] = self.pos_xy[:, 1] + vy_w_end * dt
        travel = torch.hypot(dx_local, dy_local)
        ang = torch.atan2(dy_local, dx_local)
        ray_d_along = self._interp_ray_distance(self._rays_m, ang)
        pen = travel - ray_d_along
        collided = (pen > 1e-6) & (ray_d_along > 0.0)
        d_prev = torch.hypot(
            self._global_task_xy[:, 0] - pos_prev[:, 0],
            self._global_task_xy[:, 1] - pos_prev[:, 1],
        )
        d_next = torch.hypot(
            self._global_task_xy[:, 0] - self.pos_xy[:, 0],
            self._global_task_xy[:, 1] - self.pos_xy[:, 1],
        )
        delta_d = d_next - d_prev
        denom_progress = vx_max * dt
        jerk_x = (vx_cmd - 2.0 * self.prev_cmd[:, 0] + self.prev_prev_cmd[:, 0]) / vx_max
        jerk_omega = (om_cmd - 2.0 * self.prev_cmd[:, 2] + self.prev_prev_cmd[:, 2]) / om_max
        limit_hit = (vx_cmd.abs() >= vx_max - 1e-9) | (om_cmd.abs() >= om_max - 1e-9)

        min_ray_dist_m = self._rays_m.min(dim=-1).values
        task_resampled = torch.zeros((self.B,), dtype=torch.bool, device=self.device)

        jerk_norm = ((jerk_x * jerk_x) / 16.0).clamp(0.0, 1.0)
        jerk_omega_norm = ((jerk_omega * jerk_omega) / 16.0).clamp(0.0, 1.0)
        v_lin = torch.hypot(vx_w_end, vy_w_end)
        v_ratio = (v_lin / vx_max).clamp(0.0, 1.0)
        base_progress = (-delta_d) / denom_progress

        if bool(self.cfg.orientation_verify):
            v_mag = torch.hypot(vx_w_end, vy_w_end)
            dot_hv = c1 * vx_w_end + s1 * vy_w_end
            cos_heading_vel = torch.where(
                v_mag > 1e-9,
                (dot_hv / v_mag).clamp(-1.0, 1.0),
                torch.ones_like(v_mag),
            )
        else:
            cos_heading_vel = torch.ones_like(base_progress)

        progress = base_progress
        progress = torch.where(delta_d > 0.0, -progress.abs(), progress)
        if bool(self.cfg.orientation_verify):
            allow_pos = (delta_d < 0.0) & (cos_heading_vel > 0.0)
            progress = torch.where(allow_pos, progress, -progress.abs())

        rew_progress = self.cfg.w_progress * progress
        rew_progress = torch.where(delta_d > 0.0, -rew_progress.abs(), rew_progress)

        rew = (
            rew_progress
            - self.cfg.w_collision * (v_ratio * collided.to(torch.float32))
            - self.cfg.w_jerk * jerk_norm
            - self.cfg.w_jerk_omega * jerk_omega_norm
            - self.cfg.w_limits * limit_hit.to(torch.float32)
            - float(getattr(self.cfg, "reward_time", 0.0))
        ).to(torch.float32)
        self.prev_prev_cmd.copy_(self.prev_cmd)
        zero_hist = torch.zeros_like(vx_cmd)
        self.prev_cmd.copy_(torch.stack([vx_cmd, zero_hist, om_cmd], dim=-1))
        if bool(getattr(self.cfg, "collision_done", False)):
            term = collided.clone().to(torch.bool)
            if bool(term.any()):
                mask = term
                self.t[mask] = 0
                self.yaw[mask] = 0.0
                self.vel_xy[mask] = 0.0
                self.pos_xy[mask] = 0.0
                self.prev_cmd[mask] = 0.0
                self.prev_prev_cmd[mask] = 0.0
                self._resample_fov_and_ref()
                self._sample_new_global_task_points(mask=mask)
                task_resampled |= mask
                if interval_max > 0:
                    self._task_redraw_counter[mask] = 0
                    n_collided = int(mask.sum().item())
                    if n_collided > 0:
                        self._task_redraw_target[mask] = torch.randint(
                            1,
                            interval_max + 1,
                            (n_collided,),
                            device=self.device,
                            dtype=torch.int32,
                        )
        else:
            term = torch.zeros((self.B,), dtype=torch.bool, device=self.device)
        u = (self.pos_xy - pos_prev)
        uu = (u[:, 0] * u[:, 0] + u[:, 1] * u[:, 1])
        w0 = (self._global_task_xy - pos_prev)
        t_proj = torch.zeros_like(uu)
        move_mask = uu > 0.0
        t_proj[move_mask] = (
            (w0[move_mask, 0] * u[move_mask, 0] + w0[move_mask, 1] * u[move_mask, 1]) / uu[move_mask]
        ).clamp(0.0, 1.0)
        nearest_x = pos_prev[:, 0] + t_proj * u[:, 0]
        nearest_y = pos_prev[:, 1] + t_proj * u[:, 1]
        dist2_near = (nearest_x - self._global_task_xy[:, 0]) ** 2 + (nearest_y - self._global_task_xy[:, 1]) ** 2
        r_s = float(self.cfg.task_point_success_radius_m)
        success = dist2_near <= (r_s * r_s)
        if bool(success.any()):
            self._sample_new_global_task_points(mask=success)
            task_resampled |= success
            if interval_max > 0:
                self._task_redraw_counter[success] = 0
                n_success = int(success.sum().item())
                if n_success > 0:
                    self._task_redraw_target[success] = torch.randint(
                        1,
                        interval_max + 1,
                        (n_success,),
                        device=self.device,
                        dtype=torch.int32,
                    )

        if interval_max > 0:
            redraw_mask = (self._task_redraw_counter >= self._task_redraw_target) & (~task_resampled)
            if bool(redraw_mask.any()):
                self._sample_new_global_task_points(mask=redraw_mask)
                self._task_redraw_counter[redraw_mask] = 0
                n_redraw = int(redraw_mask.sum().item())
                if n_redraw > 0:
                    self._task_redraw_target[redraw_mask] = torch.randint(
                        1,
                        interval_max + 1,
                        (n_redraw,),
                        device=self.device,
                        dtype=torch.int32,
                    )

        self._resample_fov_and_ref()
        obs_next = self._build_obs(self._rays_m, self._ref_feat)
        info: Dict[str, Any] = {
            "limits": self.get_limits(),
            "success": success,
            "timeout": torch.zeros((self.B,), dtype=torch.bool, device=self.device),
            "min_ray_dist_m": min_ray_dist_m,
            "collided": collided,
        }
        return obs_next, rew, term, info

    @torch.no_grad()
    def _resample_fov_and_ref(self) -> None:
        if self.n_rays <= 0:
            self._rays_m.zero_()
            self._update_local_task_points()
            self._update_ref_from_local()
            return

        base = float(self.cfg.blank_ratio_base)
        jitter = float(self.cfg.blank_ratio_randmax)
        std_ratio = float(getattr(self.cfg, "blank_ratio_std_ratio", 0.33))
        sigma = max(jitter * std_ratio, 1e-6)
        p_empty_raw = torch.randn((self.B,), device=self.device) * sigma + base
        p_empty = (torch.clamp(p_empty_raw, base, base + jitter) / 100.0).to(torch.float32)
        mask_empty = torch.rand((self.B, self.n_rays), device=self.device) < p_empty.view(-1, 1)
        use_gaussian = bool(getattr(self.cfg, "narrow_passage_gaussian", False))
        if use_gaussian:
            std_ratio = float(getattr(self.cfg, "narrow_passage_std_ratio", 0.3))
            sigma = max(self.view_radius_m * std_ratio, 1e-6)
            dist = torch.abs(torch.randn((self.B, self.n_rays), device=self.device, dtype=torch.float32)) * sigma
            dist = dist.clamp(min=0.0, max=self.view_radius_m)
            rays_m = torch.where(mask_empty, torch.full_like(dist, self.view_radius_m), dist)
        else:
            prop = torch.rand((self.B, self.n_rays), device=self.device)
            rays_m = prop * self.view_radius_m
            rays_m = torch.where(mask_empty, torch.full_like(rays_m, self.view_radius_m), rays_m)
        self._rays_m = rays_m.to(torch.float32)
        self._update_local_task_points()
        self._update_ref_from_local()

    @torch.no_grad()
    @torch.no_grad()
    def _update_local_task_points(self, mask: Optional[torch.Tensor] = None) -> None:
        """Project global task points into current LOS; keep the closest visible point."""
        dx = self._global_task_xy[:, 0] - self.pos_xy[:, 0]
        dy = self._global_task_xy[:, 1] - self.pos_xy[:, 1]
        dist_global = torch.hypot(dx, dy)

        dir_x = torch.zeros_like(dx)
        dir_y = torch.zeros_like(dy)
        nz = dist_global > 0.0
        dir_x[nz] = dx[nz] / dist_global[nz]
        dir_y[nz] = dy[nz] / dist_global[nz]

        ang_world = torch.atan2(dy, dx)
        ang_body = _wrap_angle_pi(ang_world - self.yaw)
        los_dist = self._interp_ray_distance(self._rays_m, ang_body)
        los_dist = torch.clamp(los_dist, min=0.0)
        travel = torch.minimum(dist_global, los_dist)

        new_x = self.pos_xy[:, 0] + travel * dir_x
        new_y = self.pos_xy[:, 1] + travel * dir_y

        if mask is None:
            self._local_task_xy[:, 0] = new_x
            self._local_task_xy[:, 1] = new_y
        else:
            self._local_task_xy[mask, 0] = new_x[mask]
            self._local_task_xy[mask, 1] = new_y[mask]

    @torch.no_grad()
    def _update_ref_from_local(self, mask: Optional[torch.Tensor] = None) -> None:
        """Align reference vectors/features to the current local task points."""
        dx = (self._local_task_xy[:, 0] - self.pos_xy[:, 0])
        dy = (self._local_task_xy[:, 1] - self.pos_xy[:, 1])
        n = torch.hypot(dx, dy)
        hx = torch.cos(self.yaw)
        hy = torch.sin(self.yaw)
        tx = torch.where(n > 1e-9, dx / n, hx)
        ty = torch.where(n > 1e-9, dy / n, hy)

        cos_th = (tx * hx + ty * hy).clamp(-1.0, 1.0)
        sin_th = (ty * hx - tx * hy).clamp(-1.0, 1.0)
        ref_feat = torch.stack([sin_th, cos_th], dim=-1)

        if mask is None:
            self._ref_vec[:, 0] = tx
            self._ref_vec[:, 1] = ty
            self._ref_feat = ref_feat
        else:
            self._ref_vec[mask, 0] = tx[mask]
            self._ref_vec[mask, 1] = ty[mask]
            self._ref_feat[mask] = ref_feat[mask]

    @torch.no_grad()
    def _sample_new_global_task_points(self, mask: torch.Tensor) -> None:
        """Sample new global task points; locals are the closest LOS points toward them."""
        if not bool(mask.any()):
            return
        r_min = max(float(self.cfg.task_point_success_radius_m), 0.0)
        r_max = float(min(float(self.cfg.task_point_max_dist_m), float(self.view_radius_m)))
        if r_max <= r_min:
            r_max = r_min
        u = torch.rand((self.B,), device=self.device, dtype=torch.float32)
        dist = r_min + (r_max - r_min) * u
        safe = self._rays_m >= float(self.cfg.safe_distance_m)
        idx = torch.empty((self.B,), dtype=torch.long, device=self.device)
        has_any = safe.any(dim=-1)
        argmax_idx = self._rays_m.argmax(dim=-1)
        rand_scores = torch.rand((self.B, self.n_rays), device=self.device, dtype=torch.float32)
        scores = torch.where(safe, rand_scores, torch.full_like(rand_scores, float("-inf")))
        pick_any = torch.argmax(scores, dim=-1)
        idx.copy_(torch.where(has_any, pick_any, argmax_idx))
        if self.n_rays > 0:
            idx.clamp_(min=0, max=self.n_rays - 1)
        dth = (2.0 * math.pi) / float(max(self.n_rays, 1))
        jitter_local = (torch.rand((self.B,), device=self.device, dtype=torch.float32) - 0.5) * float(dth)
        ang = self._ray_ang[idx]
        th = _wrap_angle_pi(self.yaw + ang + jitter_local)
        tx = torch.cos(th)
        ty = torch.sin(th)
        new_x = self.pos_xy[:, 0] + dist * tx
        new_y = self.pos_xy[:, 1] + dist * ty
        self._global_task_xy[mask, 0] = new_x[mask]
        self._global_task_xy[mask, 1] = new_y[mask]

        self._update_local_task_points(mask=mask)
        self._update_ref_from_local(mask=mask)

    @torch.no_grad()
    def _build_obs(self, rays_m: torch.Tensor, ref_feat: torch.Tensor) -> torch.Tensor:
        """Construct the observation vector following the agreed layout."""
        if self.n_rays <= 0:
            vx_lim = float(self.cfg.vx_max)
            om_lim = float(self.cfg.omega_max)
            prev_vx_n = self.prev_cmd[:, 0] / vx_lim
            prev_om_n = self.prev_cmd[:, 2] / om_lim
            prev_cmd_n = torch.stack([prev_vx_n, prev_om_n], dim=-1)
            dvx_n = (self.prev_cmd[:, 0] - self.prev_prev_cmd[:, 0]) / (2.0 * vx_lim)
            dom_n = (self.prev_cmd[:, 2] - self.prev_prev_cmd[:, 2]) / (2.0 * om_lim)
            dprev_all_n = torch.stack([dvx_n, dom_n], dim=-1)
            dist = torch.hypot(
                self._local_task_xy[:, 0] - self.pos_xy[:, 0],
                self._local_task_xy[:, 1] - self.pos_xy[:, 1],
            )
            dist_n = (dist / self.view_radius_m).clamp(0.0, 1.0).unsqueeze(-1)
            return torch.cat([ref_feat, prev_cmd_n, dprev_all_n, dist_n], dim=-1).to(torch.float32)
        Rm = self.view_radius_m
        rays_n = (rays_m / Rm).clamp(0.0, 1.0)
        vx_lim = float(self.cfg.vx_max)
        om_lim = float(self.cfg.omega_max)
        prev_vx_n = self.prev_cmd[:, 0] / vx_lim
        prev_om_n = self.prev_cmd[:, 2] / om_lim
        prev_cmd_n = torch.stack([prev_vx_n, prev_om_n], dim=-1)
        dvx_n = (self.prev_cmd[:, 0] - self.prev_prev_cmd[:, 0]) / (2.0 * vx_lim)
        dom_n = (self.prev_cmd[:, 2] - self.prev_prev_cmd[:, 2]) / (2.0 * om_lim)
        dprev_all_n = torch.stack([dvx_n, dom_n], dim=-1)
        dist = torch.hypot(
            self._local_task_xy[:, 0] - self.pos_xy[:, 0],
            self._local_task_xy[:, 1] - self.pos_xy[:, 1],
        )
        dist_n = (dist / Rm).clamp(0.0, 1.0).unsqueeze(-1)

        parts = [rays_n, ref_feat, prev_cmd_n, dprev_all_n, dist_n]
        return torch.cat(parts, dim=-1).to(torch.float32)

    @torch.no_grad()
    def _interp_ray_distance(self, rays_m: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """Interpolate ray distance along a body-frame angle on the circular samples.
        rays_m: [B,R]; angle: [B] (radians); returns [B] (meters).
        """
        if self.n_rays <= 0:
            return torch.full((self.B,), float("inf"), device=self.device, dtype=torch.float32)
        R = float(self.n_rays)
        dth = (2.0 * math.pi) / R

        a = angle % (2.0 * math.pi)
        f = a / dth
        i0 = torch.floor(f).to(torch.long)
        t = f - i0.to(torch.float32)
        if self.n_rays > 0:
            i0 = i0.clamp(min=0, max=self.n_rays - 1)
            i1 = (i0 + 1) % int(self.n_rays)
        else:
            i1 = i0
        ar = torch.arange(self.B, device=self.device)
        d0 = rays_m[ar, i0]
        d1 = rays_m[ar, i1]
        return d0 * (1.0 - t) + d1 * t


def infer_obs_dim(cfg: SimGPUEnvConfig) -> int:
    if int(cfg.n_rays) <= 0:
        return 7
    return int(cfg.n_rays) + 7
