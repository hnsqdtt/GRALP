from __future__ import annotations

"""Episodic goal-navigation env on a narrow-channel (pillar-lattice) map.

Subclasses ``SimRandomGPUBatchEnv`` so the *observation* fed to a PPO policy is
bit-identical to training (same ray normalisation, same 7-dim pose tail, same
LOS local-target projection). The only thing that changes is the source of the
ray distances: instead of resampling a random FOV every step, we raycast the
real inflated occupancy grid of ``NarrowChannelMap`` at the robot's true pose.
Collisions and goal arrival are likewise resolved against the real map, so a
run is a genuine point-to-point navigation episode through the corridors.

Why this is a faithful "use the policy the way deployment does"
---------------------------------------------------------------
The minimalist PyBullet sim (the deployment reference) feeds the policy lidar
ranges that are dilated by the robot radius, and steers toward a line-of-sight
local target derived from a global goal. We reproduce both:
  * rays come from raycasting the *inflated* map (obstacles appear ``r`` closer),
    exactly the deployment ray-dilation, so a point planner keeps the footprint;
  * the LOS local target is the base class' ``_update_local_task_points`` -- the
    same carrot the policy was trained on.

Episode protocol (fully reproducible, identical across planners)
----------------------------------------------------------------
Every env runs a fixed, pre-generated sequence of ``episodes_per_env`` (start,
goal) pairs. Env ``b``'s ``k``-th episode is global id ``gid = k*B + b``; both
the DWA baseline and the PPO policy therefore face the *same* pairs in the same
per-env order -- apples-to-apples. An episode ends on collision, arrival, or
timeout; the env records the outcome into preallocated [T] buffers (one host
sync at the end) and respawns the next pair for that env until its quota is met.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import math

import numpy as np
import torch

from env.sim_gpu_env import SimGPUEnvConfig, SimRandomGPUBatchEnv, _wrap_angle_pi

from .narrow_channel_scene import NarrowChannelMap


# Outcome codes stored in the result buffer.
OUTCOME_UNSET = 0
OUTCOME_SUCCESS = 1
OUTCOME_COLLISION = 2
OUTCOME_TIMEOUT = 3


@dataclass
class NarrowChannelEpisodeSpec:
    n_episodes: int = 192          # >= this many episodes total (rounded up to a multiple of B)
    max_episode_steps: int = 600   # timeout horizon per episode
    success_radius_m: float = 0.30  # arrival radius around the goal centre
    min_geo_nodes: int = 3         # require start/goal at least this many node-steps apart
    start_yaw_toward_goal: bool = True
    seed: int = 0
    track_path0: bool = False      # log env-0's first-episode path (for rendering)
    collision_mode: str = "footprint"  # "footprint" (center in inflated map, any
    #   direction -- realistic) or "forward_ray" (travel past the motion-direction
    #   ray only, matching the training env's point-robot collision rule).

    def __post_init__(self) -> None:
        if self.collision_mode not in ("footprint", "forward_ray"):
            raise ValueError(
                f"collision_mode must be 'footprint' or 'forward_ray', got {self.collision_mode!r}")


class NarrowChannelEnv(SimRandomGPUBatchEnv):
    """Vectorised episodic navigation over a fixed narrow-channel map."""

    def __init__(self, sim_cfg: SimGPUEnvConfig, ncmap: NarrowChannelMap,
                 ep: NarrowChannelEpisodeSpec) -> None:
        if str(sim_cfg.device) != str(ncmap.device):
            # Keep the map and the env on the same device for the raycast gather.
            sim_cfg = SimGPUEnvConfig(**{**sim_cfg.__dict__, "device": str(ncmap.device)})
        self._ncmap = ncmap
        self._ep = ep
        # Ray marching parameters: step at the grid resolution so a ray cannot
        # tunnel through a (thick) inflated obstacle.
        self._ray_step = float(ncmap.cfg.res)
        # Allocated lazily once view_radius_m is known (in _resample, post super-init).
        self._ray_S = 0
        # Chunk the [B, N, S] raycast over envs to cap peak memory.
        self._ray_chunk_cap = 4_000_000  # max B*N*S elements per chunk
        # Layer-2 (robot_radius + extra) rays, used ONLY for the LOS carrot.
        self._rays_los_m = None

        super().__init__(sim_cfg)  # sets device, B, n_rays, pose state; calls _resample_fov_and_ref

        if self.B <= 0:
            raise ValueError("n_envs must be >= 1")
        self.max_steps = int(ep.max_episode_steps)
        self.success_radius_m = float(ep.success_radius_m)
        # Fixed sub-sampling for the swept-segment collision test: a single
        # control step moves at most vx_max*dt metres, so this bound is constant
        # and avoids a per-step host sync on the realised move distance.
        max_move = float(self.cfg.vx_max) * float(self.cfg.dt)
        self._n_seg_sub = max(2, int(math.ceil(max_move / self._ray_step)) + 1)

        self._build_episode_queue()
        self._alloc_state_and_results()
        self.path0: List[List[float]] = []
        self.reset()

    # ------------------------------------------------------------------
    # Episode queue (reproducible, planner-independent)
    # ------------------------------------------------------------------

    def _build_episode_queue(self) -> None:
        ep = self._ep
        B = self.B
        self.episodes_per_env = int(math.ceil(max(1, ep.n_episodes) / B))
        self.T = self.episodes_per_env * B

        K = int(self._ncmap.nodes.shape[0])
        if not self._ncmap.passable or K < 2:
            raise ValueError(
                "Narrow-channel map has no connected free nodes (corridors too "
                "narrow for the current robot_radius / res). Widen channel_w or "
                "lower res; see NarrowChannelMap.summary()."
            )
        geo = self._ncmap.geo_nodes  # [K, K] node-steps, inf if unreachable
        nodes = self._ncmap.nodes.cpu().numpy()  # [K, 2]

        # Valid ordered pairs: distinct, mutually reachable, far enough apart.
        reach = np.isfinite(geo) & (geo >= float(ep.min_geo_nodes))
        si, gi = np.nonzero(reach)
        if si.size == 0:
            # Relax the distance floor if it was too strict for this lattice.
            reach = np.isfinite(geo) & (geo > 0)
            si, gi = np.nonzero(reach)
        if si.size == 0:
            raise ValueError("No reachable start/goal pairs on this map.")

        rng = np.random.default_rng(int(ep.seed))
        pick = rng.integers(0, si.size, size=self.T)
        s_nodes = si[pick]
        g_nodes = gi[pick]

        self.start_node = torch.tensor(s_nodes, device=self.device, dtype=torch.long)
        self.goal_node = torch.tensor(g_nodes, device=self.device, dtype=torch.long)
        self.start_xy = torch.tensor(nodes[s_nodes], device=self.device, dtype=torch.float32)
        self.goal_xy = torch.tensor(nodes[g_nodes], device=self.device, dtype=torch.float32)
        geo_m = geo[s_nodes, g_nodes] * float(self._ncmap.cfg.pitch)
        self.geo_m = torch.tensor(geo_m, device=self.device, dtype=torch.float32)

    def _alloc_state_and_results(self) -> None:
        B, T, dev = self.B, self.T, self.device
        self.ep_k = torch.zeros((B,), dtype=torch.long, device=dev)
        self.cur_steps = torch.zeros((B,), dtype=torch.long, device=dev)
        self.cur_path = torch.zeros((B,), dtype=torch.float32, device=dev)
        self.res_outcome = torch.zeros((T,), dtype=torch.int8, device=dev)
        self.res_steps = torch.zeros((T,), dtype=torch.int32, device=dev)
        self.res_path = torch.zeros((T,), dtype=torch.float32, device=dev)
        self.res_geo = torch.zeros((T,), dtype=torch.float32, device=dev)
        self.res_spl = torch.zeros((T,), dtype=torch.float32, device=dev)
        self.res_goaldist = torch.zeros((T,), dtype=torch.float32, device=dev)
        self._arangeB = torch.arange(B, device=dev, dtype=torch.long)

    def _gid(self) -> torch.Tensor:
        """Current global episode id per env (valid where not retired)."""
        return self.ep_k * self.B + self._arangeB

    @property
    def retired(self) -> torch.Tensor:
        return self.ep_k >= self.episodes_per_env

    def all_done(self) -> bool:
        return bool(self.retired.all().item())

    # ------------------------------------------------------------------
    # Episode (re)assignment
    # ------------------------------------------------------------------

    def _assign_episode(self, mask: torch.Tensor) -> None:
        """Place the (start, goal) for the current gid of every env in ``mask``."""
        if not bool(mask.any()):
            return
        gid = self._gid().clamp(max=self.T - 1)
        s = self.start_xy[gid]
        g = self.goal_xy[gid]
        self.pos_xy[mask] = s[mask]
        self._global_task_xy[mask] = g[mask]
        self.vel_xy[mask] = 0.0
        self.prev_cmd[mask] = 0.0
        self.prev_prev_cmd[mask] = 0.0
        self.t[mask] = 0
        self.cur_steps[mask] = 0
        self.cur_path[mask] = 0.0
        if self._ep.start_yaw_toward_goal:
            yaw0 = torch.atan2(g[:, 1] - s[:, 1], g[:, 0] - s[:, 0])
        else:
            yaw0 = torch.zeros((self.B,), device=self.device, dtype=torch.float32)
        self.yaw[mask] = yaw0[mask]

    def reset(self) -> torch.Tensor:
        self.ep_k.zero_()
        self.cur_steps.zero_()
        self.cur_path.zero_()
        self.res_outcome.zero_()
        self.res_steps.zero_()
        self.res_path.zero_()
        self.res_geo.zero_()
        self.res_spl.zero_()
        self.res_goaldist.zero_()
        self.path0 = []
        full = torch.ones((self.B,), dtype=torch.bool, device=self.device)
        self._assign_episode(full)
        self._resample_fov_and_ref()
        self._maybe_log_path0()
        return self.observe()

    # ------------------------------------------------------------------
    # Ray casting against the real inflated map (replaces random FOV)
    # ------------------------------------------------------------------

    def _resample_fov_and_ref(self) -> None:  # override
        self._raycast_both()
        self._update_local_task_points()   # overridden below: LOS uses layer-2 rays
        self._update_ref_from_local()

    def _raycast_both(self) -> None:
        """Raycast both inflated maps, sharing the sample points.

        Fills ``self._rays_m`` from the layer-1 grid (robot_radius -> policy rays
        + DWA + collision) and ``self._rays_los_m`` from the layer-2 grid
        (robot_radius + extra -> LOS carrot only). ray 0 is aligned with heading.
        """
        if self.n_rays <= 0:
            self._rays_m.zero_()
            self._rays_los_m = torch.zeros_like(self._rays_m)
            return
        dev = self.device
        R = float(self.view_radius_m)
        step = self._ray_step
        if self._ray_S <= 0:
            self._ray_S = int(math.ceil(R / step))
        S = self._ray_S
        s = (torch.arange(1, S + 1, device=dev, dtype=torch.float32) * step).clamp(max=R)  # [S]
        BIG = R * 10.0
        big_t = torch.full((), BIG, device=dev)

        ang = self.yaw.view(self.B, 1) + self._ray_ang.view(1, self.n_rays)  # [B, N]
        cos_a = torch.cos(ang)
        sin_a = torch.sin(ang)

        # The live working set inside the loop is ~6x (c*N*S) float32 -- px, py,
        # pts (2x), s_hit, plus occupied()'s int64 index temporaries -- so budget
        # the chunk against that multiplier to keep peak ~ _ray_chunk_cap floats.
        per_env = 6 * self.n_rays * S
        chunk = max(1, int(self._ray_chunk_cap // max(per_env, 1)))
        out1 = torch.empty((self.B, self.n_rays), device=dev, dtype=torch.float32)
        out2 = torch.empty((self.B, self.n_rays), device=dev, dtype=torch.float32)
        s_view = s.view(1, 1, S)
        for lo in range(0, self.B, chunk):
            hi = min(self.B, lo + chunk)
            cx = cos_a[lo:hi].unsqueeze(-1)  # [c, N, 1]
            cy = sin_a[lo:hi].unsqueeze(-1)
            px = self.pos_xy[lo:hi, 0].view(-1, 1, 1) + s_view * cx  # [c, N, S]
            py = self.pos_xy[lo:hi, 1].view(-1, 1, 1) + s_view * cy
            pts = torch.stack([px, py], dim=-1)  # [c, N, S, 2]
            for occ_fn, out in ((self._ncmap.occupied, out1),
                                (self._ncmap.occupied_los, out2)):
                occ = occ_fn(pts)                # [c, N, S] bool
                s_hit = torch.where(occ, s_view, big_t)
                dmin = s_hit.amin(dim=-1)        # [c, N]
                out[lo:hi] = torch.where(dmin < BIG, dmin, torch.full_like(dmin, R)).clamp(0.0, R)
        self._rays_m = out1
        self._rays_los_m = out2

    def _update_local_task_points(self, mask: Optional[torch.Tensor] = None) -> None:  # override
        """Deployment-style LOS local target (vectorised ``task._select_local_target``).

        The carrot is the visible point -- along the layer-2 LOS rays
        (``_rays_los_m``, robot_radius + extra) -- that is CLOSEST to the global
        goal. Unlike a clip-along-the-goal-line carrot, its bearing can steer the
        robot AROUND a pillar (the nearest-visible point need not lie on the
        straight goal line), so the layer-2 inflation actively shapes guidance --
        exactly how the minimalist deployment runner feeds the policy.

        Overlap fallback: when the robot centre is inside the layer-2 inflation
        (the deployment 'dist < 0' case) the nearest-visible projection degenerates,
        so we steer along the most heading-aligned open ray instead.

        Only the policy consumes this (via ref_feat / task_dist); DWA navigates
        off the global goal in ``snapshot_for_dwa`` and is unaffected.
        """
        if self.n_rays <= 0:
            new = self._global_task_xy
        else:
            ar = torch.arange(self.B, device=self.device)
            pos = self.pos_xy                                   # [B, 2]
            goal = self._global_task_xy                         # [B, 2]
            ang = self.yaw.view(self.B, 1) + self._ray_ang.view(1, self.n_rays)  # [B, N]
            dirx = torch.cos(ang)
            diry = torch.sin(ang)
            los = self._rays_los_m.clamp(min=0.0)               # [B, N]
            gx = (goal[:, 0] - pos[:, 0]).unsqueeze(1)          # [B, 1]
            gy = (goal[:, 1] - pos[:, 1]).unsqueeze(1)
            # Closest point on each ray segment [0, los_i] to the goal.
            t = (gx * dirx + gy * diry).clamp(min=0.0)
            t = torch.minimum(t, los)                           # [B, N]
            cpx = pos[:, 0].unsqueeze(1) + t * dirx             # [B, N]
            cpy = pos[:, 1].unsqueeze(1) + t * diry
            d2 = (cpx - goal[:, 0].unsqueeze(1)) ** 2 + (cpy - goal[:, 1].unsqueeze(1)) ** 2
            best = d2.argmin(dim=1)                             # [B]
            nx = cpx[ar, best]
            ny = cpy[ar, best]
            # Overlap fallback (robot inside layer-2 inflation).
            overlap = self._ncmap.occupied_los(pos)             # [B] bool
            if bool(overlap.any()):
                hx = torch.cos(self.yaw).unsqueeze(1)
                hy = torch.sin(self.yaw).unsqueeze(1)
                align = dirx * hx + diry * hy                   # [B, N]
                bh = align.argmax(dim=1)                        # [B]
                dh = los[ar, bh]
                nx = torch.where(overlap, pos[:, 0] + dh * dirx[ar, bh], nx)
                ny = torch.where(overlap, pos[:, 1] + dh * diry[ar, bh], ny)
            new = torch.stack([nx, ny], dim=-1)                 # [B, 2]

        if mask is None:
            self._local_task_xy[:, 0] = new[:, 0]
            self._local_task_xy[:, 1] = new[:, 1]
        else:
            self._local_task_xy[mask, 0] = new[mask, 0]
            self._local_task_xy[mask, 1] = new[mask, 1]

    # ------------------------------------------------------------------
    # Episodic step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, action: torch.Tensor):  # override
        if action.shape[0] != self.B or action.dim() != 2 or action.shape[1] not in (2, 3):
            raise AssertionError(
                f"expected action shape [{self.B},2] or [{self.B},3], got {tuple(action.shape)}")
        if action.device != self.device:
            action = action.to(self.device, non_blocking=True)

        retired = self.retired
        active = ~retired

        vx_max = float(self.cfg.vx_max)
        om_max = float(self.cfg.omega_max)
        dt = float(self.cfg.dt)

        vx_cmd = action[:, 0].clamp(self._vx_lo, self._vx_hi)
        om_cmd = action[:, -1].clamp(-om_max, om_max)
        # Retired envs stand still (no movement, no spurious arrivals).
        vx_cmd = torch.where(active, vx_cmd, torch.zeros_like(vx_cmd))
        om_cmd = torch.where(active, om_cmd, torch.zeros_like(om_cmd))

        pos_prev = self.pos_xy.clone()
        yaw_end = _wrap_angle_pi(self.yaw + om_cmd * dt)
        c1 = torch.cos(yaw_end)
        s1 = torch.sin(yaw_end)
        vx_w = c1 * vx_cmd
        vy_w = s1 * vx_cmd
        pos_new = torch.stack([pos_prev[:, 0] + vx_w * dt, pos_prev[:, 1] + vy_w * dt], dim=-1)

        # --- collision.
        if self._ep.collision_mode == "footprint":
            # Any sample on the swept segment lands in the inflated map (all
            # directions -- catches lateral corner clipping; the realistic rule).
            collided = self._segment_hits(pos_prev, pos_new)
        else:
            # Training-consistent: travel exceeds the motion-direction ray
            # (cast at pos_prev during the previous _resample). Forward-only
            # motion => motion angle 0 => the forward ray.
            travel = torch.hypot(vx_w * dt, vy_w * dt)
            motion_ang = torch.atan2(vy_w, vx_w)  # world; body-frame motion is along +x
            # Body-frame motion angle relative to heading is ~0 for forward-only.
            body_ang = _wrap_angle_pi(motion_ang - yaw_end)
            ray_along = self._interp_ray_distance(self._rays_m, body_ang)
            collided = (travel - ray_along > 1e-6) & (ray_along > 0.0)

        # --- arrival: closest point on the segment to the goal within radius.
        goal = self._global_task_xy
        u = pos_new - pos_prev
        uu = (u * u).sum(dim=-1)
        w0 = goal - pos_prev
        t_proj = torch.zeros_like(uu)
        moving = uu > 0.0
        t_proj[moving] = ((w0[moving, 0] * u[moving, 0] + w0[moving, 1] * u[moving, 1]) / uu[moving]).clamp(0.0, 1.0)
        near_x = pos_prev[:, 0] + t_proj * u[:, 0]
        near_y = pos_prev[:, 1] + t_proj * u[:, 1]
        dist2_near = (near_x - goal[:, 0]) ** 2 + (near_y - goal[:, 1]) ** 2
        r_s = self.success_radius_m
        arrived = dist2_near <= (r_s * r_s)

        # Commit motion and bookkeeping.
        self.pos_xy = pos_new
        self.yaw = yaw_end
        self.vel_xy[:, 0] = vx_w
        self.vel_xy[:, 1] = vy_w
        move_d = torch.hypot(u[:, 0], u[:, 1])
        self.cur_path = self.cur_path + torch.where(active, move_d, torch.zeros_like(move_d))
        self.cur_steps = self.cur_steps + active.to(torch.long)
        self.t = self.t + active.to(self.t.dtype)
        timeout = self.cur_steps >= self.max_steps

        # Secondary reward (progress toward goal minus collision), for reference.
        d_prev = torch.hypot(goal[:, 0] - pos_prev[:, 0], goal[:, 1] - pos_prev[:, 1])
        d_next = torch.hypot(goal[:, 0] - pos_new[:, 0], goal[:, 1] - pos_new[:, 1])
        progress = (d_prev - d_next) / max(vx_max * dt, 1e-9)
        reward = (self.cfg.w_progress * progress
                  - self.cfg.w_collision * collided.to(torch.float32))
        reward = torch.where(active, reward, torch.zeros_like(reward))

        self.prev_prev_cmd.copy_(self.prev_cmd)
        self.prev_cmd.copy_(torch.stack([vx_cmd, torch.zeros_like(vx_cmd), om_cmd], dim=-1))

        # Outcome priority: collision (failure) > arrival > timeout.
        done = (collided | arrived | timeout) & active
        outcome = torch.where(
            collided, torch.full_like(self.ep_k, OUTCOME_COLLISION, dtype=torch.int8),
            torch.where(arrived,
                        torch.full_like(self.ep_k, OUTCOME_SUCCESS, dtype=torch.int8),
                        torch.full_like(self.ep_k, OUTCOME_TIMEOUT, dtype=torch.int8)))

        if bool(done.any()):
            gid = self._gid().clamp(max=self.T - 1)
            d_idx = done.nonzero(as_tuple=False).squeeze(-1)
            g_sel = gid[d_idx]
            self.res_outcome[g_sel] = outcome[d_idx]
            self.res_steps[g_sel] = self.cur_steps[d_idx].to(torch.int32)
            self.res_path[g_sel] = self.cur_path[d_idx]
            geo_sel = self.geo_m[g_sel]
            self.res_geo[g_sel] = geo_sel
            final_d = torch.hypot(goal[d_idx, 0] - pos_new[d_idx, 0],
                                  goal[d_idx, 1] - pos_new[d_idx, 1])
            self.res_goaldist[g_sel] = final_d
            succ = (outcome[d_idx] == OUTCOME_SUCCESS).to(torch.float32)
            denom = torch.maximum(self.cur_path[d_idx], geo_sel).clamp_min(1e-6)
            self.res_spl[g_sel] = succ * (geo_sel / denom).clamp(0.0, 1.0)

            # Advance the finished envs to their next episode (or retire them).
            self.ep_k[d_idx] = self.ep_k[d_idx] + 1
            respawn = done & (~self.retired)
            self._assign_episode(respawn)

        self._resample_fov_and_ref()
        self._maybe_log_path0()
        obs_next = self.observe()
        info: Dict[str, Any] = {
            "collided": collided & active,
            "arrived": arrived & active,
            "timeout": timeout & active,
            "finished": done,
            "retired": retired,
            "reward": reward,
        }
        return obs_next, reward, done, info

    @torch.no_grad()
    def snapshot_for_dwa(self) -> Dict[str, torch.Tensor]:
        """DWA inputs (zero host sync): goal in robot frame + rays + prev command.

        Mirrors ``eval_env.EvalEnv.snapshot_for_dwa`` so the DWA planner consumes
        the same tensors here as on the random training env -- the only
        difference is that ``_rays_m`` comes from the real raycast.
        """
        pos = self.pos_xy
        yaw = self.yaw
        tgt = self._global_task_xy
        dx = tgt[:, 0] - pos[:, 0]
        dy = tgt[:, 1] - pos[:, 1]
        c = torch.cos(yaw)
        s = torch.sin(yaw)
        tx_local = c * dx + s * dy
        ty_local = -s * dx + c * dy
        return {
            "rays_m": self._rays_m,
            "target_x_local": tx_local,
            "target_y_local": ty_local,
            "vx_cur": self.prev_cmd[:, 0],
            "omega_cur": self.prev_cmd[:, 2],
        }

    def _segment_hits(self, p0: torch.Tensor, p1: torch.Tensor) -> torch.Tensor:
        """True per env if the swept segment p0->p1 touches the inflated map."""
        move = p1 - p0
        n_sub = self._n_seg_sub
        ts = torch.linspace(0.0, 1.0, n_sub, device=self.device).view(1, n_sub, 1)
        pts = p0.unsqueeze(1) + ts * move.unsqueeze(1)  # [B, n_sub, 2]
        return self._ncmap.occupied(pts).any(dim=1)

    def _maybe_log_path0(self) -> None:
        if not self._ep.track_path0:
            return
        if int(self.ep_k[0].item()) != 0:
            return
        p = self.pos_xy[0]
        self.path0.append([float(p[0].item()), float(p[1].item())])

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def metrics(self) -> Dict[str, Any]:
        """Aggregate the per-episode result buffers (single host sync)."""
        outcome = self.res_outcome.cpu().numpy()
        steps = self.res_steps.cpu().numpy()
        path = self.res_path.cpu().numpy()
        geo = self.res_geo.cpu().numpy()
        spl = self.res_spl.cpu().numpy()
        goaldist = self.res_goaldist.cpu().numpy()

        filled = outcome != OUTCOME_UNSET
        n = int(filled.sum())
        succ = outcome == OUTCOME_SUCCESS
        coll = outcome == OUTCOME_COLLISION
        tout = outcome == OUTCOME_TIMEOUT
        n_succ = int(succ.sum())

        def _safe_mean(x, m):
            xm = x[m]
            return float(xm.mean()) if xm.size else 0.0

        dt = float(self.cfg.dt)
        return {
            "n_episodes": n,
            "episodes_per_env": int(self.episodes_per_env),
            "n_envs": self.B,
            "success_rate": (n_succ / n) if n else 0.0,
            "collision_rate": (int(coll.sum()) / n) if n else 0.0,
            "timeout_rate": (int(tout.sum()) / n) if n else 0.0,
            "spl_mean": float(spl[filled].mean()) if n else 0.0,
            "steps_success_mean": _safe_mean(steps, succ),
            "time_success_mean_s": _safe_mean(steps, succ) * dt,
            "path_success_mean_m": _safe_mean(path, succ),
            "geo_success_mean_m": _safe_mean(geo, succ),
            "goaldist_fail_mean_m": _safe_mean(goaldist, ~succ & filled),
            "n_success": n_succ,
            "n_collision": int(coll.sum()),
            "n_timeout": int(tout.sum()),
        }
