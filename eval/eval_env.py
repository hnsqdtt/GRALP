from __future__ import annotations

"""Fixed-seed wrapper around SimRandomGPUBatchEnv for evaluation.

Why a wrapper?  Training uses huge batches (2048 envs) and rolls forever; eval
wants a small fixed batch (e.g. 24) under a known seed so DWA and every PPO
checkpoint see the same random ray traces. This wrapper:

- Builds SimGPUEnvConfig from a parsed env_config dict.
- Seeds torch (CPU + CUDA) before constructing the env, so the FOV resampling
  inside __init__/reset/step is reproducible per seed.
- Exposes a single ``snapshot_for_dwa()`` helper that returns all numpy inputs
  DWA needs (target in robot frame + ray distances + previous command) with
  one GPU->CPU sync per step instead of four scattered .cpu() calls.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from env import load_json_config
from env.sim_gpu_env import SimGPUEnvConfig, SimRandomGPUBatchEnv


def build_sim_cfg(env_cfg: Dict[str, Any], n_envs: int, device: str) -> SimGPUEnvConfig:
    obs_c = env_cfg.get("obs", {}) or {}
    sim_c = env_cfg.get("sim", {}) or {}
    lim_c = env_cfg.get("limits", {}) or {}
    rew_c = env_cfg.get("reward", {}) or {}
    safe_dist = float(sim_c.get("safe_distance", sim_c.get("warning_distance", 0.5)))
    return SimGPUEnvConfig(
        dt=float(sim_c.get("dt", 0.1)),
        n_envs=int(n_envs),
        patch_meters=float(obs_c.get("patch_meters", 10.0)),
        ray_step_m=float(obs_c.get("ray_step_m", 0.025)),
        n_rays=int(obs_c.get("n_rays", 0)),
        ray_max_gap=float(obs_c.get("ray_max_gap", 0.25)),
        safe_distance_m=safe_dist,
        vx_max=float(lim_c.get("vx_max", 1.5)),
        omega_max=float(lim_c.get("omega_max", 2.0)),
        w_collision=float(rew_c.get("reward_collision", 1.0)),
        w_progress=float(rew_c.get("reward_progress", 0.01)),
        orientation_verify=bool(rew_c.get("orientation_verify", False)),
        w_jerk=float(rew_c.get("reward_jerk", 0.0)),
        w_jerk_omega=float(rew_c.get("reward_jerk_omega", 0.0)),
        blank_ratio_base=float(obs_c.get("blank_ratio_base", 40.0)),
        blank_ratio_randmax=float(obs_c.get("blank_ratio_randmax", 40.0)),
        blank_ratio_std_ratio=float(obs_c.get("blank_ratio_std_ratio", 0.33)),
        narrow_passage_gaussian=bool(obs_c.get("narrow_passage_gaussian", False)),
        narrow_passage_std_ratio=float(obs_c.get("narrow_passage_std_ratio", 0.3)),
        device=str(device),
        task_point_max_dist_m=float(sim_c.get("task_point_max_dist_m", 8.0)),
        task_point_success_radius_m=float(sim_c.get("task_point_success_radius_m", 0.25)),
        task_point_random_interval_max=int(sim_c.get("task_point_random_interval_max", 0)),
        # For eval the env should not silently reset on collision: we want the
        # collision to register in the metric. Caller can override if needed.
        collision_done=True,
    )


@dataclass
class EvalEnvSpec:
    env_cfg_path: str
    n_envs: int = 24
    seed: int = 0
    device: Optional[str] = None  # None -> auto (cuda if available, else cpu)


class EvalEnv:
    """Thin reproducible wrapper over SimRandomGPUBatchEnv for evaluation."""

    def __init__(self, spec: EvalEnvSpec) -> None:
        self.spec = spec
        self.env_cfg = load_json_config(spec.env_cfg_path)

        device = spec.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_str = device

        # Seed everything before constructing the env so its init RNG draws are
        # deterministic for this run.
        torch.manual_seed(int(spec.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(spec.seed))

        sim_cfg = build_sim_cfg(self.env_cfg, spec.n_envs, device)
        self.env = SimRandomGPUBatchEnv(sim_cfg)
        self.device = self.env.device
        self.B = int(sim_cfg.n_envs)
        self.n_rays = int(self.env.n_rays)

    # ---- env interface forwards ---------------------------------------

    def reset(self) -> torch.Tensor:
        return self.env.reset()

    def step(self, action: torch.Tensor):
        return self.env.step(action)

    def get_limits(self) -> torch.Tensor:
        return self.env.get_limits()

    # ---- DWA-facing snapshot ------------------------------------------

    def snapshot_for_dwa(self) -> Dict[str, np.ndarray]:
        """Return the numpy float64 inputs DWA needs in a single sync.

        Computes the target offset in the robot frame on GPU first (one fused
        rotation), then copies a small batch of tensors over PCIe in one go.
        Keys: ``rays_m`` [B, N], ``target_x_local`` [B], ``target_y_local`` [B],
        ``vx_cur`` [B], ``omega_cur`` [B].
        """
        env = self.env
        pos = env.pos_xy
        yaw = env.yaw
        tgt = env._global_task_xy
        dx = tgt[:, 0] - pos[:, 0]
        dy = tgt[:, 1] - pos[:, 1]
        c = torch.cos(yaw)
        s = torch.sin(yaw)
        tx_local = c * dx + s * dy
        ty_local = -s * dx + c * dy

        rays_cpu = env._rays_m.detach().to("cpu", non_blocking=False).numpy().astype(np.float64, copy=False)
        tx_cpu = tx_local.detach().to("cpu", non_blocking=False).numpy().astype(np.float64, copy=False)
        ty_cpu = ty_local.detach().to("cpu", non_blocking=False).numpy().astype(np.float64, copy=False)
        vx_cpu = env.prev_cmd[:, 0].detach().to("cpu", non_blocking=False).numpy().astype(np.float64, copy=False)
        wc_cpu = env.prev_cmd[:, 2].detach().to("cpu", non_blocking=False).numpy().astype(np.float64, copy=False)
        return {
            "rays_m": rays_cpu,
            "target_x_local": tx_cpu,
            "target_y_local": ty_cpu,
            "vx_cur": vx_cpu,
            "omega_cur": wc_cpu,
        }

    def action_to_device(self, vx: np.ndarray, omega: np.ndarray) -> torch.Tensor:
        """Pack ``(vx, omega)`` numpy arrays into a ``[B,2]`` tensor on env.device."""
        act_np = np.stack([vx.astype(np.float32, copy=False),
                           omega.astype(np.float32, copy=False)], axis=-1)
        return torch.from_numpy(act_np).to(self.device, non_blocking=True)
