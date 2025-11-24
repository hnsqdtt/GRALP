from __future__ import annotations

"""
轻量级 PPO 推理 API（严格输入校验，对齐当前训练模型）。
要点
- 读取同目录 `config.json` 获取限幅/模型参数，保持与导出脚本一致。
- 未显式传 `ckpt_path` 时，权重解析优先级：<api>/latest.pt → <api>/final.pt。
- 观测布局：射线为米制，内部裁剪并归一化；姿态尾部 7 维：
  [sin_ref, cos_ref, prev_vx/vx_max, prev_omega/omega_max,
   Δvx/(2*vx_max), Δomega/(2*omega_max), task_dist/patch_meters]
- 动作维度与训练对齐：2 轴 (vx, omega)。

用法示例
    from ppo_api.inference import PPOInference
    api = PPOInference()
    action = api.infer(
        rays_m=...,
        sin_ref=...,
        cos_ref=...,
        prev_vx=...,
        prev_omega=...,
        prev_prev_vx=...,
        prev_prev_omega=...,
        task_dist=...,  # 米制
    )
    print(action)  # numpy 数组: [vx, omega]
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .ppo_models import PPOPolicy


ArrayLike = Union[Sequence[float], np.ndarray]


@dataclass
class APIConfig:
    vx_max: float = 1.5
    vy_max: float = 0.0
    omega_max: float = 2.0
    dt: float = 0.1
    num_queries: int = 4
    num_heads: int = 4
    patch_meters: float = 10.0
    ray_max_gap: float = 0.6
    ckpt_filename: Optional[str] = None

    @staticmethod
    def from_json(path: str) -> "APIConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}

        return APIConfig(
            vx_max=float(data.get("vx_max", 1.5)),
            vy_max=float(data.get("vy_max", 0.0)),
            omega_max=float(data.get("omega_max", 2.0)),
            dt=float(data.get("dt", 0.1)),
            num_queries=int(data.get("num_queries", 4)),
            num_heads=int(data.get("num_heads", 4)),
            patch_meters=float(data.get("patch_meters", 10.0)),
            ray_max_gap=float(data.get("ray_max_gap", 0.6)),
            ckpt_filename=data.get("ckpt_filename", None),
        )

    def expected_rays(self) -> int:
        """Derive ray count R from config using (2π * patch_meters) / ray_max_gap.

        Returns 0 if either patch_meters or ray_max_gap is non-positive.
        """
        if self.patch_meters <= 0 or self.ray_max_gap <= 0:
            return 0
        return int(np.ceil((2.0 * np.pi * float(self.patch_meters)) / max(float(self.ray_max_gap), 1e-9)))


def _ensure_2d_rays(rays: ArrayLike) -> np.ndarray:
    arr = np.asarray(rays, dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(1, -1)  # [1, R]
    if arr.ndim != 2:
        raise ValueError("rays must be 1D [R] or 2D [B,R]")
    return arr


def _validate_and_normalize_rays_m(rays_m: np.ndarray, patch_meters: float) -> np.ndarray:
    """校验射线值有限且位于 [0, patch_meters]，然后归一化到 [0,1]。"""
    if patch_meters <= 0:
        raise ValueError("patch_meters must be > 0")
    if not np.isfinite(rays_m).all():
        bad = np.where(~np.isfinite(rays_m))
        raise ValueError(f"rays_m contains non-finite values at indices {bad}")
    if (rays_m < 0).any():
        bad = np.where(rays_m < 0)
        v = float(rays_m[bad[0][0], bad[1][0]] if rays_m.ndim == 2 else rays_m[bad[0][0]])
        raise ValueError(f"rays_m must be >= 0; found {v}")
    if (rays_m > patch_meters).any():
        bad = np.where(rays_m > patch_meters)
        v = float(rays_m[bad[0][0], bad[1][0]] if rays_m.ndim == 2 else rays_m[bad[0][0]])
        raise ValueError(f"rays_m must be <= patch_meters ({patch_meters}); found {v}")
    return (rays_m.astype(np.float32) / float(patch_meters)).astype(np.float32)


def _build_pose_features(
    B: int,
    sin_ref: Optional[ArrayLike],
    cos_ref: Optional[ArrayLike],
    prev_cmd: Optional[ArrayLike],
    prev_prev_cmd: Optional[ArrayLike],
    limits: Tuple[float, float, float],
    patch_meters: float,
    task_dist: Optional[ArrayLike],
    *,
    # New-style inputs (preferred): separate components; broadcastable to [B]
    prev_vx: Optional[ArrayLike] = None,
    prev_omega: Optional[ArrayLike] = None,
    prev_prev_vx: Optional[ArrayLike] = None,
    prev_prev_omega: Optional[ArrayLike] = None,
) -> np.ndarray:
    vx_max, vy_max, om_max = limits
    if patch_meters <= 0:
        raise ValueError("patch_meters must be > 0")

    def _to_1d_or_b(x: Optional[ArrayLike], name: str, dim: int) -> np.ndarray:
        if x is None:
            raise ValueError(f"{name} must be provided; got None")
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 0:
            arr = np.full((B, dim), float(arr), dtype=np.float32)
        elif arr.ndim == 1 and arr.shape[0] == dim:
            arr = np.tile(arr.reshape(1, dim), (B, 1))
        elif arr.ndim == 1 and arr.shape[0] == B:
            arr = arr.reshape(B, 1).astype(np.float32)
            if dim != 1:
                raise ValueError(f"{name} expects dim={dim}")
        elif arr.ndim == 2 and arr.shape[0] == B and arr.shape[1] == dim:
            pass
        else:
            raise ValueError(f"{name} shape invalid; got {arr.shape}, expected [B,{dim}] or broadcastable")
        if not np.isfinite(arr).all():
            bad = np.where(~np.isfinite(arr))
            raise ValueError(f"{name} contains non-finite values at indices {bad}")
        return arr.astype(np.float32)

    sinv = _to_1d_or_b(sin_ref, "sin_ref", 1)
    cosv = _to_1d_or_b(cos_ref, "cos_ref", 1)
    # Validate sin/cos magnitude and orthogonality (tolerance)
    if (np.abs(sinv) > 1.0 + 1e-4).any() or (np.abs(cosv) > 1.0 + 1e-4).any():
        raise ValueError("sin_ref/cos_ref must be within [-1,1]")
    mag = (sinv * sinv + cosv * cosv)
    if not np.allclose(mag, 1.0, atol=5e-2):
        raise ValueError("sin_ref^2 + cos_ref^2 must be close to 1 (tolerance 0.05)")

    def _to_B(x: Optional[ArrayLike], name: str) -> np.ndarray:
        if x is None:
            raise ValueError(f"{name} must be provided; got None")
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 0:
            arr = np.full((B,), float(arr), dtype=np.float32)
        elif arr.ndim == 1 and arr.shape[0] == B:
            pass
        elif arr.ndim == 2 and arr.shape == (B, 1):
            arr = arr.reshape(B)
        else:
            raise ValueError(f"{name} shape invalid; got {arr.shape}, expected scalar or [B]")
        if not np.isfinite(arr).all():
            bad = np.where(~np.isfinite(arr))
            raise ValueError(f"{name} contains non-finite values at indices {bad}")
        return arr.astype(np.float32)

    # Accept both legacy (prev_cmd/prev_prev_cmd) and new-style inputs (separate components)
    if prev_cmd is not None and prev_prev_cmd is not None:
        prev = _to_1d_or_b(prev_cmd, "prev_cmd", 3)
        prev_prev = _to_1d_or_b(prev_prev_cmd, "prev_prev_cmd", 3)
    else:
        # New-style: explicit components must all be provided
        if any(v is None for v in (prev_vx, prev_omega, prev_prev_vx, prev_prev_omega)):
            raise ValueError(
                "Provide either prev_cmd+prev_prev_cmd (legacy) or all of prev_vx, prev_omega, prev_prev_vx, prev_prev_omega (new)."
            )
        pvx = _to_B(prev_vx, "prev_vx")
        pom = _to_B(prev_omega, "prev_omega")
        ppvx = _to_B(prev_prev_vx, "prev_prev_vx")
        ppom = _to_B(prev_prev_omega, "prev_prev_omega")
        prev = np.stack([pvx, np.zeros_like(pvx), pom], axis=-1).astype(np.float32)
        prev_prev = np.stack([ppvx, np.zeros_like(ppvx), ppom], axis=-1).astype(np.float32)

    # Normalize prev command by axis limits (vx / omega only; vy is ignored)
    def _safe_div(x: np.ndarray, m: float) -> np.ndarray:
        return (x / max(m, 1e-9)).astype(np.float32)

    prev_vx_n = _safe_div(prev[:, 0], vx_max)
    prev_om_n = _safe_div(prev[:, 2], om_max)

    dprev = (prev - prev_prev).astype(np.float32)
    dvx_n = _safe_div(dprev[:, 0], 2.0 * vx_max)
    dom_n = _safe_div(dprev[:, 2], 2.0 * om_max)

    # Normalize task distance
    if task_dist is None:
        raise ValueError("task_dist must be provided")
    td = np.asarray(task_dist, dtype=np.float32)
    if td.ndim == 0:
        td = np.full((B,), float(td), dtype=np.float32)
    elif td.ndim == 1 and td.shape[0] == B:
        td = td.astype(np.float32)
    elif td.ndim == 2 and td.shape == (B, 1):
        td = td.reshape(B).astype(np.float32)
    else:
        raise ValueError(f"task_dist shape invalid; got {td.shape}, expected scalar or [B] or [B,1]")
    if not np.isfinite(td).all():
        bad = np.where(~np.isfinite(td))
        raise ValueError(f"task_dist contains non-finite values at indices {bad}")
    if (td < 0).any():
        bad = np.where(td < 0)
        v = float(td[bad[0][0]])
        raise ValueError(f"task_dist must be >= 0; found {v}")
    if (td > patch_meters).any():
        bad = np.where(td > patch_meters)
        v = float(td[bad[0][0]])
        raise ValueError(f"task_dist must be <= patch_meters ({patch_meters}); found {v}")
    task_dist_n = (td / float(patch_meters)).astype(np.float32).reshape(B, 1)

    # Pose tail: [sin_ref, cos_ref, prev_vx, prev_omega, dvx, domega, task_dist]
    prev_block = np.stack([prev_vx_n, prev_om_n], axis=-1).astype(np.float32)
    dprev_block = np.stack([dvx_n, dom_n], axis=-1).astype(np.float32)

    pose = np.concatenate([sinv, cosv, prev_block, dprev_block, task_dist_n], axis=-1).astype(np.float32)
    return pose


class PPOInference:
    def __init__(
        self,
        *,
        config_path: Optional[str] = None,
        ckpt_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self._config_path = config_path or os.path.join(base_dir, "config.json")
        self.cfg = APIConfig.from_json(self._config_path)
        self._ckpt_path = ckpt_path or self._resolve_default_ckpt(base_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self._policy: Optional[PPOPolicy] = None
        self._vec_dim: Optional[int] = None

        # Expected ray count derived from config
        self._expected_rays: int = self.cfg.expected_rays()

        # Limits tensor cached lazily per batch size
        self._limits_np = self._build_limits_np()

    def _build_limits_np(self) -> np.ndarray:
        # Fixed layout: [vx, omega]
        return np.array([self.cfg.vx_max, self.cfg.omega_max], dtype=np.float32)

    def _resolve_default_ckpt(self, base_dir: str) -> str:
        """按优先级解析默认权重：config.ckpt_filename → latest.pt → final.pt。"""
        candidates = []
        if self.cfg.ckpt_filename:
            if isinstance(self.cfg.ckpt_filename, str):
                candidates.append(self.cfg.ckpt_filename)
            elif isinstance(self.cfg.ckpt_filename, list):
                candidates.extend([str(x) for x in self.cfg.ckpt_filename])
        candidates.extend(["latest.pt", "final.pt"])

        for name in candidates:
            p = name
            if not os.path.isabs(p):
                p = os.path.join(base_dir, p)
            if os.path.exists(p):
                return p

        raise FileNotFoundError(
            f"No checkpoint found. Tried: {', '.join(str(os.path.join(base_dir, n)) if not os.path.isabs(n) else n for n in candidates)}."
        )

    def _ensure_policy(self, vec_dim: int) -> None:
        if self._policy is None or self._vec_dim != vec_dim:
            self._vec_dim = int(vec_dim)
            # Build policy; weights are length-agnostic along the ray axis
            pol = PPOPolicy(
                vec_dim=self._vec_dim,
                action_dim=2,
                num_queries=self.cfg.num_queries,
                num_heads=self.cfg.num_heads,
            ).to(self.device)
            pol.eval()

            # Load weights if available
            if self._ckpt_path and os.path.exists(self._ckpt_path):
                try:
                    payload = torch.load(self._ckpt_path, map_location="cpu")
                    state: Optional[Dict[str, Any]] = None
                    if isinstance(payload, dict):
                        state = payload.get("policy", None)
                        if state is None:
                            keys = list(payload.keys())
                            if all(isinstance(k, str) for k in keys) and any(k.startswith("encoder") or k.startswith("mu") for k in keys):
                                state = payload
                    if state is not None:
                        pol.load_state_dict(state, strict=False)
                except Exception:
                    # Fallback: leave random weights if loading fails
                    pass
            self._policy = pol

    @torch.no_grad()
    def infer(
        self,
        *,
        rays_m: ArrayLike,
        sin_ref: Optional[ArrayLike] = None,
        cos_ref: Optional[ArrayLike] = None,
        # Legacy (still supported):
        prev_cmd: Optional[ArrayLike] = None,
        prev_prev_cmd: Optional[ArrayLike] = None,
        # New-style (preferred): pass separate components (scalar or [B])
        prev_vx: Optional[ArrayLike] = None,
        prev_omega: Optional[ArrayLike] = None,
        prev_prev_vx: Optional[ArrayLike] = None,
        prev_prev_omega: Optional[ArrayLike] = None,
        task_dist: Optional[ArrayLike] = None,
        deterministic: bool = True,
    ) -> np.ndarray:
        """执行推理并返回 SI 单位动作。

        输入
        - rays_m: 1D [R] 或 2D [B,R]，“米”为单位；内部裁剪至 [0, patch_meters] 并归一化。
        - sin_ref, cos_ref: 参考方向特征；必须提供。
        - 历史指令：legacy 的 prev_cmd/prev_prev_cmd，或新式逐分量 prev_vx/prev_omega/prev_prev_vx/prev_prev_omega。
        - task_dist: 当前局部任务点距离（米），归一化为 task_dist / patch_meters。
        - deterministic: True 用 tanh(mu)*limits；否则按策略采样。

        返回
        - np.ndarray 形状 [B, action_dim]（若输入为 1D，则为 [action_dim]）。
        """
        rays_bxR = _ensure_2d_rays(rays_m)  # [B,R]
        B, R = int(rays_bxR.shape[0]), int(rays_bxR.shape[1])

        # Enforce ray length derived from config when available
        if self._expected_rays > 0 and R != self._expected_rays:
            raise ValueError(
                f"rays_m length mismatch: got {R}, expected {self._expected_rays} from (2π*patch_meters)/ray_max_gap"
            )

        # Strict presence checks for pose components
        if sin_ref is None or cos_ref is None:
            raise ValueError("sin_ref and cos_ref must be provided (no defaults).")
        # Require either legacy pair or all new-style components
        legacy_ok = (prev_cmd is not None and prev_prev_cmd is not None)
        new_ok = all(v is not None for v in (prev_vx, prev_omega, prev_prev_vx, prev_prev_omega))
        if not (legacy_ok or new_ok):
            raise ValueError(
                "Provide either prev_cmd+prev_prev_cmd (legacy) or all of prev_vx, prev_omega, prev_prev_vx, prev_prev_omega (new)."
            )
        if task_dist is None:
            raise ValueError("task_dist must be provided (task distance is required).")
        task_dist_val = task_dist if task_dist is not None else 0.0

        # Validate and normalize rays
        rays_n = _validate_and_normalize_rays_m(rays_bxR, self.cfg.patch_meters)  # [B,R]
        pose = _build_pose_features(
            B,
            sin_ref=sin_ref,
            cos_ref=cos_ref,
            prev_cmd=prev_cmd,
            prev_prev_cmd=prev_prev_cmd,
            limits=(self.cfg.vx_max, self.cfg.vy_max, self.cfg.omega_max),
            patch_meters=self.cfg.patch_meters,
            task_dist=task_dist_val,
            prev_vx=prev_vx,
            prev_omega=prev_omega,
            prev_prev_vx=prev_prev_vx,
            prev_prev_omega=prev_prev_omega,
        )  # [B, obs_pose_dim]

        if pose.shape[1] != 7:
            raise ValueError(f"Pose tail dim mismatch: got {pose.shape[1]}, expected 7")

        obs = np.concatenate([rays_n, pose], axis=-1).astype(np.float32)  # [B, R+pose_dim]
        self._ensure_policy(vec_dim=R + pose.shape[1])
        assert self._policy is not None

        v = torch.as_tensor(obs, dtype=torch.float32, device=self.device)  # [B, vec_dim]
        lim = torch.as_tensor(
            np.tile(self._limits_np.reshape(1, self._limits_np.shape[0]), (B, 1)),
            dtype=torch.float32,
            device=self.device,
        )  # [B, 2]

        if deterministic:
            mu, _, _ = self._policy._core(v)
            a = torch.tanh(mu) * lim
        else:
            out = self._policy.act(v, lim)
            a = out.action

        act = a.detach().cpu().numpy().astype(np.float32)  # [B, 2]
        if act.shape[0] == 1 and np.asarray(rays_m).ndim == 1:
            return act[0]
        return act
