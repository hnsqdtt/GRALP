from __future__ import annotations

"""
Lightweight PPO inference (strict input validation).
- Loads limits/model config from `config.json`; default weight resolution: policy.onnx -> latest.onnx -> final.onnx.
- `execution_provider` controls ONNX Runtime EP (cpu/cuda/tensorrt), default cpu.
- Observations: rays in meters normalized to [0,1]; pose tail 7 dims [sin_ref, cos_ref, prev_vx/vx_max, prev_omega/omega_max, dvx/(2*vx_max), domega/(2*omega_max), task_dist/patch_meters].
- Actions: 2-D [vx, omega] in SI units.
"""

import json
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np


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
    execution_provider: str = "cpu"

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
            execution_provider=str(data.get("execution_provider", "cpu")),
        )

    def expected_rays(self) -> int:
        """Derive ray count R from (2*pi*patch_meters)/ray_max_gap; raises on invalid params."""
        if self.patch_meters <= 0 or self.ray_max_gap <= 0:
            raise ValueError("patch_meters and ray_max_gap must be > 0")
        return int(np.ceil((2.0 * np.pi * float(self.patch_meters)) / max(float(self.ray_max_gap), 1e-9)))


def _ensure_2d_rays(rays: ArrayLike) -> np.ndarray:
    arr = np.asarray(rays, dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(1, -1)  # shape [1, R]
    if arr.ndim != 2:
        raise ValueError("rays must be 1D [R] or 2D [B,R]")
    return arr


def _validate_and_normalize_rays_m(rays_m: np.ndarray, patch_meters: float) -> np.ndarray:
    """Validate rays are finite within [0, patch_meters] then normalize to [0,1]."""
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
    # Prefer new-style separate components; legacy pair is still accepted
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

    # Accept both legacy (prev_cmd/prev_prev_cmd) and new-style inputs
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

    # Normalize previous commands by axis limits (vx / omega)
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

    # Pose tail layout: [sin_ref, cos_ref, prev_vx, prev_omega, dvx, domega, task_dist]
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
        # Device hint: if provided it overrides config.execution_provider; otherwise config controls provider
        self.device = device

        self._session = None
        self._vec_dim: Optional[int] = None
        self._obs_input_name: Optional[str] = None
        self._limits_input_name: Optional[str] = None
        self._action_out_idx: Optional[int] = None
        self._mu_out_idx: Optional[int] = None
        self._log_std_out_idx: Optional[int] = None

        # Expected ray count derived from config
        self._expected_rays: int = self.cfg.expected_rays()

        # Limits tensor cached lazily per batch size
        self._limits_np = self._build_limits_np()

    def _build_limits_np(self) -> np.ndarray:
        # Fixed action limits layout: [vx, omega]
        return np.array([self.cfg.vx_max, self.cfg.omega_max], dtype=np.float32)

    def _resolve_default_ckpt(self, base_dir: str) -> str:
        """Resolve default weights: config.ckpt_filename -> policy.onnx -> latest.onnx -> final.onnx."""
        candidates = []
        if self.cfg.ckpt_filename:
            if isinstance(self.cfg.ckpt_filename, str):
                candidates.append(self.cfg.ckpt_filename)
            elif isinstance(self.cfg.ckpt_filename, list):
                candidates.extend([str(x) for x in self.cfg.ckpt_filename])
        candidates.extend(["policy.onnx", "latest.onnx", "final.onnx"])

        for name in candidates:
            p = name
            if not os.path.isabs(p):
                p = os.path.join(base_dir, p)
            if os.path.exists(p):
                return p

        raise FileNotFoundError(
            f"No checkpoint found. Tried: {', '.join(str(os.path.join(base_dir, n)) if not os.path.isabs(n) else n for n in candidates)}."
        )

    def _select_providers(self, ort) -> List[str]:
        """Choose ONNX Runtime providers based on config/device hint and availability."""
        available = list(ort.get_available_providers())

        pref = str(self.cfg.execution_provider or "cpu").strip().lower()
        dev = str(self.device or "").strip().lower()
        # Only override config if caller explicitly passed a device hint
        if dev:
            if "tensorrt" in dev or "tensor" in dev or "trt" in dev:
                pref = "tensorrt"
            elif dev.startswith("cuda") or dev.startswith("gpu"):
                pref = "cuda"
            elif dev.startswith("cpu"):
                pref = "cpu"

        def _ordered(names: List[str]) -> List[str]:
            out: List[str] = []
            for n in names:
                if n in available and n not in out:
                    out.append(n)
            return out

        if pref.startswith("tensor"):
            prov = _ordered(["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"])
        elif pref.startswith("cuda") or pref.startswith("gpu"):
            prov = _ordered(["CUDAExecutionProvider", "CPUExecutionProvider"])
        else:
            prov = _ordered(["CPUExecutionProvider"])

        if not prov:
            prov = available  # fallback to whatever ORT exposes
        return prov

    def _load_session(self):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for ONNX inference. Install with `pip install onnxruntime` "
                "or `pip install onnxruntime-gpu` depending on your environment."
            ) from exc

        if not self._ckpt_path.lower().endswith(".onnx"):
            raise ValueError(f"ONNX backend expects an .onnx file; got {self._ckpt_path}")

        providers = self._select_providers(ort)
        try:
            return ort.InferenceSession(self._ckpt_path, providers=providers)
        except Exception as exc:
            raise RuntimeError(f"Failed to load ONNX model from {self._ckpt_path}: {exc}") from exc

    @staticmethod
    def _find_output_idx(outputs, keywords: Tuple[str, ...]) -> Optional[int]:
        for idx, out in enumerate(outputs):
            name = out.name.lower()
            if any(k in name for k in keywords):
                return idx
        return None

    @staticmethod
    def _pick_obs_input(inputs) -> Optional[str]:
        if not inputs:
            return None
        for kw in ("obs", "input", "data", "x"):
            for inp in inputs:
                if kw in inp.name.lower():
                    return inp.name
        return inputs[0].name

    @staticmethod
    def _pick_limits_input(inputs) -> Optional[str]:
        for inp in inputs:
            nm = inp.name.lower()
            shape = inp.shape
            last_dim = None
            if isinstance(shape, (list, tuple)) and len(shape) > 0:
                last = shape[-1]
                if isinstance(last, (int, float)):
                    last_dim = int(last)
            if "limit" in nm or "bound" in nm or "scale" in nm:
                return inp.name
            if last_dim == 2:
                return inp.name
        return None

    def _introspect_onnx_io(self) -> None:
        assert self._session is not None
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        self._obs_input_name = self._pick_obs_input(inputs)
        self._limits_input_name = self._pick_limits_input(inputs)
        self._action_out_idx = self._find_output_idx(outputs, ("action", "actions", "act"))
        self._mu_out_idx = self._find_output_idx(outputs, ("mu", "mean"))
        self._log_std_out_idx = self._find_output_idx(outputs, ("log_std", "logstd", "logsigma", "log_sigma"))

        if self._obs_input_name is None:
            raise RuntimeError("Failed to locate observation input name in ONNX model.")

    def _ensure_session(self, vec_dim: int) -> None:
        if self._session is None or self._vec_dim != vec_dim:
            self._vec_dim = int(vec_dim)
            self._session = self._load_session()
            self._introspect_onnx_io()

    def _limits_for_batch(self, B: int) -> np.ndarray:
        return np.tile(self._limits_np.reshape(1, self._limits_np.shape[0]), (B, 1)).astype(np.float32)

    @staticmethod
    def _ensure_batch_2d(arr: np.ndarray) -> np.ndarray:
        out = np.asarray(arr, dtype=np.float32)
        if out.ndim == 1:
            out = out.reshape(1, -1)
        return out

    def _compute_action_from_outputs(
        self,
        outputs: List[np.ndarray],
        *,
        limits_bx2: np.ndarray,
        deterministic: bool,
    ) -> np.ndarray:
        action = None
        mu = None
        log_std = None

        if self._action_out_idx is not None and self._action_out_idx < len(outputs):
            action = outputs[self._action_out_idx]
        if self._mu_out_idx is not None and self._mu_out_idx < len(outputs):
            mu = outputs[self._mu_out_idx]
        if self._log_std_out_idx is not None and self._log_std_out_idx < len(outputs):
            log_std = outputs[self._log_std_out_idx]

        if action is None and mu is None:
            raise RuntimeError(
                "ONNX model outputs must include either an action tensor or (mu, log_std). "
                f"Found outputs: {[getattr(o, 'shape', None) for o in outputs]}"
            )

        if action is not None:
            act = self._ensure_batch_2d(action)
        else:
            mu_ba = self._ensure_batch_2d(mu)
            if deterministic or log_std is None:
                act = np.tanh(mu_ba) * limits_bx2
            else:
                log_std_ba = self._ensure_batch_2d(log_std)
                if log_std_ba.shape != mu_ba.shape:
                    log_std_ba = np.broadcast_to(log_std_ba, mu_ba.shape).astype(np.float32)
                eps = np.random.normal(size=mu_ba.shape).astype(np.float32)
                pre_tanh = mu_ba + np.exp(log_std_ba) * eps
                act = np.tanh(pre_tanh) * limits_bx2

        return act.astype(np.float32)

    def _run_onnx(self, obs_vec: np.ndarray, limits_bx2: np.ndarray, deterministic: bool) -> np.ndarray:
        assert self._session is not None
        feed = {self._obs_input_name: obs_vec.astype(np.float32)}
        if self._limits_input_name:
            feed[self._limits_input_name] = limits_bx2.astype(np.float32)
        try:
            outputs = self._session.run(None, feed)
        except Exception as exc:
            raise RuntimeError(f"ONNX inference failed: {exc}") from exc

        return self._compute_action_from_outputs(outputs, limits_bx2=limits_bx2, deterministic=deterministic)

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
        """Run inference and return actions in SI units."""
        rays_bxR = _ensure_2d_rays(rays_m)  # shape [B, R]
        B, R = int(rays_bxR.shape[0]), int(rays_bxR.shape[1])

        # Enforce ray length derived from config when available
        if self._expected_rays > 0 and R != self._expected_rays:
            raise ValueError(
                f"rays_m length mismatch: got {R}, expected {self._expected_rays} from (2*pi*patch_meters)/ray_max_gap"
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
        rays_n = _validate_and_normalize_rays_m(rays_bxR, self.cfg.patch_meters)  # shape [B, R]
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
        )  # shape [B, obs_pose_dim]

        if pose.shape[1] != 7:
            raise ValueError(f"Pose tail dim mismatch: got {pose.shape[1]}, expected 7")

        obs = np.concatenate([rays_n, pose], axis=-1).astype(np.float32)  # shape [B, R+pose_dim]
        self._ensure_session(vec_dim=R + pose.shape[1])
        lim = self._limits_for_batch(B)

        act = self._run_onnx(obs, limits_bx2=lim, deterministic=deterministic)
        if act.shape[-1] != 2:
            raise ValueError(f"Expected action dimension 2 (vx, omega); got shape {act.shape}")
        if act.shape[0] == 1 and np.asarray(rays_m).ndim == 1:
            return act[0]
        return act
