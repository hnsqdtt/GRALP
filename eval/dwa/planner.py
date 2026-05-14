from __future__ import annotations

"""Python wrapper around the C DWA planner (eval/dwa/dwa.c).

Loads the compiled shared library, exposes a typed ``plan_step`` function that
takes a flat ray observation in the robot frame and returns a chosen
``(vx, omega)`` action. The native config is built from ``env_config.json`` +
``eval/dwa_config.json``; overlapping fields (v_max / omega_max / dt / etc.)
are auto-derived from env_config so the DWA action limits always match the env.
"""

import ctypes
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import numpy as np


_HERE = Path(__file__).resolve().parent


def _resolve_library() -> Path:
    if sys.platform == "win32":
        candidate = _HERE / "dwa.dll"
    elif sys.platform == "darwin":
        candidate = _HERE / "libdwa.dylib"
    else:
        candidate = _HERE / "libdwa.so"
    if not candidate.exists():
        raise FileNotFoundError(
            f"DWA shared library not found at {candidate}. "
            "Build it first with: python -m eval.dwa.setup_dwa"
        )
    return candidate


class _DWAConfigC(ctypes.Structure):
    _fields_ = [
        ("dt", ctypes.c_double),
        ("predict_time", ctypes.c_double),
        ("v_min", ctypes.c_double),
        ("v_max", ctypes.c_double),
        ("omega_max", ctypes.c_double),
        ("v_acc_max", ctypes.c_double),
        ("omega_acc_max", ctypes.c_double),
        ("v_brake_acc", ctypes.c_double),
        ("omega_brake_acc", ctypes.c_double),
        ("v_samples", ctypes.c_int),
        ("omega_samples", ctypes.c_int),
        ("alpha_heading", ctypes.c_double),
        ("beta_clearance", ctypes.c_double),
        ("gamma_velocity", ctypes.c_double),
        ("robot_radius_m", ctypes.c_double),
        ("dist_clip_m", ctypes.c_double),
        ("rotate_away_mode", ctypes.c_int),
        ("predict_steps", ctypes.c_int),
    ]


class _DWAOutputC(ctypes.Structure):
    _fields_ = [
        ("vx_cmd", ctypes.c_double),
        ("omega_cmd", ctypes.c_double),
        ("score", ctypes.c_double),
        ("found", ctypes.c_int),
    ]


_lib: Optional[ctypes.CDLL] = None


def _load() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib
    lib = ctypes.CDLL(str(_resolve_library()))
    lib.dwa_plan.argtypes = [
        ctypes.POINTER(_DWAConfigC),
        ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(_DWAOutputC),
    ]
    lib.dwa_plan.restype = ctypes.c_int
    _lib = lib
    return lib


@dataclass
class DWAConfig:
    """Python-facing DWA config; mirrors ``DWAConfig`` in dwa.h.

    Build via ``DWAConfig.from_configs(env_cfg, dwa_cfg)`` to auto-derive the
    fields that must align with the environment (v_max/omega_max/dt/...).
    """

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

        # Dynamic-window width: env semantics allow one control step to reach
        # the full velocity range, so V_d effectively degenerates to V_s.
        v_acc = (vx_max - v_min) / max(dt, 1e-9)
        w_acc = (2.0 * omega_max) / max(dt, 1e-9)

        # Braking decelerations: tighten admissibility (V_a). Defaults follow
        # paper-scale values (Fox/Burgard/Thrun reports RHINO 60 cm/s^2 ≈
        # 0.6 m/s^2 and ~60 deg/s^2 ≈ 1.0 rad/s^2). Setting these too high
        # makes V_a vacuous: any non-immediate collision becomes "stoppable".
        v_brake = float(dwa_cfg.get("v_brake_acc", 0.5))
        w_brake = float(dwa_cfg.get("omega_brake_acc", 1.0))

        patch = float(obs.get("patch_meters", 10.0))

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
            dist_clip_m=patch,
            rotate_away_mode=bool(dwa_cfg.get("rotate_away_mode", True)),
            predict_steps=int(dwa_cfg.get("predict_steps", 20)),
        )

    def to_c(self) -> _DWAConfigC:
        return _DWAConfigC(
            dt=self.dt,
            predict_time=self.predict_time,
            v_min=self.v_min,
            v_max=self.v_max,
            omega_max=self.omega_max,
            v_acc_max=self.v_acc_max,
            omega_acc_max=self.omega_acc_max,
            v_brake_acc=self.v_brake_acc,
            omega_brake_acc=self.omega_brake_acc,
            v_samples=int(self.v_samples),
            omega_samples=int(self.omega_samples),
            alpha_heading=self.alpha_heading,
            beta_clearance=self.beta_clearance,
            gamma_velocity=self.gamma_velocity,
            robot_radius_m=self.robot_radius_m,
            dist_clip_m=self.dist_clip_m,
            rotate_away_mode=1 if self.rotate_away_mode else 0,
            predict_steps=int(self.predict_steps),
        )


class DWAPlanner:
    """Stateless DWA planner; reuses a single scratch buffer per instance."""

    def __init__(self, cfg: DWAConfig) -> None:
        self.cfg = cfg
        self._cfg_c = cfg.to_c()
        self._lib = _load()
        self._scratch: Optional[np.ndarray] = None

    def _ensure_scratch(self, n_rays: int) -> np.ndarray:
        # lines: 4 * n_rays doubles
        # per-candidate: 4 * NV * NW doubles
        need = 4 * n_rays + 4 * self.cfg.v_samples * self.cfg.omega_samples
        if self._scratch is None or self._scratch.size < need:
            self._scratch = np.empty(need, dtype=np.float64)
        return self._scratch

    def plan(self,
             vx_cur: float, omega_cur: float,
             target_x_local: float, target_y_local: float,
             ray_dists_m: np.ndarray,
             ray_angles: Optional[np.ndarray] = None) -> Tuple[float, float, float, bool]:
        """Plan one DWA step and return ``(vx, omega, score, found)``."""
        rd = np.ascontiguousarray(ray_dists_m, dtype=np.float64)
        if rd.ndim != 1:
            raise ValueError(f"ray_dists_m must be 1-D, got shape {rd.shape}")
        n_rays = int(rd.size)

        if ray_angles is not None:
            ra = np.ascontiguousarray(ray_angles, dtype=np.float64)
            if ra.shape != rd.shape:
                raise ValueError(f"ray_angles shape {ra.shape} != ray_dists {rd.shape}")
            ra_ptr = ra.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        else:
            ra_ptr = ctypes.POINTER(ctypes.c_double)()

        scratch = self._ensure_scratch(n_rays)
        out = _DWAOutputC()
        rc = self._lib.dwa_plan(
            ctypes.byref(self._cfg_c),
            ctypes.c_double(vx_cur), ctypes.c_double(omega_cur),
            ctypes.c_double(target_x_local), ctypes.c_double(target_y_local),
            rd.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ra_ptr,
            ctypes.c_int(n_rays),
            scratch.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(int(scratch.size)),
            ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError(f"dwa_plan returned {rc} (bad inputs or scratch too small)")
        return float(out.vx_cmd), float(out.omega_cmd), float(out.score), bool(out.found)
