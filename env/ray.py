from __future__ import annotations

from typing import Tuple, Sequence, Dict, Any
import numpy as np


def raycast(occ: np.ndarray, origin: Tuple[float, float], direction: Tuple[float, float], max_dist: float, step: float = 0.5) -> float:
    """Simple raycast on an occupancy grid; returns hit distance or max_dist.
    - occ: H×W boolean grid (True means obstacle)
    - origin: pixel coordinate (x, y)
    - direction: normalized direction (dx, dy)
    """
    ox, oy = origin
    dx, dy = direction
    if dx == 0 and dy == 0:
        return 0.0
    H, W = occ.shape
    dist = 0.0
    while dist < max_dist:
        x = int(round(ox + dx * dist))
        y = int(round(oy + dy * dist))
        if y < 0 or y >= H or x < 0 or x >= W:
            return dist
        if occ[y, x]:
            return dist
        dist += step
    return max_dist


def radial_scan(occ: np.ndarray,
                origin: Tuple[float, float],
                n_rays: int = 36,
                max_dist: float = 50.0,
                step: float = 0.25,
                fov_deg: float = 360.0,
                start_angle_deg: float = 0.0) -> np.ndarray:
    """Cast equally spaced rays around the origin and return distances.

    - occ: H×W boolean occupancy (True means obstacle)
    - origin: pixel coordinate (x, y)
    - n_rays: number of rays
    - max_dist: maximum distance (pixels)
    - step: sampling step along each ray (pixels)
    - fov_deg: field of view in degrees, default 360°
    - start_angle_deg: start angle in degrees; 0° points right, counter-clockwise is positive
    """
    if n_rays <= 0:
        return np.zeros((0,), dtype=np.float32)
    angles = np.deg2rad(start_angle_deg + np.linspace(0.0, max(0.0, fov_deg), num=n_rays, endpoint=False))
    dx = np.cos(angles)
    dy = np.sin(angles)
    dists = np.empty((n_rays,), dtype=np.float32)
    for i in range(n_rays):
        dists[i] = float(raycast(occ, origin, (dx[i], dy[i]), max_dist=max_dist, step=step))
    return dists


def compute_ray_defaults(obs_cfg: Dict[str, Any], patch_meters: float) -> Tuple[int, float, float]:
    """Derive default ray parameters from observation config and view distance.

    - obs_cfg:
      - ray_max_gap: used to derive the number of rays (meters). <=0 returns 0 rays.
      - ray_step_m: sampling step along each ray (meters), default 0.025.
    - patch_meters: view distance / radar radius (meters).

    Returns (n_rays, ray_step_m, view_radius_m).
    """
    step_m = float(obs_cfg.get("ray_step_m", 0.025))
    radius_m = float(patch_meters)
    max_gap = float(obs_cfg.get("ray_max_gap", 0.0))
    if max_gap > 0.0 and radius_m > 0.0:
        n_rays = int(np.ceil((2.0 * np.pi * radius_m) / max(max_gap, 1e-9)))
    else:
        n_rays = 0
    return n_rays, step_m, radius_m
