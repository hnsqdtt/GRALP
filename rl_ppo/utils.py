from __future__ import annotations

import os
import sys
from typing import Optional
import numpy as np

from env import load_json_config  # type: ignore



def infer_vec_dim(env_cfg_path: str, mission_cfg_path: Optional[str]) -> int:
    """Launch an environment via MissionPlanner and return the flattened vector observation dimension."""
    try:
        from mission import MissionPlanner
    except Exception:
        _MIS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'agent interface'))
        if os.path.isdir(_MIS_DIR) and _MIS_DIR not in sys.path:
            sys.path.append(_MIS_DIR)
        from mission import MissionPlanner  # type: ignore

    mpn = MissionPlanner(env_cfg_path, mission_cfg_path, generator_module=None, generator_kwargs=None)
    try:
        obs_cfg = mpn.env_cfg.get("obs", {}) or {}
        obs_cfg["mode"] = "vector"
        if "rays_align_yaw" not in obs_cfg:
            obs_cfg["rays_align_yaw"] = True
        mpn.env_cfg["obs"] = obs_cfg
    except Exception:
        pass
    env = mpn.respawn_env(current_env=None, rotate_map=True, seed=None)
    obs = env.reset()
    vec_dim = int(np.asarray(obs).reshape(-1).shape[0])
    return vec_dim
