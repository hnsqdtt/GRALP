"""Lightweight environment API for PPO training with only the randomized GPU simulator.

This package only exposes utilities required by `rl_ppo/ppo_train.py`:
- utils: `load_json_config`, `LOGGER`
- sim_gpu_env: `SimGPUEnvConfig`, `SimRandomGPUBatchEnv`, `infer_obs_dim`

Traditional map-based environment modules are removed to simplify the current training flow.
"""

from .utils import LOGGER, load_json_config
__all__ = [
    "LOGGER",
    "load_json_config",
]
