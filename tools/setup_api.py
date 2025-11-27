#!/usr/bin/env python3
"""
Set up the ONNX API package.

- Recreates <root>/ppo_api from tools/api_example.
- Syncs config.json with config/env_config.json and config/train_config.json.
- Exports a PPO checkpoint (.pt) to ONNX inside the new folder so it matches tools/api_example/README.md expectations.
"""

from __future__ import annotations

import json
import math
import importlib.util
import shutil
import sys
from pathlib import Path
import re
from typing import Optional, Tuple, Set

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def copy_api_example_to_root(root: Path) -> Path:
    src = root / "tools" / "api_example"
    dst = root / "ppo_api"
    if not src.exists():
        raise FileNotFoundError(f"Missing source folder: {src}")
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    shutil.copytree(src, dst)
    return dst


def _check_dependencies() -> None:
    missing = []
    if importlib.util.find_spec("onnxruntime") is None:
        missing.append("onnxruntime (or onnxruntime-gpu)")
    # onnx is commonly required alongside onnxruntime tools/CLIs
    if importlib.util.find_spec("onnx") is None:
        missing.append("onnx")
    if missing:
        raise RuntimeError(
            "Missing required dependencies: "
            + ", ".join(missing)
            + ". Please install with `pip install onnx onnxruntime` or `pip install onnx onnxruntime-gpu`."
        )


def _scan_fallback_ray_max_gap(root: Path) -> float:
    """Try to infer a reasonable ray_max_gap from source defaults."""
    try:
        sim_gpu_env_py = root / "env" / "sim_gpu_env.py"
        if sim_gpu_env_py.exists():
            text = sim_gpu_env_py.read_text(encoding="utf-8", errors="ignore")
            m = re.search(r"ray_max_gap\s*:\s*float\s*=\s*([0-9]*\.?[0-9]+)", text)
            if m:
                return float(m.group(1))
    except Exception:
        pass
    try:
        ppo_train_py = root / "rl_ppo" / "ppo_train.py"
        if ppo_train_py.exists():
            text = ppo_train_py.read_text(encoding="utf-8", errors="ignore")
            m = re.search(r"ray_max_gap'\s*,\s*([0-9]*\.?[0-9]+)\)\)", text)
            if m:
                return float(m.group(1))
    except Exception:
        pass
    return 0.6


def _derive_ray_and_obs_dim(api_cfg: dict) -> Tuple[int, int]:
    patch_meters = float(api_cfg.get("patch_meters", 0.0))
    ray_max_gap = float(api_cfg.get("ray_max_gap", 0.0))
    if patch_meters <= 0.0 or ray_max_gap <= 0.0:
        raise ValueError("patch_meters and ray_max_gap must be > 0 to derive ray count for ONNX export.")
    rays = int(math.ceil((2.0 * math.pi * patch_meters) / max(ray_max_gap, 1e-9)))
    return rays, rays + 7


def _resolve_ckpt_dir(root: Path, train_cfg: dict) -> Path:
    run_cfg = (train_cfg.get("run") or {})
    ckpt_dir = run_cfg.get("ckpt_dir", "runs")
    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.is_absolute():
        ckpt_path = root / ckpt_path
    return ckpt_path


def _select_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    if not ckpt_dir.exists():
        return None

    def _latest_by_mtime(paths):
        try:
            return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)
        except Exception:
            return list(paths)

    latest = _latest_by_mtime(ckpt_dir.rglob("latest.pt"))
    final = _latest_by_mtime(ckpt_dir.rglob("final.pt"))
    others = _latest_by_mtime(p for p in ckpt_dir.rglob("*.pt") if p.name not in ("latest.pt", "final.pt"))

    for group in (latest, final, others):
        if group:
            return group[0]
    return None


def _list_onnx_files(dir_path: Path) -> Set[Path]:
    if not dir_path.exists():
        return set()
    return {p.resolve() for p in dir_path.rglob("*.onnx")}


def _clean_new_onnx_in_ckpt_dir(ckpt_dir: Path, before: Set[Path]) -> None:
    after = _list_onnx_files(ckpt_dir)
    new_files = [p for p in after if p not in before]
    if not new_files:
        return
    for p in new_files:
        try:
            p.unlink()
            print(f"[setup_api] Removed stray ONNX in ckpt_dir: {p}")
        except Exception as exc:
            print(f"[setup_api] Warning: failed to remove new ONNX file {p}: {exc}")


def update_api_config_from_env_and_train(root: Path, env_cfg: dict, train_cfg: dict) -> dict:
    api_cfg_p = root / "ppo_api" / "config.json"

    if not api_cfg_p.exists():
        raise FileNotFoundError(f"Not found: {api_cfg_p} (did you copy api_example?)")

    api_cfg = _load_json(api_cfg_p)

    limits = (env_cfg.get("limits") or {})
    sim = (env_cfg.get("sim") or {})
    obs = (env_cfg.get("obs") or {})
    model = (train_cfg.get("model") or {})

    def fnum(v, default):
        try:
            return float(v)
        except Exception:
            return float(default)

    def inum(v, default):
        try:
            return int(v)
        except Exception:
            return int(default)

    sanitized_cfg = {
        "vx_max": fnum(limits.get("vx_max", api_cfg.get("vx_max", 1.5)), 1.5),
        "vy_max": fnum(limits.get("vy_max", api_cfg.get("vy_max", 0.0)), 0.0),
        "omega_max": fnum(limits.get("omega_max", api_cfg.get("omega_max", 2.0)), 2.0),
        "dt": fnum(sim.get("dt", api_cfg.get("dt", 0.1)), 0.1),
        "patch_meters": fnum(obs.get("patch_meters", api_cfg.get("patch_meters", 10.0)), 10.0),
        "num_queries": inum(model.get("num_queries", api_cfg.get("num_queries", 4)), 4),
        "num_heads": inum(model.get("num_heads", api_cfg.get("num_heads", 4)), 4),
    }

    if isinstance(obs.get("ray_max_gap", None), (int, float)):
        sanitized_cfg["ray_max_gap"] = fnum(obs.get("ray_max_gap"), api_cfg.get("ray_max_gap", 0.6))
    else:
        fallback_gap = _scan_fallback_ray_max_gap(root)
        sanitized_cfg["ray_max_gap"] = fnum(api_cfg.get("ray_max_gap", fallback_gap), fallback_gap)

    ckpt_name = api_cfg.get("ckpt_filename", "policy.onnx")
    if isinstance(ckpt_name, list):
        ckpt_name = ckpt_name[0] if ckpt_name else "policy.onnx"
    ckpt_name = str(ckpt_name)
    if not ckpt_name.lower().endswith(".onnx"):
        ckpt_name = "policy.onnx"
    ckpt_name = Path(ckpt_name).name
    sanitized_cfg["ckpt_filename"] = ckpt_name

    sanitized_cfg["execution_provider"] = str(api_cfg.get("execution_provider", "cpu") or "cpu")

    _dump_json(api_cfg_p, sanitized_cfg)
    return sanitized_cfg


def _load_policy_state_dict(ckpt_path: Path):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to load checkpoints and export ONNX.") from exc

    payload = torch.load(ckpt_path, map_location="cpu")
    if isinstance(payload, dict):
        if isinstance(payload.get("policy", None), dict):
            return payload["policy"]
        if isinstance(payload.get("state_dict", None), dict):
            return payload["state_dict"]
        keys = list(payload.keys())
        if keys and all(isinstance(k, str) for k in keys):
            # Heuristic: a bare state_dict usually has parameter-like keys
            if any(k.startswith(("encoder", "mu", "value", "log_std")) or "." in k for k in keys):
                return payload
    raise ValueError(f"Unrecognized checkpoint format at {ckpt_path}")


def _export_policy_to_onnx(api_dst: Path, api_cfg: dict, train_cfg: dict, ckpt_path: Path) -> Path:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to export ONNX.") from exc

    from rl_ppo.ppo_models import PPOPolicy

    rays, obs_dim = _derive_ray_and_obs_dim(api_cfg)
    model_cfg = (train_cfg.get("model") or {})
    ppo_cfg = (train_cfg.get("ppo") or {})
    num_queries = int(model_cfg.get("num_queries", api_cfg.get("num_queries", 4)))
    num_heads = int(model_cfg.get("num_heads", api_cfg.get("num_heads", 4)))
    log_std_min = float(ppo_cfg.get("log_std_min", -5.0))
    log_std_max = float(ppo_cfg.get("log_std_max", 2.0))

    policy = PPOPolicy(
        vec_dim=obs_dim,
        action_dim=2,
        num_queries=num_queries,
        num_heads=num_heads,
        log_std_min=log_std_min,
        log_std_max=log_std_max,
    )
    state_dict = _load_policy_state_dict(ckpt_path)
    policy.load_state_dict(state_dict, strict=False)
    policy.eval()

    class PPOOnnxWrapper(torch.nn.Module):
        def __init__(self, p: PPOPolicy) -> None:
            super().__init__()
            self.policy = p

        def forward(self, obs_vec: torch.Tensor, limits: torch.Tensor):
            mu, log_std, _ = self.policy._core(obs_vec)
            action = torch.tanh(mu) * limits
            return action, mu, log_std

    wrapper = PPOOnnxWrapper(policy)

    dummy_obs = torch.zeros((1, obs_dim), dtype=torch.float32)
    dummy_limits = torch.tensor(
        [
            [float(api_cfg["vx_max"]), float(api_cfg["omega_max"])],
        ],
        dtype=torch.float32,
    )

    ckpt_name = Path(str(api_cfg.get("ckpt_filename", "policy.onnx"))).name
    onnx_path = api_dst / ckpt_name

    # Clear template weights to avoid mixing with newly exported ONNX.
    for stale in (onnx_path, onnx_path.with_suffix(onnx_path.suffix + ".data")):
        try:
            if stale.exists():
                stale.unlink()
        except Exception:
            pass

    dynamic_axes = {
        "obs": {0: "batch"},
        "limits": {0: "batch"},
        "action": {0: "batch"},
        "mu": {0: "batch"},
        "log_std": {0: "batch"},
    }

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_obs, dummy_limits),
            onnx_path,
            input_names=["obs", "limits"],
            output_names=["action", "mu", "log_std"],
            dynamic_shapes=dynamic_axes,
            opset_version=18,
        )

    print(f"[setup_api] Exported ONNX with {rays} rays (obs_dim={obs_dim}) -> {onnx_path}")
    return onnx_path


def main(argv: list[str]) -> int:
    root = REPO_ROOT

    print(f"[setup_api] Repo root: {root}")

    _check_dependencies()

    env_cfg_p = root / "config" / "env_config.json"
    train_cfg_p = root / "config" / "train_config.json"
    if not env_cfg_p.exists():
        raise FileNotFoundError(f"Not found: {env_cfg_p}")
    if not train_cfg_p.exists():
        raise FileNotFoundError(f"Not found: {train_cfg_p}")

    env_cfg = _load_json(env_cfg_p)
    train_cfg = _load_json(train_cfg_p)

    api_dst = copy_api_example_to_root(root)
    print(f"[setup_api] Recreated {api_dst} from tools/api_example (old removed)")

    api_cfg = update_api_config_from_env_and_train(root, env_cfg, train_cfg)
    print(f"[setup_api] Updated {api_dst / 'config.json'} from config/env_config.json and config/train_config.json")

    ckpt_dir = _resolve_ckpt_dir(root, train_cfg)
    pre_export_onnx = _list_onnx_files(ckpt_dir)
    ckpt_path = _select_checkpoint(ckpt_dir)
    if ckpt_path is None:
        raise FileNotFoundError(f"No .pt checkpoint found under {ckpt_dir}")
    print(f"[setup_api] Using checkpoint for export: {ckpt_path}")

    onnx_path = _export_policy_to_onnx(api_dst, api_cfg, train_cfg, ckpt_path)
    print(f"[setup_api] ONNX ready at {onnx_path}")

    _clean_new_onnx_in_ckpt_dir(ckpt_dir, pre_export_onnx)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
