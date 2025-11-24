#!/usr/bin/env python3
"""
从 tools/api_example 同步 API 到 <root>/ppo_api，并用
config/env_config.json 与 config/train_config.json 的值更新 ppo_api/config.json。

用法
  python tools/setup_api.py  # 在仓库根目录或任意位置执行

行为
  - 自动定位仓库根（脚本所在目录的上一级）。
  - 删除已有 <root>/ppo_api 后整体拷贝 tools/api_example → <root>/ppo_api。
  - 读取 <root>/config/env_config.json 与 train_config.json。
  - 更新 <root>/ppo_api/config.json 中的：vx_max, vy_max, omega_max, dt,
    patch_meters, ray_max_gap, num_queries, num_heads。
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
import re


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: Path, data: dict) -> None:
    # Preserve stable, readable formatting
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def copy_api_example_to_root(root: Path) -> Path:
    src = root / "tools" / "api_example"
    dst = root / "ppo_api"
    if not src.exists():
        raise FileNotFoundError(f"Missing source folder: {src}")
    # Default behavior: clean the existing api directory before copying
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    shutil.copytree(src, dst)
    return dst


def _scan_fallback_ray_max_gap(root: Path) -> float:
    """若 env_config.json 未提供数值型 ray_max_gap，尝试从源码默认值中回退解析。

    优先级：
      1) env/sim_gpu_env.py: SimGPUEnvConfig.ray_max_gap 的默认值
      2) rl_ppo/ppo_train.py: obs_cfg.get('ray_max_gap', <default>) 的默认值
      3) 固定兜底 0.6

    返回解析到的 float 值。
    """
    # 1) parse dataclass default in env/sim_gpu_env.py
    try:
        sim_gpu_env_py = (root / "env" / "sim_gpu_env.py")
        if sim_gpu_env_py.exists():
            text = sim_gpu_env_py.read_text(encoding="utf-8", errors="ignore")
            m = re.search(r"ray_max_gap\s*:\s*float\s*=\s*([0-9]*\.?[0-9]+)", text)
            if m:
                return float(m.group(1))
    except Exception:
        pass
    # 2) parse trainer hardcoded default in rl_ppo/ppo_train.py
    try:
        ppo_train_py = (root / "rl_ppo" / "ppo_train.py")
        if ppo_train_py.exists():
            text = ppo_train_py.read_text(encoding="utf-8", errors="ignore")
            m = re.search(r"ray_max_gap'\s*,\s*([0-9]*\.?[0-9]+)\)\)", text)
            if m:
                return float(m.group(1))
    except Exception:
        pass
    # 3) final fallback
    return 0.6


def update_api_config_from_env_and_train(root: Path) -> None:
    cfg_dir = root / "config"
    env_cfg_p = cfg_dir / "env_config.json"
    train_cfg_p = cfg_dir / "train_config.json"
    api_cfg_p = root / "ppo_api" / "config.json"

    if not env_cfg_p.exists():
        raise FileNotFoundError(f"Not found: {env_cfg_p}")
    if not train_cfg_p.exists():
        raise FileNotFoundError(f"Not found: {train_cfg_p}")
    if not api_cfg_p.exists():
        raise FileNotFoundError(f"Not found: {api_cfg_p} (did you copy api_example?)")

    env_cfg = _load_json(env_cfg_p)
    train_cfg = _load_json(train_cfg_p)
    api_cfg = _load_json(api_cfg_p)

    # Extract values with safe defaults and types
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
        # 训练环境中仅使用 (vx, omega) 控制；vy_max 仅保留用于 API 兼容。
        # 若 env_config 中未提供 vy_max，则与当前网络结构对齐到 0.0。
        "vy_max": fnum(limits.get("vy_max", api_cfg.get("vy_max", 0.0)), 0.0),
        "omega_max": fnum(limits.get("omega_max", api_cfg.get("omega_max", 2.0)), 2.0),
        "dt": fnum(sim.get("dt", api_cfg.get("dt", 0.1)), 0.1),
        "patch_meters": fnum(obs.get("patch_meters", api_cfg.get("patch_meters", 10.0)), 10.0),
        "num_queries": inum(model.get("num_queries", api_cfg.get("num_queries", 4)), 4),
        "num_heads": inum(model.get("num_heads", api_cfg.get("num_heads", 4)), 4),
    }

    # 额外同步 obs.ray_max_gap（决定射线数量 R 的关键参数之一），若缺失则搜索源码默认值
    if isinstance(obs.get("ray_max_gap", None), (int, float)):
        sanitized_cfg["ray_max_gap"] = fnum(obs.get("ray_max_gap"), api_cfg.get("ray_max_gap", 0.6))
    else:
        fallback_gap = _scan_fallback_ray_max_gap(root)
        sanitized_cfg["ray_max_gap"] = fnum(api_cfg.get("ray_max_gap", fallback_gap), fallback_gap)

    # ckpt_filename 默认使用 latest.pt，若模板或已有配置提供则沿用
    ckpt_name = api_cfg.get("ckpt_filename", "latest.pt")
    if ckpt_name is None:
        ckpt_name = "latest.pt"
    sanitized_cfg["ckpt_filename"] = ckpt_name

    _dump_json(api_cfg_p, sanitized_cfg)


def _pick_newest(paths: list[Path]) -> Optional[Path]:
    if not paths:
        return None
    try:
        return max(paths, key=lambda p: p.stat().st_mtime)
    except Exception:
        # Fallback: first one
        return paths[0]


def populate_weights_from_runs(root: Path, api_dst: Path) -> None:
    """在 <root>/runs 递归查找 final.pt/latest.pt 并复制到 <api>。

    优先级：
      - 最新的 latest.pt → <api>/latest.pt（若存在）
      - 最新的 final.pt  → <api>/final.pt（若存在）
    若都不存在，保留模板中提供的 <api>/latest.pt。
    """
    runs_dir = root / "runs"
    latest_candidates: list[Path] = []
    final_candidates: list[Path] = []
    if runs_dir.exists():
        for p in runs_dir.rglob("*.pt"):
            if p.name == "latest.pt":
                latest_candidates.append(p)
            elif p.name == "final.pt":
                final_candidates.append(p)

    chosen_latest = _pick_newest(latest_candidates)
    chosen_final = _pick_newest(final_candidates)

    if chosen_latest is not None:
        shutil.copy2(chosen_latest, api_dst / "latest.pt")
        print(f"[setup_api] Copied latest.pt from {chosen_latest}")
    if chosen_final is not None:
        shutil.copy2(chosen_final, api_dst / "final.pt")
        print(f"[setup_api] Copied final.pt from {chosen_final}")

    # If none copied, ensure there's at least api/latest.pt (template already provided).
    tpl_latest = api_dst / "latest.pt"
    if not tpl_latest.exists():
        # Try to copy from template explicitly
        template_latest = root / "tools" / "api_example" / "latest.pt"
        if template_latest.exists():
            shutil.copy2(template_latest, tpl_latest)
            print("[setup_api] Using template latest.pt from tools/api_example")


def main(argv: list[str]) -> int:
    # Detect repo root = parent of this script's directory
    script_path = Path(__file__).resolve()
    root = script_path.parent.parent

    print(f"[setup_api] Repo root: {root}")

    # 1) Copy template API from tools/api_example to <root>/api
    api_dst = copy_api_example_to_root(root)
    print(f"[setup_api] Recreated {api_dst} from tools/api_example (old removed)")

    # 2) Update <root>/api/config.json using env/train configs
    update_api_config_from_env_and_train(root)
    print(f"[setup_api] Updated {api_dst / 'config.json'} from config/env_config.json and config/train_config.json")

    # 3) Populate weights into <root>/api from runs (or keep template latest.pt)
    populate_weights_from_runs(root, api_dst)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
