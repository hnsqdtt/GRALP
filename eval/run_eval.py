from __future__ import annotations

"""Evaluation entry point with three mutually exclusive modes:

A) Single checkpoint (GUI selector if --ckpt is omitted):
       python -m eval.run_eval
       python -m eval.run_eval --ckpt path/to/step-100000.pt
       python -m eval.run_eval --ckpt stray.pt --model cnn_circular

B) Scan a run directory and eval every step-*.pt against DWA:
       python -m eval.run_eval --run-dir runs/20260514-mlp_3_seed0

C) Pure DWA baseline (no learned policy):
       python -m eval.run_eval --dwa-only

DWA reward and PPO reward both come from env.step() so the comparison goes
through the exact same SimRandomGPUBatchEnv reward pipeline.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from eval.ckpt_loader import (
    build_policy_for_eval,
    load_policy_weights,
    resolve_ckpt_configs,
)
from eval.dwa.planner import DWAConfig, DWAPlanner
from eval.dwa_runner import run_dwa
from eval.eval_env import EvalEnv, EvalEnvSpec
from eval.policy_runner import run_policy


REPO = Path(__file__).resolve().parents[1]
DEFAULT_ENV_CFG = REPO / "config" / "env_config.json"
DEFAULT_DWA_CFG = REPO / "eval" / "dwa_config.json"

_STEP_CKPT_RE = re.compile(r"^step-(\d+)\.pt$")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _reseed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _gui_pick_ckpt() -> Optional[Path]:
    """Pop up a Tk file dialog to pick a .pt; return None if canceled."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as e:
        print(f"[eval] tkinter not available ({e}); pass --ckpt <path> instead", file=sys.stderr)
        return None
    root = tk.Tk()
    root.withdraw()
    init_dir = REPO / "runs"
    path = filedialog.askopenfilename(
        title="Select PPO checkpoint (.pt)",
        filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")],
        initialdir=str(init_dir) if init_dir.exists() else str(REPO),
    )
    root.destroy()
    if not path:
        return None
    return Path(path)


def _progress_factory():
    """Return a callable matching ``tqdm(iterable, total=..., desc=...)`` if
    available, else a no-op that just returns the iterable."""
    try:
        from tqdm import tqdm
        return lambda it, total=None, desc=None: tqdm(it, total=total, desc=desc)
    except ImportError:
        return lambda it, total=None, desc=None: it


def _format_row(name: str, res: Dict[str, Any], width_name: int) -> str:
    return (f"  {name:<{width_name}}"
            f"  reward={res['reward_mean']:+.4f} +/- {res['reward_std']:.4f}"
            f"  collision={res['collision_mean']:.4f} +/- {res['collision_std']:.4f}"
            f"  success={res['success_mean']:.4f}"
            f"  ({res['fps']/1000:.1f}k env-steps/s)")


def _print_summary(rows: List[Tuple[str, Dict[str, Any]]], *, title: str = "Evaluation summary") -> None:
    if not rows:
        return
    width = max(len(name) for name, _ in rows)
    bar = "=" * max(70, width + 50)
    print()
    print(bar)
    print(f" {title}")
    print(bar)
    for name, res in rows:
        print(_format_row(name, res, width))
    print(bar)


def _build_env_from_cfg(env_cfg: Dict[str, Any], args: argparse.Namespace) -> EvalEnv:
    spec = EvalEnvSpec(
        env_cfg=env_cfg,
        n_envs=args.n_envs,
        seed=args.seed,
        device=args.device,
    )
    env = EvalEnv(spec)
    print(f"[eval] Env  | device={env.device}  B={env.B}  N_rays={env.n_rays}  seed={args.seed}")
    return env


def _build_dwa_planner(env_cfg: Dict[str, Any], device: torch.device) -> DWAPlanner:
    dwa_cfg_dict = _load_json(DEFAULT_DWA_CFG)
    dwa_cfg = DWAConfig.from_configs(env_cfg, dwa_cfg_dict)
    print(f"[eval] DWA  | v=[{dwa_cfg.v_min},{dwa_cfg.v_max}]  omega=+/-{dwa_cfg.omega_max}  "
          f"alpha/beta/gamma={dwa_cfg.alpha_heading}/{dwa_cfg.beta_clearance}/{dwa_cfg.gamma_velocity}  "
          f"grid={dwa_cfg.v_samples}x{dwa_cfg.omega_samples}")
    return DWAPlanner(dwa_cfg, device=device)


# ---------------------------------------------------------------------------
# Mode C: DWA only
# ---------------------------------------------------------------------------


def cmd_dwa_only(args: argparse.Namespace) -> int:
    env_cfg = _load_json(args.env_config)
    env = _build_env_from_cfg(env_cfg, args)
    planner = _build_dwa_planner(env_cfg, env.device)

    print()
    print(f"[eval] Running DWA: {args.n_rollouts} rollouts x {args.rollout_len} steps x {args.n_envs} envs")
    _reseed(args.seed)
    res = run_dwa(env, planner,
                  rollout_len=args.rollout_len,
                  n_rollouts=args.n_rollouts,
                  verbose=not args.quiet)
    _print_summary([("DWA", res)], title="DWA baseline")
    return 0


# ---------------------------------------------------------------------------
# Mode A: single checkpoint (with optional GUI picker)
# ---------------------------------------------------------------------------


def cmd_single_ckpt(args: argparse.Namespace, ckpt: Path) -> int:
    train_cfg, env_cfg, model_configs, model_key, source = resolve_ckpt_configs(
        ckpt, model_override=args.model
    )
    print(f"[eval] Ckpt | {ckpt}  (configs from {source}, model={model_key!r})")

    env = _build_env_from_cfg(env_cfg, args)
    obs = env.reset()
    vec_dim = int(obs.shape[1])
    policy = build_policy_for_eval(train_cfg, model_configs, model_key, vec_dim, env.device)
    step = load_policy_weights(policy, ckpt, env.device)
    print(f"[eval] Pol  | vec_dim={vec_dim}  feature_dim={policy.encoder.feature_dim}  step={step:,}")

    planner = _build_dwa_planner(env_cfg, env.device)
    progress = _progress_factory()

    print()
    print(f"[eval] Running DWA baseline ...")
    _reseed(args.seed)
    dwa_res = run_dwa(env, planner,
                      rollout_len=args.rollout_len,
                      n_rollouts=args.n_rollouts,
                      verbose=False)

    print(f"[eval] Running PPO ckpt (deterministic) ...")
    _reseed(args.seed)
    ppo_res = run_policy(env, policy,
                         rollout_len=args.rollout_len,
                         n_rollouts=args.n_rollouts,
                         deterministic=True,
                         verbose=False,
                         progress_factory=progress)

    ckpt_label = f"PPO/{model_key} @ step={step:,}" if step > 0 else f"PPO/{model_key} ({ckpt.stem})"
    _print_summary([("DWA", dwa_res), (ckpt_label, ppo_res)],
                   title=f"Single-ckpt eval ({ckpt.name})")
    return 0


# ---------------------------------------------------------------------------
# Mode B: run-dir scan
# ---------------------------------------------------------------------------


def cmd_run_dir(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    for fname in ("train_config.json", "env_config.json", "model_config.json"):
        if not (run_dir / fname).is_file():
            print(f"[eval] FAILED: missing {fname} in {run_dir}", file=sys.stderr)
            return 2
    train_cfg = _load_json(run_dir / "train_config.json")
    env_cfg = _load_json(run_dir / "env_config.json")
    model_configs = _load_json(run_dir / "model_config.json")
    snapshot_key = str(train_cfg.get("model", "gralp_attn"))
    model_key = args.model or snapshot_key
    if args.model and args.model != snapshot_key:
        print(f"[eval] NOTE: --model {args.model!r} overrides snapshot model {snapshot_key!r}")

    ckpts: List[Tuple[int, Path]] = []
    for p in run_dir.iterdir():
        m = _STEP_CKPT_RE.match(p.name)
        if m:
            ckpts.append((int(m.group(1)), p))
    ckpts.sort(key=lambda t: t[0])
    if not ckpts:
        print(f"[eval] FAILED: no step-*.pt under {run_dir}", file=sys.stderr)
        return 2
    print(f"[eval] Run  | {run_dir.name}  model={model_key!r}  ckpts={len(ckpts)}  "
          f"steps=[{ckpts[0][0]:,} ... {ckpts[-1][0]:,}]")

    env = _build_env_from_cfg(env_cfg, args)
    obs = env.reset()
    vec_dim = int(obs.shape[1])
    policy = build_policy_for_eval(train_cfg, model_configs, model_key, vec_dim, env.device)
    print(f"[eval] Pol  | vec_dim={vec_dim}  feature_dim={policy.encoder.feature_dim}")

    planner = _build_dwa_planner(env_cfg, env.device)
    progress = _progress_factory()

    print()
    print(f"[eval] Running DWA baseline ...")
    _reseed(args.seed)
    dwa_res = run_dwa(env, planner,
                      rollout_len=args.rollout_len,
                      n_rollouts=args.n_rollouts,
                      verbose=False)
    print(f"  DWA: reward={dwa_res['reward_mean']:+.4f}+/-{dwa_res['reward_std']:.4f}  "
          f"collision={dwa_res['collision_mean']:.4f}")

    rows: List[Tuple[str, Dict[str, Any]]] = [("DWA (baseline)", dwa_res)]

    print(f"[eval] Sweeping {len(ckpts)} checkpoints ...")
    ckpt_iter = progress(ckpts, total=len(ckpts), desc="ckpts")
    for step, ckpt_path in ckpt_iter:
        load_policy_weights(policy, ckpt_path, env.device)
        _reseed(args.seed)
        res = run_policy(env, policy,
                         rollout_len=args.rollout_len,
                         n_rollouts=args.n_rollouts,
                         deterministic=True,
                         verbose=False)
        rows.append((f"step={step:,}", res))

    _print_summary(rows, title=f"Run-dir sweep ({run_dir.name})")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRALP evaluation: DWA baseline + PPO ckpt comparison")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--ckpt", type=Path, help="Single .pt to evaluate against DWA")
    g.add_argument("--run-dir", type=Path, help="Sweep every step-*.pt in a run directory")
    g.add_argument("--dwa-only", action="store_true", help="Just the DWA baseline (no PPO ckpt)")

    p.add_argument("--model", type=str, default=None,
                   help="Encoder name from model_config.json (required for stray .pt with no snapshot)")

    p.add_argument("--n-envs", type=int, default=24)
    p.add_argument("--rollout-len", type=int, default=256)
    p.add_argument("--n-rollouts", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None,
                   help="cuda / cuda:0 / cpu (default: cuda if available)")

    p.add_argument("--env-config", type=Path, default=DEFAULT_ENV_CFG,
                   help="Only used by --dwa-only (other modes read env_config from the run dir / ckpt snapshot)")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_cli()
    if args.dwa_only:
        return cmd_dwa_only(args)
    if args.run_dir is not None:
        return cmd_run_dir(args)

    ckpt = args.ckpt
    if ckpt is None:
        print("[eval] No --ckpt / --run-dir / --dwa-only given; opening file picker ...")
        ckpt = _gui_pick_ckpt()
        if ckpt is None:
            print("[eval] No checkpoint selected. Use --ckpt, --run-dir, or --dwa-only.", file=sys.stderr)
            return 1
    if not ckpt.is_file():
        print(f"[eval] FAILED: ckpt not found: {ckpt}", file=sys.stderr)
        return 2
    return cmd_single_ckpt(args, ckpt)


if __name__ == "__main__":
    raise SystemExit(main())
