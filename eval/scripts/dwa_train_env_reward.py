from __future__ import annotations

"""Measure DWA's reward in the TRAINING environment at a given robot radius (rr).

One script == one experiment. This script answers a single question:

    With DWA's assumed robot radius set to ``rr``
    (``eval/dwa_config.json::robot_radius_m``), what reward does DWA collect in
    the same environment the PPO policy trains in?

Why this is apples-to-apples with training
-------------------------------------------
  * The env is built from ``config/env_config.json`` -- the exact obstacle
    distribution, reward weights and limits the policy sees while training.
  * ``rollout_len`` and ``collision_done`` default to the *training* values:
    ``rollout_len`` from ``config/train_config.json::sampling.rollout_len`` and
    ``collision_done=True`` (EvalEnv forces it; matches ``ppo.collision_done``).
  * Reward is whatever ``env.step()`` returns -- never recomputed here -- so the
    number is directly comparable to the policy's training reward.

What rr does (see ``eval/dwa/planner.py``)
------------------------------------------
A candidate DWA trajectory is rejected when the robot CENTRE comes within
``rr`` of an obstacle point. Smaller rr => DWA assumes a smaller body => it is
more aggressive and hugs obstacles closer. rr is a PLANNER assumption only; it
does NOT change the env's own collision rule.

Why not batch = 2048 (training's batch_env)?
--------------------------------------------
DWA's ``plan_batch`` allocates several ``[B, NC, 3*N]`` tensors
(NC = v_samples * omega_samples). Training runs B=2048 with a cheap *policy*
forward; DWA at B=2048 with the 21x41 grid would need tens of GB and OOM. The
per-step reward is a batch-size-independent expectation, so we use a smaller,
DWA-feasible batch and more rollouts -- statistically identical to training's
env, just split differently. Raise ``--n-envs`` only as far as VRAM allows.

Output:
    ``runs/dwa_train_env_reward/<timestamp>/result.json``

Usage:
    python -m eval.scripts.dwa_train_env_reward
    python -m eval.scripts.dwa_train_env_reward --rr 0.002 0.005
    python -m eval.scripts.dwa_train_env_reward --rr 0.005 --n-envs 64 --n-rollouts 20
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch

from eval.dwa.planner import DWAConfig, DWAPlanner

from .dwa_runner import run_dwa
from .eval_env import EvalEnv, EvalEnvSpec


REPO = Path(__file__).resolve().parents[2]
DEFAULT_ENV_CFG = REPO / "config" / "env_config.json"
DEFAULT_TRAIN_CFG = REPO / "config" / "train_config.json"
DEFAULT_DWA_CFG = REPO / "eval" / "dwa_config.json"
DEFAULT_OUT_DIR = REPO / "runs" / "dwa_train_env_reward"

# Robot radii (m) to evaluate. Default brackets the chosen baseline rr=0.005
# (eval/dwa_config.json) with a smaller, more aggressive point.
DEFAULT_RR: List[float] = [0.002, 0.005]


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _reseed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DWA reward in the training environment at a given robot radius (rr)")
    p.add_argument("--env-config", type=Path, default=DEFAULT_ENV_CFG)
    p.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CFG,
                   help="Used only to inherit sampling.rollout_len (the training value).")
    p.add_argument("--dwa-config", type=Path, default=DEFAULT_DWA_CFG)
    p.add_argument("--rr", type=float, nargs="+", default=None,
                   help=f"One or more robot_radius_m values to evaluate. "
                        f"Default: {DEFAULT_RR}")
    # DWA objective-weight overrides (default: inherit from --dwa-config).
    # These change the PLANNER's behavior (unlike the env's reward weights).
    p.add_argument("--alpha", type=float, default=None,
                   help="Override DWA alpha_heading (heading term weight).")
    p.add_argument("--beta", type=float, default=None,
                   help="Override DWA beta_clearance: raise it to make DWA "
                        "weight obstacle clearance more (stronger avoidance).")
    p.add_argument("--gamma", type=float, default=None,
                   help="Override DWA gamma_velocity: lower it to make DWA "
                        "less speed-greedy.")
    p.add_argument("--n-envs", type=int, default=256,
                   help="DWA-feasible batch. NOT training's 2048 (DWA would OOM); "
                        "per-step reward is batch-independent, so use more rollouts. "
                        "Footprint ~15.5 MB/env with the 21x41 grid (256 ~ 4 GB, "
                        "512 ~ 8 GB); lower it if you share the GPU.")
    p.add_argument("--rollout-len", type=int, default=None,
                   help="Default: sampling.rollout_len from --train-config.")
    p.add_argument("--n-rollouts", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None,
                   help="cuda / cpu / cuda:0 (default: cuda if available)")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                   help="Where to write result.json. A timestamp subdir is appended.")
    p.add_argument("--no-save", action="store_true",
                   help="Skip writing result.json; just print the stdout table.")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_cli()
    for label, path in (("env_config", args.env_config),
                        ("train_config", args.train_config),
                        ("dwa_config", args.dwa_config)):
        if not path.is_file():
            print(f"{label} not found: {path}", file=sys.stderr)
            return 2

    rr_list = list(args.rr) if args.rr else list(DEFAULT_RR)
    if any(rr <= 0.0 for rr in rr_list):
        print(f"all --rr values must be > 0 (got {rr_list})", file=sys.stderr)
        return 2

    env_cfg = _load_json(args.env_config)
    train_cfg = _load_json(args.train_config)
    dwa_cfg = _load_json(args.dwa_config)

    sampling = (train_cfg.get("sampling") or {})
    rollout_len = int(args.rollout_len if args.rollout_len is not None
                      else sampling.get("rollout_len", 256))

    base_cfg = DWAConfig.from_configs(env_cfg, dwa_cfg)
    # Apply objective-weight overrides onto base_cfg so every rr iteration's
    # DWAConfig(**base_cfg.__dict__) copy inherits them.
    if args.alpha is not None:
        base_cfg.alpha_heading = float(args.alpha)
    if args.beta is not None:
        base_cfg.beta_clearance = float(args.beta)
    if args.gamma is not None:
        base_cfg.gamma_velocity = float(args.gamma)

    print(f"Env cfg:    {args.env_config}  (= training env)")
    print(f"Train cfg:  {args.train_config}  -> rollout_len={rollout_len}")
    print(f"DWA cfg:    {args.dwa_config}  "
          f"v=[{base_cfg.v_min},{base_cfg.v_max}]  omega=+/-{base_cfg.omega_max}  "
          f"grid={base_cfg.v_samples}x{base_cfg.omega_samples}  "
          f"alpha/beta/gamma={base_cfg.alpha_heading}/{base_cfg.beta_clearance}/{base_cfg.gamma_velocity}")
    print(f"rr sweep:   {rr_list}  (eval/dwa_config.json default = {base_cfg.robot_radius_m})")
    print(f"Eval:       {args.n_rollouts} rollouts x {rollout_len} steps x {args.n_envs} envs "
          f"per rr (seed={args.seed})")
    print()

    rows: List[Dict[str, Any]] = []
    for i, rr in enumerate(rr_list, start=1):
        # Override only robot_radius_m; everything else inherits from base_cfg.
        cfg = DWAConfig(**base_cfg.__dict__)
        cfg.robot_radius_m = float(rr)

        # Fresh env + reseed per rr so each rr sees the same env trajectory.
        _reseed(args.seed)
        env = EvalEnv(EvalEnvSpec(
            env_cfg=env_cfg,
            n_envs=args.n_envs,
            seed=args.seed,
            device=args.device,
        ))
        planner = DWAPlanner(cfg, device=env.device)
        planner.reset_stats()

        print(f"[{i}/{len(rr_list)}] rr={rr:.4f} m")
        t0 = time.perf_counter()
        res = run_dwa(env, planner,
                      rollout_len=rollout_len,
                      n_rollouts=args.n_rollouts,
                      verbose=not args.quiet)
        stats = planner.get_stats()
        elapsed = time.perf_counter() - t0
        print(f"   reward={res['reward_mean']:+.4f} +/- {res['reward_std']:.4f}  "
              f"collide={res['collision_mean']:.4f}  "
              f"success={res['success_mean']:.4f}  "
              f"fallback={stats['fallback_rate']:.4f}  "
              f"({elapsed:.1f}s)")

        rows.append({
            "rr": float(rr),
            "reward_mean": res["reward_mean"],
            "reward_std": res["reward_std"],
            "collision_mean": res["collision_mean"],
            "collision_std": res["collision_std"],
            "success_mean": res["success_mean"],
            "success_std": res["success_std"],
            "fallback_rate": stats["fallback_rate"],
            "fallback_envsteps": stats["fallback_envsteps"],
            "total_envsteps": stats["total_envsteps"],
            "elapsed_sec": res["elapsed_sec"],
            "fps": res["fps"],
            "n_env_steps": res["n_env_steps"],
        })

    # Pretty stdout summary.
    bar = "=" * 78
    print()
    print(bar)
    print(f" DWA reward in training env | seed={args.seed}, "
          f"{args.n_rollouts}x{rollout_len}x{args.n_envs} env-steps/rr")
    print(bar)
    print(f"{'rr (m)':>9}  {'reward':>9}  {'collide':>8}  {'success':>8}  {'fallback':>9}")
    for r in rows:
        print(f"{r['rr']:>9.4f}  {r['reward_mean']:+8.4f}  "
              f"{r['collision_mean']:8.4f}  {r['success_mean']:8.4f}  "
              f"{r['fallback_rate']:9.4f}")
    print(bar)

    if args.no_save:
        return 0

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.out_dir / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "result.json"
    payload: Dict[str, Any] = {
        "timestamp": ts,
        "env_config": str(args.env_config),
        "train_config": str(args.train_config),
        "dwa_config": str(args.dwa_config),
        "n_envs": args.n_envs,
        "rollout_len": rollout_len,
        "n_rollouts": args.n_rollouts,
        "seed": args.seed,
        "device": str(args.device) if args.device else "auto",
        "grid": [base_cfg.v_samples, base_cfg.omega_samples],
        "weights": {
            "alpha_heading": base_cfg.alpha_heading,
            "beta_clearance": base_cfg.beta_clearance,
            "gamma_velocity": base_cfg.gamma_velocity,
        },
        "rr": rows,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
