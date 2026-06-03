from __future__ import annotations

"""Sweep DWA candidate-grid density on EvalEnv.

For each ``(v_samples, omega_samples)`` pair, the sweep:
  1. Reseeds and builds a fresh EvalEnv at the requested seed so every grid
     sees the same env trajectory (apples-to-apples).
  2. Builds a DWAPlanner with that grid density (everything else inherited
     from ``eval/dwa_config.json``).
  3. Runs ``run_dwa`` for ``n_rollouts`` episodes of ``rollout_len`` steps.
  4. Reads ``planner.get_stats()`` for the fallback counter (stays on GPU
     during the rollout and syncs only once per grid).

Output:
    ``runs/dwa_grid_sweep/<timestamp>/sweep.json``

Usage:
    python -m eval.scripts.dwa_grid_sweep
    python -m eval.scripts.dwa_grid_sweep --grids 3,5 5,9 11,21 21,41
    python -m eval.scripts.dwa_grid_sweep --n-rollouts 4 --quiet
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from eval.dwa.planner import DWAConfig, DWAPlanner

from .dwa_runner import run_dwa
from .eval_env import EvalEnv, EvalEnvSpec


REPO = Path(__file__).resolve().parents[2]
DEFAULT_ENV_CFG = REPO / "config" / "env_config.json"
DEFAULT_DWA_CFG = REPO / "eval" / "dwa_config.json"
DEFAULT_OUT_DIR = REPO / "runs" / "dwa_grid_sweep"

# A roughly 1.5-2x candidate-density ladder, coarse (15) up to the
# eval/dwa_config.json default (21*41 = 861). Override with --grids.
DEFAULT_GRIDS: List[Tuple[int, int]] = [
    (3, 5),
    (5, 9),
    (7, 13),
    (11, 21),
    (21, 41),
]


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _reseed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _parse_grids(argv: List[str]) -> List[Tuple[int, int]]:
    """Parse ['3,5', '5x9', ...] -> [(3,5), (5,9), ...]."""
    out: List[Tuple[int, int]] = []
    for tok in argv:
        parts = tok.replace("x", ",").split(",")
        if len(parts) != 2:
            raise ValueError(f"grid token {tok!r} must look like 'NV,NW' or 'NVxNW'")
        nv, nw = int(parts[0]), int(parts[1])
        if nv < 2 or nw < 2:
            raise ValueError(f"grid {nv}x{nw}: v_samples/omega_samples must each be >= 2")
        out.append((nv, nw))
    return out


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DWA candidate-grid density sweep on EvalEnv")
    p.add_argument("--env-config", type=Path, default=DEFAULT_ENV_CFG)
    p.add_argument("--dwa-config", type=Path, default=DEFAULT_DWA_CFG)
    p.add_argument("--grids", type=str, nargs="*", default=None,
                   help=f"List of grid sizes 'NV,NW' (or 'NVxNW'). "
                        f"Default: {DEFAULT_GRIDS}")
    p.add_argument("--n-envs", type=int, default=24)
    p.add_argument("--rollout-len", type=int, default=256)
    p.add_argument("--n-rollouts", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None,
                   help="cuda / cpu / cuda:0 (default: cuda if available)")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                   help="Where to write sweep.json. A timestamp subdir is appended.")
    p.add_argument("--no-save", action="store_true",
                   help="Skip writing sweep.json; just print the stdout table.")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_cli()
    if not args.env_config.is_file():
        print(f"env_config not found: {args.env_config}", file=sys.stderr)
        return 2
    if not args.dwa_config.is_file():
        print(f"dwa_config not found: {args.dwa_config}", file=sys.stderr)
        return 2

    grids = _parse_grids(args.grids) if args.grids else list(DEFAULT_GRIDS)
    if not grids:
        print("no grids to sweep", file=sys.stderr)
        return 2

    env_cfg = _load_json(args.env_config)
    dwa_cfg = _load_json(args.dwa_config)
    base_cfg = DWAConfig.from_configs(env_cfg, dwa_cfg)

    print(f"Env cfg:  {args.env_config}")
    print(f"DWA cfg:  {args.dwa_config}  "
          f"v=[{base_cfg.v_min},{base_cfg.v_max}]  omega=+/-{base_cfg.omega_max}  "
          f"alpha/beta/gamma={base_cfg.alpha_heading}/{base_cfg.beta_clearance}/{base_cfg.gamma_velocity}")
    print(f"Sweep:    {len(grids)} grids = " + ", ".join(f"{v}x{w}" for v, w in grids))
    print(f"Eval:     {args.n_rollouts} rollouts x {args.rollout_len} steps x {args.n_envs} envs "
          f"per grid (seed={args.seed})")
    print()

    rows: List[Dict[str, Any]] = []
    for i, (nv, nw) in enumerate(grids, start=1):
        # Override only grid density; everything else inherits from base_cfg.
        cfg = DWAConfig(**base_cfg.__dict__)
        cfg.v_samples = int(nv)
        cfg.omega_samples = int(nw)

        # Fresh env + reseed per grid so each grid sees the same env trajectory.
        _reseed(args.seed)
        env = EvalEnv(EvalEnvSpec(
            env_cfg=env_cfg,
            n_envs=args.n_envs,
            seed=args.seed,
            device=args.device,
        ))
        planner = DWAPlanner(cfg, device=env.device)
        planner.reset_stats()

        print(f"[{i}/{len(grids)}] grid {nv}x{nw} = {nv*nw} candidates")
        t0 = time.perf_counter()
        res = run_dwa(env, planner,
                      rollout_len=args.rollout_len,
                      n_rollouts=args.n_rollouts,
                      verbose=not args.quiet)
        stats = planner.get_stats()
        elapsed = time.perf_counter() - t0
        print(f"   reward={res['reward_mean']:+.4f} +/- {res['reward_std']:.4f}  "
              f"collide={res['collision_mean']:.4f}  "
              f"fallback={stats['fallback_rate']:.4f}  "
              f"({elapsed:.1f}s)")

        rows.append({
            "v_samples": int(nv),
            "omega_samples": int(nw),
            "nc": int(nv * nw),
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
    print(f" DWA grid-density sweep | seed={args.seed}, "
          f"{args.n_rollouts}x{args.rollout_len}x{args.n_envs} env-steps/grid")
    print(bar)
    print(f"{'grid':>9}  {'NC':>5}  {'reward':>9}  {'collide':>8}  {'fallback':>9}")
    for r in rows:
        print(f"{r['v_samples']:>4}x{r['omega_samples']:<4}  {r['nc']:>5}  "
              f"{r['reward_mean']:+8.4f}  {r['collision_mean']:8.4f}  "
              f"{r['fallback_rate']:9.4f}")
    print(bar)

    if args.no_save:
        return 0

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.out_dir / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sweep.json"
    payload: Dict[str, Any] = {
        "timestamp": ts,
        "env_config": str(args.env_config),
        "dwa_config": str(args.dwa_config),
        "n_envs": args.n_envs,
        "rollout_len": args.rollout_len,
        "n_rollouts": args.n_rollouts,
        "seed": args.seed,
        "device": str(args.device) if args.device else "auto",
        "grids": rows,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
