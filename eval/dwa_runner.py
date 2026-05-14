from __future__ import annotations

"""Run DWA on EvalEnv for N rollouts and aggregate metrics.

Usage as a script (DWA-only baseline):
    python -m eval.dwa_runner
    python -m eval.dwa_runner --rollout-len 256 --n-rollouts 10 --n-envs 24 --seed 0

Importable: ``run_dwa(eval_env, planner, rollout_len, n_rollouts) -> dict``.
All ops stay on the env's device (no CPU<->GPU sync inside the inner loop).
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

from eval.dwa.planner import DWAConfig, DWAPlanner
from eval.eval_env import EvalEnv, EvalEnvSpec


REPO = Path(__file__).resolve().parents[1]
DEFAULT_ENV_CFG = REPO / "config" / "env_config.json"
DEFAULT_DWA_CFG = REPO / "eval" / "dwa_config.json"


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@torch.no_grad()
def run_dwa(env: EvalEnv,
            planner: DWAPlanner,
            *,
            rollout_len: int,
            n_rollouts: int,
            verbose: bool = True) -> Dict[str, Any]:
    """Run DWA for ``n_rollouts`` episodes of length ``rollout_len`` each.

    Reward / collision / success accumulate as GPU scalars across the rollout
    and only sync to the host once at the end (per rollout), so the inner loop
    is a pure GPU pipeline: env.step -> planner.plan_batch -> env.step.
    Final report aggregates mean / std across rollouts.
    """
    B = env.B
    device = env.device

    per_roll_reward: List[float] = []
    per_roll_collide: List[float] = []
    per_roll_success: List[float] = []

    t_start = time.perf_counter()
    n_env_steps = 0

    for r in range(n_rollouts):
        env.reset()
        sum_reward = torch.zeros((), device=device, dtype=torch.float32)
        sum_collide = torch.zeros((), device=device, dtype=torch.float32)
        sum_success = torch.zeros((), device=device, dtype=torch.float32)
        for _ in range(rollout_len):
            snap = env.snapshot_for_dwa()
            vx, w = planner.plan_batch(
                snap["vx_cur"], snap["omega_cur"],
                snap["target_x_local"], snap["target_y_local"],
                snap["rays_m"],
            )
            action = torch.stack([vx, w], dim=-1)
            _obs, reward, _term, info = env.step(action)
            sum_reward += reward.sum()
            sum_collide += info["collided"].to(torch.float32).sum()
            sum_success += info["success"].to(torch.float32).sum()
            n_env_steps += B

        denom = float(rollout_len * B)
        roll_r = float(sum_reward.item()) / denom
        roll_c = float(sum_collide.item()) / denom
        roll_s = float(sum_success.item()) / denom
        per_roll_reward.append(roll_r)
        per_roll_collide.append(roll_c)
        per_roll_success.append(roll_s)

        if verbose:
            print(f"  rollout {r+1:2d}/{n_rollouts}: "
                  f"reward={roll_r:+.4f}  collide={roll_c:.4f}  success={roll_s:.4f}")

    elapsed = time.perf_counter() - t_start

    def _stats(xs: List[float]):
        if not xs:
            return 0.0, 0.0
        if len(xs) == 1:
            return float(xs[0]), 0.0
        return float(statistics.mean(xs)), float(statistics.pstdev(xs))

    r_mean, r_std = _stats(per_roll_reward)
    c_mean, c_std = _stats(per_roll_collide)
    s_mean, s_std = _stats(per_roll_success)

    return {
        "rollouts": {
            "reward": per_roll_reward,
            "collision": per_roll_collide,
            "success": per_roll_success,
        },
        "reward_mean": r_mean,
        "reward_std": r_std,
        "collision_mean": c_mean,
        "collision_std": c_std,
        "success_mean": s_mean,
        "success_std": s_std,
        "elapsed_sec": elapsed,
        "n_env_steps": n_env_steps,
        "fps": n_env_steps / elapsed if elapsed > 0 else float("inf"),
    }


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DWA-only baseline runner on EvalEnv")
    p.add_argument("--env-config", type=Path, default=DEFAULT_ENV_CFG)
    p.add_argument("--dwa-config", type=Path, default=DEFAULT_DWA_CFG)
    p.add_argument("--n-envs", type=int, default=24)
    p.add_argument("--rollout-len", type=int, default=256)
    p.add_argument("--n-rollouts", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None,
                   help="cuda / cpu / cuda:0 (default: cuda if available)")
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

    env = EvalEnv(EvalEnvSpec(
        env_cfg_path=str(args.env_config),
        n_envs=args.n_envs,
        seed=args.seed,
        device=args.device,
    ))
    print(f"Env: {args.env_config.name}  B={env.B}  N_rays={env.n_rays}  device={env.device}")

    env_cfg = _load_json(args.env_config)
    dwa_cfg = _load_json(args.dwa_config)
    cfg = DWAConfig.from_configs(env_cfg, dwa_cfg)
    print(f"DWA: v=[{cfg.v_min},{cfg.v_max}] omega=+/-{cfg.omega_max}  "
          f"alpha/beta/gamma={cfg.alpha_heading}/{cfg.beta_clearance}/{cfg.gamma_velocity}  "
          f"brake=(v={cfg.v_brake_acc}, w={cfg.omega_brake_acc})  "
          f"grid={cfg.v_samples}x{cfg.omega_samples}")
    print(f"Stopping distance @ v_max: {cfg.v_max**2 / (2*cfg.v_brake_acc):.3f}m")
    planner = DWAPlanner(cfg, device=env.device)

    print()
    print(f"Running {args.n_rollouts} rollouts x {args.rollout_len} steps "
          f"(= {args.n_envs * args.n_rollouts * args.rollout_len} env-steps)...")
    res = run_dwa(env, planner,
                  rollout_len=args.rollout_len,
                  n_rollouts=args.n_rollouts,
                  verbose=not args.quiet)

    print()
    print("=" * 60)
    print(f"DWA baseline | {args.n_rollouts} rollouts x {args.rollout_len} steps x {args.n_envs} envs")
    print("=" * 60)
    print(f"  reward      = {res['reward_mean']:+.4f} +/- {res['reward_std']:.4f}")
    print(f"  collision   = {res['collision_mean']:.4f} +/- {res['collision_std']:.4f}  "
          f"(per-step rate)")
    print(f"  success     = {res['success_mean']:.4f} +/- {res['success_std']:.4f}  "
          f"(per-step rate)")
    print(f"  elapsed     = {res['elapsed_sec']:.2f} s  "
          f"(throughput = {res['fps']/1000:.1f} k env-steps/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
