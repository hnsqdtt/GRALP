from __future__ import annotations

"""Run a PPO policy on EvalEnv with the same reward / collision / success
accounting as dwa_runner.py.

Reward source of truth: ``env.step()`` return value. We never reconstruct or
post-process reward inside the runner — that's the GPU env's contract and
keeps DWA vs PPO comparisons honest (both go through the exact same
SimRandomGPUBatchEnv reward pipeline).

Actions are deterministic by default (``tanh(mu) * limits``) so the reported
mean reflects the policy's expected behavior rather than a stochastic sample.
Pass ``deterministic=False`` to recover sampling behavior.
"""

import statistics
import time
from typing import Any, Callable, Dict, List, Optional

import torch

from eval.eval_env import EvalEnv
from models import PPOPolicy


@torch.no_grad()
def run_policy(env: EvalEnv,
               policy: PPOPolicy,
               *,
               rollout_len: int,
               n_rollouts: int,
               deterministic: bool = True,
               verbose: bool = True,
               progress_factory: Optional[Callable] = None,
               ) -> Dict[str, Any]:
    """Run ``policy`` in ``env`` for ``n_rollouts`` episodes of ``rollout_len`` steps.

    Returns the same metric layout as ``dwa_runner.run_dwa``: per-rollout
    arrays plus ``reward_mean / reward_std`` aggregated across rollouts. Each
    per-rollout average divides by ``rollout_len * B`` so the unit is
    per-env-step (identical to ``rl_ppo/train.py``'s mean_reward logging).
    """
    policy.eval()
    B = env.B
    limits = env.get_limits()
    limits_b = limits.view(1, -1).expand(B, -1)

    per_roll_reward: List[float] = []
    per_roll_collide: List[float] = []
    per_roll_success: List[float] = []

    t_start = time.perf_counter()
    n_env_steps = 0

    iterator = range(n_rollouts)
    if progress_factory is not None:
        iterator = progress_factory(iterator, total=n_rollouts, desc="rollouts")

    for r in iterator:
        obs = env.reset()
        sum_reward = 0.0
        sum_collide = 0.0
        sum_success = 0.0
        for _ in range(rollout_len):
            if deterministic:
                action = policy.act_deterministic(obs, limits_b)
            else:
                action = policy.act(obs, limits_b).action
            obs, reward, _term, info = env.step(action)
            sum_reward += float(reward.sum().item())
            sum_collide += float(info["collided"].to(torch.float32).sum().item())
            sum_success += float(info["success"].to(torch.float32).sum().item())
            n_env_steps += B

        denom = float(rollout_len * B)
        per_roll_reward.append(sum_reward / denom)
        per_roll_collide.append(sum_collide / denom)
        per_roll_success.append(sum_success / denom)

        if verbose and progress_factory is None:
            print(f"  rollout {r+1:2d}/{n_rollouts}: "
                  f"reward={per_roll_reward[-1]:+.4f}  "
                  f"collide={per_roll_collide[-1]:.4f}  "
                  f"success={per_roll_success[-1]:.4f}")

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
