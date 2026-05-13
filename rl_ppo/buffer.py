from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch


@dataclass
class RolloutBatch:
    obs: torch.Tensor         # [N, obs_dim]

    actions: torch.Tensor     # [N, act_dim]

    logp: torch.Tensor        # [N, 1]

    advantages: torch.Tensor  # [N, 1]

    returns: torch.Tensor     # [N, 1]

    values: torch.Tensor      # [N, 1]

    limits: torch.Tensor      # [N, act_dim]



class RolloutBuffer:
    """On-policy experience buffer for PPO (GAE-Lambda).

    Stored as flat tensors for easy slicing/minibatching.
    Layout: T steps × B environments → N = T*B rows.
    """

    def __init__(self, T: int, B: int, obs_dim: int, act_dim: int, device: torch.device) -> None:
        N = int(T * B)
        self.T = int(T)
        self.B = int(B)
        self.N = int(N)
        self.device = device

        self.obs = torch.zeros((N, obs_dim), device=device, dtype=torch.float32)
        self.actions = torch.zeros((N, act_dim), device=device, dtype=torch.float32)
        self.logp = torch.zeros((N, 1), device=device, dtype=torch.float32)
        self.rewards = torch.zeros((N, 1), device=device, dtype=torch.float32)
        self.dones = torch.zeros((N, 1), device=device, dtype=torch.float32)
        self.values = torch.zeros((N, 1), device=device, dtype=torch.float32)
        self.limits = torch.zeros((N, act_dim), device=device, dtype=torch.float32)
        self._ptr = 0

    def add(self, *, obs: torch.Tensor, act: torch.Tensor, logp: torch.Tensor, rew: torch.Tensor,
            done: torch.Tensor, val: torch.Tensor, limits: torch.Tensor) -> None:
        n = obs.shape[0]
        i0 = self._ptr
        i1 = i0 + n
        self.obs[i0:i1].copy_(obs)
        self.actions[i0:i1].copy_(act)
        self.logp[i0:i1].copy_(logp)
        self.rewards[i0:i1].copy_(rew.view(-1, 1))
        self.dones[i0:i1].copy_(done.view(-1, 1).to(torch.float32))
        self.values[i0:i1].copy_(val.view(-1, 1))
        self.limits[i0:i1].copy_(limits)
        self._ptr = i1

    @torch.no_grad()
    def compute_gae(self, last_value: torch.Tensor, gamma: float, lam: float, count_timeout_as_done: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute advantages and returns in-place.

        last_value: [B,1] for bootstrapping at the final step of each env.
        If timeouts are not considered done, `dones` should exclude them and the caller should provide the correct mask.
        """
        T, B = self.T, self.B
        adv = torch.zeros((T, B, 1), device=self.device, dtype=torch.float32)
        ret = torch.zeros((T, B, 1), device=self.device, dtype=torch.float32)

        r = self.rewards.view(T, B, 1)
        d = self.dones.view(T, B, 1)
        v = self.values.view(T, B, 1)
        next_v = torch.cat([v[1:], last_value.view(1, B, 1)], dim=0)

        gae = torch.zeros((B, 1), device=self.device, dtype=torch.float32)
        for t in reversed(range(T)):
            not_done = 1.0 - d[t]
            delta = r[t] + gamma * next_v[t] * not_done - v[t]
            gae = delta + gamma * lam * not_done * gae
            adv[t] = gae
            ret[t] = adv[t] + v[t]

        advantages = adv.view(T * B, 1)
        returns = ret.view(T * B, 1)
        adv_mean = advantages.mean()
        adv_std = advantages.std().clamp_min(1e-8)
        advantages = (advantages - adv_mean) / adv_std
        self.advantages = advantages
        self.returns = returns
        return advantages, returns

    @torch.no_grad()
    def compute_mc_returns(self, gamma: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Monte Carlo returns without bootstrap and advantages as (return - value).

        - No value bootstrap at horizon end: ret_T = 0 and rolls back with discounts.
        - Episode terminations in `self.dones` break the return chain for that env.
        - Normalizes advantages to mean 0, std 1 (like compute_gae).
        """
        T, B = self.T, self.B
        r = self.rewards.view(T, B, 1)
        d = self.dones.view(T, B, 1)
        v = self.values.view(T, B, 1)

        ret = torch.zeros((T, B, 1), device=self.device, dtype=torch.float32)
        next_ret = torch.zeros((B, 1), device=self.device, dtype=torch.float32)
        for t in reversed(range(T)):
            not_done = 1.0 - d[t]
            next_ret = r[t] + gamma * not_done * next_ret
            ret[t] = next_ret

        advantages = (ret - v).view(T * B, 1)
        returns = ret.view(T * B, 1)

        adv_mean = advantages.mean()
        adv_std = advantages.std().clamp_min(1e-8)
        advantages = (advantages - adv_mean) / adv_std
        self.advantages = advantages
        self.returns = returns
        return advantages, returns

    def minibatches(self, batch_size: int):
        idx = torch.randperm(self.N, device=self.device)
        for i in range(0, self.N, batch_size):
            j = min(self.N, i + batch_size)
            sl = idx[i:j]
            yield RolloutBatch(
                obs=self.obs[sl],
                actions=self.actions[sl],
                logp=self.logp[sl],
                advantages=self.advantages[sl],
                returns=self.returns[sl],
                values=self.values[sl],
                limits=self.limits[sl],
            )
