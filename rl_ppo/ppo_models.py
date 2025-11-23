from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from rl_ppo.encoder import RayEncoder


@dataclass
class PPOActOut:
    action: torch.Tensor  # scaled action in SI limits, shape [B, A]

    logp: torch.Tensor    # log-prob of action under current policy, shape [B, 1]

    mu: torch.Tensor      # pre-squash mean, shape [B, A]

    std: torch.Tensor     # pre-squash std, shape [B, A]



def _tanh_log_det_jac(pre_tanh: torch.Tensor) -> torch.Tensor:
    return 2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))


def _squash(mu: torch.Tensor, log_std: torch.Tensor, eps: torch.Tensor, limits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    std = log_std.exp()
    pre_tanh = mu + std * eps
    a = torch.tanh(pre_tanh)
    log_det = _tanh_log_det_jac(pre_tanh)
    dist = Normal(mu, std)
    logp = (dist.log_prob(pre_tanh) - log_det).sum(-1, keepdim=True)
    a_scaled = a * limits
    return a_scaled, logp, std


def _inverse_squash(action_scaled: torch.Tensor, limits: torch.Tensor) -> torch.Tensor:
    a = (action_scaled / limits.clamp_min(1e-12)).clamp(-0.999999, 0.999999)
    return 0.5 * (torch.log1p(a) - torch.log1p(-a))  # atanh(a)


class PPOPolicy(nn.Module):
    """Gaussian policy with a shared encoder, tanh squashing, and a separate value head.

    - Each step uses a vector observation of dimension `vec_dim`
    - Actions are scaled by per-axis `limits` provided by the environment
    """

    def __init__(self, vec_dim: int, action_dim: int = 3, hidden: int = 64, d_model: int = 128, *, num_queries: int = 4, num_heads: int = 4, learnable_queries: bool = True, log_std_min: float = -5.0, log_std_max: float = 2.0):
        super().__init__()
        self.encoder = RayEncoder(vec_dim, hidden=hidden, d_model=d_model, num_queries=num_queries, num_heads=num_heads, learnable_queries=learnable_queries)
        self.mu = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # global, stable in PPO
        lo = float(log_std_min); hi = float(log_std_max)
        if hi < lo:
            lo, hi = hi, lo
        self._log_std_min = float(lo)
        self._log_std_max = float(hi)

        self.value = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))

    def _core(self, obs_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        g, _, _ = self.encoder(obs_vec)
        mu = self.mu(g)
        log_std = self.log_std.view(1, -1).expand_as(mu).clamp(self._log_std_min, self._log_std_max)
        v = self.value(g)
        return mu, log_std, v

    @torch.no_grad()
    def act(self, obs_vec: torch.Tensor, limits: torch.Tensor) -> PPOActOut:
        mu, log_std, _ = self._core(obs_vec)
        eps = torch.randn_like(mu)
        a_scaled, logp, std = _squash(mu, log_std, eps, limits)
        return PPOActOut(action=a_scaled, logp=logp, mu=mu, std=std)

    def evaluate_actions(self, obs_vec: torch.Tensor, actions_scaled: torch.Tensor, limits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (logp, entropy, value) for given actions.

        - actions_scaled are real actions already limited in SI units
        - logp uses the same tanh-corrected squashed Normal as sampling
        - entropy uses the base Normal approximation
        """
        mu, log_std, v = self._core(obs_vec)
        std = log_std.exp()
        y = _inverse_squash(actions_scaled, limits)
        dist = Normal(mu, std)
        log_det = _tanh_log_det_jac(y)
        logp = (dist.log_prob(y) - log_det).sum(-1, keepdim=True)
        ent = dist.entropy().sum(-1, keepdim=True)
        return logp, ent, v
