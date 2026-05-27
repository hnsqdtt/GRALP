from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .encoders.base import EncoderBase


@dataclass
class PPOActOut:
    action: torch.Tensor  # scaled action in SI limits, shape [B, A]

    logp: torch.Tensor    # log-prob of action under current policy, shape [B, 1]

    mu: torch.Tensor      # pre-squash mean, shape [B, A]

    std: torch.Tensor     # pre-squash std, shape [B, A]


def _tanh_log_det_jac(pre_tanh: torch.Tensor) -> torch.Tensor:
    return 2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))


def _bounds_to_center_scale(limits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unpack action bounds ``[..., A, 2]`` (lo, hi) into (center, scale)."""
    lo = limits[..., 0]
    hi = limits[..., 1]
    center = 0.5 * (lo + hi)
    scale = 0.5 * (hi - lo)
    return center, scale


def _squash(mu: torch.Tensor, log_std: torch.Tensor, eps: torch.Tensor, limits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a squashed Gaussian action mapped to per-axis bounds.

    ``limits`` is shape ``[..., A, 2]`` with the last dim being (lo, hi). The
    affine map ``a_scaled = center + tanh(pre) * scale`` adds a constant
    ``-Σ log|scale|`` to logp that cancels in PPO's ratio (scale is fixed per
    env). We deliberately omit it, matching the legacy symmetric form which
    similarly omitted ``log|limits|``.
    """
    std = log_std.exp()
    pre_tanh = mu + std * eps
    a = torch.tanh(pre_tanh)
    log_det = _tanh_log_det_jac(pre_tanh)
    dist = Normal(mu, std)
    logp = (dist.log_prob(pre_tanh) - log_det).sum(-1, keepdim=True)
    center, scale = _bounds_to_center_scale(limits)
    a_scaled = center + a * scale
    return a_scaled, logp, std


def _inverse_squash(action_scaled: torch.Tensor, limits: torch.Tensor) -> torch.Tensor:
    """Inverse of the affine + tanh map; ``limits`` is ``[..., A, 2]``."""
    center, scale = _bounds_to_center_scale(limits)
    a = ((action_scaled - center) / scale.clamp_min(1e-12)).clamp(-0.999999, 0.999999)
    return 0.5 * (torch.log1p(a) - torch.log1p(-a))  # atanh(a)


class PPOPolicy(nn.Module):
    """Gaussian policy with a pluggable encoder, tanh squashing, and a separate value head.

    The encoder is any ``EncoderBase`` subclass; this policy only depends on
    ``encoder.feature_dim``. Actions are scaled by per-axis ``limits`` provided
    by the environment at sampling time.
    """

    def __init__(self, encoder: EncoderBase, action_dim: int = 2, *,
                 value_hidden: int = 256,
                 log_std_min: float = -5.0, log_std_max: float = 2.0) -> None:
        super().__init__()
        self.encoder = encoder
        feat = int(encoder.feature_dim)
        self.mu = nn.Linear(feat, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        lo = float(log_std_min); hi = float(log_std_max)
        if hi < lo:
            lo, hi = hi, lo
        self._log_std_min = float(lo)
        self._log_std_max = float(hi)

        self.value = nn.Sequential(
            nn.Linear(feat, int(value_hidden)), nn.ReLU(),
            nn.Linear(int(value_hidden), 1),
        )

    def _core(self, obs_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.encoder(obs_vec)
        mu = self.mu(feat)
        log_std = self.log_std.view(1, -1).expand_as(mu).clamp(self._log_std_min, self._log_std_max)
        v = self.value(feat)
        return mu, log_std, v

    @torch.no_grad()
    def act(self, obs_vec: torch.Tensor, limits: torch.Tensor) -> PPOActOut:
        mu, log_std, _ = self._core(obs_vec)
        eps = torch.randn_like(mu)
        a_scaled, logp, std = _squash(mu, log_std, eps, limits)
        return PPOActOut(action=a_scaled, logp=logp, mu=mu, std=std)

    @torch.no_grad()
    def act_deterministic(self, obs_vec: torch.Tensor, limits: torch.Tensor) -> torch.Tensor:
        """Return ``center + tanh(mu) * scale`` with no exploration noise.

        ``limits`` is ``[..., A, 2]`` (lo, hi). Symmetric bounds reduce to
        ``tanh(mu) * hi`` (bit-exact with the legacy form); forward-only vx
        maps tanh(mu)∈[-1,1] into [0, vx_max].

        Use this for evaluation rollouts so reported rewards reflect the
        learned policy's expected behavior rather than a sample of it.
        """
        mu, _, _ = self._core(obs_vec)
        center, scale = _bounds_to_center_scale(limits)
        return center + torch.tanh(mu) * scale

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
