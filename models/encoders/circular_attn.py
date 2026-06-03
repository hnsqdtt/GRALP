from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import EncoderBase, circular_pad1d


class _SqueezeExcite1D(nn.Module):
    def __init__(self, ch: int, r: int = 4) -> None:
        super().__init__()
        hid = max(8, ch // r)
        self.fc1 = nn.Linear(ch, hid)
        self.fc2 = nn.Linear(hid, ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=-1)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(-1)


class _DepthwiseSeparable1D(nn.Module):
    def __init__(self, ch: int, kernel: int = 5, dilation: int = 1) -> None:
        super().__init__()
        self.kernel = int(kernel)
        self.dil = int(dilation)
        self.dw = nn.Conv1d(ch, ch, kernel_size=kernel, groups=ch, bias=False, dilation=self.dil)
        self.pw = nn.Conv1d(ch, ch, kernel_size=1)
        self.bn = nn.BatchNorm1d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = ((self.kernel - 1) * self.dil) // 2
        if pad > 0:
            x = circular_pad1d(x, pad)
        out = self.dw(x)
        out = F.gelu(out)
        out = self.pw(out)
        out = self.bn(out)
        return out


class _RayBranch(nn.Module):
    def __init__(self, in_ch: int = 1, hidden: int = 64, layers: int = 4, kernel: int = 5) -> None:
        super().__init__()
        self.in_ch = int(in_ch)
        self.expand = nn.Conv1d(self.in_ch, hidden, kernel_size=1)
        dilations = [1, 2, 4, 8][:layers]
        blocks = []
        for d in dilations:
            blocks += [
                _DepthwiseSeparable1D(hidden, kernel=kernel, dilation=d),
                nn.GELU(),
                _SqueezeExcite1D(hidden, r=4),
            ]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.expand(x)
        x = self.blocks(x)
        return x


class CircularAttnEncoder(EncoderBase):
    """Ray encoder with circular dilated conv + multi-query, multi-head attention.

    Splits ``vec`` into ``[B, N]`` ray distances and ``[B, pose_dim]`` pose
    features. Conv branch produces keys/values per ray bin; the pose MLP plus
    learnable (or pose-conditioned) queries attend over those bins. The pooled
    attention output, pooled values, and pooled queries are concatenated and
    projected to ``feature_dim``.

    - ``num_queries`` (M): how many queries to attend with.
    - ``num_heads`` (H): multi-head attention heads (requires d_model % H == 0).
    - ``learnable_queries``: when True, query vectors are free parameters added
      to the pose embedding; when False, queries are produced by a linear head.
    """

    def __init__(self, vec_dim: int, *, feature_dim: int = 256, hidden: int = 64,
                 d_model: int = 128, num_queries: int = 4, num_heads: int = 4,
                 learnable_queries: bool = True, pose_dim: int = 7,
                 ray_pos_encoding: bool = False) -> None:
        super().__init__(vec_dim, feature_dim=feature_dim)
        self.num_queries = int(num_queries)
        self.num_heads = int(num_heads)
        self.learnable_queries = bool(learnable_queries)
        self.pose_dim = int(pose_dim)
        if self.vec_dim < self.pose_dim:
            raise ValueError(f"vec_dim must be >= pose_dim ({self.pose_dim}), got {vec_dim}")
        self.N = max(0, self.vec_dim - self.pose_dim)
        # When True, ``_pos_emb`` is concatenated as a second input channel to
        # break the ring symmetry of circular_pad + permutation-invariant attn.
        # See ``RayPoseCNN1D`` for the same flag and rationale.
        self.ray_pos_encoding = bool(ray_pos_encoding)
        self.ray_in_ch = 2 if self.ray_pos_encoding else 1
        self.hidden = int(hidden)
        self.d_model = int(d_model)
        if self.d_model % max(1, self.num_heads) != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.br_obs = _RayBranch(in_ch=self.ray_in_ch, hidden=self.hidden)
        if self.ray_pos_encoding and self.N > 0:
            pos = torch.arange(self.N, dtype=torch.float32) / max(self.N - 1, 1)
            self.register_buffer("_pos_emb", pos.view(1, 1, self.N), persistent=False)
        self.to_k = nn.Conv1d(self.hidden, self.d_model, kernel_size=1)
        self.to_v = nn.Conv1d(self.hidden, self.d_model, kernel_size=1)
        self.pose_mlp = nn.Sequential(
            nn.Linear(self.pose_dim, self.d_model), nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        if self.learnable_queries:
            init_scale = 1.0 / math.sqrt(max(1, self.d_model))
            self.q_params = nn.Parameter(torch.randn(self.num_queries, self.d_model) * init_scale)
            self.to_q = nn.Identity()
        else:
            if self.num_queries > 1:
                self.to_q = nn.Linear(self.d_model, self.d_model * self.num_queries)
            else:
                self.to_q = nn.Identity()
        self.post = nn.Sequential(
            nn.Linear(self.d_model * 3, self.feature_dim), nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim), nn.ReLU(),
        )

    def _split(self, vec: torch.Tensor):
        d_obs = vec[:, :self.N]
        pose = vec[:, self.N:self.N + self.pose_dim]
        return d_obs, pose

    def forward(self, vec: torch.Tensor) -> torch.Tensor:
        d_obs, pose = self._split(vec)
        if self.ray_pos_encoding:
            ray_in = torch.cat(
                [d_obs.unsqueeze(1), self._pos_emb.expand(d_obs.size(0), -1, -1)], dim=1
            )
        else:
            ray_in = d_obs  # _RayBranch unsqueezes a 2-D input to [B, 1, N]
        Fmap = self.br_obs(ray_in)
        K = self.to_k(Fmap).transpose(1, 2)
        V = self.to_v(Fmap).transpose(1, 2)

        q_pose = self.pose_mlp(pose)
        if self.learnable_queries:
            q = self.q_params.unsqueeze(0) + q_pose.unsqueeze(1)
        else:
            if self.num_queries > 1:
                qM = self.to_q(q_pose)
                q = qM.view(qM.size(0), self.num_queries, self.d_model)
            else:
                q = q_pose.view(q_pose.size(0), 1, self.d_model)

        H = max(1, self.num_heads)
        Dh = self.d_model // H
        K_h = K.view(K.size(0), K.size(1), H, Dh)
        V_h = V.view(V.size(0), V.size(1), H, Dh)
        Q_h = q.view(q.size(0), q.size(1), H, Dh)

        attn_logits = torch.einsum("bmhd,bnhd->bmhn", Q_h, K_h) / math.sqrt(Dh)
        attn = torch.softmax(attn_logits, dim=-1)
        z_h = torch.einsum("bmhn,bnhd->bmhd", attn, V_h)
        z = z_h.reshape(z_h.size(0), z_h.size(1), self.d_model)

        z_mean = z.mean(dim=1)
        q_mean = q.mean(dim=1)
        gavg = V.mean(dim=1)
        g = torch.cat([z_mean, gavg, q_mean], dim=-1)
        return self.post(g)
