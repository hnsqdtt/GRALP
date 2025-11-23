from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _circular_pad1d(x: torch.Tensor, pad: int) -> torch.Tensor:
    if pad <= 0:
        return x
    left = x[..., -pad:]
    right = x[..., :pad]
    return torch.cat([left, x, right], dim=-1)


class SqueezeExcite1D(nn.Module):
    def __init__(self, ch: int, r: int = 4):
        super().__init__()
        hid = max(8, ch // r)
        self.fc1 = nn.Linear(ch, hid)
        self.fc2 = nn.Linear(hid, ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=-1)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(-1)


class DepthwiseSeparable1D(nn.Module):
    def __init__(self, ch: int, kernel: int = 5, dilation: int = 1):
        super().__init__()
        self.kernel = int(kernel)
        self.dil = int(dilation)
        self.dw = nn.Conv1d(ch, ch, kernel_size=kernel, groups=ch, bias=False, dilation=self.dil)
        self.pw = nn.Conv1d(ch, ch, kernel_size=1)
        self.bn = nn.BatchNorm1d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = ((self.kernel - 1) * self.dil) // 2
        if pad > 0:
            x = _circular_pad1d(x, pad)
        out = self.dw(x)
        out = F.gelu(out)
        out = self.pw(out)
        out = self.bn(out)
        return out


class RayBranch(nn.Module):
    def __init__(self, in_ch: int = 1, hidden: int = 64, layers: int = 4, kernel: int = 5):
        super().__init__()
        self.in_ch = int(in_ch)
        self.expand = nn.Conv1d(self.in_ch, hidden, kernel_size=1)
        dilations = [1, 2, 4, 8][:layers]
        blocks = []
        for d in dilations:
            blocks += [
                DepthwiseSeparable1D(hidden, kernel=kernel, dilation=d),
                nn.GELU(),
                SqueezeExcite1D(hidden, r=4),
            ]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.expand(x)
        x = self.blocks(x)
        return x


class RayEncoder(nn.Module):
    """射线编码器 + 多查询、多头注意力。

    - num_queries (M): 查询数量
    - num_heads (H): 注意力头个数；需满足 d_model % H == 0
    - learnable_queries: True 时查询向量为可学习参数
    """

    def __init__(self, vec_dim: int, hidden: int = 64, d_model: int = 128, *, num_queries: int = 1, num_heads: int = 1, learnable_queries: bool = True):
        super().__init__()
        self.num_queries = int(num_queries)
        self.num_heads = int(num_heads)
        self.learnable_queries = bool(learnable_queries)
        # Pose features now include 7 dims:
        # [sin_ref, cos_ref,
        #  prev_vx/lim, prev_omega/lim,
        #  dprev_vx/(2*lim), dprev_omega/(2*omega_max),
        #  dist_to_task/patch_meters]
        pose_dim = 7
        assert vec_dim >= pose_dim, f"vec_dim must be N + {pose_dim}, got {vec_dim}"
        self.vec_dim = int(vec_dim)
        self.pose_dim = pose_dim
        d_len = vec_dim - self.pose_dim
        # Angle channels removed; treat rays as a single channel of length N

        self.ray_in_ch = 1
        self.N = max(0, d_len)
        self.hidden = int(hidden)
        self.d_model = int(d_model)
        assert self.d_model % max(1, self.num_heads) == 0, "d_model must be divisible by num_heads"

        self.br_obs = RayBranch(in_ch=self.ray_in_ch, hidden=hidden)
        self.to_k = nn.Conv1d(hidden, d_model, kernel_size=1)
        self.to_v = nn.Conv1d(hidden, d_model, kernel_size=1)
        self.pose_mlp = nn.Sequential(
            nn.Linear(self.pose_dim, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        # Queries

        if self.learnable_queries:
            # Learnable query embeddings [M, d_model]

            init_scale = 1.0 / math.sqrt(max(1, d_model))
            self.q_params = nn.Parameter(torch.randn(self.num_queries, d_model) * init_scale)
            # Keep a lightweight identity to avoid conditionals later

            self.to_q = nn.Identity()
        else:
            # If using input-conditioned queries, map the pose embedding to M*d_model

            if self.num_queries > 1:
                self.to_q = nn.Linear(d_model, d_model * self.num_queries)
            else:
                self.to_q = nn.Identity()
        self.post = nn.Sequential(
            nn.Linear(d_model * 2 + d_model, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU()
        )

    def split(self, vec: torch.Tensor):
        d_len = self.N * self.ray_in_ch
        d_obs = vec[:, :d_len]
        pose = vec[:, d_len:d_len + self.pose_dim]
        if self.ray_in_ch == 1:
            return d_obs, pose
        x = d_obs.view(d_obs.size(0), self.ray_in_ch, self.N)
        return x, pose

    def forward(self, vec: torch.Tensor):
        d_obs, pose = self.split(vec)
        Fmap = self.br_obs(d_obs)  # [B, hidden, N]
        K = self.to_k(Fmap).transpose(1, 2)  # [B, N, d_model]
        V = self.to_v(Fmap).transpose(1, 2)  # [B, N, d_model]

        # Queries: learnable or input-conditioned (always pose-aware)

        # Embed the 8-dim pose features (sin_ref,cos_ref, prev_cmd[3], dprev[3])

        q_pose = self.pose_mlp(pose)  # [B, d_model]
        if self.learnable_queries:
            # Pose-conditioned learned queries: add pose embedding as a per-batch bias

            # Result shape: [B, M, d_model]

            q = self.q_params.unsqueeze(0) + q_pose.unsqueeze(1)
        else:
            # Input-conditioned queries from pose embedding

            if self.num_queries > 1:
                qM = self.to_q(q_pose)  # [B, M*d_model]
                q = qM.view(qM.size(0), self.num_queries, self.d_model)  # [B, M, d_model]
            else:
                q = q_pose.view(q_pose.size(0), 1, self.d_model)  # [B, 1, d_model]

        # Multi-head projection via reshape (K/V already at d_model)

        H = max(1, self.num_heads)
        Dh = self.d_model // H
        K_h = K.view(K.size(0), K.size(1), H, Dh)        # [B, N, H, Dh]
        V_h = V.view(V.size(0), V.size(1), H, Dh)        # [B, N, H, Dh]
        Q_h = q.view(q.size(0), q.size(1), H, Dh)        # [B, M, H, Dh]

        # Attention: softmax over sequence length N

        attn_logits = torch.einsum('bmhd,bnhd->bmhn', Q_h, K_h) / math.sqrt(Dh)
        attn = torch.softmax(attn_logits, dim=-1)        # [B, M, H, N]
        z_h = torch.einsum('bmhn,bnhd->bmhd', attn, V_h) # [B, M, H, Dh]
        z = z_h.reshape(z_h.size(0), z_h.size(1), self.d_model)  # [B, M, d_model]

        # Aggregate queries (mean). Keep global avg from V without heads for compatibility

        z_mean = z.mean(dim=1)        # [B, d_model]
        q_mean = q.mean(dim=1)        # [B, d_model]
        gavg = V.mean(dim=1)          # [B, d_model]

        g = torch.cat([z_mean, gavg, q_mean], dim=-1)
        g = self.post(g)
        return g, K, V
