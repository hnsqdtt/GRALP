from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def circular_pad1d(x: torch.Tensor, pad: int) -> torch.Tensor:
    if pad <= 0:
        return x
    left = x[..., -pad:]
    right = x[..., :pad]
    return torch.cat([left, x, right], dim=-1)


def zero_pad1d(x: torch.Tensor, pad: int) -> torch.Tensor:
    if pad <= 0:
        return x
    return F.pad(x, (pad, pad), mode="constant", value=0.0)


class EncoderBase(nn.Module):
    """Common interface: vec [B, vec_dim] -> feature [B, feature_dim].

    Subclasses must set ``self.feature_dim`` (declared via the constructor) and
    implement ``forward``. ``PPOPolicy`` only depends on this contract.
    """

    def __init__(self, vec_dim: int, *, feature_dim: int = 256) -> None:
        super().__init__()
        self.vec_dim = int(vec_dim)
        self.feature_dim = int(feature_dim)

    def forward(self, vec: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError


class DilatedConv1DBlock(nn.Module):
    """Single Conv1d + BN + GELU with explicit padding (circular or zero)."""

    def __init__(self, ch: int, *, kernel: int = 5, dilation: int = 1,
                 padding_mode: str = "circular") -> None:
        super().__init__()
        if padding_mode not in ("circular", "zero"):
            raise ValueError(f"padding_mode must be 'circular' or 'zero', got {padding_mode!r}")
        self.kernel = int(kernel)
        self.dilation = int(dilation)
        self.padding_mode = padding_mode
        self.conv = nn.Conv1d(ch, ch, kernel_size=self.kernel, dilation=self.dilation, bias=False)
        self.bn = nn.BatchNorm1d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = ((self.kernel - 1) * self.dilation) // 2
        if self.padding_mode == "circular":
            x = circular_pad1d(x, pad)
        else:
            x = zero_pad1d(x, pad)
        return F.gelu(self.bn(self.conv(x)))


class RayPoseCNN1D(EncoderBase):
    """Shared 1D-CNN backbone for ray distances + pose features.

    Splits ``vec`` into ``[B, N]`` ray distances and ``[B, pose_dim]`` pose
    features. The ray branch runs a stack of dilated 1D convolutions with the
    configured padding mode, then global-mean-pools across the ray axis. The
    pose branch is a small MLP. Both are concatenated and projected to
    ``feature_dim`` (the contract consumed by ``PPOPolicy``).

    ``ray_pos_encoding``: when True, appends a constant ``[0, 1/(N-1), ..., 1]``
    channel alongside the ray distances before ``expand``. This breaks the ring
    symmetry that ``circular_pad + mean pool`` otherwise imposes (the ray branch
    of a pure circular CNN is bit-exactly invariant to circular shifts of the
    input). ray[0] is the robot-front bin and gets pos=0, so the model can tell
    "in front" from "behind" without losing the equivariance benefit of the
    convolution itself. Default False keeps the previous 1-channel architecture,
    so existing checkpoints load with ``strict=True``.
    """

    PADDING_MODE = "circular"

    def __init__(self, vec_dim: int, *, feature_dim: int = 256, channels: int = 32,
                 kernel: int = 5, dilations: tuple = (1, 2, 4, 8, 16),
                 pose_dim: int = 7, pose_hidden: int = 64,
                 ray_pos_encoding: bool = False) -> None:
        super().__init__(vec_dim, feature_dim=feature_dim)
        self.pose_dim = int(pose_dim)
        self.N = self.vec_dim - self.pose_dim
        if self.N <= 0:
            raise ValueError(
                f"vec_dim={vec_dim} too small for pose_dim={pose_dim} (need vec_dim > pose_dim)"
            )
        self.ray_pos_encoding = bool(ray_pos_encoding)
        in_ch = 2 if self.ray_pos_encoding else 1
        self.expand = nn.Conv1d(in_ch, int(channels), kernel_size=1)
        if self.ray_pos_encoding:
            pos = torch.arange(self.N, dtype=torch.float32) / max(self.N - 1, 1)
            self.register_buffer("_pos_emb", pos.view(1, 1, self.N), persistent=False)
        self.blocks = nn.Sequential(*[
            DilatedConv1DBlock(int(channels), kernel=int(kernel), dilation=int(d),
                               padding_mode=self.PADDING_MODE)
            for d in dilations
        ])
        self.pose_mlp = nn.Sequential(
            nn.Linear(self.pose_dim, int(pose_hidden)), nn.ReLU(),
            nn.Linear(int(pose_hidden), int(pose_hidden)), nn.ReLU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(int(channels) + int(pose_hidden), self.feature_dim), nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim), nn.ReLU(),
        )

    def forward(self, vec: torch.Tensor) -> torch.Tensor:
        rays = vec[:, :self.N].unsqueeze(1)
        if self.ray_pos_encoding:
            rays = torch.cat([rays, self._pos_emb.expand(rays.size(0), -1, -1)], dim=1)
        pose = vec[:, self.N:self.N + self.pose_dim]
        ray_feat = self.blocks(self.expand(rays)).mean(dim=-1)
        pose_feat = self.pose_mlp(pose)
        return self.fuse(torch.cat([ray_feat, pose_feat], dim=-1))
