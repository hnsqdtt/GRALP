from __future__ import annotations

import torch
import torch.nn as nn

from .base import EncoderBase


class MLPEncoder(EncoderBase):
    """Plain fully-connected encoder over the flat observation vector.

    ``depth`` is the total number of Linear layers (>= 2). The final layer
    projects to ``feature_dim``; intermediate layers have ``hidden`` width.
    Use ``depth=2/3/4/5`` to instantiate the MLP-2/3/4/5 ablation arms.
    """

    def __init__(self, vec_dim: int, *, feature_dim: int = 256,
                 hidden: int = 256, depth: int = 3) -> None:
        super().__init__(vec_dim, feature_dim=feature_dim)
        if depth < 2:
            raise ValueError(f"MLPEncoder requires depth >= 2, got {depth}")
        layers = []
        in_dim = self.vec_dim
        for _ in range(int(depth) - 1):
            layers += [nn.Linear(in_dim, int(hidden)), nn.ReLU()]
            in_dim = int(hidden)
        layers += [nn.Linear(in_dim, self.feature_dim), nn.ReLU()]
        self.net = nn.Sequential(*layers)

    def forward(self, vec: torch.Tensor) -> torch.Tensor:
        return self.net(vec)
