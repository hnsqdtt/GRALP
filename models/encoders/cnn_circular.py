from __future__ import annotations

from .base import RayPoseCNN1D


class CNNCircularEncoder(RayPoseCNN1D):
    """1D CNN over ray distances with circular padding.

    Wraps the ring of rays so the receptive field is continuous across the
    first/last bin boundary — the topology-aware ablation arm.
    """

    PADDING_MODE = "circular"
