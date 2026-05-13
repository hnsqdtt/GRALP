from __future__ import annotations

from .base import RayPoseCNN1D


class CNNZeroPadEncoder(RayPoseCNN1D):
    """1D CNN over ray distances with zero padding.

    Treats the ray sequence as an open interval — the baseline that ignores
    the ring topology of the 360-degree observation.
    """

    PADDING_MODE = "zero"
