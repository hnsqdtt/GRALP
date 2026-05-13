from __future__ import annotations

from .writer import MetricWriter, close_writer, get_writer, init_writer

__all__ = [
    "MetricWriter",
    "init_writer",
    "get_writer",
    "close_writer",
]
