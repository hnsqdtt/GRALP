from __future__ import annotations

from typing import Any, Dict, Mapping

from .encoders import (
    CNNCircularEncoder,
    CNNZeroPadEncoder,
    EncoderBase,
    CircularAttnEncoder,
    MLPEncoder,
)
from .policy import PPOActOut, PPOPolicy

ENCODER_REGISTRY: Dict[str, type] = {
    "mlp": MLPEncoder,
    "cnn_zeropad": CNNZeroPadEncoder,
    "cnn_circular": CNNCircularEncoder,
    "circular_attn": CircularAttnEncoder,
}


def build_encoder(model_cfg: Mapping[str, Any], vec_dim: int) -> EncoderBase:
    """Instantiate an encoder from a ``{name, params}`` dict.

    ``model_cfg`` is a single entry of ``model_config.json`` (the one selected
    by ``train_config.model``). The ``name`` field must match a key in
    ``ENCODER_REGISTRY``; ``params`` is forwarded as kwargs.
    """
    if not isinstance(model_cfg, Mapping):
        raise TypeError(f"model_cfg must be a mapping, got {type(model_cfg).__name__}")
    name = model_cfg.get("name")
    if name is None:
        raise KeyError("model_cfg missing required 'name' field")
    if name not in ENCODER_REGISTRY:
        raise KeyError(
            f"Unknown encoder {name!r}; available: {sorted(ENCODER_REGISTRY)}"
        )
    params = dict(model_cfg.get("params") or {})
    return ENCODER_REGISTRY[name](vec_dim=int(vec_dim), **params)


def resolve_model_entry(model_configs: Mapping[str, Any], key: str) -> Dict[str, Any]:
    """Look up a model entry by ``key`` in the parsed ``model_config.json``."""
    if key not in model_configs:
        raise KeyError(
            f"Model {key!r} not found in model_config.json; "
            f"available: {sorted(model_configs)}"
        )
    entry = model_configs[key]
    if not isinstance(entry, Mapping):
        raise TypeError(
            f"model_config[{key!r}] must be a mapping, got {type(entry).__name__}"
        )
    return dict(entry)


__all__ = [
    "EncoderBase",
    "MLPEncoder",
    "CNNZeroPadEncoder",
    "CNNCircularEncoder",
    "CircularAttnEncoder",
    "PPOPolicy",
    "PPOActOut",
    "ENCODER_REGISTRY",
    "build_encoder",
    "resolve_model_entry",
]
