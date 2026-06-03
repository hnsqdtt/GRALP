from __future__ import annotations

"""Resolve a PPO checkpoint's accompanying configs and build a PPOPolicy.

Two cases:

1. The .pt sits in a snapshot run directory (the layout `python -m rl_ppo.train
   --fresh` produces): the sibling `train_config.json`, `env_config.json`, and
   `model_config.json` are authoritative. The model key defaults to
   `train_config["model"]` but can be overridden by ``--model``.

2. The .pt is a stray file (copied elsewhere): no sibling snapshots. The
   caller must pass ``--model <name>``. We fall back to the global
   ``config/train_config.json`` + ``config/env_config.json`` + ``config/
   model_config.json`` for everything else.

We deliberately do NOT build the env or policy in this module — the caller
constructs EvalEnv first (because vec_dim is inferred from a live env.reset())
and then asks us for the configured PPOPolicy via ``build_policy_for_eval``.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from models import PPOPolicy, build_encoder, resolve_model_entry

_REPO = Path(__file__).resolve().parents[1]
_GLOBAL_TRAIN = _REPO / "config" / "train_config.json"
_GLOBAL_ENV = _REPO / "config" / "env_config.json"
_GLOBAL_MODEL = _REPO / "config" / "model_config.json"


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_ckpt_configs(ckpt_path: Path, *,
                         model_override: Optional[str] = None
                         ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], str, str]:
    """Return ``(train_cfg, env_cfg, model_configs, model_key, source)``.

    ``source`` is either ``"snapshot:<run_dir_name>"`` or ``"global"``.
    """
    ckpt_path = Path(ckpt_path).resolve()
    run_dir = ckpt_path.parent
    train_p = run_dir / "train_config.json"
    env_p = run_dir / "env_config.json"
    model_p = run_dir / "model_config.json"

    has_snapshot = train_p.is_file() and env_p.is_file() and model_p.is_file()

    if has_snapshot:
        train_cfg = _load_json(train_p)
        env_cfg = _load_json(env_p)
        model_configs = _load_json(model_p)
        snapshot_key = str(train_cfg.get("model", "circular_attn"))
        if model_override and model_override != snapshot_key:
            print(f"[eval] WARNING: --model {model_override!r} overrides snapshot "
                  f"value {snapshot_key!r}; weight load may fail if architectures differ")
        model_key = model_override or snapshot_key
        source = f"snapshot:{run_dir.name}"
    else:
        if not model_override:
            raise ValueError(
                f"Checkpoint {ckpt_path} has no train/env/model_config.json siblings "
                "in its run directory; pass --model <name> to use global config."
            )
        for p in (_GLOBAL_TRAIN, _GLOBAL_ENV, _GLOBAL_MODEL):
            if not p.is_file():
                raise FileNotFoundError(f"Required global config missing: {p}")
        train_cfg = _load_json(_GLOBAL_TRAIN)
        env_cfg = _load_json(_GLOBAL_ENV)
        model_configs = _load_json(_GLOBAL_MODEL)
        model_key = model_override
        source = "global"

    return train_cfg, env_cfg, model_configs, model_key, source


def build_policy_for_eval(train_cfg: Dict[str, Any],
                          model_configs: Dict[str, Any],
                          model_key: str,
                          vec_dim: int,
                          device: torch.device) -> PPOPolicy:
    """Build PPOPolicy with the same constructor arguments as rl_ppo/train.py."""
    model_entry = resolve_model_entry(model_configs, model_key)
    encoder = build_encoder(model_entry, vec_dim=vec_dim)
    ppo_cfg = (train_cfg.get("ppo") or {})
    policy = PPOPolicy(
        encoder=encoder,
        action_dim=2,
        log_std_min=float(ppo_cfg.get("log_std_min", -5.0)),
        log_std_max=float(ppo_cfg.get("log_std_max", 2.0)),
    ).to(device)
    return policy


def load_policy_weights(policy: PPOPolicy, ckpt_path: Path, device: torch.device) -> int:
    """Load weights from ``ckpt_path`` into ``policy``. Returns the embedded step (0 if none)."""
    payload = torch.load(str(ckpt_path), map_location=device)
    if isinstance(payload, dict):
        if isinstance(payload.get("policy", None), dict):
            state = payload["policy"]
        elif isinstance(payload.get("state_dict", None), dict):
            state = payload["state_dict"]
        else:
            state = payload
        step = int(payload.get("step", payload.get("global_step", 0)))
    else:
        state = payload
        step = 0
    missing, unexpected = policy.load_state_dict(state, strict=False)
    if missing:
        print(f"[eval] WARNING: missing keys in ckpt: {sorted(missing)[:5]}"
              + (f" (+{len(missing)-5} more)" if len(missing) > 5 else ""))
    if unexpected:
        print(f"[eval] WARNING: unexpected keys in ckpt: {sorted(unexpected)[:5]}"
              + (f" (+{len(unexpected)-5} more)" if len(unexpected) > 5 else ""))
    return step
