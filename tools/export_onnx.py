#!/usr/bin/env python3
"""Export a trained PPO checkpoint (.pt) into a self-contained ``model/`` folder.

The folder this script produces is everything the simulation side needs to run
the policy without any GRALP source on hand:

    model/
      policy.pt    a copy of the source checkpoint
      policy.onnx  the exported ONNX graph (inputs: obs, limits; outputs: action, mu, log_std)
      meta.json    self-describing metadata: encoder kind + params + obs/action contract

Usage:
    # a single checkpoint file; cnn_* needs --model to disambiguate circular/zeropad
    python tools/export_onnx.py --ckpt runs/20250101-120000/latest.pt
    python tools/export_onnx.py --ckpt some_policy.pt --model cnn_circular

    # a training run located by tag
    python tools/export_onnx.py --tag my_run
    python tools/export_onnx.py --tag my_run --step 2000000

Config resolution:
    env_config.json and model_config.json are required; each is taken from the
    run directory beside the checkpoint when present, otherwise from config/.
    train_config.json is optional -- it only supplies the encoder kind and the
    log_std bounds, both of which fall back to weight inference / config/.

Options:
    --ckpt PATH    a checkpoint .pt file
    --tag NAME     a training run; resolves the newest runs/*-<NAME> folder
    --step N       with --tag, load step-<N>.pt instead of latest.pt
    --model KEY    encoder key in model_config.json (required for single-file cnn_* exports)
    -o, --output   output folder (default: ./model)
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONFIG_DIR = REPO_ROOT / "config"
POSE_DIM = 7
ACTION_DIM = 2  # GRALP actions are (vx, omega); limits below assume this layout


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _derive_rays(patch_meters: float, ray_max_gap: float) -> int:
    """Ray count R = ceil(2*pi*patch_meters / ray_max_gap)."""
    if patch_meters <= 0.0 or ray_max_gap <= 0.0:
        raise ValueError("patch_meters and ray_max_gap must be > 0 to derive the ray count")
    return int(math.ceil((2.0 * math.pi * patch_meters) / ray_max_gap))


def _resolve_config(snap_dir: Path, name: str, *, required: bool) -> Tuple[Optional[dict], Optional[Path]]:
    """Load a config file, preferring the snapshot beside the checkpoint, then config/."""
    for base in (snap_dir, CONFIG_DIR):
        path = base / name
        if path.is_file():
            return _load_json(path), path
    if required:
        raise FileNotFoundError(f"{name} not found in {snap_dir} or {CONFIG_DIR}")
    return None, None


def _resolve_run_dir_by_tag(runs_root: Path, tag: str) -> Path:
    """Newest runs/*-<tag> folder (matches the train.py tag convention)."""
    if not runs_root.is_dir():
        raise FileNotFoundError(f"runs root not found: {runs_root}")
    matches = [d for d in runs_root.iterdir()
               if d.is_dir() and (d.name == tag or d.name.endswith(f"-{tag}"))]
    if not matches:
        raise FileNotFoundError(f"no run folder matching tag {tag!r} under {runs_root}")
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def _ckpt_in_run(run_dir: Path, step: Optional[int]) -> Path:
    ckpt = run_dir / (f"step-{step}.pt" if step is not None else "latest.pt")
    if not ckpt.is_file():
        hint = "" if step is not None else "; pass --step <N>"
        raise FileNotFoundError(f"checkpoint not found: {ckpt}{hint}")
    return ckpt


def _load_state_dict(ckpt_path: Path) -> Tuple[Dict[str, "object"], Optional[int]]:
    """Return (policy state_dict, training step) from a checkpoint payload."""
    import torch

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and isinstance(payload.get("policy"), dict):
        return payload["policy"], payload.get("step")
    if isinstance(payload, dict) and payload and all(isinstance(k, str) for k in payload):
        if any(k.startswith(("encoder", "mu", "value", "log_std")) for k in payload):
            return payload, None  # bare state_dict
    raise ValueError(f"unrecognized checkpoint format: {ckpt_path}")


def _infer_model_keys(state_dict: Dict[str, "object"], model_configs: dict) -> List[str]:
    """Infer which model_config entries are structurally compatible with the weights.

    The state_dict key layout is a fingerprint of the encoder kind. cnn_circular
    and cnn_zeropad share identical parameters (padding is not learned), so a cnn
    checkpoint matches both -- the caller must then disambiguate via --model.
    """
    keys = set(state_dict.keys())
    names = {k: (v.get("name") if isinstance(v, dict) else None)
             for k, v in model_configs.items()}

    if "encoder.q_params" in keys:
        return [k for k, n in names.items() if n == "circular_attn"]
    if "encoder.net.0.weight" in keys:
        depth = sum(1 for k in keys if k.startswith("encoder.net.") and k.endswith(".weight"))
        return [k for k, v in model_configs.items()
                if names.get(k) == "mlp" and int((v.get("params") or {}).get("depth", -1)) == depth]
    if "encoder.expand.weight" in keys and "encoder.fuse.0.weight" in keys:
        return [k for k, n in names.items() if n in ("cnn_circular", "cnn_zeropad")]
    return []


def _export_onnx(policy, vec_dim: int, limits_row: List[float], onnx_path: Path,
                 *, vx_forward_only: bool = False) -> None:
    """Export a deterministic policy graph: (obs, limits) -> (action, mu, log_std).

    ``limits`` input semantics for the consumer are unchanged: ``[vx_max, omega_max]``
    (physical half-amplitudes). The forward-only choice is baked into the graph
    at export time via the ``vx_forward_only`` flag; the if/else is folded by
    the tracer into a single static path, so symmetric exports remain bit-exact
    with the legacy ``tanh(mu) * limits`` form.

    Uses the legacy TorchScript exporter (``dynamo=False``) on purpose: it emits a
    single-file graph with stable, explicitly-named I/O that the simulation side
    loads by name. The dynamo exporter is the future default but renames I/O and
    can spill external-data sidecars; its deprecation notice is silenced here
    since the choice is deliberate.
    """
    import warnings

    import torch

    class _Wrapper(torch.nn.Module):
        def __init__(self, p, vx_forward_only: bool):
            super().__init__()
            self.policy = p
            self.vx_forward_only = bool(vx_forward_only)

        def forward(self, obs, limits):
            # limits[..., 2] = (vx_max, omega_max). Trace-time bool folds away.
            mu, log_std, _ = self.policy._core(obs)
            vx_max = limits[..., 0:1]
            om_max = limits[..., 1:2]
            if self.vx_forward_only:
                vx_scale = 0.5 * vx_max
                vx_center = 0.5 * vx_max
            else:
                vx_scale = vx_max
                vx_center = torch.zeros_like(vx_max)
            scale = torch.cat([vx_scale, om_max], dim=-1)
            center = torch.cat([vx_center, torch.zeros_like(om_max)], dim=-1)
            return center + torch.tanh(mu) * scale, mu, log_std

    wrapper = _Wrapper(policy, vx_forward_only).eval()
    dummy_obs = torch.zeros((1, vec_dim), dtype=torch.float32)
    dummy_limits = torch.tensor([limits_row], dtype=torch.float32)

    for stale in (onnx_path, Path(str(onnx_path) + ".data")):
        if stale.exists():
            stale.unlink()

    with torch.no_grad(), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        torch.onnx.export(
            wrapper,
            (dummy_obs, dummy_limits),
            str(onnx_path),
            input_names=["obs", "limits"],
            output_names=["action", "mu", "log_std"],
            dynamic_axes={n: {0: "batch"} for n in ("obs", "limits", "action", "mu", "log_std")},
            opset_version=18,
            dynamo=False,
        )


def _verify(onnx_path: Path, policy, vec_dim: int, limits_row: List[float],
            *, vx_forward_only: bool = False,
            atol: float = 1e-4, rtol: float = 1e-3) -> None:
    """Run torch and onnxruntime on the same random batch and assert they agree."""
    import numpy as np
    import onnxruntime as ort
    import torch

    rng = np.random.default_rng(0)
    obs = rng.random((4, vec_dim), dtype=np.float32)
    limits = np.tile(np.asarray(limits_row, dtype=np.float32), (4, 1))

    with torch.no_grad():
        mu, log_std, _ = policy._core(torch.from_numpy(obs))
        lim_t = torch.from_numpy(limits)
        vx_max = lim_t[..., 0:1]
        om_max = lim_t[..., 1:2]
        if vx_forward_only:
            vx_scale = 0.5 * vx_max
            vx_center = 0.5 * vx_max
        else:
            vx_scale = vx_max
            vx_center = torch.zeros_like(vx_max)
        scale = torch.cat([vx_scale, om_max], dim=-1)
        center = torch.cat([vx_center, torch.zeros_like(om_max)], dim=-1)
        ref = {
            "action": (center + torch.tanh(mu) * scale).numpy(),
            "mu": mu.numpy(),
            "log_std": log_std.numpy(),
        }

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    got = dict(zip(out_names, sess.run(None, {"obs": obs, "limits": limits})))

    for name, ref_val in ref.items():
        max_err = float(np.max(np.abs(got[name] - ref_val)))
        ok = np.allclose(got[name], ref_val, atol=atol, rtol=rtol)
        print(f"[export_onnx]   {name:<8s} max|delta|={max_err:.2e}  {'OK' if ok else 'MISMATCH'}")
        if not ok:
            raise RuntimeError(f"ONNX/torch mismatch on {name!r}: max abs error {max_err:.3e}")


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a PPO checkpoint into a self-contained model/ folder.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", type=str, help="path to a checkpoint .pt file")
    src.add_argument("--tag", type=str, help="training run tag; resolves the newest runs/*-<tag> folder")
    parser.add_argument("--step", type=int, default=None,
                        help="with --tag: load step-<N>.pt instead of latest.pt")
    parser.add_argument("--model", type=str, default=None,
                        help="encoder key in model_config.json; required for cnn_* single-file exports")
    parser.add_argument("-o", "--output", type=str, default="model",
                        help="output folder (default: ./model)")
    args = parser.parse_args(argv)
    if args.step is not None and not args.tag:
        parser.error("--step is only valid together with --tag")
    return args


def main(argv: List[str]) -> int:
    args = _parse_args(argv)
    from models import PPOPolicy, build_encoder, resolve_model_entry

    # --- locate the checkpoint ---
    if args.tag:
        base_train = _load_json(CONFIG_DIR / "train_config.json")
        runs_root = Path((base_train.get("run") or {}).get("ckpt_dir", "runs"))
        if not runs_root.is_absolute():
            runs_root = REPO_ROOT / runs_root
        run_dir = _resolve_run_dir_by_tag(runs_root, args.tag)
        ckpt_path = _ckpt_in_run(run_dir, args.step)
    else:
        ckpt_path = Path(args.ckpt).expanduser().resolve()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    snap_dir = ckpt_path.parent
    print(f"[export_onnx] checkpoint : {ckpt_path}")

    # --- configs: env + model are required; train is optional ---
    # Each is read from the run directory if present, else from config/.
    env_cfg, env_src = _resolve_config(snap_dir, "env_config.json", required=True)
    model_configs, mc_src = _resolve_config(snap_dir, "model_config.json", required=True)
    run_train_path = snap_dir / "train_config.json"
    run_train_cfg = _load_json(run_train_path) if run_train_path.is_file() else None
    print(f"[export_onnx] env_config : {env_src}")
    print(f"[export_onnx] model_cfg  : {mc_src}")
    if run_train_cfg is None:
        print("[export_onnx] train_cfg  : not beside checkpoint "
              "(encoder kind inferred from weights; log_std bounds from config/)")

    # --- weights ---
    state_dict, step = _load_state_dict(ckpt_path)
    if "mu.weight" not in state_dict:
        raise ValueError("checkpoint has no 'mu.weight'; not a PPOPolicy state_dict")
    action_dim = int(state_dict["mu.weight"].shape[0])
    if action_dim != ACTION_DIM:
        raise SystemExit(
            f"[export_onnx] expected a {ACTION_DIM}-D (vx, omega) policy; got action_dim={action_dim}")

    # --- encoder kind: --model > run-dir train_config.model > weight fingerprint ---
    if args.model:
        model_key = args.model
    elif run_train_cfg is not None and run_train_cfg.get("model"):
        model_key = str(run_train_cfg["model"])
    else:
        candidates = _infer_model_keys(state_dict, model_configs)
        if len(candidates) == 1:
            model_key = candidates[0]
        elif len(candidates) > 1:
            raise SystemExit(
                f"[export_onnx] cnn weights cannot be told apart ({', '.join(sorted(candidates))}); "
                f"re-run with --model <key>")
        else:
            raise SystemExit("[export_onnx] could not infer the encoder kind; re-run with --model <key>")
    model_entry = resolve_model_entry(model_configs, model_key)  # raises if key is unknown

    # --- obs / action contract ---
    obs_cfg = env_cfg.get("obs") or {}
    lim_cfg = env_cfg.get("limits") or {}
    sim_cfg = env_cfg.get("sim") or {}
    # log_std bounds: run-dir train_config, else config/ train_config, else defaults.
    ppo_cfg = (run_train_cfg or {}).get("ppo")
    if ppo_cfg is None:
        cfg_train = CONFIG_DIR / "train_config.json"
        ppo_cfg = (_load_json(cfg_train).get("ppo") or {}) if cfg_train.is_file() else {}
    patch_meters = float(obs_cfg["patch_meters"])
    ray_max_gap = float(obs_cfg["ray_max_gap"])
    vx_max = float(lim_cfg["vx_max"])
    vx_forward_only = bool(lim_cfg.get("vx_forward_only", False))
    omega_max = float(lim_cfg["omega_max"])
    dt = float(sim_cfg.get("dt", 0.1))
    log_std_min = float(ppo_cfg.get("log_std_min", -5.0))
    log_std_max = float(ppo_cfg.get("log_std_max", 2.0))
    rays = _derive_rays(patch_meters, ray_max_gap)
    vec_dim = rays + POSE_DIM

    # --- build, strict-load (mismatch => the weights do not fit this model) ---
    encoder = build_encoder(model_entry, vec_dim=vec_dim)
    policy = PPOPolicy(encoder=encoder, action_dim=action_dim,
                       log_std_min=log_std_min, log_std_max=log_std_max)
    try:
        policy.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise SystemExit(
            f"[export_onnx] weights do not match model {model_key!r} "
            f"({model_entry.get('name')}):\n{exc}")
    policy.eval()
    print(f"[export_onnx] model      : {model_key} ({model_entry.get('name')}) | "
          f"rays={rays} obs_dim={vec_dim} action_dim={action_dim}")

    # --- export + verify ---
    out_dir = Path(args.output).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "policy.onnx"
    limits_row = [vx_max, omega_max]
    _export_onnx(policy, vec_dim, limits_row, onnx_path, vx_forward_only=vx_forward_only)
    print(f"[export_onnx] exported   : {onnx_path}  "
          f"(vx_forward_only={vx_forward_only})")
    print("[export_onnx] verifying onnxruntime vs torch ...")
    _verify(onnx_path, policy, vec_dim, limits_row, vx_forward_only=vx_forward_only)

    # --- finish the model/ folder ---
    shutil.copy2(ckpt_path, out_dir / "policy.pt")
    meta = {
        "model": model_key,
        "encoder": {"name": model_entry.get("name"), "params": model_entry.get("params") or {}},
        "action_dim": action_dim,
        "log_std_min": log_std_min,
        "log_std_max": log_std_max,
        "obs": {
            "patch_meters": patch_meters,
            "ray_max_gap": ray_max_gap,
            "rays": rays,
            "pose_dim": POSE_DIM,
            "obs_dim": vec_dim,
        },
        "limits": {"vx_max": vx_max, "omega_max": omega_max,
                   "vx_forward_only": vx_forward_only},
        "dt": dt,
        "source": ckpt_path.name,
    }
    if step is not None:
        meta["step"] = int(step)
    _dump_json(out_dir / "meta.json", meta)

    print(f"[export_onnx] model/ folder ready: {out_dir.resolve()}")
    print("[export_onnx]   policy.pt  policy.onnx  meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
