from __future__ import annotations

"""Episodic narrow-channel navigation: DWA baseline vs trained PPO policy.

Both planners face the SAME pre-generated set of (start, goal) episodes on the
SAME pillar-lattice map, so the comparison is apples-to-apples. For each planner
we report success / collision / timeout rates, SPL, and time/path on successes.

Usage:
    python -m eval.scripts.narrow_channel_eval
    python -m eval.scripts.narrow_channel_eval --channel-w 0.5 --pillars 6 --render
    python -m eval.scripts.narrow_channel_eval --planners dwa --n-episodes 256
    python -m eval.scripts.narrow_channel_eval --ckpt runs/<run>/latest.pt --channel-w 0.45 --res 0.02

Geometry knobs (metres): --pillars N, --pillar-w, --channel-w, --robot-radius,
--res. The lattice is passable iff channel_w > 2*robot_radius; start and goal
sit at corridor crossroad centres ("channel centres"). Lower --res for very
narrow channels so the raster keeps the passage open.

Output: runs/narrow_channel/<timestamp>/result.json (+ scenario.png with --render).
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from eval.ckpt_loader import (build_policy_for_eval, load_policy_weights,
                              resolve_ckpt_configs)
from eval.dwa.planner import DWAConfig, DWAPlanner

from .eval_env import build_sim_cfg
from .narrow_channel_env import (NarrowChannelEnv, NarrowChannelEpisodeSpec)
from .narrow_channel_scene import (NarrowChannelConfig, NarrowChannelMap,
                                   build_narrow_channel_map)


REPO = Path(__file__).resolve().parents[2]
DEFAULT_ENV_CFG = REPO / "config" / "env_config.json"
DEFAULT_DWA_CFG = REPO / "eval" / "dwa_config.json"
DEFAULT_OUT_DIR = REPO / "runs" / "narrow_channel"


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _reseed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _default_ckpt() -> Optional[Path]:
    """Newest runs/*/latest.pt, else None."""
    cands = sorted(DEFAULT_OUT_DIR.parent.glob("*/latest.pt"))
    return cands[-1] if cands else None


# ---------------------------------------------------------------------------
# Planner action functions
# ---------------------------------------------------------------------------


def make_dwa_act(env: NarrowChannelEnv, planner: DWAPlanner) -> Callable[[], torch.Tensor]:
    @torch.no_grad()
    def act() -> torch.Tensor:
        snap = env.snapshot_for_dwa()
        vx, w = planner.plan_batch(
            snap["vx_cur"], snap["omega_cur"],
            snap["target_x_local"], snap["target_y_local"], snap["rays_m"],
        )
        return torch.stack([vx, w], dim=-1)
    return act


def make_policy_act(env: NarrowChannelEnv, policy) -> Callable[[], torch.Tensor]:
    limits = env.get_limits()

    @torch.no_grad()
    def act() -> torch.Tensor:
        obs = env.observe()
        return policy.act_deterministic(obs, limits)
    return act


# ---------------------------------------------------------------------------
# Episodic rollout driver (shared by both planners)
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_episodic(env: NarrowChannelEnv, act_fn: Callable[[], torch.Tensor],
                 *, check_every: int = 16) -> Dict[str, Any]:
    """Drive ``env`` to completion (all envs retired) or until the step cap."""
    env.reset()
    max_total = env.episodes_per_env * env.max_steps + 4
    t0 = time.perf_counter()
    steps = 0
    while steps < max_total:
        env.step(act_fn())
        steps += 1
        if steps % check_every == 0 and env.all_done():
            break
    elapsed = time.perf_counter() - t0
    m = env.metrics()
    m["wall_sec"] = elapsed
    m["driver_steps"] = steps
    m["env_step_per_sec"] = (steps * env.B) / elapsed if elapsed > 0 else float("inf")
    return m


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------


def build_env(env_cfg: Dict[str, Any], ncmap: NarrowChannelMap, *,
              n_envs: int, device: str, ep: NarrowChannelEpisodeSpec) -> NarrowChannelEnv:
    sim_cfg = build_sim_cfg(env_cfg, n_envs=n_envs, device=str(ncmap.device))
    return NarrowChannelEnv(sim_cfg, ncmap, ep)


def load_policy(ckpt: Path, env: NarrowChannelEnv, *, model_override: Optional[str]):
    """Resolve configs around ``ckpt`` and build a PPOPolicy matching the env obs."""
    train_cfg, _env_cfg, model_configs, model_key, source = resolve_ckpt_configs(
        ckpt, model_override=model_override)
    vec_dim = int(env.observe().shape[-1])
    policy = build_policy_for_eval(train_cfg, model_configs, model_key, vec_dim, env.device)
    step = load_policy_weights(policy, ckpt, env.device)
    policy.eval()  # critical: use BatchNorm running stats, deterministic forward
    return policy, {"source": source, "model_key": model_key, "ckpt_step": step}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Narrow-channel goal navigation: DWA vs PPO policy")
    p.add_argument("--env-config", type=Path, default=DEFAULT_ENV_CFG)
    p.add_argument("--dwa-config", type=Path, default=DEFAULT_DWA_CFG)
    p.add_argument("--ckpt", type=Path, default=None,
                   help="PPO checkpoint .pt (default: newest runs/*/latest.pt).")
    p.add_argument("--model", type=str, default=None,
                   help="Override encoder key (default: from the ckpt's snapshot config).")
    p.add_argument("--planners", type=str, default="dwa,policy",
                   help="Comma list: dwa, policy. Default both.")
    # map geometry
    p.add_argument("--pillars", type=int, default=6, help="Pillars per side.")
    p.add_argument("--pillar-w", type=float, default=1.0, help="Pillar side length (m).")
    p.add_argument("--channel-w", type=float, default=0.5, help="Corridor width (m).")
    p.add_argument("--robot-radius", type=float, default=0.18, help="Footprint half-extent (m).")
    p.add_argument("--obstacle-inflation-extra", type=float, default=0.02,
                   help="Layer-2 EXTRA inflation for the LOS carrot only (m). "
                        "Must stay below (channel_w - 2*robot_radius)/2 or the carrot "
                        "loses forward guidance in the corridor.")
    p.add_argument("--wall-thickness", type=float, default=0.2)
    p.add_argument("--res", type=float, default=0.02, help="Raster resolution (m/cell).")
    # episode protocol
    p.add_argument("--n-envs", type=int, default=48)
    p.add_argument("--n-episodes", type=int, default=192,
                   help="Rounded up to a multiple of --n-envs.")
    p.add_argument("--max-steps", type=int, default=600, help="Per-episode timeout (steps).")
    p.add_argument("--success-radius", type=float, default=0.30, help="Arrival radius (m).")
    p.add_argument("--min-geo-nodes", type=int, default=3,
                   help="Min start/goal separation in corridor node-steps.")
    p.add_argument("--collision-mode", type=str, default="footprint",
                   choices=("footprint", "forward_ray"),
                   help="footprint = center in inflated map any direction (realistic); "
                        "forward_ray = motion-direction ray only (training-consistent).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None, help="cuda / cpu (default: auto).")
    # output
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--render", action="store_true",
                   help="Save scenario.png with env-0's first-episode trajectory per planner.")
    p.add_argument("--no-save", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_cli()
    for label, path in (("env_config", args.env_config), ("dwa_config", args.dwa_config)):
        if not path.is_file():
            print(f"{label} not found: {path}", file=sys.stderr)
            return 2

    planners = [s.strip().lower() for s in args.planners.split(",") if s.strip()]
    bad = [p for p in planners if p not in ("dwa", "policy")]
    if bad:
        print(f"unknown planner(s): {bad} (choose from dwa, policy)", file=sys.stderr)
        return 2

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    env_cfg = _load_json(args.env_config)
    dwa_cfg = _load_json(args.dwa_config)

    # ---- scenario ----
    map_cfg = NarrowChannelConfig(
        n_pillars=args.pillars, pillar_w=args.pillar_w, channel_w=args.channel_w,
        robot_radius=args.robot_radius,
        obstacle_inflation_extra=args.obstacle_inflation_extra,
        wall_thickness=args.wall_thickness, res=args.res,
    )
    _reseed(args.seed)
    ncmap = build_narrow_channel_map(map_cfg, device=device)
    print(ncmap.summary())
    if not ncmap.passable:
        print("\nMap is not passable at this resolution; aborting. "
              "Widen --channel-w or lower --res.", file=sys.stderr)
        return 3
    print()
    print(ncmap.ascii_art(max_cols=60))
    print()

    ep = NarrowChannelEpisodeSpec(
        n_episodes=args.n_episodes, max_episode_steps=args.max_steps,
        success_radius_m=args.success_radius, min_geo_nodes=args.min_geo_nodes,
        seed=args.seed, track_path0=bool(args.render),
        collision_mode=args.collision_mode,
    )

    # ---- resolve policy ckpt up front (so we fail fast) ----
    ckpt = args.ckpt or (_default_ckpt() if "policy" in planners else None)
    if "policy" in planners and (ckpt is None or not Path(ckpt).is_file()):
        print(f"policy requested but no checkpoint found "
              f"(--ckpt {ckpt}); running DWA only.", file=sys.stderr)
        planners = [p for p in planners if p != "policy"]

    results: Dict[str, Dict[str, Any]] = {}
    render_tracks = []  # (label, path_xy, start, goal)
    policy_meta: Dict[str, Any] = {}

    for name in planners:
        print(f"\n>>> Running planner: {name}")
        _reseed(args.seed)
        env = build_env(env_cfg, ncmap, n_envs=args.n_envs, device=device, ep=ep)

        if name == "dwa":
            cfg = DWAConfig.from_configs(env_cfg, dwa_cfg)
            print(f"    DWA grid={cfg.v_samples}x{cfg.omega_samples}  "
                  f"alpha/beta/gamma={cfg.alpha_heading}/{cfg.beta_clearance}/{cfg.gamma_velocity}  "
                  f"rr={cfg.robot_radius_m}")
            planner = DWAPlanner(cfg, device=env.device)
            act = make_dwa_act(env, planner)
        else:
            policy, policy_meta = load_policy(ckpt, env, model_override=args.model)
            print(f"    policy: {policy_meta['model_key']} from {policy_meta['source']} "
                  f"(step {policy_meta['ckpt_step']})")
            act = make_policy_act(env, policy)

        m = run_episodic(env, act)
        results[name] = m
        print(f"    success={m['success_rate']:.3f}  collision={m['collision_rate']:.3f}  "
              f"timeout={m['timeout_rate']:.3f}  SPL={m['spl_mean']:.3f}  "
              f"({m['n_episodes']} eps, {m['wall_sec']:.1f}s)")

        if args.render and env.path0:
            s = env.start_xy[0].cpu().numpy().tolist()
            g = env.goal_xy[0].cpu().numpy().tolist()
            render_tracks.append((name, np.asarray(env.path0, dtype=np.float32), s, g))

    # ---- comparison table ----
    bar = "=" * 88
    print("\n" + bar)
    print(f" Narrow-channel navigation | {args.pillars}x{args.pillars} pillars, "
          f"channel={args.channel_w}m, free={map_cfg.free_corridor_w:.3f}m | seed={args.seed}")
    print(bar)
    hdr = (f"{'planner':>8} {'success':>8} {'collide':>8} {'timeout':>8} {'SPL':>6} "
           f"{'t_succ(s)':>9} {'path(m)':>8} {'geo(m)':>7} {'episodes':>8}")
    print(hdr)
    for name in planners:
        m = results[name]
        print(f"{name:>8} {m['success_rate']:>8.3f} {m['collision_rate']:>8.3f} "
              f"{m['timeout_rate']:>8.3f} {m['spl_mean']:>6.3f} "
              f"{m['time_success_mean_s']:>9.2f} {m['path_success_mean_m']:>8.2f} "
              f"{m['geo_success_mean_m']:>7.2f} {m['n_episodes']:>8d}")
    print(bar)

    if args.no_save:
        return 0

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.out_dir / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    png_rel = None
    if args.render:
        png = out_dir / "scenario.png"
        title = (f"Narrow channel {args.pillars}x{args.pillars}  "
                 f"channel={args.channel_w}m free={map_cfg.free_corridor_w:.2f}m")
        written = ncmap.render(str(png), trajectories=render_tracks or None,
                               title=title, show_nodes=not render_tracks)
        png_rel = png.name if written else None

    payload = {
        "timestamp": ts,
        "device": device,
        "seed": args.seed,
        "env_config": str(args.env_config),
        "dwa_config": str(args.dwa_config),
        "ckpt": str(ckpt) if ckpt else None,
        "policy_meta": policy_meta,
        "map": {
            "n_pillars": args.pillars, "pillar_w": args.pillar_w,
            "channel_w": args.channel_w, "robot_radius": args.robot_radius,
            "obstacle_inflation_extra": args.obstacle_inflation_extra,
            "wall_thickness": args.wall_thickness, "res": args.res,
            "interior_m": map_cfg.interior, "pitch_m": map_cfg.pitch,
            "free_corridor_w_m": map_cfg.free_corridor_w,
            "los_corridor_w_m": map_cfg.los_corridor_w,
            "n_nodes": int(ncmap.nodes.shape[0]),
        },
        "episode": {
            "n_envs": args.n_envs, "n_episodes_requested": args.n_episodes,
            "episodes_per_env": int(np.ceil(max(1, args.n_episodes) / args.n_envs)),
            "max_steps": args.max_steps, "success_radius_m": args.success_radius,
            "min_geo_nodes": args.min_geo_nodes, "collision_mode": args.collision_mode,
        },
        "results": results,
        "render": png_rel,
    }
    out_path = out_dir / "result.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_path}")
    if png_rel:
        print(f"Wrote {out_dir / png_rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
