from __future__ import annotations

"""Entry point for PPO training using the randomized GPU ray environment (map-free).

Usage:
    python -m rl_ppo.ppo_train --fresh [--tag NAME] [--train_config PATH]
    python -m rl_ppo.ppo_train --resume <run_dir-or-step-N.pt>
"""

import os
import re
import shutil
import time
import argparse
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import torch

from .ppo_models import PPOPolicy
from .ppo_buffer import RolloutBuffer
from env import load_json_config
from env.sim_gpu_env import SimGPUEnvConfig, SimRandomGPUBatchEnv, infer_obs_dim as _infer_obs_dim_sim


DEFAULT_TRAIN_CONFIG = os.path.join("config", "train_config.json")
_STEP_CKPT_PATTERN = re.compile(r"^step-(\d+)\.pt$")


def load_train_config(path: Optional[str]) -> Dict[str, Any]:
    cfg = load_json_config(path) if path else {}
    cfg.setdefault("device", "cuda:0")
    cfg.setdefault("env_config", "env_config.json")
    cfg.setdefault("mission_config", None)
    samp = cfg.setdefault("sampling", {})
    samp.setdefault("batch_env", 256)
    samp.setdefault("rollout_len", 128)
    samp.setdefault("reset_each_rollout", True)
    ppo = cfg.setdefault("ppo", {})
    ppo.setdefault("gamma", 0.99)
    ppo.setdefault("gae_lambda", 0.95)
    ppo.setdefault("clip_range", 0.2)
    ppo.setdefault("lr", 3e-4)
    ppo.setdefault("value_lr", 3e-4)
    ppo.setdefault("entropy_coef", 0.0)
    ppo.setdefault("value_coef", 0.5)
    ppo.setdefault("max_grad_norm", 0.5)
    ppo.setdefault("epochs", 4)
    ppo.setdefault("minibatch_size", 2048)
    ppo.setdefault("amp", True)
    ppo.setdefault("amp_bf16", True)
    ppo.setdefault("bootstrap", True)
    ppo.setdefault("log_std_min", -5.0)
    ppo.setdefault("log_std_max", 2.0)
    ppo.setdefault("collision_done", True)
    model = cfg.setdefault("model", {})
    model.setdefault("num_queries", 4)
    model.setdefault("num_heads", 4)

    run = cfg.setdefault("run", {})
    run.setdefault("total_env_steps", 2_000_000)
    run.setdefault("ckpt_dir", "runs")
    run.setdefault("log_interval", 20000)
    run.setdefault("eval_every", 100000)
    return cfg


def _extract_seed_from_train_or_env(train_cfg: Dict[str, Any], env_cfg: Dict[str, Any]) -> Optional[int]:
    try:
        run = train_cfg.get("run", {}) or {}
        seed_v = run.get("seed", None)
        if seed_v in (None, "", "null"):
            seed_v = None
        if seed_v is not None:
            return int(seed_v)
        sim = env_cfg.get("sim", {}) or {}
        s2 = sim.get("seed", None)
        return None if s2 in (None, "", "null") else int(s2)
    except Exception:
        return None


def _resolve_env_cfg_path(train_cfg_path: str, train_cfg: Dict[str, Any]) -> Optional[str]:
    env_cfg_path = train_cfg.get("env_config", None)
    if not env_cfg_path:
        return None
    if os.path.isabs(env_cfg_path):
        return env_cfg_path
    return os.path.join(os.path.dirname(os.path.abspath(train_cfg_path)), env_cfg_path)


def _setup_fresh_run_dir(runs_root: str, tag: Optional[str],
                         train_cfg_src: str, env_cfg_src: str) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{ts}-{tag}" if tag else ts
    run_dir = os.path.join(runs_root, name)
    os.makedirs(run_dir, exist_ok=False)
    shutil.copy2(train_cfg_src, os.path.join(run_dir, "train_config.json"))
    shutil.copy2(env_cfg_src, os.path.join(run_dir, "env_config.json"))
    return run_dir


def _resolve_resume(path: str) -> Tuple[str, str]:
    if os.path.isdir(path):
        run_dir = os.path.abspath(path)
        latest = os.path.join(run_dir, "latest.pt")
        if os.path.isfile(latest):
            return run_dir, latest
        candidates = []
        for name in os.listdir(run_dir):
            m = _STEP_CKPT_PATTERN.match(name)
            if m:
                candidates.append((int(m.group(1)), os.path.join(run_dir, name)))
        if not candidates:
            raise FileNotFoundError(f"No latest.pt or step-<N>.pt found in {run_dir}")
        candidates.sort(reverse=True)
        return run_dir, candidates[0][1]
    if os.path.isfile(path):
        return os.path.dirname(os.path.abspath(path)), os.path.abspath(path)
    raise FileNotFoundError(f"--resume target not found: {path}")


def _load_checkpoint(ckpt_file: str, policy: PPOPolicy, opt_pi: torch.optim.Optimizer) -> int:
    payload = torch.load(ckpt_file, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected checkpoint format at {ckpt_file}")
    step_v = int(payload.get("step", payload.get("global_step", 0)))
    state = payload.get("policy", None)
    if state is None:
        keys = list(payload.keys())
        if all(isinstance(k, str) for k in keys) and any(
            k.startswith(("encoder", "mu", "value", "log_std")) for k in keys
        ):
            state = payload
    if state is not None:
        policy.load_state_dict(state, strict=False)
    if "opt" in payload:
        try:
            opt_pi.load_state_dict(payload["opt"])
        except Exception:
            print("[PPO] Warning: failed to load optimizer state; continuing without it.")
    return step_v


def _save_checkpoint(run_dir: str, global_step: int,
                     policy: PPOPolicy, opt_pi: torch.optim.Optimizer) -> str:
    step_path = os.path.join(run_dir, f"step-{int(global_step)}.pt")
    torch.save({
        "step": int(global_step),
        "policy": policy.state_dict(),
        "opt": opt_pi.state_dict(),
    }, step_path)
    shutil.copy2(step_path, os.path.join(run_dir, "latest.pt"))
    return step_path


def _parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config", type=str, default=None,
                        help="Path to train_config.json (only used with --fresh)")
    parser.add_argument("--fresh", action="store_true",
                        help="Start a new run in <ckpt_dir>/<timestamp>[-<tag>]/")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a run directory or a specific step-<N>.pt")
    parser.add_argument("--tag", type=str, default=None,
                        help="Optional tag appended to the new run directory name")
    args = parser.parse_args()
    if bool(args.fresh) == bool(args.resume):
        parser.error("exactly one of --fresh or --resume is required")
    if args.resume and args.train_config:
        parser.error("--train_config is not accepted with --resume (config is read from the run directory)")
    if args.resume and args.tag:
        parser.error("--tag is only valid with --fresh")
    return args


def main():
    args = _parse_cli()

    if args.fresh:
        train_cfg_src = args.train_config or DEFAULT_TRAIN_CONFIG
        if not os.path.isfile(train_cfg_src):
            raise FileNotFoundError(f"train_config not found: {train_cfg_src}")
        cfg_bootstrap = load_train_config(train_cfg_src)
        env_cfg_src = _resolve_env_cfg_path(train_cfg_src, cfg_bootstrap)
        if not env_cfg_src or not os.path.isfile(env_cfg_src):
            raise FileNotFoundError(f"env_config not found: {env_cfg_src}")
        runs_root = cfg_bootstrap["run"]["ckpt_dir"]
        run_dir = _setup_fresh_run_dir(runs_root, args.tag, train_cfg_src, env_cfg_src)
        print(f"[PPO] Fresh run at {run_dir}")
        resume_ckpt: Optional[str] = None
    else:
        run_dir, resume_ckpt = _resolve_resume(args.resume)
        print(f"[PPO] Resuming from {resume_ckpt}")

    train_cfg_path = os.path.join(run_dir, "train_config.json")
    env_cfg_path = os.path.join(run_dir, "env_config.json")
    if not os.path.isfile(train_cfg_path):
        raise FileNotFoundError(f"Missing train_config.json in {run_dir}")
    if not os.path.isfile(env_cfg_path):
        raise FileNotFoundError(f"Missing env_config.json in {run_dir}")
    cfg = load_train_config(train_cfg_path)
    env_cfg = load_json_config(env_cfg_path)

    device = torch.device(
        cfg.get("device", "cuda:0") if (cfg.get("device", "cuda:0") == "cpu" or torch.cuda.is_available()) else "cpu"
    )

    obs_cfg = env_cfg.get("obs", {}) or {}
    sim_cfg = env_cfg.get("sim", {}) or {}
    lim_cfg = env_cfg.get("limits", {}) or {}
    rew_cfg = env_cfg.get("reward", {}) or {}
    ppo_cfg = cfg.get("ppo", {}) or {}

    B_env = int((cfg.get("sampling", {}) or {}).get("batch_env", 256))
    T_roll = int((cfg.get("sampling", {}) or {}).get("rollout_len", 128))
    reset_each_rollout = bool((cfg.get("sampling", {}) or {}).get("reset_each_rollout", True))
    run_seed = _extract_seed_from_train_or_env(cfg, env_cfg)
    if run_seed is not None:
        torch.manual_seed(int(run_seed))
    safe_dist = float(sim_cfg.get("safe_distance", sim_cfg.get("warning_distance", 0.5)))
    sim = SimGPUEnvConfig(
        dt=float(sim_cfg.get("dt", 0.1)),
        n_envs=B_env,
        patch_meters=float(obs_cfg.get("patch_meters", 10.0)),
        ray_step_m=float(obs_cfg.get("ray_step_m", 0.025)),
        n_rays=int(obs_cfg.get("n_rays", 0)),
        ray_max_gap=float(obs_cfg.get("ray_max_gap", 0.25)),
        safe_distance_m=safe_dist,
        vx_max=float(lim_cfg.get("vx_max", 1.5)),
        omega_max=float(lim_cfg.get("omega_max", 2.0)),
        w_collision=float(rew_cfg.get("reward_collision", 1.0)),
        w_progress=float(rew_cfg.get("reward_progress", 0.01)),
        w_limits=float(rew_cfg.get("reward_limits", 0.1)),
        orientation_verify=bool(rew_cfg.get("orientation_verify", False)),
        w_jerk=float(rew_cfg.get("reward_jerk", 0.0)),
        w_jerk_omega=float(rew_cfg.get("reward_jerk_omega", 0.0)),
        reward_time=float(rew_cfg.get("reward_time", 0.0)),
        blank_ratio_base=float((obs_cfg.get("blank_ratio_base", 40.0))),
        blank_ratio_randmax=float((obs_cfg.get("blank_ratio_randmax", 40.0))),
        blank_ratio_std_ratio=float(obs_cfg.get("blank_ratio_std_ratio", 0.33)),
        narrow_passage_gaussian=bool(obs_cfg.get("narrow_passage_gaussian", False)),
        narrow_passage_std_ratio=float(obs_cfg.get("narrow_passage_std_ratio", 0.3)),
        device=str(device),
        task_point_max_dist_m=float(sim_cfg.get("task_point_max_dist_m", 8.0)),
        task_point_success_radius_m=float(sim_cfg.get("task_point_success_radius_m", 0.25)),
        task_point_random_interval_max=int(sim_cfg.get("task_point_random_interval_max", 0)),
        collision_done=bool(ppo_cfg.get("collision_done", True)),
    )
    env = SimRandomGPUBatchEnv(sim)
    obs = env.reset()
    vec_dim = int(obs.shape[1]) if obs.dim() == 2 else int(_infer_obs_dim_sim(sim))
    act_dim = 2

    model_cfg = cfg.get("model", {}) or {}
    policy = PPOPolicy(
        vec_dim=vec_dim,
        action_dim=act_dim,
        num_queries=int(model_cfg.get("num_queries", 4)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        log_std_min=float((cfg.get("ppo", {}) or {}).get("log_std_min", -5.0)),
        log_std_max=float((cfg.get("ppo", {}) or {}).get("log_std_max", 2.0)),
    ).to(device)
    ppo_cfg = cfg.get("ppo", {}) or {}
    opt_pi = torch.optim.Adam([
        {"params": list(policy.encoder.parameters()) + list(policy.mu.parameters()) + [policy.log_std], "lr": float(ppo_cfg.get("lr", 3e-4))},
        {"params": policy.value.parameters(), "lr": float(ppo_cfg.get("value_lr", ppo_cfg.get("lr", 3e-4)))}
    ])

    use_cuda = device.type == "cuda" and torch.cuda.is_available()
    use_amp = bool(ppo_cfg.get("amp", True)) and use_cuda
    prefer_bf16 = bool(ppo_cfg.get("amp_bf16", True))
    amp_dtype = torch.bfloat16 if (use_amp and prefer_bf16 and getattr(torch.cuda, "is_bf16_supported", lambda: False)()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    global_step = 0
    if resume_ckpt is not None:
        global_step = _load_checkpoint(resume_ckpt, policy, opt_pi)
        print(f"[PPO] Loaded step={global_step:,}")
    log_interval = int((cfg.get("run", {}) or {}).get("log_interval", 20000))

    total_env_steps = int(cfg["run"]["total_env_steps"])
    if bool((cfg.get("run", {}) or {}).get("resume_as_additional", False)) and global_step > 0:
        total_env_steps = int(global_step) + int(cfg["run"]["total_env_steps"])
        print(f"[PPO] resume_as_additional=True => target_total_env_steps={total_env_steps:,} (loaded {global_step:,} + add {int(cfg['run']['total_env_steps']):,})")
    clip_eps = float(ppo_cfg.get("clip_range", 0.2))
    ent_coef = float(ppo_cfg.get("entropy_coef", 0.0))
    vf_coef = float(ppo_cfg.get("value_coef", 0.5))
    max_grad_norm = float(ppo_cfg.get("max_grad_norm", 0.5))
    epochs = int(ppo_cfg.get("epochs", 4))
    mb_size = int(ppo_cfg.get("minibatch_size", 2048))
    gamma = float(ppo_cfg.get("gamma", 0.99))
    lam = float(ppo_cfg.get("gae_lambda", 0.95))

    limits = env.get_limits()
    limits_b = limits.view(1, -1).expand(B_env, -1)
    last_log_step = global_step
    last_log_time = time.time()

    while global_step < total_env_steps:
        if reset_each_rollout:
            obs = env.reset()
        buf = RolloutBuffer(T_roll, B_env, obs_dim=obs.shape[1], act_dim=act_dim, device=device)
        with torch.no_grad():
            for _ in range(T_roll):
                out = policy.act(obs, limits_b)
                logp = out.logp
                act = out.action
                _, _, v = policy._core(obs)
                next_obs, reward_t, done_t, info = env.step(act)
                d = done_t
                buf.add(
                    obs=obs.detach(),
                    act=act.detach(),
                    logp=logp.detach(),
                    rew=reward_t.detach(),
                    done=d.detach().to(torch.float32),
                    val=v.detach(),
                    limits=limits_b.detach(),
                )

                global_step += B_env
                obs = next_obs

        if bool(ppo_cfg.get("bootstrap", True)):
            with torch.no_grad():
                _, _, last_v = policy._core(obs)
                last_v = last_v.view(B_env, 1)
            buf.compute_gae(last_v, gamma=gamma, lam=lam, count_timeout_as_done=True)
        else:
            buf.compute_mc_returns(gamma=gamma)
        for _ in range(epochs):
            for mb in buf.minibatches(mb_size):
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    new_logp, ent, v_pred = policy.evaluate_actions(mb.obs, mb.actions, mb.limits)
                    ratio = (new_logp - mb.logp).exp().clamp(0.0, 10.0)
                    surr1 = ratio * mb.advantages
                    surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * mb.advantages
                    pg_loss = -torch.min(surr1, surr2).mean()
                    v_loss = 0.5 * (mb.returns - v_pred).pow(2).mean()
                    ent_bonus = ent.mean()
                    loss = pg_loss + vf_coef * v_loss - ent_coef * ent_bonus

                opt_pi.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt_pi)
                else:
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                if scaler.is_enabled():
                    scaler.step(opt_pi)
                    scaler.update()
                else:
                    opt_pi.step()
        if global_step - last_log_step >= log_interval:
            now = time.time()
            fps = (global_step - last_log_step) / max(1e-3, now - last_log_time)
            print(f"[PPO] step={global_step:,} | fps={fps:.1f}")
            _save_checkpoint(run_dir, global_step, policy, opt_pi)
            last_log_step = global_step
            last_log_time = now

    _save_checkpoint(run_dir, global_step, policy, opt_pi)


if __name__ == "__main__":
    main()
