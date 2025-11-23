from __future__ import annotations

"""Entry point for PPO training using the randomized GPU ray environment (map-free).

Usage:
    python -m rl_ppo.ppo_train --train_config config/train_config.json
"""

import os
import time
import argparse
from typing import Any, Dict, Optional

import torch

from .ppo_models import PPOPolicy
from .ppo_buffer import RolloutBuffer
from env import load_json_config
from env.sim_gpu_env import SimGPUEnvConfig, SimRandomGPUBatchEnv, infer_obs_dim as _infer_obs_dim_sim


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
    run.setdefault("ckpt_dir", "runs/ppo_exp1")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config", type=str, default=os.path.join("config", "train_config.json"))
    args = parser.parse_args()

    cfg = load_train_config(args.train_config)
    cfg_dir = os.path.dirname(os.path.abspath(args.train_config))
    env_cfg_path = cfg.get("env_config", None)
    if env_cfg_path and not os.path.isabs(env_cfg_path):
        env_cfg_path = os.path.join(cfg_dir, env_cfg_path)

    device = torch.device(cfg.get("device", "cuda:0") if (cfg.get("device", "cuda:0") == "cpu" or torch.cuda.is_available()) else "cpu")

    env_cfg = load_json_config(env_cfg_path) if env_cfg_path else {}
    obs_cfg = env_cfg.get('obs', {}) or {}
    sim_cfg = env_cfg.get('sim', {}) or {}
    lim_cfg = env_cfg.get('limits', {}) or {}
    rew_cfg = env_cfg.get('reward', {}) or {}
    ppo_cfg = cfg.get('ppo', {}) or {}

    B_env = int((cfg.get('sampling', {}) or {}).get('batch_env', 256))
    T_roll = int((cfg.get('sampling', {}) or {}).get('rollout_len', 128))
    reset_each_rollout = bool((cfg.get('sampling', {}) or {}).get('reset_each_rollout', True))
    run_seed = _extract_seed_from_train_or_env(cfg, env_cfg)
    if run_seed is not None:
        torch.manual_seed(int(run_seed))
    safe_dist = float(sim_cfg.get('safe_distance', sim_cfg.get('warning_distance', 0.5)))
    sim = SimGPUEnvConfig(
        dt=float(sim_cfg.get('dt', 0.1)),
        n_envs=B_env,
        patch_meters=float(obs_cfg.get('patch_meters', 10.0)),
        ray_step_m=float(obs_cfg.get('ray_step_m', 0.025)),
        n_rays=int(obs_cfg.get('n_rays', 0)),
        ray_max_gap=float(obs_cfg.get('ray_max_gap', 0.25)),
        safe_distance_m=safe_dist,
        vx_max=float(lim_cfg.get('vx_max', 1.5)),
        omega_max=float(lim_cfg.get('omega_max', 2.0)),
        w_collision=float(rew_cfg.get('reward_collision', 1.0)),
        w_progress=float(rew_cfg.get('reward_progress', 0.01)),
        w_limits=float(rew_cfg.get('reward_limits', 0.1)),
        orientation_verify=bool(rew_cfg.get('orientation_verify', False)),
        w_jerk=float(rew_cfg.get('reward_jerk', 0.0)),
        w_jerk_omega=float(rew_cfg.get('reward_jerk_omega', 0.0)),
        reward_time=float(rew_cfg.get('reward_time', 0.0)),
        blank_ratio_base=float((obs_cfg.get('blank_ratio_base', 40.0))),
        blank_ratio_randmax=float((obs_cfg.get('blank_ratio_randmax', 40.0))),
        blank_ratio_std_ratio=float(obs_cfg.get('blank_ratio_std_ratio', 0.33)),
        narrow_passage_gaussian=bool(obs_cfg.get('narrow_passage_gaussian', False)),
        narrow_passage_std_ratio=float(obs_cfg.get('narrow_passage_std_ratio', 0.3)),
        device=str(device),
        task_point_max_dist_m=float(sim_cfg.get('task_point_max_dist_m', 8.0)),
        task_point_success_radius_m=float(sim_cfg.get('task_point_success_radius_m', 0.25)),
        task_point_random_interval_max=int(sim_cfg.get('task_point_random_interval_max', 0)),
        collision_done=bool(ppo_cfg.get('collision_done', True)),
    )
    env = SimRandomGPUBatchEnv(sim)
    obs = env.reset()
    vec_dim = int(obs.shape[1]) if obs.dim() == 2 else int(_infer_obs_dim_sim(sim))
    act_dim = 2

    model_cfg = cfg.get('model', {}) or {}
    policy = PPOPolicy(
        vec_dim=vec_dim,
        action_dim=act_dim,
        num_queries=int(model_cfg.get('num_queries', 4)),
        num_heads=int(model_cfg.get('num_heads', 4)),
        log_std_min=float((cfg.get('ppo', {}) or {}).get('log_std_min', -5.0)),
        log_std_max=float((cfg.get('ppo', {}) or {}).get('log_std_max', 2.0)),
    ).to(device)
    ppo_cfg = cfg.get('ppo', {}) or {}
    opt_pi = torch.optim.Adam([
        {"params": list(policy.encoder.parameters()) + list(policy.mu.parameters()) + [policy.log_std], "lr": float(ppo_cfg.get('lr', 3e-4))},
        {"params": policy.value.parameters(), "lr": float(ppo_cfg.get('value_lr', ppo_cfg.get('lr', 3e-4)))}
    ])

    use_cuda = device.type == 'cuda' and torch.cuda.is_available()
    use_amp = bool(ppo_cfg.get("amp", True)) and use_cuda
    prefer_bf16 = bool(ppo_cfg.get("amp_bf16", True))
    amp_dtype = torch.bfloat16 if (use_amp and prefer_bf16 and getattr(torch.cuda, "is_bf16_supported", lambda: False)()) else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and amp_dtype == torch.float16))
    try:
        _choice = input("Retrain from scratch? Create a fresh latest.pt overriding ckpt_dir/latest.pt [y/n]: ").strip().lower()
    except Exception:
        _choice = 'n'

    global_step = 0
    last_log = 0

    if _choice == 'y':
        ckpt_dir = cfg['run']['ckpt_dir']
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, 'latest.pt')
        torch.save({
            'step': 0,
            'policy': policy.state_dict(),
            'opt': opt_pi.state_dict(),
        }, ckpt_path)
        print(f"[PPO] Created fresh checkpoint at {ckpt_path} (step=0)")
        global_step = 0
        last_log = 0
    else:
        ckpt_dir = cfg['run']['ckpt_dir']
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_candidates = []
        prefer = os.path.join(ckpt_dir, 'latest.pt')
        if os.path.exists(prefer):
            ckpt_candidates.append(prefer)
        fallback_final = os.path.join(ckpt_dir, 'final.pt')
        if os.path.exists(fallback_final):
            ckpt_candidates.append(fallback_final)
        try:
            other_pts = []
            for name in os.listdir(ckpt_dir):
                if name.endswith('.pt') and name not in ('latest.pt', 'final.pt'):
                    other_pts.append(os.path.join(ckpt_dir, name))
            other_pts.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            ckpt_candidates.extend(other_pts)
        except Exception:
            pass

        ckpt_path = ckpt_candidates[0] if ckpt_candidates else os.path.join(ckpt_dir, 'latest.pt')

        if os.path.exists(ckpt_path):
            try:
                payload = torch.load(ckpt_path, map_location='cpu')
                step_v = 0
                try:
                    if isinstance(payload, dict):
                        if 'step' in payload:
                            step_v = int(payload['step'])
                        elif 'global_step' in payload:
                            step_v = int(payload['global_step'])
                        elif 'env_steps' in payload:
                            step_v = int(payload['env_steps'])
                except Exception:
                    step_v = 0

                state = None
                if isinstance(payload, dict):
                    state = payload.get('policy', None)
                    if state is None:
                        keys = list(payload.keys())
                        if all(isinstance(k, str) for k in keys) and any(k.startswith('encoder') or k.startswith('mu') for k in keys):
                            state = payload
                if state is not None:
                    policy.load_state_dict(state, strict=False)

                if isinstance(payload, dict) and 'opt' in payload:
                    try:
                        opt_pi.load_state_dict(payload['opt'])
                    except Exception:
                        print("[PPO] Warning: failed to load optimizer state; continuing without it.")

                global_step = int(step_v)
                try:
                    log_int = int((cfg.get('run', {}) or {}).get('log_interval', 20000))
                except Exception:
                    log_int = 20000
                last_log = max(0, int(global_step) - log_int)

                print(f"[PPO] Resumed from {ckpt_path} | loaded_step={global_step:,}")
            except Exception as e:
                print(f"[PPO] Failed to resume from {ckpt_path}: {e}")
        else:
            print(f"[PPO] No existing checkpoint to resume at {ckpt_path}; starting fresh.")

    total_env_steps = int(cfg['run']['total_env_steps'])
    if bool((cfg.get('run', {}) or {}).get('resume_as_additional', False)) and global_step > 0:
        total_env_steps = int(global_step) + int(cfg['run']['total_env_steps'])
        print(f"[PPO] resume_as_additional=True => target_total_env_steps={total_env_steps:,} (loaded {global_step:,} + add {int(cfg['run']['total_env_steps']):,})")
    clip_eps = float(ppo_cfg.get('clip_range', 0.2))
    ent_coef = float(ppo_cfg.get('entropy_coef', 0.0))
    vf_coef = float(ppo_cfg.get('value_coef', 0.5))
    max_grad_norm = float(ppo_cfg.get('max_grad_norm', 0.5))
    epochs = int(ppo_cfg.get('epochs', 4))
    mb_size = int(ppo_cfg.get('minibatch_size', 2048))
    gamma = float(ppo_cfg.get('gamma', 0.99))
    lam = float(ppo_cfg.get('gae_lambda', 0.95))
    collision_done = bool((cfg.get('ppo', {}) or {}).get('collision_done', True))

    limits = env.get_limits()
    limits_b = limits.view(1, -1).expand(B_env, -1)
    t0 = time.time()

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

        if bool(ppo_cfg.get('bootstrap', True)):
            with torch.no_grad():
                _, _, last_v = policy._core(obs)
                last_v = last_v.view(B_env, 1)
            buf.compute_gae(last_v, gamma=gamma, lam=lam, count_timeout_as_done=True)
        else:
            buf.compute_mc_returns(gamma=gamma)
        for _ in range(epochs):
            for mb in buf.minibatches(mb_size):
                with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
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
        if global_step - last_log >= int(cfg['run']['log_interval']):
            elapsed = time.time() - t0
            fps = global_step / max(1e-3, elapsed)
            print(f"[PPO] step={global_step:,} | fps={fps:.1f} | BxT={B_env}x{T_roll} | epochs={epochs}")
            try:
                with torch.no_grad():
                    log_std_vec = policy.log_std.clamp(policy._log_std_min, policy._log_std_max)
                    log_std_vals = [float(x) for x in log_std_vec.detach().cpu().view(-1)]
                print(f"[PPO] action log_std per-dim: {log_std_vals}")
            except Exception as _e:
                print(f"[PPO] (warn) failed to print log_std: {_e}")
            os.makedirs(cfg['run']['ckpt_dir'], exist_ok=True)
            torch.save({
                'step': int(global_step),
                'policy': policy.state_dict(),
                'opt': opt_pi.state_dict(),
            }, os.path.join(cfg['run']['ckpt_dir'], 'latest.pt'))
            last_log = global_step

    os.makedirs(cfg['run']['ckpt_dir'], exist_ok=True)
    torch.save({
        'step': int(global_step),
        'policy': policy.state_dict(),
        'opt': opt_pi.state_dict(),
    }, os.path.join(cfg['run']['ckpt_dir'], 'final.pt'))


if __name__ == "__main__":
    main()
