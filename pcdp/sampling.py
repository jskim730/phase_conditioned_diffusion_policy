"""DDIM action sampling and rollout utilities for diffusion policies."""

import numpy as np
import torch
from typing import Callable, Optional

from .dataset import encode_phase_continuation_cossin, encode_phase_cossin


# =====================================================================
# NaN-safe statistics shared by rollout summaries and evaluation tables
# =====================================================================

def nanmean(values: np.ndarray) -> float:
    """NaN-safe mean that returns NaN without emitting all-NaN warnings."""
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else float('nan')


def nanstd(values: np.ndarray) -> float:
    """NaN-safe std that returns NaN without emitting all-NaN warnings."""
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    return float(finite.std()) if finite.size else float('nan')


# =====================================================================
# Conditioning extractors at sampling time
# =====================================================================

def vanilla_sample_cond_fn(obs_window: torch.Tensor, phase_chunk=None, device='cuda'):
    """Vanilla DP: global_cond = flattened obs window. phase는 무시."""
    obs_window = obs_window.to(device)
    global_cond = obs_window.flatten(start_dim=1)
    return global_cond, None


def periodic_phase_sample_cond_fn(obs_window: torch.Tensor, phase_chunk: torch.Tensor,
                                   device='cuda'):
    """Step 3: global_cond = [flattened obs, (cosφ_0, sinφ_0)].

    phase_chunk: (B, PH) — chunk 첫 step의 phase만 사용.
    """
    obs_window = obs_window.to(device)
    phase_chunk = phase_chunk.to(device)
    phi0 = phase_chunk[:, 0]
    phase_enc = encode_phase_cossin(phi0)
    global_cond = torch.cat([obs_window.flatten(start_dim=1), phase_enc], dim=-1)
    return global_cond, None


def trajectory_phase_sample_cond_fn(obs_window: torch.Tensor,
                                     phase_chunk: torch.Tensor,
                                     device: str = 'cuda'):
    """Step 4 sampling: per-step phase trajectory.

    phase_chunk: (B, PH) — full chunk phase trajectory.
    Returns (global_cond, per_step_cond) where per_step_cond is (B, PH, 2).
    """
    obs_window = obs_window.to(device)
    phase_chunk = phase_chunk.to(device)

    global_cond = obs_window.flatten(start_dim=1)
    per_step_cond = encode_phase_cossin(phase_chunk)
    return global_cond, per_step_cond


def trajectory_phase_continuation_sample_cond_fn(
    obs_window: torch.Tensor,
    phase_chunk: torch.Tensor,
    device: str = 'cuda',
):
    """Sampling-time phase continuation condition with four per-step channels."""
    obs_window = obs_window.to(device)
    phase_chunk = phase_chunk.to(device)

    global_cond = obs_window.flatten(start_dim=1)
    per_step_cond = encode_phase_continuation_cossin(phase_chunk)
    return global_cond, per_step_cond


# =====================================================================
# DDIM sampling
# =====================================================================

@torch.no_grad()
def sample_action_chunk(
    model,
    ema,
    noise_scheduler_config: dict,
    obs_window: torch.Tensor,
    phase_chunk: Optional[torch.Tensor] = None,
    cond_fn: Callable = vanilla_sample_cond_fn,
    pred_horizon: int = 16,
    act_dim: int = 8,
    num_inference_steps: int = 16,
    use_ema: bool = True,
    device: str = 'cuda',
    seed: Optional[int] = None,
) -> torch.Tensor:
    """DDIM sampling. EMA weights를 inference에 사용 (DP 표준).

    seed: torch.manual_seed 적용. 재현성 필요 시 사용 (paper evaluation).
    """
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if use_ema:
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
    model.eval()

    # Batch 차원 처리
    if obs_window.dim() == 2:
        obs_window = obs_window.unsqueeze(0)
    if phase_chunk is not None and phase_chunk.dim() == 1:
        phase_chunk = phase_chunk.unsqueeze(0)
    obs_window = obs_window.to(device)
    B = obs_window.shape[0]

    global_cond, per_step_cond = cond_fn(obs_window, phase_chunk, device=device)

    sched = DDIMScheduler(
        num_train_timesteps=noise_scheduler_config['num_train_timesteps'],
        beta_schedule=noise_scheduler_config['beta_schedule'],
        prediction_type=noise_scheduler_config['prediction_type'],
        clip_sample=noise_scheduler_config['clip_sample'],
    )
    sched.set_timesteps(num_inference_steps)

    x = torch.randn(B, pred_horizon, act_dim, device=device)

    for t in sched.timesteps:
        if per_step_cond is None:
            noise_pred = model(x, t, global_cond)
        else:
            noise_pred = model(x, t, global_cond, per_step_cond)
        x = sched.step(noise_pred, t, x).prev_sample

    if use_ema:
        ema.restore(model.parameters())

    return x.cpu()


# =====================================================================
# MuJoCo Ant rollout
# =====================================================================

@torch.no_grad()
def rollout(
    model,
    ema,
    env,
    noise_scheduler_config: dict,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    act_min: np.ndarray,
    act_range: np.ndarray,
    cond_fn: Callable = vanilla_sample_cond_fn,
    obs_horizon: int = 2,
    pred_horizon: int = 16,
    action_horizon: int = 8,
    obs_dim: int = 105,
    act_dim: int = 8,
    num_inference_steps: int = 16,
    max_steps: int = 300,
    seed: int = 0,
    render: bool = False,
    phase_trajectory_fn: Optional[Callable] = None,
    device: str = 'cuda',
    sampling_seed_base: Optional[int] = None,
) -> dict:
    """학습된 DP를 MuJoCo Ant 환경에서 rollout.

    Receding-horizon control: chunk 16 step 샘플링 → 앞 8 step 실행 → re-plan.

    phase_trajectory_fn: Step 3+에서 사용. (current_t_global, pred_horizon) → (PH,) phase array.
        None이면 Vanilla DP.
    """
    # EMA weights 적용
    ema.store(model.parameters())
    ema.copy_to(model.parameters())
    model.eval()

    obs, _ = env.reset(seed=seed)
    obs = obs[:obs_dim]
    obs_history = [obs.copy() for _ in range(obs_horizon)]   # 초기엔 같은 obs 반복

    total_reward = 0.0
    survival = 0
    obs_log, act_log, frames = [], [], []
    x_velocity_log, reward_forward_log, x_position_log = [], [], []

    step = 0
    while step < max_steps:
        # Obs window 정규화
        obs_arr = np.stack(obs_history[-obs_horizon:])         # (OH, OD)
        obs_n = (obs_arr - obs_mean) / obs_std
        obs_tensor = torch.from_numpy(obs_n).float()

        # Phase trajectory (Step 3+에서 사용)
        phase_chunk = None
        if phase_trajectory_fn is not None:
            phase_arr = phase_trajectory_fn(step, pred_horizon)  # (PH,)
            phase_chunk = torch.from_numpy(phase_arr.astype(np.float32))

        # Sample chunk (use_ema=False → 위에서 이미 EMA 적용 중이라 중복 방지)
        # Sampling seed: 매 chunk마다 deterministic + 다른 값 (rollout 내 다양성 유지)
        chunk_seed = None
        if sampling_seed_base is not None:
            chunk_seed = sampling_seed_base * 100000 + step
        
        chunk_n = sample_action_chunk(
            model, ema, noise_scheduler_config,
            obs_tensor, phase_chunk=phase_chunk,
            cond_fn=cond_fn,
            pred_horizon=pred_horizon, act_dim=act_dim,
            num_inference_steps=num_inference_steps,
            use_ema=False, device=device,
            seed=chunk_seed,                     # ← 추가
        ).numpy()[0]

        # Unnormalize
        chunk = (chunk_n + 1.0) / 2.0 * act_range + act_min
        low, high = env.action_space.low, env.action_space.high
        chunk = np.clip(chunk, low, high)

        # Execute action_horizon steps
        for k in range(min(action_horizon, max_steps - step)):
            action = chunk[k]
            obs_next, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            survival += 1
            step += 1

            obs_next_t = obs_next[:obs_dim]
            obs_history.append(obs_next_t)
            obs_log.append(obs_next_t)
            act_log.append(action)
            x_velocity_log.append(float(info.get('x_velocity', np.nan)))
            reward_forward_log.append(float(info.get('reward_forward', np.nan)))
            x_position_log.append(float(info.get('x_position', np.nan)))

            if render:
                frames.append(env.render())
            if terminated or truncated:
                break
        if terminated or truncated:
            break

    ema.restore(model.parameters())

    x_velocity_arr = np.asarray(x_velocity_log, dtype=np.float32)
    reward_forward_arr = np.asarray(reward_forward_log, dtype=np.float32)
    x_position_arr = np.asarray(x_position_log, dtype=np.float32)

    return {
        'total_reward': float(total_reward),
        'survival':     int(survival),
        'reward_per_step': float(total_reward / max(survival, 1)),
        'mean_x_velocity': nanmean(x_velocity_arr),
        'std_x_velocity': nanstd(x_velocity_arr),
        'x_displacement': float(x_position_arr[-1] - x_position_arr[0]) if x_position_arr.size >= 2 and np.isfinite(x_position_arr[[0, -1]]).all() else float('nan'),
        'mean_reward_forward': nanmean(reward_forward_arr),
        'obs_log':      np.asarray(obs_log),
        'act_log':      np.asarray(act_log),
        'x_velocity_log': x_velocity_arr,
        'reward_forward_log': reward_forward_arr,
        'x_position_log': x_position_arr,
        'frames':       frames,
    }


def rollout_multi_seed(model, ema, env, n_seeds: int = 5, deterministic_sampling: bool = True, **kwargs) -> list:
    """여러 seed로 rollout. 통계용 (mean ± std).
    deterministic_sampling=True: 각 rollout의 sampling이 환경 seed에 deterministic.
    Paper evaluation에 권장."""
    results = []
    for s in range(n_seeds):
        sampling_seed_base = s if deterministic_sampling else None
        res = rollout(model, ema, env, seed=s, sampling_seed_base=sampling_seed_base, **kwargs)
        results.append(res)
        x_vel = res.get('mean_x_velocity', float('nan'))
        print(f"  seed {s}: survival={res['survival']:>3d} steps, "
              f"total_reward={res['total_reward']:>8.1f}, "
              f"reward/step={res.get('reward_per_step', float('nan')):>6.3f}, "
              f"x_vel={x_vel:>6.3f}")

    surv = np.array([r['survival'] for r in results])
    rew  = np.array([r['total_reward'] for r in results])
    rps = np.array([r.get('reward_per_step', r['total_reward'] / max(r['survival'], 1)) for r in results])
    xvel = np.array([r.get('mean_x_velocity', np.nan) for r in results], dtype=np.float32)
    print(f"\nSurvival:  mean={surv.mean():.0f} ± {surv.std():.0f} steps")
    print(f"Reward:    mean={rew.mean():.1f} ± {rew.std():.1f}")
    print(f"Reward/step: mean={rps.mean():.3f} ± {rps.std():.3f}")
    print(f"X velocity: mean={nanmean(xvel):.3f} ± {nanstd(xvel):.3f}")

    return results
