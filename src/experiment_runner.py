"""High-level experiment routines used by training notebooks 02--04.

The notebooks should orchestrate these functions only; reusable training,
checkpoint, rollout, sampling, and sanity-check logic lives here or in narrower
``src`` modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from configs import ExperimentConfig
from dataset import build_loaders, load_project_data
from models import count_params
from phase import make_phase_trajectory_fn, summarize_sweep_results
from reproducibility import print_data_summary, set_global_seed
from sampling import rollout_multi_seed, sample_action_chunk
from training import load_checkpoint, save_checkpoint, train_diffusion_policy


def load_data_and_build_loaders(cfg: ExperimentConfig, data_dir: str | Path):
    """Load project arrays, seed all RNGs, and construct train/validation loaders."""
    data = load_project_data(data_dir)
    print_data_summary(data)
    seed = set_global_seed(data["seed"])
    train_ds, val_ds, train_loader, val_loader = build_loaders(
        data,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )
    print(f"✓ Reproducibility seed: {seed}")
    return data, train_ds, val_ds, train_loader, val_loader


def build_model_with_sanity_check(
    cfg: ExperimentConfig,
    data: dict,
    *,
    device: str,
    train_loader=None,
):
    """Build the configured model and run a variant-appropriate forward sanity check."""
    model = cfg.build_model(data, device=device)
    n = count_params(model)
    print(f"Total params:     {n['total'] / 1e6:.2f}M")
    print(f"Trainable params: {n['trainable'] / 1e6:.2f}M")

    batch_size = 4
    fake_action = torch.randn(batch_size, data["PRED_HORIZON"], data["ACT_DIM"], device=device)
    fake_t = torch.randint(0, cfg.diffusion.num_train_timesteps, (batch_size,), device=device)
    global_dim = data["OBS_HORIZON"] * data["OBS_DIM"]
    if cfg.name.startswith("periodic_phase"):
        global_dim += 2
    fake_global = torch.randn(batch_size, global_dim, device=device)

    with torch.no_grad():
        if cfg.name.startswith("phase_trajectory"):
            fake_phase = torch.randn(batch_size, data["PRED_HORIZON"], cfg.model.per_step_cond_dim, device=device)
            out = model(fake_action, fake_t, fake_global, fake_phase)
        else:
            out = model(fake_action, fake_t, fake_global)
    assert out.shape == fake_action.shape, (out.shape, fake_action.shape)
    print(f"✓ Forward pass: in {tuple(fake_action.shape)} → out {tuple(out.shape)}")

    if train_loader is not None:
        train_cond_fn = cfg.resolve_train_cond_fn()
        sample_batch = next(iter(train_loader))
        global_cond, per_step_cond = train_cond_fn(sample_batch, device)
        print(f"✓ train cond global: {tuple(global_cond.shape)}")
        print(f"✓ train cond per-step: {None if per_step_cond is None else tuple(per_step_cond.shape)}")

    return model


def build_noise_scheduler(cfg: ExperimentConfig):
    """Build and describe the DDPM scheduler plus serializable DDIM config."""
    noise_scheduler = cfg.build_noise_scheduler()
    print(
        f"DDPM: {cfg.diffusion.num_train_timesteps} train steps, "
        f"{noise_scheduler.config.beta_schedule}, "
        f"{noise_scheduler.config.prediction_type}-prediction"
    )
    return noise_scheduler, cfg.noise_scheduler_config(), cfg.diffusion.num_inference_steps


def train_or_load_checkpoint(
    *,
    train: bool,
    cfg: ExperimentConfig,
    model,
    ema,
    noise_scheduler,
    train_loader,
    val_loader,
    checkpoints_dir: str | Path,
    device: str,
):
    """Run training or restore the configured checkpoint, returning checkpoint metadata."""
    ckpt_path = cfg.checkpoint_path(checkpoints_dir)
    train_cond_fn = cfg.resolve_train_cond_fn()
    if train:
        train_losses, val_log, best_ema_state = train_diffusion_policy(
            model,
            ema,
            noise_scheduler,
            train_loader,
            val_loader,
            cond_fn=train_cond_fn,
            device=device,
            **cfg.training_kwargs(),
        )
        save_checkpoint(
            ckpt_path,
            model,
            ema,
            train_losses,
            val_log,
            best_ema_state,
            config=cfg.to_dict(),
        )
        print(f"✓ Training complete: {ckpt_path}")
    else:
        meta = load_checkpoint(ckpt_path, model, ema, device=device, use_best_ema=True)
        train_losses = meta["train_losses"]
        val_log = meta["val_log"]
        best_ema_state = meta["best_ema_state"]
        print(f"✓ Loaded checkpoint: {ckpt_path}")
    return train_losses, val_log, best_ema_state, ckpt_path


def apply_best_ema_after_training(train: bool, best_ema_state, ema) -> None:
    """Load the best validation EMA state into ``ema`` after a fresh training run."""
    if train and best_ema_state is not None:
        ema_state = ema.state_dict()
        ema_state.update(best_ema_state)
        ema.load_state_dict(ema_state)
        print("✓ Best EMA applied")
    else:
        print("✓ EMA already restored from checkpoint or best EMA is unavailable")


def sample_single_batch(
    model,
    ema,
    ns_config: dict,
    val_ds,
    data: dict,
    cond_fn: Callable,
    *,
    device: str,
    num_inference_steps: int,
    phase_chunk: Optional[torch.Tensor] = None,
    batch_size: int = 64,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Sample normalized action chunks for validation observations."""
    obs_batch = torch.stack([val_ds[i]["obs"] for i in range(min(batch_size, len(val_ds)))])
    if phase_chunk is None and "phase" in val_ds[0]:
        phase_chunk = torch.stack([val_ds[i]["phase"] for i in range(min(batch_size, len(val_ds)))])
    samples = sample_action_chunk(
        model,
        ema,
        ns_config,
        obs_window=obs_batch,
        phase_chunk=phase_chunk,
        cond_fn=cond_fn,
        pred_horizon=data["PRED_HORIZON"],
        act_dim=data["ACT_DIM"],
        num_inference_steps=num_inference_steps,
        device=device,
        seed=seed,
    ).numpy()
    print(f"Samples: shape={samples.shape}, min={samples.min():.3f}, max={samples.max():.3f}")
    return samples


def sample_frequency_variants(
    model,
    ema,
    ns_config: dict,
    obs_window: torch.Tensor,
    data: dict,
    sample_cond_fn: Callable,
    freqs: list[float],
    labels: list[str],
    *,
    device: str,
    num_inference_steps: int,
    dt: float = 0.05,
    seed: Optional[int] = None,
) -> dict[str, tuple[float, np.ndarray]]:
    """Sample one action chunk for each target frequency with a fixed noise seed."""
    samples_by_freq: dict[str, tuple[float, np.ndarray]] = {}
    for freq_hz, label in zip(freqs, labels):
        phase = make_phase_trajectory_fn(freq_hz, dt=dt)(0, data["PRED_HORIZON"])
        phase_t = torch.from_numpy(phase).unsqueeze(0)
        sample = sample_action_chunk(
            model,
            ema,
            ns_config,
            obs_window=obs_window,
            phase_chunk=phase_t,
            cond_fn=sample_cond_fn,
            pred_horizon=data["PRED_HORIZON"],
            act_dim=data["ACT_DIM"],
            num_inference_steps=num_inference_steps,
            device=device,
            seed=seed,
        ).numpy()[0]
        samples_by_freq[label] = (float(freq_hz), sample)
    _print_frequency_rmse(samples_by_freq)
    return samples_by_freq


def _print_frequency_rmse(samples_by_freq: dict[str, tuple[float, np.ndarray]], reference_label: str = "mean") -> None:
    ref_freq, ref_sample = samples_by_freq[reference_label]
    print(f"=== Output difference from '{reference_label}' baseline ===")
    for label, (freq_hz, sample) in samples_by_freq.items():
        rmse = np.linalg.norm(sample - ref_sample) / np.sqrt(sample.size)
        print(f"  {label:>14s} (Δf={freq_hz - ref_freq:+.3f} Hz): RMSE = {rmse:.4f}")


def rollout_at_frequency(
    model,
    ema,
    env,
    ns_config: dict,
    data: dict,
    sample_cond_fn: Callable,
    *,
    freq_hz: float,
    n_seeds: int,
    max_steps: int,
    num_inference_steps: int,
    deterministic_sampling: bool,
    device: str,
    dt: float = 0.05,
) -> list[dict]:
    """Evaluate one target frequency with shared rollout defaults."""
    print(f"=== Rollout @ f={freq_hz:.3f} Hz ===")
    return rollout_multi_seed(
        model,
        ema,
        env,
        n_seeds=n_seeds,
        deterministic_sampling=deterministic_sampling,
        noise_scheduler_config=ns_config,
        obs_mean=data["obs_mean"],
        obs_std=data["obs_std"],
        act_min=data["act_min"],
        act_range=data["act_range"],
        cond_fn=sample_cond_fn,
        obs_horizon=data["OBS_HORIZON"],
        pred_horizon=data["PRED_HORIZON"],
        action_horizon=data["ACTION_HORIZON"],
        obs_dim=data["OBS_DIM"],
        act_dim=data["ACT_DIM"],
        num_inference_steps=num_inference_steps,
        max_steps=max_steps,
        phase_trajectory_fn=make_phase_trajectory_fn(freq_hz, dt=dt),
        device=device,
    )


def run_frequency_sweep(
    model,
    ema,
    env,
    ns_config: dict,
    data: dict,
    sample_cond_fn: Callable,
    sweep_freqs,
    *,
    n_seeds: int,
    max_steps: int,
    num_inference_steps: int,
    deterministic_sampling: bool,
    device: str,
    dt: float = 0.05,
) -> dict[float, list[dict]]:
    """Run a rollout sweep and return raw results keyed by target frequency."""
    results: dict[float, list[dict]] = {}
    for freq_hz in sweep_freqs:
        freq_hz = float(freq_hz)
        print(f"\n--- freq={freq_hz:.3f} Hz ---")
        results[freq_hz] = rollout_at_frequency(
            model,
            ema,
            env,
            ns_config,
            data,
            sample_cond_fn,
            freq_hz=freq_hz,
            n_seeds=n_seeds,
            max_steps=max_steps,
            num_inference_steps=num_inference_steps,
            deterministic_sampling=deterministic_sampling,
            device=device,
            dt=dt,
        )
    summary = summarize_sweep_results(results)
    print("\n=== Sweep summary ===")
    for freq_hz, surv, rew in zip(summary.freqs, summary.survival_mean, summary.reward_mean):
        print(f"  {freq_hz:.3f} Hz: survival={surv:.0f}, reward={rew:.1f}")
    return results
