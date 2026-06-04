<<<<<<< Updated upstream
"""High-level experiment routines used by training notebooks 02--04.

Notebooks orchestrate these helpers; the full env-rollout evaluation protocol
lives in :mod:`pcdp.evaluation` and is driven by ``05_evaluation.ipynb``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from .configs import ExperimentConfig, set_global_seed
from .dataset import build_loaders, load_project_data
from .models import count_params
from .phase import make_phase_trajectory_fn
from .sampling import sample_action_chunk
from .training import load_checkpoint, save_checkpoint, train_diffusion_policy


def load_data_and_build_loaders(cfg: ExperimentConfig, data_dir: str | Path):
    """Load project arrays, seed all RNGs, and construct train/validation loaders."""
    data = load_project_data(data_dir)
    seed = set_global_seed(cfg.seed, deterministic=cfg.deterministic)
    train_ds, val_ds, train_loader, val_loader = build_loaders(
        data,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )
    print(f"✓ Experiment seed: {seed}")
    return data, train_ds, val_ds, train_loader, val_loader


def build_model(
    cfg: ExperimentConfig,
    data: dict,
    *,
    device: str,
):
    """Build the configured model and print the parameter count."""
    model = cfg.build_model(data, device=device)
    n = count_params(model)
    print(f"Total params:     {n['total'] / 1e6:.2f}M")
    print(f"Trainable params: {n['trainable'] / 1e6:.2f}M")
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
    """Train (and save) or load the configured checkpoint, then apply best EMA.

    When ``train=True`` the best-validation EMA weights captured during
    training are copied into the live ``ema`` model so subsequent sampling
    matches the saved checkpoint.  When ``train=False`` ``load_checkpoint``
    already applies the best EMA weights, so no extra step is needed.
    """
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
        if best_ema_state is not None:
            ema_state = ema.state_dict()
            ema_state.update(best_ema_state)
            ema.load_state_dict(ema_state)
            print("✓ Best EMA applied")
        print(f"✓ Training complete: {ckpt_path}")
    else:
        meta = load_checkpoint(ckpt_path, model, ema, device=device, use_best_ema=True)
        train_losses = meta["train_losses"]
        val_log = meta["val_log"]
        best_ema_state = meta["best_ema_state"]
        print(f"✓ Loaded checkpoint: {ckpt_path}")
    return train_losses, val_log, best_ema_state, ckpt_path


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
    """Sample one action chunk for each target frequency with a fixed noise seed.

    Used by the Phase Sensitivity cells in notebooks 03/04 to show how the
    sampled action chunk responds to different command frequencies while obs
    and noise seed are held fixed.
    """
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
=======
"""High-level experiment routines used by training notebooks 02--04.

Notebooks orchestrate these helpers; the full env-rollout evaluation protocol
lives in :mod:`pcdp.evaluation` and is driven by ``06_evaluation.ipynb``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from .configs import ExperimentConfig, set_global_seed
from .dataset import build_loaders, load_project_data
from .models import count_params
from .phase import make_phase_trajectory_fn
from .sampling import sample_action_chunk
from .training import load_checkpoint, save_checkpoint, train_diffusion_policy


def load_data_and_build_loaders(cfg: ExperimentConfig, data_dir: str | Path):
    """Load project arrays, seed all RNGs, and construct train/validation loaders."""
    data = load_project_data(data_dir)
    seed = set_global_seed(cfg.seed, deterministic=cfg.deterministic)
    train_ds, val_ds, train_loader, val_loader = build_loaders(
        data,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )
    print(f"✓ Experiment seed: {seed}")
    return data, train_ds, val_ds, train_loader, val_loader


def build_model(
    cfg: ExperimentConfig,
    data: dict,
    *,
    device: str,
):
    """Build the configured model and print the parameter count."""
    model = cfg.build_model(data, device=device)
    n = count_params(model)
    print(f"Total params:     {n['total'] / 1e6:.2f}M")
    print(f"Trainable params: {n['trainable'] / 1e6:.2f}M")
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
    """Train (and save) or load the configured checkpoint, then apply best EMA.

    When ``train=True`` the best-validation EMA weights captured during
    training are copied into the live ``ema`` model so subsequent sampling
    matches the saved checkpoint.  When ``train=False`` ``load_checkpoint``
    already applies the best EMA weights, so no extra step is needed.
    """
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
        if best_ema_state is not None:
            ema_state = ema.state_dict()
            ema_state.update(best_ema_state)
            ema.load_state_dict(ema_state)
            print("✓ Best EMA applied")
        print(f"✓ Training complete: {ckpt_path}")
    else:
        meta = load_checkpoint(ckpt_path, model, ema, device=device, use_best_ema=True)
        train_losses = meta["train_losses"]
        val_log = meta["val_log"]
        best_ema_state = meta["best_ema_state"]
        print(f"✓ Loaded checkpoint: {ckpt_path}")
    return train_losses, val_log, best_ema_state, ckpt_path


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
    reference_label: Optional[str] = None,
) -> dict[str, tuple[float, np.ndarray]]:
    """Sample one action chunk for each target frequency with a fixed noise seed.

    Used by the Phase Sensitivity cells in notebooks 03/04 to show how the
    sampled action chunk responds to different command frequencies while obs
    and noise seed are held fixed.
    """
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
    _print_frequency_rmse(samples_by_freq, reference_label=reference_label)
    return samples_by_freq


def _print_frequency_rmse(
    samples_by_freq: dict[str, tuple[float, np.ndarray]],
    reference_label: Optional[str] = None,
) -> None:
    if reference_label is None:
        labels = list(samples_by_freq)
        reference_label = (
            "mean" if "mean" in samples_by_freq else labels[len(labels) // 2]
        )
    ref_freq, ref_sample = samples_by_freq[reference_label]
    print(f"=== Output difference from '{reference_label}' baseline ===")
    for label, (freq_hz, sample) in samples_by_freq.items():
        rmse = np.linalg.norm(sample - ref_sample) / np.sqrt(sample.size)
        print(f"  {label:>14s} (Δf={freq_hz - ref_freq:+.3f} Hz): RMSE = {rmse:.4f}")
>>>>>>> Stashed changes
