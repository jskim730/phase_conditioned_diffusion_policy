"""Reusable data-pipeline helpers for Phase-Conditioned Diffusion Policy.

The 01 notebook should execute pipeline steps, not redefine split,
normalization, Dataset, DataLoader, and plotting logic.  This module owns those
steps so downstream notebooks and tests use the same implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import build_loaders


@dataclass(frozen=True)
class DataPipelineConfig:
    """Configuration for demo splitting, normalization, and loader creation."""

    obs_horizon: int = 2
    pred_horizon: int = 16
    action_horizon: int = 8
    val_ratio: float = 0.15
    batch_size: int = 256
    num_workers: int = 2
    seed: int = 42
    action_pad_ratio: float = 0.02


def set_reproducibility(seed: int = 42) -> str:
    """Seed NumPy/PyTorch and return the active torch device string."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"PyTorch {torch.__version__}, device={device}, seed={seed}")
    return device


def load_demo_artifact(data_path: str | Path) -> dict[str, Any]:
    """Load a Plan C demo NPZ and expose arrays plus frequency metadata."""
    data_path = Path(data_path).expanduser().resolve()
    assert data_path.exists(), f"파일 없음: {data_path}"
    demos = np.load(data_path)
    print(f"Keys: {list(demos.keys())}")

    data = {
        "demos": demos,
        "obs_data": demos["observations"],
        "act_data": demos["actions"],
        "phase_data": demos["phases"],
        "ep_lengths": demos["episode_lengths"],
        "freq_window_mean": float(demos["freq_window_mean"]),
        "freq_window_std": float(demos["freq_window_std"]),
        "freq_window_min": float(demos["freq_window_min"]),
        "freq_window_max": float(demos["freq_window_max"]),
    }
    print_demo_summary(data)
    return data


def print_demo_summary(data: dict[str, Any]) -> None:
    """Print structure and frequency metadata for a loaded demo artifact."""
    obs_data = data["obs_data"]
    act_data = data["act_data"]
    phase_data = data["phase_data"]
    ep_lengths = data["ep_lengths"]
    print(f"\nobservations: {obs_data.shape}, dtype={obs_data.dtype}")
    print(f"actions:      {act_data.shape}, dtype={act_data.dtype}")
    print(f"phases:       {phase_data.shape}, dtype={phase_data.dtype}")
    print(f"episode_lengths: min={ep_lengths.min()}, max={ep_lengths.max()}")

    f_mean = data["freq_window_mean"]
    f_std = data["freq_window_std"]
    f_min = data["freq_window_min"]
    f_max = data["freq_window_max"]
    print(f"\nLearned freq window: {f_mean:.3f} ± {f_std:.3f} Hz, range [{f_min:.3f}, {f_max:.3f}]")
    print(f"In-distribution (mean ± 2σ): [{f_mean - 2 * f_std:.3f}, {f_mean + 2 * f_std:.3f}] Hz")


def split_episodes(n_episodes: int, val_ratio: float = 0.15, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic episode-level train/validation split."""
    n_val = int(np.round(n_episodes * val_ratio))
    n_train = n_episodes - n_val
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_episodes)
    train_eps = perm[:n_train].astype(np.int32)
    val_eps = perm[n_train:].astype(np.int32)
    print(f"Train episodes: {len(train_eps)}")
    print(f"Val episodes:   {len(val_eps)}")
    return train_eps, val_eps


def compute_normalization_stats(
    obs_data: np.ndarray,
    act_data: np.ndarray,
    ep_lengths: np.ndarray,
    train_eps: np.ndarray,
    action_pad_ratio: float = 0.02,
) -> dict[str, np.ndarray]:
    """Compute train-only observation/action normalization statistics."""
    train_obs_list = [obs_data[i, : int(ep_lengths[i])] for i in train_eps]
    train_act_list = [act_data[i, : int(ep_lengths[i])] for i in train_eps]
    train_obs_flat = np.concatenate(train_obs_list, axis=0)
    train_act_flat = np.concatenate(train_act_list, axis=0)

    print(f"Train flat obs: {train_obs_flat.shape}")
    print(f"Train flat act: {train_act_flat.shape}")

    obs_mean = train_obs_flat.mean(axis=0).astype(np.float32)
    obs_std = np.maximum(train_obs_flat.std(axis=0).astype(np.float32), 1e-6)

    act_min = train_act_flat.min(axis=0).astype(np.float32)
    act_max = train_act_flat.max(axis=0).astype(np.float32)
    act_pad = action_pad_ratio * (act_max - act_min)
    act_min = act_min - act_pad
    act_max = act_max + act_pad
    act_range = np.maximum(act_max - act_min, 1e-6).astype(np.float32)

    print(f"\nobs_mean.shape={obs_mean.shape}, global mean of means={obs_mean.mean():.3f}")
    print(f"obs_std.shape={obs_std.shape}, global mean of stds={obs_std.mean():.3f}")
    print(f"act_min={act_min}")
    print(f"act_max={act_max}")
    return {
        "obs_mean": obs_mean,
        "obs_std": obs_std,
        "act_min": act_min,
        "act_max": act_max.astype(np.float32),
        "act_range": act_range,
    }


def build_norm_stats(data: dict[str, Any], train_eps: np.ndarray, val_eps: np.ndarray, config: DataPipelineConfig) -> dict[str, np.ndarray]:
    """Build the complete norm_stats payload saved for all downstream notebooks."""
    stats = compute_normalization_stats(
        data["obs_data"],
        data["act_data"],
        data["ep_lengths"],
        train_eps,
        action_pad_ratio=config.action_pad_ratio,
    )
    stats.update(
        {
            "freq_window_mean": np.float32(data["freq_window_mean"]),
            "freq_window_std": np.float32(data["freq_window_std"]),
            "freq_window_min": np.float32(data["freq_window_min"]),
            "freq_window_max": np.float32(data["freq_window_max"]),
            "obs_horizon": np.int32(config.obs_horizon),
            "pred_horizon": np.int32(config.pred_horizon),
            "action_horizon": np.int32(config.action_horizon),
            "obs_dim": np.int32(data["obs_data"].shape[-1]),
            "act_dim": np.int32(data["act_data"].shape[-1]),
            "train_eps": train_eps.astype(np.int32),
            "val_eps": val_eps.astype(np.int32),
            "seed": np.int32(config.seed),
        }
    )
    return stats


def save_norm_stats(norm_stats: dict[str, np.ndarray], norm_path: str | Path) -> Path:
    """Save normalization and split metadata as an NPZ artifact."""
    norm_path = Path(norm_path).expanduser().resolve()
    norm_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(norm_path, **norm_stats)
    print(f"✓ 저장: {norm_path}")
    print(f"  파일 크기: {norm_path.stat().st_size / 1024:.1f} KB")
    return norm_path


def run_data_pipeline(data_path: str | Path, norm_path: str | Path, config: DataPipelineConfig) -> dict[str, Any]:
    """Run loading, splitting, train-only stats, saving, and loader construction."""
    set_reproducibility(config.seed)
    data = load_demo_artifact(data_path)
    print(
        f"OBS_DIM={data['obs_data'].shape[-1]}, ACT_DIM={data['act_data'].shape[-1]}\n"
        f"obs_horizon={config.obs_horizon}, pred_horizon={config.pred_horizon}, "
        f"action_horizon={config.action_horizon}\n"
        f"batch_size={config.batch_size}, val_ratio={config.val_ratio}"
    )

    train_eps, val_eps = split_episodes(data["obs_data"].shape[0], config.val_ratio, config.seed)
    demos = data["demos"]
    if "estimated_freqs" in demos:
        train_freqs = demos["estimated_freqs"][train_eps]
        val_freqs = demos["estimated_freqs"][val_eps]
        print(f"\nTrain freq: mean={train_freqs.mean():.3f}, std={train_freqs.std():.3f}")
        print(f"Val freq:   mean={val_freqs.mean():.3f}, std={val_freqs.std():.3f}")

    norm_stats = build_norm_stats(data, train_eps, val_eps, config)
    save_norm_stats(norm_stats, norm_path)

    project_data = {
        "obs_data": data["obs_data"],
        "act_data": data["act_data"],
        "phase_data": data["phase_data"],
        "ep_lengths": data["ep_lengths"],
        "obs_mean": norm_stats["obs_mean"],
        "obs_std": norm_stats["obs_std"],
        "act_min": norm_stats["act_min"],
        "act_max": norm_stats["act_max"],
        "act_range": norm_stats["act_range"],
        "OBS_HORIZON": int(norm_stats["obs_horizon"]),
        "PRED_HORIZON": int(norm_stats["pred_horizon"]),
        "ACTION_HORIZON": int(norm_stats["action_horizon"]),
        "OBS_DIM": int(norm_stats["obs_dim"]),
        "ACT_DIM": int(norm_stats["act_dim"]),
        "train_eps": norm_stats["train_eps"],
        "val_eps": norm_stats["val_eps"],
        "seed": int(norm_stats["seed"]),
        "freq_window_mean": float(norm_stats["freq_window_mean"]),
        "freq_window_std": float(norm_stats["freq_window_std"]),
        "freq_window_min": float(norm_stats["freq_window_min"]),
        "freq_window_max": float(norm_stats["freq_window_max"]),
    }
    train_dataset, val_dataset, train_loader, val_loader = build_loaders(
        project_data,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        verbose=True,
    )
    return {
        "data": data,
        "norm_stats": norm_stats,
        "project_data": project_data,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
    }


def validate_batch(train_loader: Any) -> dict[str, Any]:
    """Pull one batch and assert expected normalized ranges."""
    batch = next(iter(train_loader))
    print("=== Batch shapes & ranges ===")
    for key, value in batch.items():
        print(
            f"  {key:8s}: shape={tuple(value.shape)}, "
            f"min={value.min():.3f}, max={value.max():.3f}, "
            f"mean={value.mean():.3f}, std={value.std():.3f}"
        )

    print("\n=== Normalization 검증 ===")
    print("obs (목표: mean≈0, std≈1):")
    print(f"  per-dim mean (앞 5개): {batch['obs'].mean(dim=(0, 1))[:5].numpy().round(3)}")
    print(f"  per-dim std  (앞 5개): {batch['obs'].std(dim=(0, 1))[:5].numpy().round(3)}")

    print("\naction (목표: [-1, 1] 안):")
    print(f"  min={batch['action'].min():.4f}, max={batch['action'].max():.4f}")
    assert batch["action"].min() >= -1.05 and batch["action"].max() <= 1.05, "Action 범위 이상 — normalization 확인 필요"

    print("\nphase (목표: [0, 2π)):")
    print(f"  min={batch['phase'].min():.3f}, max={batch['phase'].max():.3f}")
    assert batch["phase"].min() >= 0 and batch["phase"].max() < 2 * np.pi + 0.1, "Phase 범위 이상"
    print("\n✓ 모든 검증 통과")
    return batch


def plot_pipeline_sanity(train_dataset: Any, figures_dir: str | Path, filename: str = "data_pipeline_sanity.png") -> Path:
    """Save a visualization of one normalized obs/action/phase chunk."""
    figures_dir = Path(figures_dir).expanduser().resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)
    sample = train_dataset[0]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    obs_show = sample["obs"].numpy()
    for dim in range(min(8, obs_show.shape[1])):
        axes[0].plot(obs_show[:, dim], "o-", label=f"dim {dim}", markersize=4)
    axes[0].set_title(f"Obs window (정규화) — shape {tuple(sample['obs'].shape)}")
    axes[0].set_xlabel("Step (within window)")
    axes[0].set_ylabel("Normalized value")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(True, alpha=0.3)

    act_show = sample["action"].numpy()
    for dim in range(act_show.shape[1]):
        axes[1].plot(act_show[:, dim], "o-", label=f"a{dim}", markersize=4)
    axes[1].axhline(1, color="r", ls="--", alpha=0.3)
    axes[1].axhline(-1, color="r", ls="--", alpha=0.3)
    axes[1].set_title(f"Action chunk (정규화) — shape {tuple(sample['action'].shape)}")
    axes[1].set_xlabel("Step (within chunk)")
    axes[1].set_ylabel("Normalized action")
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(True, alpha=0.3)

    phase_show = sample["phase"].numpy()
    axes[2].plot(phase_show, "o-", color="purple", markersize=5)
    axes[2].axhline(2 * np.pi, color="r", ls="--", alpha=0.3, label="2π")
    axes[2].set_title(f"Phase chunk (raw φ) — shape {tuple(sample['phase'].shape)}")
    axes[2].set_xlabel("Step (within chunk)")
    axes[2].set_ylabel("φ (rad)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    output_path = figures_dir / filename
    fig.savefig(output_path, dpi=80, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"✓ 저장: {output_path}")
    return output_path


def plot_phase_advance(
    train_dataset: Any,
    figures_dir: str | Path,
    f_mean: float,
    seed: int = 42,
    dt: float = 0.05,
    n_check: int = 200,
    filename: str = "data_pipeline_phase_advance.png",
) -> dict[str, Any]:
    """Save a histogram of per-step phase advance across random chunks."""
    figures_dir = Path(figures_dir).expanduser().resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)
    n_check = min(n_check, len(train_dataset))
    rng = np.random.RandomState(seed)
    sample_idxs = rng.choice(len(train_dataset), size=n_check, replace=False)

    step_advances = []
    for idx in sample_idxs:
        phase = train_dataset[idx]["phase"].numpy()
        phase_unwrap = np.unwrap(phase)
        step_advances.append((phase_unwrap[-1] - phase_unwrap[0]) / (len(phase) - 1))
    step_advances = np.asarray(step_advances)
    expected = 2 * np.pi * f_mean * dt

    print("Phase 진행률 (rad/step):")
    print(f"  관측: mean={step_advances.mean():.4f}, std={step_advances.std():.4f}")
    print(f"  이론: 2π × {f_mean:.3f} × {dt} = {expected:.4f}")
    print(f"  비율: {step_advances.mean() / expected:.3f}× (1.0에 가까울수록 일관)")

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    bins = 1 if np.allclose(step_advances.min(), step_advances.max()) else 30
    ax.hist(step_advances, bins=bins, edgecolor="black", alpha=0.7)
    ax.axvline(expected, color="r", ls="--", linewidth=2, label=f"Theoretical (f={f_mean:.2f}Hz): {expected:.3f}")
    ax.axvline(step_advances.mean(), color="g", ls="--", linewidth=2, label=f"Observed mean: {step_advances.mean():.3f}")
    ax.set_xlabel("Per-step phase advance (rad/step)")
    ax.set_ylabel("Count")
    ax.set_title(f"Phase 진행률 분포 ({n_check} chunks)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_path = figures_dir / filename
    fig.savefig(output_path, dpi=80, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"\n✓ 저장: {output_path}")
    return {"path": output_path, "step_advances": step_advances, "expected": expected}
