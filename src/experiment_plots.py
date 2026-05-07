"""Plotting helpers for training notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from phase import summarize_sweep_results


def plot_loss_curve(train_losses, val_log, output_path: str | Path, *, title: str) -> Path:
    """Save train/validation loss curves."""
    output_path = Path(output_path)
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    if train_losses:
        ax.plot(np.arange(1, len(train_losses) + 1), train_losses, label="train", alpha=0.8)
    if val_log:
        ax.plot([v[0] for v in val_log], [v[1] for v in val_log], "o-", label="val (EMA)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"✓ {output_path}")
    return output_path


def plot_sample_histogram(samples: np.ndarray, output_path: str | Path, *, title: str) -> Path:
    """Save a histogram of normalized sampled actions."""
    output_path = Path(output_path)
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.hist(samples.reshape(-1), bins=80, alpha=0.85)
    ax.axvline(-1, color="r", ls=":", alpha=0.5)
    ax.axvline(1, color="r", ls=":", alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("normalized action")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"✓ {output_path}")
    return output_path


def plot_action_chunks(samples_by_label: dict[str, np.ndarray] | dict[str, tuple[float, np.ndarray]],
                       output_path: str | Path,
                       *,
                       title: str,
                       act_dim: int,
                       colors: Optional[dict[str, str]] = None) -> Path:
    """Plot sampled action chunks for all action dimensions."""
    output_path = Path(output_path)
    colors = colors or {}
    n_cols = min(4, act_dim)
    n_rows = int(np.ceil(act_dim / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
    for dim in range(act_dim):
        ax = axes.flat[dim]
        for label, payload in samples_by_label.items():
            if isinstance(payload, tuple):
                freq_hz, sample = payload
                legend_label = f"{label} ({freq_hz:.2f}Hz)"
            else:
                sample = payload
                legend_label = label
            ax.plot(
                sample[:, dim],
                "-",
                alpha=0.75,
                linewidth=2 if label == "mean" else 1,
                label=legend_label if dim == 0 else None,
                color=colors.get(label),
            )
        ax.axhline(1, color="r", ls=":", alpha=0.3)
        ax.axhline(-1, color="r", ls=":", alpha=0.3)
        ax.set_title(f"action dim {dim}")
        ax.grid(True, alpha=0.3)
    for j in range(act_dim, n_rows * n_cols):
        axes.flat[j].axis("off")
    axes.flat[0].legend(fontsize=7, ncol=2)
    plt.suptitle(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.show()
    print(f"✓ {output_path}")
    return output_path


def plot_frequency_sweep(sweep_results: dict[float, list[dict]],
                         data: dict,
                         output_path: str | Path,
                         *,
                         title_prefix: str,
                         max_steps: int) -> Path:
    """Save survival/reward curves for a frequency sweep."""
    output_path = Path(output_path)
    summary = summarize_sweep_results(sweep_results)
    f_mean = float(data["freq_window_mean"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].errorbar(summary.freqs, summary.survival_mean, yerr=summary.survival_std,
                     fmt="o-", capsize=5, linewidth=2, markersize=7)
    axes[0].axvspan(data["freq_window_min"], data["freq_window_max"], alpha=0.15, color="green", label="In-dist")
    axes[0].axvline(f_mean, color="gray", ls="--", alpha=0.5, label="Train mean")
    axes[0].set_xlabel("Sampling-time phase freq (Hz)")
    axes[0].set_ylabel("Survival (steps)")
    axes[0].set_title(f"{title_prefix}: Survival")
    axes[0].set_ylim(0, max_steps * 1.05)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].errorbar(summary.freqs, summary.reward_mean, yerr=summary.reward_std,
                     fmt="o-", capsize=5, linewidth=2, markersize=7)
    axes[1].axvspan(data["freq_window_min"], data["freq_window_max"], alpha=0.15, color="green", label="In-dist")
    axes[1].axvline(f_mean, color="gray", ls="--", alpha=0.5, label="Train mean")
    axes[1].set_xlabel("Sampling-time phase freq (Hz)")
    axes[1].set_ylabel("Total reward")
    axes[1].set_title(f"{title_prefix}: Reward")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"✓ {output_path}")
    return output_path


def print_step3_vs_step4_table(step3_sweep: dict[float, dict], step4_results: dict[float, list[dict]]) -> None:
    """Print a compact ablation table comparing periodic and trajectory conditioning."""
    print(f"{'freq':>7s} | {'Step 3 (Periodic)':>30s} | {'Step 4 (Trajectory)':>30s}")
    print(f"{'':7s} | {'surv':>10s} {'reward':>15s} | {'surv':>10s} {'reward':>15s}")
    print("-" * 85)
    for freq_hz in sorted(step4_results.keys()):
        key = round(float(freq_hz), 3)
        f3 = step3_sweep.get(key)
        s4 = step4_results[freq_hz]
        surv4 = np.array([r["survival"] for r in s4])
        rew4 = np.array([r["total_reward"] for r in s4])
        if f3 is None:
            left = f"{'n/a':>10s} {'n/a':>15s}"
        else:
            left = f"{f3['survival_mean']:>10.0f} {f3['reward_mean']:>10.1f}±{f3['reward_std']:<4.1f}"
        right = f"{surv4.mean():>10.0f} {rew4.mean():>10.1f}±{rew4.std():<4.1f}"
        print(f"{freq_hz:7.3f} | {left} | {right}")
