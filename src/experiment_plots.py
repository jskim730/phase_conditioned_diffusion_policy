"""Plotting helpers for training notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from phase import summarize_sweep_results


def plot_loss_curve(
    train_losses, val_log, output_path: str | Path, *, title: str
) -> Path:
    """Save train/validation loss curves."""
    output_path = Path(output_path)
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    if train_losses:
        ax.plot(
            np.arange(1, len(train_losses) + 1), train_losses, label="train", alpha=0.8
        )
    if val_log:
        ax.plot(
            [v[0] for v in val_log], [v[1] for v in val_log], "o-", label="val (EMA)"
        )
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


def plot_sample_histogram(
    samples: np.ndarray, output_path: str | Path, *, title: str
) -> Path:
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


def plot_action_chunks(
    samples_by_label: dict[str, np.ndarray] | dict[str, tuple[float, np.ndarray]],
    output_path: str | Path,
    *,
    title: str,
    act_dim: int,
    colors: Optional[dict[str, str]] = None,
) -> Path:
    """Plot sampled action chunks for all action dimensions."""
    output_path = Path(output_path)
    colors = colors or {}
    n_cols = min(4, act_dim)
    n_rows = int(np.ceil(act_dim / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False
    )
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


def plot_frequency_sweep(
    sweep_results: dict[float, list[dict]],
    data: dict,
    output_path: str | Path,
    *,
    title_prefix: str,
    max_steps: int,
) -> Path:
    """Save survival/reward curves for a frequency sweep."""
    output_path = Path(output_path)
    summary = summarize_sweep_results(sweep_results)
    f_mean = float(data["freq_window_mean"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].errorbar(
        summary.freqs,
        summary.survival_mean,
        yerr=summary.survival_std,
        fmt="o-",
        capsize=5,
        linewidth=2,
        markersize=7,
    )
    axes[0].axvspan(
        data["freq_window_min"],
        data["freq_window_max"],
        alpha=0.15,
        color="green",
        label="In-dist",
    )
    axes[0].axvline(f_mean, color="gray", ls="--", alpha=0.5, label="Train mean")
    axes[0].set_xlabel("Sampling-time phase freq (Hz)")
    axes[0].set_ylabel("Survival (steps)")
    axes[0].set_title(f"{title_prefix}: Survival")
    axes[0].set_ylim(0, max_steps * 1.05)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].errorbar(
        summary.freqs,
        summary.reward_mean,
        yerr=summary.reward_std,
        fmt="o-",
        capsize=5,
        linewidth=2,
        markersize=7,
    )
    axes[1].axvspan(
        data["freq_window_min"],
        data["freq_window_max"],
        alpha=0.15,
        color="green",
        label="In-dist",
    )
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


def print_step3_vs_step4_table(
    step3_sweep: dict[float, dict], step4_results: dict[float, list[dict]]
) -> None:
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


def plot_evaluation_frequency_comparison(
    table1_results: dict[str, list[dict]],
    sweep_results: dict[str, dict[float, list[dict]]],
    data: dict,
    output_path: str | Path,
    *,
    n_seeds_sweep: int,
) -> Path:
    """Save survival and reward-per-step curves over commanded frequency."""
    output_path = Path(output_path)
    freqs = sorted(float(f) for f in sweep_results["periodic"].keys())

    def metric_value(result: dict, metric: str) -> float:
        if metric == "reward_per_step":
            return float(
                result.get(
                    "reward_per_step",
                    result["total_reward"] / max(int(result["survival"]), 1),
                )
            )
        return float(result[metric])

    def means_stds(model_key: str, metric: str) -> tuple[list[float], list[float]]:
        means = [
            float(
                np.mean([metric_value(r, metric) for r in sweep_results[model_key][f]])
            )
            for f in freqs
        ]
        stds = [
            float(
                np.std([metric_value(r, metric) for r in sweep_results[model_key][f]])
            )
            for f in freqs
        ]
        return means, stds

    p_surv, p_surv_std = means_stds("periodic", "survival")
    p_rew, p_rew_std = means_stds("periodic", "reward_per_step")
    t_surv, t_surv_std = means_stds("trajectory", "survival")
    t_rew, t_rew_std = means_stds("trajectory", "reward_per_step")
    v_surv = np.array([r["survival"] for r in table1_results["vanilla"]])
    v_rew = np.array(
        [metric_value(r, "reward_per_step") for r in table1_results["vanilla"]]
    )

    f_mean = float(data["freq_window_mean"])
    f_min = float(data["freq_window_min"])
    f_max = float(data["freq_window_max"])

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax = axes[0]
    ax.errorbar(
        freqs,
        p_surv,
        yerr=p_surv_std,
        fmt="s-",
        capsize=4,
        linewidth=1.8,
        markersize=7,
        color="tab:green",
        label="Periodic Phase",
    )
    ax.errorbar(
        freqs,
        t_surv,
        yerr=t_surv_std,
        fmt="o-",
        capsize=4,
        linewidth=2.2,
        markersize=8,
        color="tab:red",
        label="Trajectory (ours)",
    )
    ax.errorbar(
        [f_mean],
        [v_surv.mean()],
        yerr=[v_surv.std()],
        fmt="D",
        capsize=5,
        markersize=10,
        color="tab:blue",
        label="Vanilla (no phase, ref. only)",
    )
    ax.axvspan(f_min, f_max, alpha=0.12, color="green", label="In-dist range")
    ax.axvline(f_mean, color="gray", ls="--", alpha=0.4)
    ax.set_xlabel("Sampling-time phase freq (Hz)")
    ax.set_ylabel("Survival (steps)")
    ax.set_title(f"Survival vs Frequency (3 in-dist + 2 OOD, n={n_seeds_sweep})")
    ax.legend(loc="lower center", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-50, 1080)

    ax = axes[1]
    ax.errorbar(
        freqs,
        p_rew,
        yerr=p_rew_std,
        fmt="s-",
        capsize=4,
        linewidth=1.8,
        markersize=7,
        color="tab:green",
        label="Periodic Phase",
    )
    ax.errorbar(
        freqs,
        t_rew,
        yerr=t_rew_std,
        fmt="o-",
        capsize=4,
        linewidth=2.2,
        markersize=8,
        color="tab:red",
        label="Trajectory (ours)",
    )
    ax.errorbar(
        [f_mean],
        [v_rew.mean()],
        yerr=[v_rew.std()],
        fmt="D",
        capsize=5,
        markersize=10,
        color="tab:blue",
        label="Vanilla (no phase, ref. only)",
    )
    ax.axvspan(f_min, f_max, alpha=0.12, color="green", label="In-dist range")
    ax.axvline(f_mean, color="gray", ls="--", alpha=0.4)
    ax.set_xlabel("Sampling-time phase freq (Hz)")
    ax.set_ylabel("Reward / step")
    ax.set_title(f"Reward per step vs Frequency (3 in-dist + 2 OOD, n={n_seeds_sweep})")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.show()
    print(f"✓ {output_path}")
    return output_path


def plot_frequency_tracking_alignment(
    sweep_results: dict[str, dict[float, list[dict]]],
    data: dict,
    output_path: str | Path,
    *,
    n_seeds_sweep: int | None = None,
    interval: str = "ci95",
) -> Path:
    """Plot commanded frequency against measured gait frequency.

    The dashed diagonal is perfect tracking; points closer to it provide direct
    evidence that the phase-conditioned sampler follows the requested gait
    frequency.
    """
    output_path = Path(output_path)
    freqs = sorted(float(f) for f in sweep_results["periodic"].keys())
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.5))
    styles = {
        "periodic": dict(fmt="s-", color="tab:green", label="Periodic Phase"),
        "trajectory": dict(fmt="o-", color="tab:red", label="Trajectory (ours)"),
    }
    for model_key, style in styles.items():
        means, spreads = _sweep_metric_summary(
            sweep_results[model_key], freqs, "measured_freq_hz", interval=interval
        )
        ax.errorbar(
            freqs,
            means,
            yerr=spreads,
            capsize=4,
            linewidth=2.0 if model_key == "trajectory" else 1.7,
            markersize=8 if model_key == "trajectory" else 7,
            **style,
        )
    lo = min(min(freqs), float(data["freq_window_min"]))
    hi = max(max(freqs), float(data["freq_window_max"]))
    pad = max((hi - lo) * 0.08, 0.02)
    diag = np.linspace(lo - pad, hi + pad, 100)
    ax.plot(diag, diag, "--", color="black", alpha=0.45, label="perfect tracking")
    ax.axvspan(
        data["freq_window_min"],
        data["freq_window_max"],
        alpha=0.12,
        color="green",
        label="In-dist range",
    )
    ax.axvline(float(data["freq_window_mean"]), color="gray", ls=":", alpha=0.5)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Commanded frequency (Hz)")
    ax.set_ylabel("Measured gait frequency (Hz)")
    suffix = f", n={n_seeds_sweep}" if n_seeds_sweep is not None else ""
    ax.set_title(f"Frequency command tracking{suffix}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.show()
    print(f"✓ {output_path}")
    return output_path


def plot_zone_aggregated_tracking_metrics(
    freq_protocol,
    sweep_results: dict[str, dict[float, list[dict]]],
    output_path: str | Path,
    *,
    interval: str = "ci95",
) -> Path:
    """Save OOD/in-distribution aggregate tracking metrics by frequency zone."""
    output_path = Path(output_path)
    groups = _zone_frequency_groups(freq_protocol)
    models = ["periodic", "trajectory"]
    labels = {"periodic": "Periodic Phase", "trajectory": "Trajectory (ours)"}
    colors = {"periodic": "tab:green", "trajectory": "tab:red"}
    metrics = [
        ("abs_freq_error_hz", "|Frequency error| (Hz) ↓"),
        ("phase_locking_value", "Phase locking value ↑"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(groups))
    width = 0.34
    for ax, (metric, ylabel) in zip(axes, metrics):
        for idx, model_key in enumerate(models):
            means = []
            spreads = []
            for freqs in groups.values():
                values = []
                for freq in freqs:
                    values.extend(
                        _metric_value(result, metric)
                        for result in sweep_results[model_key][float(freq)]
                    )
                mean, spread = _mean_spread(values, interval=interval)
                means.append(mean)
                spreads.append(spread)
            offset = (idx - 0.5) * width
            ax.bar(
                x + offset,
                means,
                width=width,
                yerr=spreads,
                capsize=4,
                label=labels[model_key],
                color=colors[model_key],
                alpha=0.85,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(groups.keys())
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_title("Tracking error by command zone")
    axes[1].set_title("Phase locking by command zone")
    axes[1].legend(loc="best", fontsize=9)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.show()
    print(f"✓ {output_path}")
    return output_path


def _sweep_metric_summary(
    model_results: dict[float, list[dict]],
    freqs: list[float],
    metric: str,
    *,
    interval: str,
) -> tuple[list[float], list[float]]:
    means = []
    spreads = []
    for freq in freqs:
        values = [_metric_value(result, metric) for result in model_results[freq]]
        mean, spread = _mean_spread(values, interval=interval)
        means.append(mean)
        spreads.append(spread)
    return means, spreads


def _metric_value(result: dict, metric: str) -> float:
    if metric == "reward_per_step":
        return float(
            result.get(
                "reward_per_step",
                result["total_reward"] / max(int(result["survival"]), 1),
            )
        )
    return float(result.get(metric, np.nan))


def _mean_spread(values, *, interval: str) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(arr.mean())
    std = float(arr.std())
    if interval == "ci95":
        return mean, float(1.96 * std / np.sqrt(float(arr.size)))
    if interval == "std":
        return mean, std
    raise ValueError("interval must be 'ci95' or 'std'")


def _zone_frequency_groups(freq_protocol) -> dict[str, list[float]]:
    groups = {"OOD-low": [], "In-dist": [], "OOD-high": []}
    for freq, zone in zip(freq_protocol.sweep_freqs, freq_protocol.zone_labels):
        zone_text = str(zone)
        if zone_text.startswith("OOD-low"):
            groups["OOD-low"].append(float(freq))
        elif zone_text.startswith("OOD-high"):
            groups["OOD-high"].append(float(freq))
        else:
            groups["In-dist"].append(float(freq))
    return groups
