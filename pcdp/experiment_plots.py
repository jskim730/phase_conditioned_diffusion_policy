"""Plotting helpers for training notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .evaluation import (
    MODEL_KEYS as DEFAULT_MODEL_KEYS,
    PHASE_MODEL_KEYS as DEFAULT_PHASE_MODEL_KEYS,
    SUMMARY_MODEL_LABELS,
    metric_array,
    summary_stats,
)


# ---------------------------------------------------------------------
# Per-model visual styling. ``trajectory`` and ``trajectory_sync``
# share the red family so the sync model reads as the ours-strengthened
# version of the trajectory baseline.
# ---------------------------------------------------------------------
MODEL_COLORS: dict[str, str] = {
    "vanilla": "tab:blue",
    "periodic": "tab:green",
    "trajectory": "#ff9999",            # light red — sync-less ablation
    "trajectory_sync": "tab:red",  # strong red — ours (sync fine-tune)
    "trajectory_sync_v2": "tab:orange",
    "phase_continuation": "tab:purple",
    "phase_continuation_sync": "tab:cyan",
    "phase_continuation_sync_v2": "tab:olive",
    "trajectory_sync_abs_l018": "#b2182b",
    "trajectory_sync_v2_l008": "#fdae61",
    "trajectory_sync_v2_vel025_snr10": "#d6604d",
    "phase_continuation_sync_soft": "#5e3c99",
}
MODEL_LINEWIDTH: dict[str, float] = {
    "vanilla": 1.6,
    "periodic": 1.7,
    "trajectory": 1.7,
    "trajectory_sync": 2.6,
    "trajectory_sync_v2": 2.0,
    "phase_continuation": 1.8,
    "phase_continuation_sync": 2.0,
    "phase_continuation_sync_v2": 2.2,
    "trajectory_sync_abs_l018": 2.2,
    "trajectory_sync_v2_l008": 2.0,
    "trajectory_sync_v2_vel025_snr10": 2.0,
    "phase_continuation_sync_soft": 2.0,
}
MODEL_MARKERSIZE: dict[str, float] = {
    "vanilla": 7,
    "periodic": 7,
    "trajectory": 7,
    "trajectory_sync": 9,
    "trajectory_sync_v2": 8,
    "phase_continuation": 7,
    "phase_continuation_sync": 8,
    "phase_continuation_sync_v2": 8,
    "trajectory_sync_abs_l018": 8,
    "trajectory_sync_v2_l008": 8,
    "trajectory_sync_v2_vel025_snr10": 8,
    "phase_continuation_sync_soft": 8,
}
MODEL_FMT: dict[str, str] = {
    "vanilla": "D-",
    "periodic": "s-",
    "trajectory": "^-",
    "trajectory_sync": "o-",
    "trajectory_sync_v2": "X-",
    "phase_continuation": "P-",
    "phase_continuation_sync": "v-",
    "phase_continuation_sync_v2": "*-",
    "trajectory_sync_abs_l018": "h-",
    "trajectory_sync_v2_l008": "X-",
    "trajectory_sync_v2_vel025_snr10": "d-",
    "phase_continuation_sync_soft": "p-",
}
_FALLBACK_COLOR_CYCLE = ("tab:purple", "tab:brown", "tab:cyan", "tab:olive")


def _color_for(key: str, idx: int = 0) -> str:
    return MODEL_COLORS.get(key, _FALLBACK_COLOR_CYCLE[idx % len(_FALLBACK_COLOR_CYCLE)])


def _linewidth_for(key: str) -> float:
    return MODEL_LINEWIDTH.get(key, 1.7)


def _markersize_for(key: str) -> float:
    return MODEL_MARKERSIZE.get(key, 7)


def _fmt_for(key: str) -> str:
    return MODEL_FMT.get(key, "o-")


def _label_for(key: str, state=None) -> str:
    if state is not None and key in getattr(state, "model_specs", {}):
        return state.model_specs[key].label
    return SUMMARY_MODEL_LABELS.get(key, key)


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


def plot_table1_reward_per_step_comparison(
    table1_results: dict[str, list[dict]],
    output_path: str | Path,
    *,
    model_keys: Optional[Sequence[str]] = None,
    state=None,
    interval: str = "ci95",
) -> Path:
    """Save a Table-1 reward-per-step bar chart for the configured models."""
    output_path = Path(output_path)
    keys = list(model_keys) if model_keys is not None else list(DEFAULT_MODEL_KEYS)
    labels = [_label_for(k, state) for k in keys]
    colors = [_color_for(k, i) for i, k in enumerate(keys)]

    means = []
    spreads = []
    for key in keys:
        values = [
            _metric_value(result, "reward_per_step") for result in table1_results[key]
        ]
        mean, spread = _mean_spread(values, interval=interval)
        means.append(mean)
        spreads.append(spread)

    fig, ax = plt.subplots(1, 1, figsize=(max(7.2, 1.6 * len(keys)), 4.8))
    x = np.arange(len(keys))
    ax.bar(x, means, yerr=spreads, capsize=5, color=colors, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Reward / step")
    spread_label = "95% CI" if interval == "ci95" else "std"
    ax.set_title(f"Table 1 reward per step comparison ({spread_label})")
    ax.grid(True, axis="y", alpha=0.3)

    finite_means = [mean for mean in means if np.isfinite(mean)]
    if finite_means:
        y_offset = max(abs(max(finite_means)) * 0.02, 0.02)
        for xpos, mean in zip(x, means):
            if np.isfinite(mean):
                ax.text(
                    xpos,
                    mean + y_offset,
                    f"{mean:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.show()
    print(f"✓ {output_path}")
    return output_path


def plot_evaluation_frequency_comparison(
    table1_results: dict[str, list[dict]],
    sweep_results: dict[str, dict[float, list[dict]]],
    data: dict,
    output_path: str | Path,
    *,
    n_seeds_sweep: int,
    phase_model_keys: Optional[Sequence[str]] = None,
    state=None,
) -> Path:
    """Save the Figure-2 reward-per-step curve over commanded frequency.

    ``table1_results`` is retained for notebook/backward-call compatibility, but
    Figure 2 intentionally contains only phase-conditioned sweep results.
    """
    del table1_results
    output_path = Path(output_path)
    keys = list(phase_model_keys) if phase_model_keys is not None else list(DEFAULT_PHASE_MODEL_KEYS)
    reference_key = keys[0]
    freqs = sorted(float(f) for f in sweep_results[reference_key].keys())

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.0))
    for i, key in enumerate(keys):
        means, stds = _sweep_metric_summary(
            sweep_results[key], freqs, "reward_per_step", interval="std"
        )
        ax.errorbar(
            freqs,
            means,
            yerr=stds,
            fmt=_fmt_for(key),
            capsize=4,
            linewidth=_linewidth_for(key),
            markersize=_markersize_for(key),
            color=_color_for(key, i),
            label=_label_for(key, state),
        )

    f_mean = float(data["freq_window_mean"])
    f_min = float(data["freq_window_min"])
    f_max = float(data["freq_window_max"])
    ax.axvspan(f_min, f_max, alpha=0.12, color="green", label="In-dist range")
    ax.axvline(f_mean, color="gray", ls="--", alpha=0.4)
    ax.set_xlabel("Sampling-time phase freq (Hz)")
    ax.set_ylabel("Reward / step")
    ax.set_title(f"Reward per step vs Frequency (3 in-dist + 2 OOD, n={n_seeds_sweep})")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.show()
    print(f"✓ {output_path}")
    return output_path


def plot_frequency_tracking_alignment(
    sweep_results: dict[str, dict[float, list[dict]]],
    data: dict,
    output_path: str | Path,
    *,
    n_seeds_sweep: int | None = None,
    phase_model_keys: Optional[Sequence[str]] = None,
    state=None,
    interval: str = "ci95",
    periodic_collapse_note: bool = False,
) -> Path:
    """Plot commanded frequency against measured gait frequency.

    When ``periodic_collapse_note`` is True, ``sweep_results['periodic']`` is
    inspected for a separate mode-collapse annotation (rendered as a corner
    text box reporting the mean measured frequency). This is independent of
    ``phase_model_keys``: typically the caller filters ``periodic`` out of
    ``phase_model_keys`` so its line is not drawn, and then enables this flag
    so the collapse is still acknowledged in the figure.
    """
    output_path = Path(output_path)
    keys = list(phase_model_keys) if phase_model_keys is not None else list(DEFAULT_PHASE_MODEL_KEYS)
    reference_key = keys[0]
    freqs = sorted(float(f) for f in sweep_results[reference_key].keys())
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.5))
    for i, key in enumerate(keys):
        means, spreads = _sweep_metric_summary(
            sweep_results[key], freqs, "measured_freq_hz", interval=interval
        )
        ax.errorbar(
            freqs,
            means,
            yerr=spreads,
            fmt=_fmt_for(key),
            capsize=4,
            linewidth=_linewidth_for(key),
            markersize=_markersize_for(key),
            color=_color_for(key, i),
            label=_label_for(key, state),
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
    ax.legend(loc="upper left", fontsize=9)

    if periodic_collapse_note and "periodic" in sweep_results:
        periodic_vals = []
        for _freq, rollouts in sweep_results["periodic"].items():
            for r in rollouts:
                v = r.get("measured_freq_hz", float("nan"))
                if np.isfinite(v):
                    periodic_vals.append(float(v))
        if periodic_vals:
            mean_freq = float(np.mean(periodic_vals))
            ax.text(
                0.98,
                0.02,
                f"Periodic Phase: mode collapse\n(measured ≈ {mean_freq:.2f} Hz across all commands,\n outside plot range)",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                style="italic",
                bbox=dict(
                    boxstyle="round,pad=0.35",
                    facecolor="#f5f5f5",
                    edgecolor="gray",
                    alpha=0.85,
                ),
            )

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
    phase_model_keys: Optional[Sequence[str]] = None,
    state=None,
    interval: str = "ci95",
) -> Path:
    """Save OOD/in-distribution aggregate tracking metrics by frequency zone."""
    output_path = Path(output_path)
    groups = _zone_frequency_groups(freq_protocol)
    keys = list(phase_model_keys) if phase_model_keys is not None else list(DEFAULT_PHASE_MODEL_KEYS)
    metrics = [
        ("abs_freq_error_hz", "|Frequency error| (Hz) ↓"),
        ("phase_locking_value", "Phase locking value ↑"),
    ]
    n_models = len(keys)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(groups))
    total_width = 0.8
    width = total_width / max(n_models, 1)
    offsets = (np.arange(n_models) - (n_models - 1) / 2.0) * width
    for ax, (metric, ylabel) in zip(axes, metrics):
        for idx, key in enumerate(keys):
            means = []
            spreads = []
            for freqs in groups.values():
                values = []
                for freq in freqs:
                    values.extend(
                        _metric_value(result, metric)
                        for result in sweep_results[key][float(freq)]
                    )
                mean, spread = _mean_spread(values, interval=interval)
                means.append(mean)
                spreads.append(spread)
            ax.bar(
                x + offsets[idx],
                means,
                width=width,
                yerr=spreads,
                capsize=4,
                label=_label_for(key, state),
                color=_color_for(key, idx),
                alpha=0.9,
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


def plot_phase_tracking_timeseries(
    sweep_results: dict[str, dict[float, list[dict]]],
    freq_protocol,
    output_path: str | Path,
    *,
    seed_idx: int = 0,
    dt: float = 0.05,
    phase_model_keys: Optional[Sequence[str]] = None,
    state=None,
) -> Path:
    """Plot command vs measured phase over time for one rollout per model."""
    output_path = Path(output_path)
    keys = list(phase_model_keys) if phase_model_keys is not None else list(DEFAULT_PHASE_MODEL_KEYS)

    in_freqs = list(freq_protocol.in_freqs)
    target_freq = float(in_freqs[len(in_freqs) // 2])

    n_panels = len(keys)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 3.8), sharex=True, squeeze=False)
    axes = axes[0]
    for col, key in enumerate(keys):
        label = _label_for(key, state)
        color = _color_for(key, col)
        rollout = sweep_results[key][target_freq][seed_idx]
        measured = np.asarray(rollout.get("measured_phase", []), dtype=np.float64)
        command = np.asarray(rollout.get("command_phase", []), dtype=np.float64)
        plv = float(rollout.get("phase_locking_value", float("nan")))
        n = min(len(measured), len(command))
        ax = axes[col]
        if n < 2:
            ax.text(0.5, 0.5, "no measured phase", ha="center", va="center")
            ax.set_title(f"{label}  |  f_cmd={target_freq:.3f} Hz")
            ax.set_xlabel("Time (s)")
            continue
        t = np.arange(n) * float(dt)
        measured_u = np.unwrap(measured[:n])
        command_u = np.unwrap(command[:n])
        offset = measured_u[0] - command_u[0]
        command_aligned = command_u + offset

        ax.plot(t, command_aligned, "--", color="black", alpha=0.6, label="command (target)")
        ax.plot(t, measured_u, "-", color=color, linewidth=_linewidth_for(key), label=f"measured ({label})")
        ax.set_ylabel("Unwrapped phase (rad)")
        ax.set_title(f"{label}  |  f_cmd={target_freq:.3f} Hz, PLV={plv:.3f}")
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)

    plt.suptitle(
        f"Phase tracking @ f_cmd={target_freq:.3f} Hz (seed {seed_idx})", fontsize=11
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.show()
    print(f"✓ {output_path}")
    return output_path


PAPER_FIGURE_NAMES = {
    "reward_per_step_comparison":  "eval_figure1_reward_per_step_comparison.png",
    "reward_per_step_vs_freq":     "eval_figure2_reward_per_step_vs_freq.png",
    "target_vs_measured_freq":     "eval_figure3_target_vs_measured_freq.png",
    "zone_tracking_metrics":       "eval_figure4_zone_tracking_metrics.png",
    "phase_tracking_timeseries":   "eval_figure5_phase_tracking_timeseries.png",
}


def plot_paper_figures(
    table1_results: dict[str, list[dict]],
    sweep_results: dict[str, dict[float, list[dict]]],
    freq_protocol,
    data: dict,
    figures_dir: str | Path,
    *,
    n_seeds_sweep: int,
    model_keys: Optional[Sequence[str]] = None,
    phase_model_keys: Optional[Sequence[str]] = None,
    state=None,
    interval: str = "ci95",
) -> dict[str, Path]:
    """Emit the five paper-quality evaluation figures for the configured models."""
    figures_dir = Path(figures_dir)
    paths = {}
    paths["reward_per_step_comparison"] = plot_table1_reward_per_step_comparison(
        table1_results,
        figures_dir / PAPER_FIGURE_NAMES["reward_per_step_comparison"],
        model_keys=model_keys,
        state=state,
        interval=interval,
    )
    paths["reward_per_step_vs_freq"] = plot_evaluation_frequency_comparison(
        table1_results,
        sweep_results,
        data,
        figures_dir / PAPER_FIGURE_NAMES["reward_per_step_vs_freq"],
        n_seeds_sweep=n_seeds_sweep,
        phase_model_keys=phase_model_keys,
        state=state,
    )
    # Figure 3 only: exclude 'periodic' from the lines because its measured_freq
    # mode-collapses near ~1.2 Hz and falls outside the commanded-frequency ylim.
    # Other figures keep periodic for full comparison.
    base_phase_keys = phase_model_keys if phase_model_keys is not None else DEFAULT_PHASE_MODEL_KEYS
    phase_keys_for_freq_alignment = tuple(k for k in base_phase_keys if k != "periodic")
    paths["target_vs_measured_freq"] = plot_frequency_tracking_alignment(
        sweep_results,
        data,
        figures_dir / PAPER_FIGURE_NAMES["target_vs_measured_freq"],
        n_seeds_sweep=n_seeds_sweep,
        phase_model_keys=phase_keys_for_freq_alignment,
        state=state,
        periodic_collapse_note=True,
    )
    paths["zone_tracking_metrics"] = plot_zone_aggregated_tracking_metrics(
        freq_protocol,
        sweep_results,
        figures_dir / PAPER_FIGURE_NAMES["zone_tracking_metrics"],
        phase_model_keys=phase_model_keys,
        state=state,
        interval=interval,
    )
    paths["phase_tracking_timeseries"] = plot_phase_tracking_timeseries(
        sweep_results,
        freq_protocol,
        figures_dir / PAPER_FIGURE_NAMES["phase_tracking_timeseries"],
        phase_model_keys=phase_model_keys,
        state=state,
    )
    return paths


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
    mean, spread, _ = summary_stats(values, interval=interval)
    return mean, spread


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
