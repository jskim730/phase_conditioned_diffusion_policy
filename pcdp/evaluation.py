"""Official evaluation orchestration routines for notebook 05.

The evaluation notebook should only orchestrate these functions.  Model loading,
rollout protocol construction, frequency sweeps, tabulation, and result
serialization live here so the paper-quality experiment is reproducible from a
single source module.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from .configs import ExperimentConfig, get_experiment_config
from .models import count_params
from .phase import (
    annotate_rollout_with_gait_metrics,
    controllability_sweep_frequencies,
    controllability_sweep_frequency_groups,
    frequency_zone,
    make_phase_trajectory_fn,
)
from .sampling import nanmean, nanstd, rollout_multi_seed
from .training import load_checkpoint

EVAL_CONFIG_NAMES: tuple[str, ...] = (
    "vanilla",
    "periodic_phase",
    "phase_trajectory",
    "phase_trajectory_sync",
    "phase_trajectory_sync_low_noise",
    "phase_trajectory_sync_velocity_dominant",
    "phase_trajectory_sync_strong_velocity",
    "phase_trajectory_sync_velocity_only",
)
CONFIG_TO_MODEL_KEY: dict[str, str] = {
    "vanilla": "vanilla",
    "periodic_phase": "periodic",
    "phase_trajectory": "trajectory",
    "phase_trajectory_sync": "trajectory_sync",
    "phase_trajectory_sync_low_noise": "trajectory_sync_low_noise",
    "phase_trajectory_sync_velocity_dominant": "trajectory_sync_velocity_dominant",
    "phase_trajectory_sync_strong_velocity": "trajectory_sync_strong_velocity",
    "phase_trajectory_sync_velocity_only": "trajectory_sync_velocity_only",
}
MODEL_KEYS: tuple[str, ...] = tuple(CONFIG_TO_MODEL_KEY[name] for name in EVAL_CONFIG_NAMES)
PHASE_MODEL_KEYS: tuple[str, ...] = tuple(key for key in MODEL_KEYS if key != "vanilla")
SUMMARY_MODEL_LABELS: dict[str, str] = {
    "vanilla": "Vanilla DP",
    "periodic": "Periodic Phase",
    "trajectory": "Phase Trajectory",
    "trajectory_sync": "Phase Trajectory + Sync v2 (Velocity + SNR)",
    "trajectory_sync_low_noise": "Sync Low-Noise Only",
    "trajectory_sync_velocity_dominant": "Sync Low-Noise Velocity-Dominant",
    "trajectory_sync_strong_velocity": "Sync Low-Noise Strong Velocity",
    "trajectory_sync_velocity_only": "Sync Low-Noise Velocity-Only",
}
CONFIG_USES_PHASE_TRAJECTORY: dict[str, bool] = {
    "vanilla": False,
    "periodic_phase": True,
    "phase_trajectory": True,
    "phase_trajectory_sync": True,
    "phase_trajectory_sync_low_noise": True,
    "phase_trajectory_sync_velocity_dominant": True,
    "phase_trajectory_sync_strong_velocity": True,
    "phase_trajectory_sync_velocity_only": True,
}


@dataclass(frozen=True)
class LoadedEvalModel:
    """One checkpointed model and the sampling metadata needed for rollout."""

    config_name: str
    label: str
    model: object
    ema: object
    cond_fn: object


@dataclass(frozen=True)
class ModelEvalSpec:
    """Notebook-facing model spec keyed by the names shown in tables."""

    loaded_name: str
    label: str
    model: object
    ema: object
    cond_fn: object
    uses_phase_trajectory: bool


@dataclass(frozen=True)
class EvaluationState:
    """Loaded configs/models plus common DDIM sampling settings."""

    configs: Mapping[str, ExperimentConfig]
    loaded_models: Mapping[str, LoadedEvalModel]
    model_specs: Mapping[str, ModelEvalSpec]
    noise_scheduler_config: dict
    num_inference_steps: int


@dataclass(frozen=True)
class FrequencySweepProtocol:
    """Frequency grid used by the controllability experiment."""

    in_freqs: np.ndarray
    ood_freqs: np.ndarray
    sweep_freqs: np.ndarray
    zone_labels: np.ndarray


def load_evaluation_state(
    data: Mapping[str, object],
    *,
    device: str,
    checkpoints_dir: str | Path,
    config_names: Sequence[str] = EVAL_CONFIG_NAMES,
) -> EvaluationState:
    """Load all notebook-05 checkpoints with best EMA weights applied."""
    configs = {name: get_experiment_config(name) for name in config_names}
    base_cfg = configs["vanilla"]
    loaded_models: dict[str, LoadedEvalModel] = {}

    for name, cfg in configs.items():
        model = cfg.build_model(data, device=device)
        ema = cfg.build_ema(model)
        load_checkpoint(
            cfg.checkpoint_path(checkpoints_dir),
            model,
            ema,
            device=device,
            use_best_ema=True,
        )
        loaded_models[name] = LoadedEvalModel(
            config_name=name,
            label=cfg.display_name,
            model=model,
            ema=ema,
            cond_fn=cfg.resolve_sample_cond_fn(),
        )

    unknown_configs = [name for name in configs if name not in CONFIG_TO_MODEL_KEY]
    if unknown_configs:
        raise KeyError(f"Missing evaluation mapping for configs: {unknown_configs}")

    model_specs = {}
    for config_name in config_names:
        model_key = CONFIG_TO_MODEL_KEY[config_name]
        model_specs[model_key] = _spec_from_loaded(
            loaded_models[config_name],
            label=SUMMARY_MODEL_LABELS[model_key],
            uses_phase=CONFIG_USES_PHASE_TRAJECTORY[config_name],
        )

    print("\n✓ Evaluation checkpoints loaded")
    for key in MODEL_KEYS:
        if key not in model_specs:
            continue
        spec = model_specs[key]
        print(
            f"  {spec.label:<18s}: {count_params(spec.model)['trainable'] / 1e6:.2f}M trainable"
        )

    return EvaluationState(
        configs=configs,
        loaded_models=loaded_models,
        model_specs=model_specs,
        noise_scheduler_config=base_cfg.noise_scheduler_config(),
        num_inference_steps=base_cfg.diffusion.num_inference_steps,
    )


def _spec_from_loaded(
    loaded: LoadedEvalModel, *, label: str, uses_phase: bool
) -> ModelEvalSpec:
    return ModelEvalSpec(
        loaded_name=loaded.config_name,
        label=label,
        model=loaded.model,
        ema=loaded.ema,
        cond_fn=loaded.cond_fn,
        uses_phase_trajectory=uses_phase,
    )


def evaluate_model_spec_at_frequency(
    spec: ModelEvalSpec,
    *,
    freq_hz: float,
    env,
    data: Mapping[str, object],
    device: str,
    noise_scheduler_config: Mapping[str, object],
    num_inference_steps: int,
    n_seeds: int,
    max_steps: int,
    dt: float = 0.05,
    phase0: float = 0.0,
    deterministic_sampling: bool = True,
) -> list[dict]:
    """Run one model spec at one target frequency condition.

    This is the single-model evaluation helper used by both the official
    evaluation orchestration in this module and compatibility wrappers in
    training-oriented modules.  Keep the ``rollout_multi_seed`` argument
    assembly here so rollout protocol changes have one source of truth.
    """
    phase_fn = None
    if spec.uses_phase_trajectory:
        phase_fn = make_phase_trajectory_fn(freq_hz, dt=dt, phase0=phase0)

    raw_results = rollout_multi_seed(
        spec.model,
        spec.ema,
        env,
        n_seeds=n_seeds,
        deterministic_sampling=deterministic_sampling,
        noise_scheduler_config=dict(noise_scheduler_config),
        obs_mean=data["obs_mean"],
        obs_std=data["obs_std"],
        act_min=data["act_min"],
        act_range=data["act_range"],
        cond_fn=spec.cond_fn,
        obs_horizon=data["OBS_HORIZON"],
        pred_horizon=data["PRED_HORIZON"],
        action_horizon=data["ACTION_HORIZON"],
        obs_dim=data["OBS_DIM"],
        act_dim=data["ACT_DIM"],
        num_inference_steps=num_inference_steps,
        max_steps=max_steps,
        phase_trajectory_fn=phase_fn,
        device=device,
    )
    return [
        annotate_rollout_with_gait_metrics(
            result,
            command_freq_hz=freq_hz,
            dt=dt,
            phase0=phase0,
        )
        for result in raw_results
    ]


def evaluate_model_at_frequency(
    state: EvaluationState,
    model_key: str,
    freq_hz: float,
    *,
    env,
    data: Mapping[str, object],
    device: str,
    n_seeds: int,
    max_steps: int,
    dt: float = 0.05,
    phase0: float = 0.0,
    deterministic_sampling: bool = True,
) -> list[dict]:
    """Run one registered model at one target frequency condition."""
    return evaluate_model_spec_at_frequency(
        state.model_specs[model_key],
        freq_hz=freq_hz,
        env=env,
        data=data,
        device=device,
        noise_scheduler_config=state.noise_scheduler_config,
        num_inference_steps=state.num_inference_steps,
        n_seeds=n_seeds,
        max_steps=max_steps,
        dt=dt,
        phase0=phase0,
        deterministic_sampling=deterministic_sampling,
    )


def run_in_distribution_evaluation(
    state: EvaluationState,
    *,
    env,
    data: Mapping[str, object],
    device: str,
    n_seeds: int,
    max_steps: int,
    dt: float = 0.05,
) -> dict[str, list[dict]]:
    """Evaluate all three models at the training mean frequency."""
    freq_hz = float(data["freq_window_mean"])
    print(f"=== Table 1: In-dist @ f={freq_hz:.3f} Hz, n={n_seeds} seeds ===\n")
    results: dict[str, list[dict]] = {}
    t0 = time.time()
    for model_key in MODEL_KEYS:
        print(f"\n--- {state.model_specs[model_key].label} ---")
        results[model_key] = evaluate_model_at_frequency(
            state,
            model_key,
            freq_hz,
            env=env,
            data=data,
            device=device,
            n_seeds=n_seeds,
            max_steps=max_steps,
            dt=dt,
        )
    print(f"\n총 시간: {(time.time() - t0) / 60:.1f} min")
    return results


def build_frequency_sweep_protocol(
    data: Mapping[str, object],
    *,
    n_in_dist: int = 3,
    ood_iqr_scale: float = 1.5,
) -> FrequencySweepProtocol:
    """Build the 3 in-distribution + 2 IQR-based OOD Table 2 grid."""
    in_freqs, ood_freqs = controllability_sweep_frequency_groups(
        data,
        ood_iqr_scale=ood_iqr_scale,
    )
    sweep_freqs = controllability_sweep_frequencies(
        data,
        n_in_dist=n_in_dist,
        ood_iqr_scale=ood_iqr_scale,
    )
    iqr_label = f"{ood_iqr_scale:g}IQR"
    zone_labels = np.array(
        [
            f"OOD-low (q25-{iqr_label})",
            "in-dist (q25)",
            "in-dist (q50)",
            "in-dist (q75)",
            f"OOD-high (q75+{iqr_label})",
        ]
    )
    return FrequencySweepProtocol(
        in_freqs=in_freqs,
        ood_freqs=ood_freqs,
        sweep_freqs=sweep_freqs,
        zone_labels=zone_labels,
    )


def run_frequency_sweep_evaluation(
    state: EvaluationState,
    protocol: FrequencySweepProtocol,
    *,
    env,
    data: Mapping[str, object],
    device: str,
    model_keys: Optional[Sequence[str]] = None,
    n_seeds: int,
    max_steps: int,
    dt: float = 0.05,
    deterministic_sampling: bool = True,
) -> dict[str, dict[float, list[dict]]]:
    """Run target-frequency controllability sweeps for phase-conditioned models.

    ``model_keys`` defaults to the current ``PHASE_MODEL_KEYS`` module-level
    list, so notebooks can register additional phase-conditioned models (e.g.
    phase-sync fine-tuned variants) by extending that tuple before calling.
    """
    keys = tuple(model_keys) if model_keys is not None else PHASE_MODEL_KEYS
    all_results: dict[str, dict[float, list[dict]]] = {}
    for model_key in keys:
        print(f"=== {state.model_specs[model_key].label} frequency sweep ===")
        t0 = time.time()
        model_results: dict[float, list[dict]] = {}
        for freq in protocol.sweep_freqs:
            freq = float(freq)
            zone = frequency_zone(freq, data)
            print(f"\n--- freq={freq:.3f} Hz ({zone}) ---")
            model_results[freq] = evaluate_model_at_frequency(
                state,
                model_key,
                freq,
                env=env,
                data=data,
                device=device,
                n_seeds=n_seeds,
                max_steps=max_steps,
                dt=dt,
                deterministic_sampling=deterministic_sampling,
            )
        print(
            f"\n{state.model_specs[model_key].label} sweep 시간: {(time.time() - t0) / 60:.1f} min\n"
        )
        all_results[model_key] = model_results
    return all_results


PHASE_CONDITION_LABELS: dict[str, str] = {
    "vanilla": "none",
    "periodic": "first phase only",
    "trajectory": "full phase trajectory",
    "trajectory_sync": "full phase trajectory + velocity/SNR sync loss",
    "trajectory_sync_low_noise": "full phase trajectory + low-noise sync loss",
    "trajectory_sync_velocity_dominant": "full phase trajectory + velocity-dominant sync loss",
    "trajectory_sync_strong_velocity": "full phase trajectory + strong velocity sync loss",
    "trajectory_sync_velocity_only": "full phase trajectory + velocity-only sync loss",
}


def summary_stats(values, *, interval: str) -> tuple[float, float, int]:
    """Return ``(mean, spread, n_finite)`` over the finite entries of ``values``.

    ``interval='ci95'`` returns ``1.96 σ/√n``; ``interval='std'`` returns ``σ``.
    Shared with ``experiment_plots`` so table cells and bar/error plots agree.
    """
    arr = np.asarray(values, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan"), float("nan"), 0
    mean = float(finite.mean())
    std = float(finite.std())
    if interval == "ci95":
        spread = 1.96 * std / np.sqrt(float(finite.size))
    elif interval == "std":
        spread = std
    else:
        raise ValueError("interval must be 'ci95' or 'std'")
    return mean, float(spread), int(finite.size)


def _cell(values, *, decimals: int, interval: str) -> str:
    mean, spread, _ = summary_stats(values, interval=interval)
    if not np.isfinite(mean):
        return "n/a"
    return f"{mean:.{decimals}f} ± {spread:.{decimals}f}"


def _bold_best(df, columns, *, higher_is_better: bool = True) -> None:
    """Wrap the best mean in each ``columns`` entry with markdown bold."""
    for col in columns:
        means = df[col].apply(lambda s: float(s.split(" ± ")[0]) if "±" in s else np.nan)
        if not means.notna().any():
            continue
        best_i = means.idxmax() if higher_is_better else means.idxmin()
        df.at[best_i, col] = f"**{df.at[best_i, col]}**"


def _write(df, path: Path, *, title: str) -> Path:
    """Render ``df`` as a markdown table.

    ``disable_numparse=True`` keeps numeric-looking strings (``+0.770``,
    ``1.000``) verbatim instead of letting ``tabulate`` strip trailing zeros.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    md = df.to_markdown(index=False, disable_numparse=True)
    path.write_text(f"# {title}\n\n{md}\n", encoding="utf-8")
    print(f"✓ {path}")
    return path


def write_table1_summary_markdown(
    state: EvaluationState,
    table1_results: Mapping[str, list[dict]],
    output_path: str | Path,
    *,
    freq_hz: float,
    interval: str = "ci95",
) -> Path:
    """Export Table 1 (in-distribution gait quality) with best values bolded.

    Includes ``Measured freq (Hz)`` so vanilla's spontaneous gait frequency is
    visible alongside the phase-conditioned variants.  Best values are bolded
    on directional metrics only (the measured-freq column is a descriptor, not
    a target to maximize, so it is left un-bolded).
    """
    import pandas as pd

    vanilla_rps_mean = float(np.nanmean(metric_array(table1_results["vanilla"], "reward_per_step")))
    rows = []
    for key in MODEL_KEYS:
        surv, _rew = result_arrays(table1_results[key])
        rps = metric_array(table1_results[key], "reward_per_step")
        xvel = metric_array(table1_results[key], "mean_x_velocity")
        fmeas = metric_array(table1_results[key], "measured_freq_hz")
        rps_delta = float(np.nanmean(rps)) - vanilla_rps_mean
        rows.append({
            "Model": state.model_specs[key].label,
            "Phase condition": PHASE_CONDITION_LABELS[key],
            "Survival ↑": _cell(surv, decimals=0, interval=interval),
            "Reward / step ↑": _cell(rps, decimals=3, interval=interval),
            "Mean x-velocity ↑": _cell(xvel, decimals=3, interval=interval),
            "Measured freq (Hz)": _cell(fmeas, decimals=3, interval=interval),
            "Δ reward/step vs Vanilla ↑": f"{rps_delta:+.3f}" if np.isfinite(rps_delta) else "n/a",
        })
    df = pd.DataFrame(rows)
    _bold_best(df, ["Survival ↑", "Reward / step ↑", "Mean x-velocity ↑"],
               higher_is_better=True)
    return _write(df, Path(output_path),
                  title=f"Table 1. In-distribution rollout quality @ {freq_hz:.3f} Hz")


def write_frequency_tracking_table_markdown(
    protocol: FrequencySweepProtocol,
    sweep_results: Mapping[str, Mapping[float, list[dict]]],
    output_path: str | Path,
    *,
    interval: str = "ci95",
) -> Path:
    """Export Table 2 (command-frequency tracking) with best values bolded per row pair.

    ``Measured freq`` + ``|freq error|`` express the same tracking quality from
    two angles, so ``Freq ratio`` is not exported.  ``Survival`` is dropped here
    because the in-distribution analysis (Table 1) already covers survival —
    Table 2 focuses on command-tracking metrics.
    """
    import pandas as pd

    rows = []
    for freq, zone in zip(protocol.sweep_freqs, protocol.zone_labels):
        freq = float(freq)
        for model_key in PHASE_MODEL_KEYS:
            rollouts = sweep_results[model_key][freq]
            rows.append({
                "Target freq (Hz)": f"{freq:.3f}",
                "Zone": str(zone),
                "Model": SUMMARY_MODEL_LABELS[model_key],
                "Measured freq (Hz)": _cell(metric_array(rollouts, "measured_freq_hz"), decimals=3, interval=interval),
                "|freq error| (Hz) ↓": _cell(metric_array(rollouts, "abs_freq_error_hz"), decimals=3, interval=interval),
                "PLV ↑": _cell(metric_array(rollouts, "phase_locking_value"), decimals=3, interval=interval),
                "Reward / step ↑": _cell(metric_array(rollouts, "reward_per_step"), decimals=3, interval=interval),
            })
    df = pd.DataFrame(rows)
    # Bold best per freq pair (rows 2i and 2i+1)
    for start in range(0, len(df), len(PHASE_MODEL_KEYS)):
        chunk = df.iloc[start:start + len(PHASE_MODEL_KEYS)]
        _bold_best(chunk, ["|freq error| (Hz) ↓"], higher_is_better=False)
        _bold_best(chunk, ["PLV ↑", "Reward / step ↑"], higher_is_better=True)
        df.iloc[start:start + len(PHASE_MODEL_KEYS)] = chunk
    return _write(df, Path(output_path), title="Table 2. Frequency command tracking")


def print_table1_summary(
    state: EvaluationState, table1_results: Mapping[str, list[dict]], *, freq_hz: float
) -> None:
    """Print Table 1 to the console using the same pandas formatting."""
    import pandas as pd

    n_seeds = len(next(iter(table1_results.values())))
    rows = []
    for key in MODEL_KEYS:
        surv, _rew = result_arrays(table1_results[key])
        rps = metric_array(table1_results[key], "reward_per_step")
        xvel = metric_array(table1_results[key], "mean_x_velocity")
        fmeas = metric_array(table1_results[key], "measured_freq_hz")
        rows.append({
            "Model": state.model_specs[key].label,
            "Survival": _cell(surv, decimals=0, interval="std"),
            "Reward/step": _cell(rps, decimals=3, interval="std"),
            "Forward vel.": _cell(xvel, decimals=3, interval="std"),
            "Measured freq": _cell(fmeas, decimals=3, interval="std"),
        })
    print(f"\n=== Table 1: In-distribution @ f={freq_hz:.3f} Hz, n={n_seeds} ===")
    print(pd.DataFrame(rows).to_string(index=False))


def print_frequency_sweep_summary(
    data: Mapping[str, object],
    protocol: FrequencySweepProtocol,
    sweep_results: Mapping[str, Mapping[float, list[dict]]],
) -> None:
    """Print Table 2 to the console using the same pandas formatting."""
    import pandas as pd

    rows = []
    for freq, zone in zip(protocol.sweep_freqs, protocol.zone_labels):
        freq = float(freq)
        for model_key in PHASE_MODEL_KEYS:
            rollouts = sweep_results[model_key][freq]
            rows.append({
                "freq_cmd": f"{freq:.3f}",
                "zone": str(zone),
                "model": SUMMARY_MODEL_LABELS[model_key],
                "freq_meas": _cell(metric_array(rollouts, "measured_freq_hz"), decimals=3, interval="std"),
                "|freq_err|": _cell(metric_array(rollouts, "abs_freq_error_hz"), decimals=3, interval="std"),
                "PLV": _cell(metric_array(rollouts, "phase_locking_value"), decimals=3, interval="std"),
                "reward/step": _cell(metric_array(rollouts, "reward_per_step"), decimals=3, interval="std"),
            })
    print("=== Table 2: Frequency command tracking ===")
    print(pd.DataFrame(rows).to_string(index=False))


def result_arrays(
    results_list: Sequence[Mapping[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    """Return survival/reward arrays for one list of rollout dicts."""
    survival = np.array([r["survival"] for r in results_list], dtype=np.float32)
    reward = np.array([r["total_reward"] for r in results_list], dtype=np.float32)
    return survival, reward


def metric_array(results_list: Sequence[Mapping[str, object]], key: str) -> np.ndarray:
    """Return an optional per-rollout metric array with NaN fallback."""
    if key == "reward_per_step":
        values = [
            r.get(key, r["total_reward"] / max(int(r["survival"]), 1))
            for r in results_list
        ]
    else:
        values = [r.get(key, np.nan) for r in results_list]
    return np.asarray(values, dtype=np.float32)


def build_eval_results_payload(
    data: Mapping[str, object],
    table1_results: Mapping[str, list[dict]],
    freq_protocol: FrequencySweepProtocol,
    freq_results: Mapping[str, Mapping[float, list[dict]]],
    *,
    n_seeds_indist: int,
    n_seeds_sweep: int,
) -> dict[str, np.ndarray]:
    """Convert raw rollout dicts into serializable arrays for ``np.savez``."""
    payload: dict[str, np.ndarray] = {
        "n_seeds_indist": np.asarray(n_seeds_indist, dtype=np.int32),
        "n_seeds_sweep": np.asarray(n_seeds_sweep, dtype=np.int32),
        "f_mean": np.asarray(float(data["freq_window_mean"]), dtype=np.float32),
        "freq_window_min": np.asarray(float(data["freq_window_min"]), dtype=np.float32),
        "freq_window_max": np.asarray(float(data["freq_window_max"]), dtype=np.float32),
        "in_freqs": np.asarray(freq_protocol.in_freqs, dtype=np.float32),
        "ood_freqs": np.asarray(freq_protocol.ood_freqs, dtype=np.float32),
        "sweep_freqs": np.asarray(freq_protocol.sweep_freqs, dtype=np.float32),
        "sweep_zone_labels": np.asarray(freq_protocol.zone_labels),
    }

    for key in MODEL_KEYS:
        survival, reward = result_arrays(table1_results[key])
        payload[f"table1_{key}_survival"] = survival
        payload[f"table1_{key}_reward"] = reward
        _add_metric_vectors(payload, f"table1_{key}", table1_results[key])

    _add_grid_results(
        payload, prefix="freq", grid=freq_protocol.sweep_freqs, results=freq_results
    )
    return payload


def _add_metric_vectors(
    payload: dict[str, np.ndarray],
    prefix: str,
    results_list: Sequence[Mapping[str, object]],
) -> None:
    metric_keys = (
        "reward_per_step",
        "mean_x_velocity",
        "x_displacement",
        "mean_reward_forward",
        "measured_freq_hz",
        "freq_error_hz",
        "abs_freq_error_hz",
        "freq_ratio",
        "mean_abs_phase_error",
        "phase_locking_value",
        "phase_offset_error",
        "abs_phase_offset_error",
    )
    for key in metric_keys:
        payload[f"{prefix}_{key}"] = metric_array(results_list, key).astype(np.float32)


def _add_grid_results(
    payload: dict[str, np.ndarray],
    *,
    prefix: str,
    grid: Sequence[float],
    results: Mapping[str, Mapping[float, list[dict]]],
) -> None:
    metric_keys = (
        "survival",
        "reward",
        "reward_per_step",
        "mean_x_velocity",
        "x_displacement",
        "mean_reward_forward",
        "measured_freq_hz",
        "freq_error_hz",
        "abs_freq_error_hz",
        "freq_ratio",
        "mean_abs_phase_error",
        "phase_locking_value",
        "phase_offset_error",
        "abs_phase_offset_error",
    )
    for model_key in PHASE_MODEL_KEYS:
        rows_by_metric: dict[str, list[np.ndarray]] = {key: [] for key in metric_keys}
        for value in grid:
            rows = results[model_key][float(value)]
            survival, reward = result_arrays(rows)
            rows_by_metric["survival"].append(survival)
            rows_by_metric["reward"].append(reward)
            for key in metric_keys[2:]:
                rows_by_metric[key].append(metric_array(rows, key))
        for key, rows in rows_by_metric.items():
            payload[f"{prefix}_{model_key}_{key}"] = np.stack(rows).astype(np.float32)


def save_eval_results_npz(
    payload: Mapping[str, np.ndarray], output_path: str | Path
) -> Path:
    """Persist evaluation arrays and print a reproducible manifest."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)
    print(f"✓ Saved: {output_path} ({os.path.getsize(output_path) / 1024:.1f} KB)")
    print(f"\nKeys: {list(payload.keys())}")
    return output_path
