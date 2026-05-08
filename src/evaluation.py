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
from typing import Mapping, Sequence

import numpy as np

from configs import ExperimentConfig, get_experiment_config
from models import count_params
from gait_metrics import annotate_rollout_with_gait_metrics
from phase import (
    controllability_sweep_frequencies,
    controllability_sweep_frequency_groups,
    frequency_zone,
    make_phase_trajectory_fn,
)
from sampling import rollout_multi_seed
from training import load_checkpoint

EVAL_CONFIG_NAMES: tuple[str, ...] = ("vanilla", "periodic_phase", "phase_trajectory")
MODEL_KEYS: tuple[str, ...] = ("vanilla", "periodic", "trajectory")
PHASE_MODEL_KEYS: tuple[str, ...] = ("periodic", "trajectory")
SUMMARY_MODEL_LABELS: dict[str, str] = {
    "vanilla": "Vanilla DP",
    "periodic": "Periodic Phase",
    "trajectory": "Trajectory (ours)",
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

    model_specs = {
        "vanilla": _spec_from_loaded(
            loaded_models["vanilla"], label="Vanilla DP", uses_phase=False
        ),
        "periodic": _spec_from_loaded(
            loaded_models["periodic_phase"], label="Periodic Phase", uses_phase=True
        ),
        "trajectory": _spec_from_loaded(
            loaded_models["phase_trajectory"],
            label="Trajectory (ours)",
            uses_phase=True,
        ),
    }

    print("\n✓ Evaluation checkpoints loaded")
    for key in MODEL_KEYS:
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
    phase_joint_idx = _phase_joint_idx_from_data(data)
    return [
        annotate_rollout_with_gait_metrics(
            result,
            command_freq_hz=freq_hz,
            dt=dt,
            phase0=phase0,
            phase_joint_idx=phase_joint_idx,
        )
        for result in raw_results
    ]


def _phase_joint_idx_from_data(data: Mapping[str, object], default: int = 19) -> int:
    """Return the expert-demo phase joint index used for Hilbert phase labels."""
    if "phase_joint_idx" in data:
        return int(data["phase_joint_idx"])
    demos = data.get("demos") if isinstance(data, Mapping) else None
    if demos is not None and "phase_joint_idx" in demos:
        return int(demos["phase_joint_idx"])
    return int(default)


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
    model_keys: Sequence[str] = PHASE_MODEL_KEYS,
    n_seeds: int,
    max_steps: int,
    dt: float = 0.05,
    deterministic_sampling: bool = True,
) -> dict[str, dict[float, list[dict]]]:
    """Run target-frequency controllability sweeps for phase-conditioned models."""
    all_results: dict[str, dict[float, list[dict]]] = {}
    for model_key in model_keys:
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


def print_table1_summary(
    state: EvaluationState, table1_results: Mapping[str, list[dict]], *, freq_hz: float
) -> None:
    """Print in-distribution gait quality, not just survival/return."""
    n_seeds = len(next(iter(table1_results.values())))
    print(f"\n=== Table 1: In-distribution @ f={freq_hz:.3f} Hz, n={n_seeds} ===")
    print(
        f"{'Model':>20s} | {'Survival':>15s} | {'Reward':>17s} | "
        f"{'Reward/Step':>17s} | {'Forward Vel.':>17s}"
    )
    print("-" * 96)
    for key in MODEL_KEYS:
        surv, rew = result_arrays(table1_results[key])
        rps = metric_array(table1_results[key], "reward_per_step")
        x_vel = metric_array(table1_results[key], "mean_x_velocity")
        print(
            f"{state.model_specs[key].label:>20s} | "
            f"{surv.mean():>5.0f} ± {surv.std():>4.0f}    | "
            f"{rew.mean():>7.1f} ± {rew.std():>5.1f} | "
            f"{nanmean(rps):>7.3f} ± {nanstd(rps):>5.3f} | "
            f"{nanmean(x_vel):>7.3f} ± {nanstd(x_vel):>5.3f}"
        )

    print("\n=== Standard Error (std/√n) ===")
    for key in MODEL_KEYS:
        _, rew = result_arrays(table1_results[key])
        rps = metric_array(table1_results[key], "reward_per_step")
        x_vel = metric_array(table1_results[key], "mean_x_velocity")
        print(
            f"{state.model_specs[key].label:>20s}: "
            f"reward SE={rew.std() / np.sqrt(len(rew)):.1f}, "
            f"reward/step SE={nanstd(rps) / np.sqrt(len(rps)):.3f}, "
            f"x_vel SE={nanstd(x_vel) / np.sqrt(len(x_vel)):.3f}"
        )


def print_frequency_sweep_summary(
    data: Mapping[str, object],
    protocol: FrequencySweepProtocol,
    sweep_results: Mapping[str, Mapping[float, list[dict]]],
) -> None:
    """Print command-frequency tracking metrics for phase-conditioned models."""
    print("=== Table 2: Frequency command tracking ===")
    print(
        f"{'freq_cmd':>8s} | {'zone':>22s} | {'model':>18s} | {'survival':>12s} | "
        f"{'reward/step':>13s} | {'x_vel':>11s} | {'freq_meas':>13s} | {'|freq_err|':>12s} | {'PLV':>9s}"
    )
    print("-" * 142)
    for freq, zone in zip(protocol.sweep_freqs, protocol.zone_labels):
        freq = float(freq)
        for model_key in PHASE_MODEL_KEYS:
            rows = sweep_results[model_key][freq]
            surv, _ = result_arrays(rows)
            rps = metric_array(rows, "reward_per_step")
            x_vel = metric_array(rows, "mean_x_velocity")
            f_meas = metric_array(rows, "measured_freq_hz")
            f_err = metric_array(rows, "abs_freq_error_hz")
            plv = metric_array(rows, "phase_locking_value")
            print(
                f"{freq:>8.3f} | {zone:>22s} | {SUMMARY_MODEL_LABELS[model_key]:>18s} | "
                f"{surv.mean():>5.0f}±{surv.std():<4.0f} | "
                f"{nanmean(rps):>6.3f}±{nanstd(rps):<5.3f} | "
                f"{nanmean(x_vel):>5.3f}±{nanstd(x_vel):<5.3f} | "
                f"{nanmean(f_meas):>6.3f}±{nanstd(f_meas):<5.3f} | "
                f"{nanmean(f_err):>6.3f}±{nanstd(f_err):<5.3f} | "
                f"{nanmean(plv):>5.3f}±{nanstd(plv):<5.3f}"
            )


PHASE_CONDITION_LABELS: dict[str, str] = {
    "vanilla": "none",
    "periodic": "first phase only",
    "trajectory": "full phase trajectory",
}


def write_table1_summary_markdown(
    state: EvaluationState,
    table1_results: Mapping[str, list[dict]],
    output_path: str | Path,
    *,
    freq_hz: float,
    interval: str = "ci95",
) -> Path:
    """Export Table 1 as Markdown with gait-quality metrics and best values bolded.

    Table 1 supports the first paper claim: the trajectory-conditioned policy
    should preserve in-distribution locomotion quality while adding controllable
    phase/frequency inputs.  Values are formatted as mean plus either a 95% CI
    or standard deviation across rollout seeds.
    """
    output_path = Path(output_path)
    rows = build_table1_summary_rows(
        state, table1_results, freq_hz=freq_hz, interval=interval
    )
    return _write_markdown_table(
        rows,
        output_path,
        title=f"Table 1. In-distribution rollout quality @ {freq_hz:.3f} Hz",
    )


def build_table1_summary_rows(
    state: EvaluationState,
    table1_results: Mapping[str, list[dict]],
    *,
    freq_hz: float,
    interval: str = "ci95",
) -> list[dict[str, str]]:
    """Build export-ready Table 1 rows for in-distribution locomotion quality."""
    numeric_rows: list[dict[str, object]] = []
    vanilla_rps = metric_array(table1_results["vanilla"], "reward_per_step")
    vanilla_rps_mean = nanmean(vanilla_rps)
    for key in MODEL_KEYS:
        surv, rew = result_arrays(table1_results[key])
        rps = metric_array(table1_results[key], "reward_per_step")
        x_vel = metric_array(table1_results[key], "mean_x_velocity")
        numeric_rows.append(
            {
                "model_key": key,
                "Model": state.model_specs[key].label,
                "Phase condition": PHASE_CONDITION_LABELS[key],
                "Survival ↑": _summary_cell(surv, decimals=0, interval=interval),
                "Total reward ↑": _summary_cell(rew, decimals=1, interval=interval),
                "Reward / step ↑": _summary_cell(rps, decimals=3, interval=interval),
                "Mean x-velocity ↑": _summary_cell(
                    x_vel, decimals=3, interval=interval
                ),
                "Δ reward/step vs Vanilla ↑": _format_delta(
                    nanmean(rps) - vanilla_rps_mean, decimals=3
                ),
                "survival_mean": nanmean(surv),
                "reward_mean": nanmean(rew),
                "reward_per_step_mean": nanmean(rps),
                "x_vel_mean": nanmean(x_vel),
            }
        )

    best_by_col = {
        "Survival ↑": _best_model_key(numeric_rows, "survival_mean", higher=True),
        "Total reward ↑": _best_model_key(numeric_rows, "reward_mean", higher=True),
        "Reward / step ↑": _best_model_key(
            numeric_rows, "reward_per_step_mean", higher=True
        ),
        "Mean x-velocity ↑": _best_model_key(numeric_rows, "x_vel_mean", higher=True),
    }
    rows: list[dict[str, str]] = []
    for row in numeric_rows:
        formatted = {
            key: str(row[key])
            for key in (
                "Model",
                "Phase condition",
                "Survival ↑",
                "Total reward ↑",
                "Reward / step ↑",
                "Mean x-velocity ↑",
                "Δ reward/step vs Vanilla ↑",
            )
        }
        for col, best_key in best_by_col.items():
            if row["model_key"] == best_key:
                formatted[col] = f"**{formatted[col]}**"
        rows.append(formatted)
    return rows


def write_frequency_tracking_table_markdown(
    protocol: FrequencySweepProtocol,
    sweep_results: Mapping[str, Mapping[float, list[dict]]],
    output_path: str | Path,
    *,
    interval: str = "ci95",
) -> Path:
    """Export Table 2 as Markdown with tracking-first metrics.

    The metric order highlights the main contribution: full phase-trajectory
    conditioning should reduce command-frequency error and improve phase locking
    relative to conditioning only on the first phase.
    """
    output_path = Path(output_path)
    rows = build_frequency_tracking_rows(protocol, sweep_results, interval=interval)
    return _write_markdown_table(
        rows,
        output_path,
        title="Table 2. Frequency command tracking",
    )


def build_frequency_tracking_rows(
    protocol: FrequencySweepProtocol,
    sweep_results: Mapping[str, Mapping[float, list[dict]]],
    *,
    interval: str = "ci95",
) -> list[dict[str, str]]:
    """Build export-ready Table 2 rows for phase-conditioned frequency control."""
    rows: list[dict[str, str]] = []
    for freq, zone in zip(protocol.sweep_freqs, protocol.zone_labels):
        freq = float(freq)
        per_model: dict[str, dict[str, object]] = {}
        for model_key in PHASE_MODEL_KEYS:
            rollouts = sweep_results[model_key][freq]
            surv, _ = result_arrays(rollouts)
            per_model[model_key] = {
                "Target freq (Hz)": f"{freq:.3f}",
                "Zone": str(zone),
                "Model": SUMMARY_MODEL_LABELS[model_key],
                "Measured freq (Hz) ≈": _summary_cell(
                    metric_array(rollouts, "measured_freq_hz"),
                    decimals=3,
                    interval=interval,
                ),
                "|freq error| (Hz) ↓": _summary_cell(
                    metric_array(rollouts, "abs_freq_error_hz"),
                    decimals=3,
                    interval=interval,
                ),
                "Freq ratio ≈1": _summary_cell(
                    metric_array(rollouts, "freq_ratio"), decimals=3, interval=interval
                ),
                "PLV ↑": _summary_cell(
                    metric_array(rollouts, "phase_locking_value"),
                    decimals=3,
                    interval=interval,
                ),
                "Reward / step ↑": _summary_cell(
                    metric_array(rollouts, "reward_per_step"),
                    decimals=3,
                    interval=interval,
                ),
                "Survival ↑": _summary_cell(surv, decimals=0, interval=interval),
                "abs_freq_error_mean": nanmean(
                    metric_array(rollouts, "abs_freq_error_hz")
                ),
                "plv_mean": nanmean(metric_array(rollouts, "phase_locking_value")),
                "reward_per_step_mean": nanmean(
                    metric_array(rollouts, "reward_per_step")
                ),
                "survival_mean": nanmean(surv),
            }

        best_error = _best_model_key(
            [
                {"model_key": key, "value": per_model[key]["abs_freq_error_mean"]}
                for key in PHASE_MODEL_KEYS
            ],
            "value",
            higher=False,
        )
        best_plv = _best_model_key(
            [
                {"model_key": key, "value": per_model[key]["plv_mean"]}
                for key in PHASE_MODEL_KEYS
            ],
            "value",
            higher=True,
        )
        best_rps = _best_model_key(
            [
                {"model_key": key, "value": per_model[key]["reward_per_step_mean"]}
                for key in PHASE_MODEL_KEYS
            ],
            "value",
            higher=True,
        )
        best_surv = _best_model_key(
            [
                {"model_key": key, "value": per_model[key]["survival_mean"]}
                for key in PHASE_MODEL_KEYS
            ],
            "value",
            higher=True,
        )

        for model_key in PHASE_MODEL_KEYS:
            row = {
                key: str(per_model[model_key][key])
                for key in (
                    "Target freq (Hz)",
                    "Zone",
                    "Model",
                    "Measured freq (Hz) ≈",
                    "|freq error| (Hz) ↓",
                    "Freq ratio ≈1",
                    "PLV ↑",
                    "Reward / step ↑",
                    "Survival ↑",
                )
            }
            if model_key == best_error:
                row["|freq error| (Hz) ↓"] = f"**{row['|freq error| (Hz) ↓']}**"
            if model_key == best_plv:
                row["PLV ↑"] = f"**{row['PLV ↑']}**"
            if model_key == best_rps:
                row["Reward / step ↑"] = f"**{row['Reward / step ↑']}**"
            if model_key == best_surv:
                row["Survival ↑"] = f"**{row['Survival ↑']}**"
            rows.append(row)
    return rows


def _summary_cell(values: Sequence[object], *, decimals: int, interval: str) -> str:
    mean, spread, _ = _summary_stats(values, interval=interval)
    return f"{_format_float(mean, decimals)} ± {_format_float(spread, decimals)}"


def _summary_stats(
    values: Sequence[object], *, interval: str
) -> tuple[float, float, int]:
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


def _format_float(value: float, decimals: int) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.{decimals}f}"


def _format_delta(value: float, *, decimals: int) -> str:
    if not np.isfinite(value):
        return "n/a"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.{decimals}f}"


def _best_model_key(
    rows: Sequence[Mapping[str, object]], metric_key: str, *, higher: bool
) -> str | None:
    best_key: str | None = None
    best_value = -np.inf if higher else np.inf
    for row in rows:
        value = float(row[metric_key])
        if not np.isfinite(value):
            continue
        is_better = value > best_value if higher else value < best_value
        if is_better:
            best_value = value
            best_key = str(row["model_key"])
    return best_key


def _write_markdown_table(
    rows: Sequence[Mapping[str, str]], output_path: Path, *, title: str
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text(f"# {title}\n\n_No rows._\n", encoding="utf-8")
        return output_path
    headers = list(rows[0].keys())
    lines = [
        f"# {title}",
        "",
        "| " + " | ".join(_escape_markdown_cell(header) for header in headers) + " |",
    ]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(_escape_markdown_cell(str(row[h])) for h in headers)
            + " |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✓ {output_path}")
    return output_path


def _escape_markdown_cell(value: str) -> str:
    return value.replace("|", "\\|")


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


def nanmean(values: np.ndarray) -> float:
    """NaN-safe mean that returns NaN without emitting all-NaN warnings."""
    values = np.asarray(values, dtype=np.float32)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def nanstd(values: np.ndarray) -> float:
    """NaN-safe std that returns NaN without emitting all-NaN warnings."""
    values = np.asarray(values, dtype=np.float32)
    finite = values[np.isfinite(values)]
    return float(finite.std()) if finite.size else float("nan")


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
        "phase_joint_idx": np.asarray(_phase_joint_idx_from_data(data), dtype=np.int32),
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
