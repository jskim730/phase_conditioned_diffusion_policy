"""Official evaluation orchestration routines for notebook 05.

The evaluation notebook should only orchestrate these functions.  Model loading,
rollout protocol construction, frequency/phase sweeps, tabulation, and result
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
from phase import controllability_sweep_frequencies, frequency_zone, make_phase_trajectory_fn
from sampling import rollout_multi_seed
from training import load_checkpoint


EVAL_CONFIG_NAMES: tuple[str, ...] = ("vanilla", "periodic_phase", "phase_trajectory")
MODEL_KEYS: tuple[str, ...] = ("vanilla", "periodic", "trajectory")
PHASE_MODEL_KEYS: tuple[str, ...] = ("periodic", "trajectory")


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


@dataclass(frozen=True)
class PhaseSweepProtocol:
    """Phase-offset grid used to test whether phase information affects rollout."""

    freq_hz: float
    phase_offsets: np.ndarray
    phase_labels: np.ndarray


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
        "vanilla": _spec_from_loaded(loaded_models["vanilla"], label="Vanilla DP", uses_phase=False),
        "periodic": _spec_from_loaded(
            loaded_models["periodic_phase"], label="Periodic Phase", uses_phase=True
        ),
        "trajectory": _spec_from_loaded(
            loaded_models["phase_trajectory"], label="Trajectory (ours)", uses_phase=True
        ),
    }

    print("\n✓ Evaluation checkpoints loaded")
    for key in MODEL_KEYS:
        spec = model_specs[key]
        print(f"  {spec.label:<18s}: {count_params(spec.model)['trainable'] / 1e6:.2f}M trainable")

    return EvaluationState(
        configs=configs,
        loaded_models=loaded_models,
        model_specs=model_specs,
        noise_scheduler_config=base_cfg.noise_scheduler_config(),
        num_inference_steps=base_cfg.diffusion.num_inference_steps,
    )


def _spec_from_loaded(loaded: LoadedEvalModel, *, label: str, uses_phase: bool) -> ModelEvalSpec:
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
    """Run one model spec at one target frequency/phase-offset combination.

    This is the single-model evaluation helper used by both the official
    evaluation orchestration in this module and compatibility wrappers in
    training-oriented modules.  Keep the ``rollout_multi_seed`` argument
    assembly here so rollout protocol changes have one source of truth.
    """
    phase_fn = None
    if spec.uses_phase_trajectory:
        phase_fn = make_phase_trajectory_fn(freq_hz, dt=dt, phase0=phase0)

    return rollout_multi_seed(
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
    """Run one registered model at one target frequency/phase-offset combination."""
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


def build_frequency_sweep_protocol(data: Mapping[str, object], *, n_in_dist: int = 5) -> FrequencySweepProtocol:
    """Build the 5 in-distribution + 4 OOD frequency grid."""
    f_mean = float(data["freq_window_mean"])
    in_freqs = np.linspace(float(data["freq_window_min"]), float(data["freq_window_max"]), n_in_dist, dtype=np.float32)
    ood_freqs = (f_mean * np.array([0.50, 0.75, 1.25, 1.50], dtype=np.float32)).astype(np.float32)
    ood_freqs = ood_freqs[ood_freqs > 0.2]
    sweep_freqs = controllability_sweep_frequencies(data, n_in_dist=n_in_dist)
    zone_labels = np.array([frequency_zone(float(freq), data) for freq in sweep_freqs])
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
        print(f"\n{state.model_specs[model_key].label} sweep 시간: {(time.time() - t0) / 60:.1f} min\n")
        all_results[model_key] = model_results
    return all_results


def build_phase_sweep_protocol(data: Mapping[str, object]) -> PhaseSweepProtocol:
    """Build a fixed-frequency phase-offset grid for phase-information evaluation."""
    phase_offsets = np.array([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi], dtype=np.float32)
    phase_labels = np.array(["0", "pi/2", "pi", "3pi/2"])
    return PhaseSweepProtocol(
        freq_hz=float(data["freq_window_mean"]),
        phase_offsets=phase_offsets,
        phase_labels=phase_labels,
    )


def run_phase_sweep_evaluation(
    state: EvaluationState,
    protocol: PhaseSweepProtocol,
    *,
    env,
    data: Mapping[str, object],
    device: str,
    model_keys: Sequence[str] = PHASE_MODEL_KEYS,
    n_seeds: int,
    max_steps: int,
    dt: float = 0.05,
) -> dict[str, dict[float, list[dict]]]:
    """Run phase-offset sweeps to verify phase conditioning is not ignored."""
    all_results: dict[str, dict[float, list[dict]]] = {}
    for model_key in model_keys:
        print(f"=== {state.model_specs[model_key].label} phase-offset sweep ===")
        t0 = time.time()
        model_results: dict[float, list[dict]] = {}
        for phase0, label in zip(protocol.phase_offsets, protocol.phase_labels):
            phase0 = float(phase0)
            print(f"\n--- phase0={label} rad @ f={protocol.freq_hz:.3f} Hz ---")
            model_results[phase0] = evaluate_model_at_frequency(
                state,
                model_key,
                protocol.freq_hz,
                env=env,
                data=data,
                device=device,
                n_seeds=n_seeds,
                max_steps=max_steps,
                dt=dt,
                phase0=phase0,
            )
        print(f"\n{state.model_specs[model_key].label} phase sweep 시간: {(time.time() - t0) / 60:.1f} min\n")
        all_results[model_key] = model_results
    return all_results


def print_table1_summary(state: EvaluationState, table1_results: Mapping[str, list[dict]], *, freq_hz: float) -> None:
    """Print in-distribution performance and standard-error diagnostics."""
    n_seeds = len(next(iter(table1_results.values())))
    print(f"\n=== Table 1: In-distribution @ f={freq_hz:.3f} Hz, n={n_seeds} ===")
    print(f"{'Model':>20s} | {'Survival':>15s} | {'Reward':>17s}")
    print("-" * 60)
    for key in MODEL_KEYS:
        surv, rew = result_arrays(table1_results[key])
        print(
            f"{state.model_specs[key].label:>20s} | "
            f"{surv.mean():>5.0f} ± {surv.std():>4.0f}    | "
            f"{rew.mean():>7.1f} ± {rew.std():>5.1f}"
        )

    print("\n=== Standard Error (std/√n) ===")
    for key in MODEL_KEYS:
        _, rew = result_arrays(table1_results[key])
        se = rew.std() / np.sqrt(len(rew))
        print(f"{state.model_specs[key].label:>20s}: SE = {se:.1f}  ({se / rew.mean() * 100:.1f}% of mean)")


def print_frequency_sweep_summary(
    data: Mapping[str, object],
    protocol: FrequencySweepProtocol,
    sweep_results: Mapping[str, Mapping[float, list[dict]]],
) -> None:
    """Print Periodic-vs-Trajectory frequency controllability table."""
    print("=== Table 2: Frequency sweep ===")
    print(f"{'freq':>6s} | {'zone':>8s} | {'Periodic':>26s} | {'Trajectory':>26s} | Δreward")
    print(f"{'':6s} | {'':8s} | {'surv':>11s} {'reward':>13s} | {'surv':>11s} {'reward':>13s}")
    print("-" * 96)
    for freq in protocol.sweep_freqs:
        freq = float(freq)
        p_surv, p_rew = result_arrays(sweep_results["periodic"][freq])
        t_surv, t_rew = result_arrays(sweep_results["trajectory"][freq])
        print(
            f"{freq:>6.3f} | {frequency_zone(freq, data):>8s} | "
            f"{p_surv.mean():>5.0f}±{p_surv.std():>3.0f}  {p_rew.mean():>5.0f}±{p_rew.std():>4.0f} | "
            f"{t_surv.mean():>5.0f}±{t_surv.std():>3.0f}  {t_rew.mean():>5.0f}±{t_rew.std():>4.0f} | "
            f"{t_rew.mean() - p_rew.mean():+7.0f}"
        )


def print_phase_sweep_summary(
    protocol: PhaseSweepProtocol,
    phase_results: Mapping[str, Mapping[float, list[dict]]],
) -> None:
    """Print phase-offset robustness/sensitivity table."""
    print(f"=== Table 3: Phase-offset sweep @ f={protocol.freq_hz:.3f} Hz ===")
    print(f"{'phase0':>7s} | {'Periodic':>26s} | {'Trajectory':>26s} | Δreward")
    print(f"{'':7s} | {'surv':>11s} {'reward':>13s} | {'surv':>11s} {'reward':>13s}")
    print("-" * 84)
    for phase0, label in zip(protocol.phase_offsets, protocol.phase_labels):
        phase0 = float(phase0)
        p_surv, p_rew = result_arrays(phase_results["periodic"][phase0])
        t_surv, t_rew = result_arrays(phase_results["trajectory"][phase0])
        print(
            f"{label:>7s} | "
            f"{p_surv.mean():>5.0f}±{p_surv.std():>3.0f}  {p_rew.mean():>5.0f}±{p_rew.std():>4.0f} | "
            f"{t_surv.mean():>5.0f}±{t_surv.std():>3.0f}  {t_rew.mean():>5.0f}±{t_rew.std():>4.0f} | "
            f"{t_rew.mean() - p_rew.mean():+7.0f}"
        )


def result_arrays(results_list: Sequence[Mapping[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    """Return survival/reward arrays for one list of rollout dicts."""
    survival = np.array([r["survival"] for r in results_list], dtype=np.float32)
    reward = np.array([r["total_reward"] for r in results_list], dtype=np.float32)
    return survival, reward


def build_eval_results_payload(
    data: Mapping[str, object],
    table1_results: Mapping[str, list[dict]],
    freq_protocol: FrequencySweepProtocol,
    freq_results: Mapping[str, Mapping[float, list[dict]]],
    phase_protocol: PhaseSweepProtocol,
    phase_results: Mapping[str, Mapping[float, list[dict]]],
    *,
    n_seeds_indist: int,
    n_seeds_sweep: int,
    n_seeds_phase: int,
) -> dict[str, np.ndarray]:
    """Convert raw rollout dicts into serializable arrays for ``np.savez``."""
    payload: dict[str, np.ndarray] = {
        "n_seeds_indist": np.asarray(n_seeds_indist, dtype=np.int32),
        "n_seeds_sweep": np.asarray(n_seeds_sweep, dtype=np.int32),
        "n_seeds_phase": np.asarray(n_seeds_phase, dtype=np.int32),
        "f_mean": np.asarray(float(data["freq_window_mean"]), dtype=np.float32),
        "freq_window_min": np.asarray(float(data["freq_window_min"]), dtype=np.float32),
        "freq_window_max": np.asarray(float(data["freq_window_max"]), dtype=np.float32),
        "in_freqs": np.asarray(freq_protocol.in_freqs, dtype=np.float32),
        "ood_freqs": np.asarray(freq_protocol.ood_freqs, dtype=np.float32),
        "sweep_freqs": np.asarray(freq_protocol.sweep_freqs, dtype=np.float32),
        "sweep_zone_labels": np.asarray(freq_protocol.zone_labels),
        "phase_sweep_freq": np.asarray(phase_protocol.freq_hz, dtype=np.float32),
        "phase_offsets": np.asarray(phase_protocol.phase_offsets, dtype=np.float32),
        "phase_labels": np.asarray(phase_protocol.phase_labels),
    }

    for key in MODEL_KEYS:
        survival, reward = result_arrays(table1_results[key])
        payload[f"table1_{key}_survival"] = survival
        payload[f"table1_{key}_reward"] = reward

    _add_grid_results(payload, prefix="freq", grid=freq_protocol.sweep_freqs, results=freq_results)
    _add_grid_results(payload, prefix="phase", grid=phase_protocol.phase_offsets, results=phase_results)
    return payload


def _add_grid_results(
    payload: dict[str, np.ndarray],
    *,
    prefix: str,
    grid: Sequence[float],
    results: Mapping[str, Mapping[float, list[dict]]],
) -> None:
    for model_key in PHASE_MODEL_KEYS:
        survival_rows = []
        reward_rows = []
        for value in grid:
            survival, reward = result_arrays(results[model_key][float(value)])
            survival_rows.append(survival)
            reward_rows.append(reward)
        payload[f"{prefix}_{model_key}_survival"] = np.stack(survival_rows).astype(np.float32)
        payload[f"{prefix}_{model_key}_reward"] = np.stack(reward_rows).astype(np.float32)


def save_eval_results_npz(payload: Mapping[str, np.ndarray], output_path: str | Path) -> Path:
    """Persist evaluation arrays and print a reproducible manifest."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)
    print(f"✓ Saved: {output_path} ({os.path.getsize(output_path) / 1024:.1f} KB)")
    print(f"\nKeys: {list(payload.keys())}")
    return output_path
