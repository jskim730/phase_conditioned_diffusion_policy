"""Phase trajectory and frequency-grid utilities for phase-conditioned policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


PhaseTrajectoryFn = Callable[[int, int], np.ndarray]


def make_phase_trajectory_fn(freq_hz: float, *, dt: float = 0.05, phase0: float = 0.0) -> PhaseTrajectoryFn:
    """Return ``fn(start_step, horizon)`` that emits an unwrapped phase trajectory."""
    freq_hz = float(freq_hz)
    dt = float(dt)
    phase0 = float(phase0)

    def phase_trajectory(start_step: int, horizon: int) -> np.ndarray:
        steps = np.arange(int(start_step), int(start_step) + int(horizon), dtype=np.float32)
        return (phase0 + 2.0 * np.pi * freq_hz * steps * dt).astype(np.float32)

    return phase_trajectory


def training_frequency_triplet(data: dict) -> tuple[float, float, float]:
    """Return ``(mean, min, max)`` frequency values from loaded project data."""
    return (
        float(data["freq_window_mean"]),
        float(data["freq_window_min"]),
        float(data["freq_window_max"]),
    )


def periodic_offline_frequencies(data: dict) -> tuple[list[float], list[str]]:
    """Frequency list used for periodic-conditioning offline sample checks."""
    f_mean, f_min, f_max = training_frequency_triplet(data)
    f_std = float(data["freq_window_std"])
    return (
        [f_mean, f_min, f_max, f_mean - 3.0 * f_std, f_mean + 3.0 * f_std],
        ["mean", "min", "max", "OOD-low", "OOD-high"],
    )


def trajectory_offline_frequencies(data: dict) -> tuple[list[float], list[str]]:
    """Frequency list used for trajectory-conditioning offline sample checks."""
    freqs, labels = periodic_offline_frequencies(data)
    f_mean = float(data["freq_window_mean"])
    f_std = float(data["freq_window_std"])
    return (
        [*freqs, f_mean - 6.0 * f_std, f_mean + 6.0 * f_std],
        [*labels, "far-OOD-low", "far-OOD-high"],
    )


def controllability_sweep_frequencies(data: dict, *, n_in_dist: int = 5) -> np.ndarray:
    """Return the 9-frequency in-distribution/OOD sweep used by Step 4."""
    f_mean, f_min, f_max = training_frequency_triplet(data)
    in_freqs = np.linspace(f_min, f_max, n_in_dist, dtype=np.float32)
    ood_freqs = (f_mean * np.array([0.50, 0.75, 1.25, 1.50], dtype=np.float32)).astype(np.float32)
    ood_freqs = ood_freqs[ood_freqs > 0.2]
    sweep = np.concatenate([ood_freqs[:2], in_freqs, ood_freqs[2:]]).astype(np.float32)
    if len(sweep) != n_in_dist + 4:
        raise ValueError(f"Expected {n_in_dist + 4} sweep freqs, got {len(sweep)}: {sweep}")
    return sweep


def legacy_periodic_sweep_frequencies(data: dict) -> np.ndarray:
    """Return the original five-frequency sweep used in Step 3."""
    f_mean, f_min, f_max = training_frequency_triplet(data)
    f_std = float(data["freq_window_std"])
    return np.array(
        [f_mean - 3.0 * f_std, f_min, f_mean, f_max, f_mean + 3.0 * f_std],
        dtype=np.float32,
    )


def frequency_zone(freq_hz: float, data: dict, *, atol: float = 1e-6) -> str:
    """Classify a target frequency relative to the training frequency support.

    The sweep grid is usually stored as ``float32`` for compact artifacts, so
    boundary values can round a few ULPs outside the ``float64`` dataset stats.
    A small absolute tolerance keeps exact min/max grid points in-distribution.
    """
    _, f_min, f_max = training_frequency_triplet(data)
    freq = float(freq_hz)
    if (f_min - atol) <= freq <= (f_max + atol):
        return "in-dist"
    return "OOD-low" if freq < f_min else "OOD-high"


@dataclass(frozen=True)
class SweepSummary:
    freqs: list[float]
    survival_mean: list[float]
    survival_std: list[float]
    reward_mean: list[float]
    reward_std: list[float]


def summarize_sweep_results(sweep_results: dict[float, list[dict]]) -> SweepSummary:
    """Convert rollout results by frequency into plotting/table arrays."""
    freqs = sorted(float(f) for f in sweep_results.keys())
    return SweepSummary(
        freqs=freqs,
        survival_mean=[float(np.mean([r["survival"] for r in sweep_results[f]])) for f in freqs],
        survival_std=[float(np.std([r["survival"] for r in sweep_results[f]])) for f in freqs],
        reward_mean=[float(np.mean([r["total_reward"] for r in sweep_results[f]])) for f in freqs],
        reward_std=[float(np.std([r["total_reward"] for r in sweep_results[f]])) for f in freqs],
    )
