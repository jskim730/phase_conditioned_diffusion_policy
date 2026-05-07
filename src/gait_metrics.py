"""Gait-quality and command-tracking metrics for Ant rollouts.

These helpers intentionally reuse the same Hilbert-transform phase extraction
used during expert-demo construction, so evaluation-time measured phase/frequency
has the same convention as the training labels.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from data_extraction import extract_phase


DEFAULT_PHASE_JOINT_IDX = 19


def circular_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return wrapped circular difference ``a - b`` in ``[-pi, pi]``."""
    return np.angle(np.exp(1j * (np.asarray(a) - np.asarray(b))))


def circular_mean_angle(angles: np.ndarray) -> float:
    """Return the circular mean angle, or NaN for empty/all-invalid inputs."""
    angles = np.asarray(angles, dtype=np.float64)
    angles = angles[np.isfinite(angles)]
    if angles.size == 0:
        return float("nan")
    return float(np.angle(np.mean(np.exp(1j * angles))))


def extract_rollout_phase(
    obs_log: np.ndarray,
    *,
    phase_joint_idx: int = DEFAULT_PHASE_JOINT_IDX,
    smooth_sigma: float = 2.0,
    min_steps: int = 20,
) -> tuple[np.ndarray, bool, str]:
    """Extract motion phase from a rollout observation log.

    The phase signal is the same observation dimension used by
    ``data_extraction.extract_demos_from_episodes`` for the training RL dataset.
    """
    obs_log = np.asarray(obs_log)
    if obs_log.ndim != 2:
        return np.asarray([], dtype=np.float32), False, "obs_log is not a 2D array"
    if obs_log.shape[0] < min_steps:
        return np.asarray([], dtype=np.float32), False, f"too few steps ({obs_log.shape[0]} < {min_steps})"
    if phase_joint_idx >= obs_log.shape[1]:
        return np.asarray([], dtype=np.float32), False, (
            f"phase_joint_idx {phase_joint_idx} outside obs dim {obs_log.shape[1]}"
        )

    joint_signal = obs_log[:, phase_joint_idx]
    if not np.all(np.isfinite(joint_signal)):
        return np.asarray([], dtype=np.float32), False, "phase joint signal contains non-finite values"
    if float(np.std(joint_signal)) < 1e-6:
        return np.asarray([], dtype=np.float32), False, "phase joint signal is nearly constant"

    phases = extract_phase(joint_signal, smooth_sigma=smooth_sigma)
    return phases.astype(np.float32), True, "ok"


def estimate_frequency_from_phase(phases: np.ndarray, dt: float) -> float:
    """Estimate frequency in Hz from the slope of an unwrapped phase trajectory."""
    phases = np.asarray(phases, dtype=np.float64)
    if phases.size < 2 or dt <= 0:
        return float("nan")
    unwrapped = np.unwrap(phases)
    t = np.arange(phases.size, dtype=np.float64) * float(dt)
    if np.ptp(t) <= 0:
        return float("nan")
    slope = np.polyfit(t, unwrapped, deg=1)[0]
    return float(slope / (2.0 * np.pi))


def command_phase_trajectory(length: int, *, freq_hz: float, dt: float, phase0: float = 0.0) -> np.ndarray:
    """Build the command phase trajectory aligned to rollout samples."""
    steps = np.arange(int(length), dtype=np.float64)
    return (float(phase0) + 2.0 * np.pi * float(freq_hz) * steps * float(dt)).astype(np.float32)


def compute_phase_tracking_metrics(
    *,
    measured_phase: np.ndarray,
    command_phase: np.ndarray,
    command_freq_hz: float,
    dt: float,
) -> dict[str, float]:
    """Compute frequency/phase-command tracking metrics for one rollout."""
    n = min(len(measured_phase), len(command_phase))
    if n < 2:
        return {
            "measured_freq_hz": float("nan"),
            "freq_error_hz": float("nan"),
            "abs_freq_error_hz": float("nan"),
            "freq_ratio": float("nan"),
            "mean_abs_phase_error": float("nan"),
            "phase_locking_value": float("nan"),
            "phase_offset_error": float("nan"),
            "abs_phase_offset_error": float("nan"),
        }

    measured_phase = np.asarray(measured_phase[:n], dtype=np.float64)
    command_phase = np.asarray(command_phase[:n], dtype=np.float64)
    phase_error = circular_difference(measured_phase, command_phase)
    offset_error = circular_mean_angle(phase_error)
    measured_freq = estimate_frequency_from_phase(measured_phase, dt=dt)
    freq_error = measured_freq - float(command_freq_hz) if np.isfinite(measured_freq) else float("nan")
    ratio = measured_freq / float(command_freq_hz) if command_freq_hz > 0 and np.isfinite(measured_freq) else float("nan")

    return {
        "measured_freq_hz": float(measured_freq),
        "freq_error_hz": float(freq_error),
        "abs_freq_error_hz": float(abs(freq_error)) if np.isfinite(freq_error) else float("nan"),
        "freq_ratio": float(ratio),
        "mean_abs_phase_error": float(np.mean(np.abs(phase_error))),
        "phase_locking_value": float(abs(np.mean(np.exp(1j * phase_error)))),
        "phase_offset_error": float(offset_error),
        "abs_phase_offset_error": float(abs(offset_error)) if np.isfinite(offset_error) else float("nan"),
    }


def annotate_rollout_with_gait_metrics(
    result: Mapping[str, object],
    *,
    command_freq_hz: float,
    dt: float,
    phase0: float = 0.0,
    phase_joint_idx: int = DEFAULT_PHASE_JOINT_IDX,
    smooth_sigma: float = 2.0,
    min_duration_s: float = 2.0,
) -> dict:
    """Return a rollout dict augmented with motion-derived tracking metrics."""
    out = dict(result)
    survival = int(out.get("survival", 0))
    total_reward = float(out.get("total_reward", np.nan))
    out["reward_per_step"] = total_reward / max(survival, 1)
    out["command_freq_hz"] = float(command_freq_hz)
    out["command_phase0"] = float(phase0)

    min_steps = max(20, int(np.ceil(min_duration_s / float(dt))))
    measured_phase, valid, reason = extract_rollout_phase(
        np.asarray(out.get("obs_log", [])),
        phase_joint_idx=phase_joint_idx,
        smooth_sigma=smooth_sigma,
        min_steps=min_steps,
    )
    out["measured_phase"] = measured_phase
    out["phase_valid"] = bool(valid)
    out["phase_extraction_reason"] = reason

    if valid:
        command_phase = command_phase_trajectory(
            len(measured_phase), freq_hz=command_freq_hz, dt=dt, phase0=phase0
        )
        out["command_phase"] = command_phase
        out.update(
            compute_phase_tracking_metrics(
                measured_phase=measured_phase,
                command_phase=command_phase,
                command_freq_hz=command_freq_hz,
                dt=dt,
            )
        )
    else:
        out["command_phase"] = np.asarray([], dtype=np.float32)
        out.update(
            compute_phase_tracking_metrics(
                measured_phase=np.asarray([], dtype=np.float32),
                command_phase=np.asarray([], dtype=np.float32),
                command_freq_hz=command_freq_hz,
                dt=dt,
            )
        )
    return out
