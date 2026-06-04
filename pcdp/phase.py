<<<<<<< Updated upstream
"""Phase utilities for phase-conditioned diffusion policies.

This module owns both the **command side** (target phase trajectories the
policy is told to follow) and the **measured side** (phases extracted from
rollout observations and the metrics used to score command tracking).
Keeping them together avoids a parallel ``gait_metrics`` module that duplicates
the same Hilbert/frequency conventions.
"""

from __future__ import annotations

from typing import Callable, Mapping

import numpy as np

from .data_extraction import ANT_PHASE_JOINT_IDX, extract_phase

PhaseTrajectoryFn = Callable[[int, int], np.ndarray]


# =====================================================================
# Command side: target phase trajectory generation
# =====================================================================

def make_phase_trajectory_fn(
    freq_hz: float, *, dt: float = 0.05, phase0: float = 0.0
) -> PhaseTrajectoryFn:
    """Return ``fn(start_step, horizon)`` that emits an unwrapped phase trajectory."""
    freq_hz = float(freq_hz)
    dt = float(dt)
    phase0 = float(phase0)

    def phase_trajectory(start_step: int, horizon: int) -> np.ndarray:
        steps = np.arange(
            int(start_step), int(start_step) + int(horizon), dtype=np.float32
        )
        return (phase0 + 2.0 * np.pi * freq_hz * steps * dt).astype(np.float32)

    return phase_trajectory


def command_phase_trajectory(
    length: int, *, freq_hz: float, dt: float, phase0: float = 0.0
) -> np.ndarray:
    """Build the command phase trajectory aligned to rollout samples.

    Thin wrapper over :func:`make_phase_trajectory_fn` for the common
    "start at step 0" case used by tracking-metric computation.
    """
    return make_phase_trajectory_fn(freq_hz, dt=dt, phase0=phase0)(0, length)


# =====================================================================
# Frequency-grid helpers shared by training and evaluation notebooks
# =====================================================================

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


def controllability_sweep_frequency_groups(
    data: dict,
    *,
    ood_iqr_scale: float = 1.5,
    min_freq_hz: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the 3 in-distribution + 2 IQR-based OOD Table 2 groups.

    The in-distribution commands are the three interior points of a five-point
    training-support grid, which avoids evaluating exactly at the observed
    min/max boundaries.  The OOD commands are Tukey-style outer fences around
    those interior commands: ``q25 - 1.5 * IQR`` and ``q75 + 1.5 * IQR`` by
    default.
    """
    _, f_min, f_max = training_frequency_triplet(data)
    support_grid = np.linspace(f_min, f_max, 5, dtype=np.float32)
    in_freqs = support_grid[1:-1].astype(np.float32)

    q25, _, q75 = [float(freq) for freq in in_freqs]
    iqr = q75 - q25
    ood_freqs = np.array(
        [
            max(float(min_freq_hz), q25 - float(ood_iqr_scale) * iqr),
            q75 + float(ood_iqr_scale) * iqr,
        ],
        dtype=np.float32,
    )
    return in_freqs, ood_freqs


def controllability_sweep_frequencies(
    data: dict, *, n_in_dist: int = 3, ood_iqr_scale: float = 1.5
) -> np.ndarray:
    """Return the 5-frequency Table 2 sweep: OOD-low + 3 in-dist + OOD-high."""
    if n_in_dist != 3:
        raise ValueError(
            "Table 2 uses exactly three interior in-distribution frequencies."
        )
    in_freqs, ood_freqs = controllability_sweep_frequency_groups(
        data, ood_iqr_scale=ood_iqr_scale
    )
    return np.concatenate([ood_freqs[:1], in_freqs, ood_freqs[1:]]).astype(np.float32)


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


# =====================================================================
# Measured side: rollout phase extraction and command-tracking metrics
# =====================================================================

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
    phase_joint_idx: int = ANT_PHASE_JOINT_IDX,
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
    phase_joint_idx: int = ANT_PHASE_JOINT_IDX,
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
=======
"""Phase utilities for phase-conditioned diffusion policies.

This module owns both the **command side** (target phase trajectories the
policy is told to follow) and the **measured side** (phases extracted from
rollout observations and the metrics used to score command tracking).
Keeping them together avoids a parallel ``gait_metrics`` module that duplicates
the same Hilbert/frequency conventions.
"""

from __future__ import annotations

from typing import Callable, Mapping

import numpy as np

from .data_extraction import ANT_PHASE_JOINT_IDX, extract_phase

PhaseTrajectoryFn = Callable[[int, int], np.ndarray]


# =====================================================================
# Command side: target phase trajectory generation
# =====================================================================

def make_phase_trajectory_fn(
    freq_hz: float,
    *,
    dt: float = 0.05,
    phase0: float = 0.0,
    ramp_steps: int = 0,
) -> PhaseTrajectoryFn:
    """Return ``fn(start_step, horizon)`` that emits an unwrapped phase trajectory.

    Args:
        freq_hz: Target gait frequency in Hz.
        dt: Simulation timestep in seconds.
        phase0: Initial phase offset in radians at ``start_step = 0``.
        ramp_steps: If > 0, the command frequency linearly ramps from 0 to
            ``freq_hz`` over the first ``ramp_steps`` steps.  This prevents a
            sudden phase-velocity jump at rollout start when the body is
            initially stationary.

            Phase integral during ramp (t < ramp_steps):
                φ(t) = phase0 + 2π · freq_hz · dt · t² / (2 · ramp_steps)

            Phase after ramp (t ≥ ramp_steps):
                φ(t) = φ_ramp_end + 2π · freq_hz · dt · (t − ramp_steps)
    """
    freq_hz = float(freq_hz)
    dt = float(dt)
    phase0 = float(phase0)
    ramp_steps = int(ramp_steps)
    # Phase value at the end of the ramp (used for continuity)
    phase_ramp_end = phase0 + 2.0 * np.pi * freq_hz * dt * ramp_steps / 2.0

    def phase_trajectory(start_step: int, horizon: int) -> np.ndarray:
        steps = np.arange(
            int(start_step), int(start_step) + int(horizon), dtype=np.float64
        )
        if ramp_steps <= 0:
            return (phase0 + 2.0 * np.pi * freq_hz * steps * dt).astype(np.float32)

        phase = np.where(
            steps < ramp_steps,
            phase0 + 2.0 * np.pi * freq_hz * dt * steps ** 2 / (2.0 * ramp_steps),
            phase_ramp_end + 2.0 * np.pi * freq_hz * dt * (steps - ramp_steps),
        )
        return phase.astype(np.float32)

    return phase_trajectory


def command_phase_trajectory(
    length: int, *, freq_hz: float, dt: float, phase0: float = 0.0, ramp_steps: int = 0
) -> np.ndarray:
    """Build the command phase trajectory aligned to rollout samples.

    Thin wrapper over :func:`make_phase_trajectory_fn` for the common
    "start at step 0" case used by tracking-metric computation.
    """
    return make_phase_trajectory_fn(freq_hz, dt=dt, phase0=phase0, ramp_steps=ramp_steps)(0, length)


class PhaseContinuityTracker:
    """Stateful tracker for seamless phase continuity across rollout chunks.

    Solves three problems in the default ``make_phase_trajectory_fn`` usage:

    1. **Phase initialisation**: the ant body has a random initial phase at
       ``env.reset()``.  Calling :meth:`initialize_from_obs` aligns the command
       phase to the measured body phase before the rollout starts.

    2. **Frequency ramping**: pass ``ramp_steps > 0`` to avoid a hard step-change
       in commanded phase velocity at t=0.

    3. **Chunk-to-chunk continuity**: the tracker is a callable
       ``(start_step, horizon) → phase_array``, so it can be passed directly as
       ``phase_trajectory_fn`` to :func:`~sampling.rollout`.  Because it delegates
       to the same underlying :func:`make_phase_trajectory_fn` with a fixed
       ``phase0``, the resulting trajectory is automatically continuous across
       receding-horizon re-plan boundaries.

    Example::

        tracker = PhaseContinuityTracker(freq_hz=1.5, dt=0.05, ramp_steps=20)
        tracker.initialize_from_obs(obs_window)
        result = rollout(model, ema, env, ..., phase_trajectory_fn=tracker)
    """

    def __init__(
        self,
        freq_hz: float,
        *,
        dt: float = 0.05,
        ramp_steps: int = 0,
        phase0: float = 0.0,
    ) -> None:
        self.freq_hz = float(freq_hz)
        self.dt = float(dt)
        self.ramp_steps = int(ramp_steps)
        self.phase0 = float(phase0)
        self._traj_fn = make_phase_trajectory_fn(
            self.freq_hz, dt=self.dt, phase0=self.phase0, ramp_steps=self.ramp_steps
        )

    def initialize_from_obs(
        self,
        obs_window: np.ndarray,
        *,
        phase_joint_idx: int = ANT_PHASE_JOINT_IDX,
        smooth_sigma: float = 2.0,
    ) -> float:
        """Align command phase to the estimated body phase from an obs window.

        Args:
            obs_window: Array of shape ``(obs_horizon, obs_dim)`` or
                ``(obs_dim,)`` for a single step.

        Returns:
            Estimated ``phase0`` in radians (also stored as ``self.phase0``).
        """
        obs_window = np.asarray(obs_window)
        joint_signal = (
            obs_window[:, phase_joint_idx] if obs_window.ndim == 2
            else obs_window[np.newaxis, phase_joint_idx]
        )
        if joint_signal.size < 3 or float(np.std(joint_signal)) < 1e-6:
            return self.phase0

        phases = extract_phase(joint_signal, smooth_sigma=smooth_sigma)
        self.phase0 = float(phases[-1])
        self._traj_fn = make_phase_trajectory_fn(
            self.freq_hz, dt=self.dt, phase0=self.phase0, ramp_steps=self.ramp_steps
        )
        return self.phase0

    def __call__(self, start_step: int, horizon: int) -> np.ndarray:
        """Return the phase trajectory for ``horizon`` steps starting at ``start_step``."""
        return self._traj_fn(start_step, horizon)


# =====================================================================
# Frequency-grid helpers shared by training and evaluation notebooks
# =====================================================================

def training_frequency_triplet(data: dict) -> tuple[float, float, float]:
    """Return ``(mean, min, max)`` frequency values from loaded project data."""
    return (
        float(data["freq_window_mean"]),
        float(data["freq_window_min"]),
        float(data["freq_window_max"]),
    )



def controllability_sweep_frequency_groups(
    data: dict,
    *,
    ood_iqr_scale: float = 1.5,
    min_freq_hz: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the 3 in-distribution + 2 IQR-based OOD Table 2 groups.

    The in-distribution commands are the three interior points of a five-point
    training-support grid, which avoids evaluating exactly at the observed
    min/max boundaries.  The OOD commands are Tukey-style outer fences around
    those interior commands: ``q25 - 1.5 * IQR`` and ``q75 + 1.5 * IQR`` by
    default.
    """
    _, f_min, f_max = training_frequency_triplet(data)
    support_grid = np.linspace(f_min, f_max, 5, dtype=np.float32)
    in_freqs = support_grid[1:-1].astype(np.float32)

    q25, _, q75 = [float(freq) for freq in in_freqs]
    iqr = q75 - q25
    ood_freqs = np.array(
        [
            max(float(min_freq_hz), q25 - float(ood_iqr_scale) * iqr),
            q75 + float(ood_iqr_scale) * iqr,
        ],
        dtype=np.float32,
    )
    return in_freqs, ood_freqs


def controllability_sweep_frequencies(
    data: dict, *, n_in_dist: int = 3, ood_iqr_scale: float = 1.5
) -> np.ndarray:
    """Return the 5-frequency Table 2 sweep: OOD-low + 3 in-dist + OOD-high."""
    if n_in_dist != 3:
        raise ValueError(
            "Table 2 uses exactly three interior in-distribution frequencies."
        )
    in_freqs, ood_freqs = controllability_sweep_frequency_groups(
        data, ood_iqr_scale=ood_iqr_scale
    )
    return np.concatenate([ood_freqs[:1], in_freqs, ood_freqs[1:]]).astype(np.float32)


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


# =====================================================================
# Measured side: rollout phase extraction and command-tracking metrics
# =====================================================================

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
    phase_joint_idx: int = ANT_PHASE_JOINT_IDX,
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
    phase_joint_idx: int = ANT_PHASE_JOINT_IDX,
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
>>>>>>> Stashed changes
