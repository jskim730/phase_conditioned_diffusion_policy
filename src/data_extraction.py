"""Demo extraction utilities for Minari/D4RL Ant datasets.

The 00 data-extraction notebook is intentionally thin: it configures a run,
then delegates dataset discovery, phase labeling, quality filtering, artifact
saving, and diagnostic plotting to this module.  Keeping this logic in ``src``
makes the preprocessing choices reusable and version-controlled.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert


DEFAULT_MINARI_ANT_CANDIDATES: tuple[str, ...] = (
    "mujoco/ant/medium-replay-v0",
    "mujoco/ant/medium-replay-v1",
    "mujoco/ant/medium-expert-v0",
    "mujoco/ant/expert-v0",
    "D4RL/ant/medium-replay-v2",
    "D4RL/ant/medium-replay-v0",
)


@dataclass(frozen=True)
class DemoExtractionConfig:
    """Configuration for Plan C Ant demo extraction."""

    episode_length: int = 200
    min_episode_length: int = 200
    phase_joint_idx: int = 19
    expected_obs_dim: int = 105
    smooth_sigma: float = 2.0
    dt: float = 0.05
    min_monotonicity: float = 0.90
    min_peak_sharpness: float = 8.0
    max_freq_stability: float = 0.50
    max_demos: int = 120
    max_episodes_to_process: int = 5000


def list_ant_remote_datasets(minari_module: Any, max_fallback: int = 30) -> list[str]:
    """Print and return remote Minari datasets whose name contains ``ant``."""
    print("=== Minari에서 사용 가능한 Ant dataset ===")
    remote = minari_module.list_remote_datasets()
    ant_datasets = [name for name in remote.keys() if "ant" in name.lower()]

    if not ant_datasets:
        print("⚠ Ant dataset 없음. D4RL 직접 시도 필요.")
        print("\n전체 dataset (참고):")
        for name in list(remote.keys())[:max_fallback]:
            print(f"  {name}")
    else:
        for name in ant_datasets:
            info = remote[name]
            print(f"  {name}")
            if hasattr(info, "description"):
                print(f"    {str(info.description)[:80]}")
            if hasattr(info, "total_episodes"):
                print(f"    episodes: {info.total_episodes}")
    return ant_datasets


def load_first_available_minari_dataset(
    minari_module: Any,
    candidates: Sequence[str] = DEFAULT_MINARI_ANT_CANDIDATES,
    download: bool = True,
) -> tuple[Any, str | None]:
    """Try candidate Minari dataset names in order and return the first success."""
    dataset = None
    used_name = None
    for name in candidates:
        try:
            print(f"시도: {name}")
            dataset = minari_module.load_dataset(name, download=download)
            used_name = name
            print("  ✓ 성공!")
            break
        except Exception as exc:  # external dataset availability differs by Minari version
            print(f"  ✗ 실패: {str(exc)[:80]}")

    if dataset is None:
        print("\n⚠ 자동 후보 모두 실패. 위 목록의 정확한 이름으로 수동 시도 필요.")
        return None, None

    print(f"\n=== Loaded: {used_name} ===")
    print(f"Total episodes: {dataset.total_episodes}")
    print(f"Total steps: {dataset.total_steps}")
    print(f"Observation space: {dataset.observation_space}")
    print(f"Action space: {dataset.action_space}")
    return dataset, used_name


def materialize_episodes(dataset: Any, expected_obs_dim: int = 105, sample_size: int = 100) -> list[Any]:
    """Load Minari's episode iterator into a list and print an episode-length summary."""
    episodes = list(dataset.iterate_episodes())
    print(f"Total episodes loaded: {len(episodes)}")

    ep0 = episodes[0]
    actual_obs_dim = ep0.observations.shape[-1]
    if actual_obs_dim < expected_obs_dim:
        print(f"⚠ Dataset이 예상보다 적은 관측 feature를 제공합니다. {actual_obs_dim}개 feature만 사용합니다.")
    elif actual_obs_dim > expected_obs_dim:
        print(f"⚠ Dataset이 예상보다 많은 관측 feature를 제공합니다. 첫 {expected_obs_dim}개 feature만 사용합니다.")

    lengths = [len(ep.actions) for ep in episodes[:sample_size]]
    print(f"Episode length sample: min={min(lengths)}, mean={np.mean(lengths):.0f}, max={max(lengths)}")
    return episodes


def extract_phase(joint_signal: np.ndarray, smooth_sigma: float = 2.0) -> np.ndarray:
    """Extract instantaneous phase from a 1D joint signal using a Hilbert transform."""
    signal = np.asarray(joint_signal, dtype=np.float32) - np.mean(joint_signal)
    if smooth_sigma > 0:
        signal = gaussian_filter1d(signal, sigma=smooth_sigma)
    phases = np.angle(hilbert(signal))
    return ((phases + 2 * np.pi) % (2 * np.pi)).astype(np.float32)


def measure_phase_quality(joint_signal: np.ndarray, phases: np.ndarray, dt: float = 0.05) -> dict[str, float]:
    """Measure monotonicity, spectral sharpness, estimated frequency, and stability."""
    length = len(phases)
    phases_unwrapped = np.unwrap(phases)
    diffs = np.diff(phases_unwrapped)
    monotonicity = float((diffs > 0).mean()) if len(diffs) else 0.0

    signal_centered = np.asarray(joint_signal) - np.mean(joint_signal)
    fft = np.abs(np.fft.rfft(signal_centered))
    if len(fft) > 1:
        peak_val = fft[1:].max()
        mean_val = fft[1:].mean()
        peak_sharpness = float(peak_val / (mean_val + 1e-8))
        peak_idx = fft[1:].argmax() + 1
        estimated_freq = float(peak_idx / (length * dt))
    else:
        peak_sharpness = 0.0
        estimated_freq = 0.0

    if len(diffs):
        inst_freq = diffs / (2 * np.pi * dt)
        freq_stability = float(np.std(inst_freq))
    else:
        freq_stability = float("inf")

    return {
        "monotonicity": monotonicity,
        "peak_sharpness": peak_sharpness,
        "estimated_freq": estimated_freq,
        "freq_stability": freq_stability,
    }


def compare_phase_joint_candidates(
    episodes: Sequence[Any],
    hip_candidates: Sequence[int] = (13, 15, 17, 19),
    min_length: int = 200,
    max_sample_episodes: int = 3,
    smooth_sigma: float = 2.0,
    dt: float = 0.05,
) -> dict[int, list[dict[str, float]]]:
    """Print phase-quality diagnostics for candidate Ant hip-joint observation indices."""
    sample_eps = [ep for ep in episodes[:20] if len(ep.actions) >= min_length][:max_sample_episodes]
    print(f"Sample episodes (length >= {min_length}): {len(sample_eps)}")
    print("\n=== 각 sample episode + 각 hip joint별 quality ===")
    quality_summary: dict[int, list[dict[str, float]]] = {idx: [] for idx in hip_candidates}

    for ep_i, ep in enumerate(sample_eps):
        obs_seq = np.asarray(ep.observations)[: len(ep.actions)]
        print(f"\nEpisode {ep_i} (length {len(obs_seq)}):")
        for joint_idx in hip_candidates:
            if joint_idx >= obs_seq.shape[1]:
                continue
            sig = obs_seq[:, joint_idx]
            ph = extract_phase(sig, smooth_sigma=smooth_sigma)
            q = measure_phase_quality(sig, ph, dt=dt)
            quality_summary[joint_idx].append(q)
            print(
                f"  joint {joint_idx}: mono={q['monotonicity']:.2f}, "
                f"sharp={q['peak_sharpness']:.2f}, "
                f"freq={q['estimated_freq']:.2f}Hz, "
                f"stab={q['freq_stability']:.2f}"
            )

    print("\n=== Hip joint별 평균 quality ===")
    for joint_idx, qualities in quality_summary.items():
        if not qualities:
            continue
        mean_mono = np.mean([q["monotonicity"] for q in qualities])
        mean_sharp = np.mean([q["peak_sharpness"] for q in qualities])
        mean_stab = np.mean([q["freq_stability"] for q in qualities])
        print(f"  joint {joint_idx}: mono={mean_mono:.2f}, sharp={mean_sharp:.2f}, stab={mean_stab:.2f}")
    return quality_summary


def quality_score(entry: dict[str, Any]) -> float:
    """Rank extracted episodes after filtering."""
    quality = entry["quality"]
    return quality["peak_sharpness"] + 5.0 * quality["monotonicity"] - quality["freq_stability"]


def extract_demos_from_episodes(episodes: Iterable[Any], config: DemoExtractionConfig) -> dict[str, np.ndarray] | None:
    """Extract phase-coherent Ant demonstrations from materialized Minari episodes."""
    print(f"Phase 1: episodes 처리 (max {config.max_episodes_to_process})")
    enriched: list[dict[str, Any]] = []
    n_processed = 0
    n_too_short = 0

    for ep_i, ep in enumerate(episodes):
        if ep_i >= config.max_episodes_to_process:
            break

        length = min(len(ep.actions), config.episode_length)
        if length < config.min_episode_length:
            n_too_short += 1
            continue

        obs_seq = np.asarray(ep.observations)[:length, : config.expected_obs_dim]
        act_seq = np.asarray(ep.actions)[:length]
        joint_signal = obs_seq[:, config.phase_joint_idx]
        phases = extract_phase(joint_signal, smooth_sigma=config.smooth_sigma)
        quality = measure_phase_quality(joint_signal, phases, dt=config.dt)

        enriched.append(
            {
                "obs": obs_seq,
                "act": act_seq,
                "phases": phases,
                "length": length,
                "reward_sum": float(np.sum(ep.rewards[:length])),
                "quality": quality,
            }
        )
        n_processed += 1

        if (n_processed + n_too_short) % 500 == 0:
            print(f"  처리됨: {n_processed} (too short: {n_too_short})")

    print(f"\n총 처리: {n_processed} episodes (too short: {n_too_short})")
    if n_processed == 0:
        return None

    _print_quality_distribution(enriched, n_processed)

    print("\nPhase 3: 주기성 필터")
    print(
        f"  mono >= {config.min_monotonicity}, sharp >= {config.min_peak_sharpness}, "
        f"stab <= {config.max_freq_stability}"
    )
    passed = [
        e
        for e in enriched
        if e["quality"]["monotonicity"] >= config.min_monotonicity
        and e["quality"]["peak_sharpness"] >= config.min_peak_sharpness
        and e["quality"]["freq_stability"] <= config.max_freq_stability
    ]
    print(f"  통과: {len(passed)}/{len(enriched)} ({len(passed) / len(enriched) * 100:.1f}%)")

    if not passed:
        print("  → 통과 0개. 상위 10% fallback 적용.")
        scored = sorted(enriched, key=quality_score, reverse=True)
        passed = scored[: max(20, len(scored) // 10)]
        print(f"  → {len(passed)} 선택 (fallback)")

    print(f"\nPhase 4: Quality 기준 top-{config.max_demos} 선택")
    passed.sort(key=quality_score, reverse=True)
    selected = passed[: config.max_demos]
    if not selected:
        print("\n⚠ 선택 0개. 임계값 완화 필요.")
        return None

    return _pack_selected_demos(selected, config)


def save_demos(demos: dict[str, np.ndarray], output_path: str | Path) -> Path:
    """Save extracted demos as a compressed NPZ artifact."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **demos)
    print(f"\n✓ 저장: {output_path} ({len(demos['observations'])} episodes)")
    print(
        "  학습 freq window: "
        f"{float(demos['freq_window_mean']):.3f} ± {float(demos['freq_window_std']):.3f} Hz "
        f"(range [{float(demos['freq_window_min']):.3f}, {float(demos['freq_window_max']):.3f}])"
    )
    return output_path


def plot_demo_quality(demos: Any, figures_dir: str | Path) -> tuple[Path, Path]:
    """Save Plan C demo-quality histograms and representative episode plots."""
    figures_dir = Path(figures_dir).expanduser().resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded: {list(demos.keys())}")
    print(f"Episodes: {len(demos['observations'])}")
    print(f"평균 length: {demos['episode_lengths'].mean():.1f}")

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, key, color in zip(
        axes,
        ["monotonicity", "peak_sharpness", "freq_stability", "estimated_freqs"],
        ["green", "blue", "orange", "purple"],
    ):
        ax.hist(demos[key], bins=20, edgecolor="black", color=color, alpha=0.7)
        ax.axvline(demos[key].mean(), color="red", linestyle="--", label=f"mean {demos[key].mean():.2f}")
        ax.set_xlabel(key)
        ax.set_title(key)
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    quality_path = figures_dir / "planC_quality_dist.png"
    fig.savefig(quality_path, dpi=80)
    print(f"✓ 저장: {quality_path}")
    plt.show()
    plt.close(fig)

    phase_joint_idx = int(demos["phase_joint_idx"]) if "phase_joint_idx" in demos else 19
    sorted_idx = np.argsort(demos["estimated_freqs"])
    sample_indices = [sorted_idx[0], sorted_idx[len(sorted_idx) // 2], sorted_idx[-1]]

    fig, axes = plt.subplots(3, 3, figsize=(14, 8))
    for row, ep_idx in enumerate(sample_indices):
        length = int(demos["episode_lengths"][ep_idx])
        freq = float(demos["estimated_freqs"][ep_idx])
        axes[row, 0].plot(demos["observations"][ep_idx, :length, phase_joint_idx])
        axes[row, 0].set_title(f"Ep{ep_idx}: joint signal (freq={freq:.2f}Hz)")
        axes[row, 0].grid(True)

        axes[row, 1].plot(demos["phases"][ep_idx, :length])
        axes[row, 1].set_title("Extracted phase")
        axes[row, 1].grid(True)

        axes[row, 2].scatter(demos["phases"][ep_idx, :length], demos["actions"][ep_idx, :length, 0], s=4, alpha=0.5)
        axes[row, 2].set_xlabel("Phase")
        axes[row, 2].set_ylabel("Action[0]")
        axes[row, 2].set_title("Phase ↔ Action")
        axes[row, 2].grid(True)

    fig.tight_layout()
    viz_path = figures_dir / "planC_demo_visualization.png"
    fig.savefig(viz_path, dpi=80)
    print(f"✓ 저장: {viz_path}")
    plt.show()
    plt.close(fig)
    return quality_path, viz_path


def print_demo_quality_report(demos: Any) -> None:
    """Print a compact quality report for extracted Plan C demos."""
    print("=== Plan C Demo 품질 자동 평가 ===\n")
    n_demos = len(demos["observations"])
    print(f"1. Episode 수: {n_demos}")
    print(f"   {'✓ 충분 (50+)' if n_demos >= 50 else '○ 적당 (30-50)' if n_demos >= 30 else '⚠ 부족 (<30)'}")

    mean_len = demos["episode_lengths"].mean()
    print(f"\n2. 평균 length: {mean_len:.1f}")
    print(f"   {'✓ 우수' if mean_len >= 200 else '○ 충분' if mean_len >= 150 else '⚠ 짧음'}")

    freq_std = demos["estimated_freqs"].std()
    freq_range = demos["estimated_freqs"].max() - demos["estimated_freqs"].min()
    print(f"\n3. Frequency 다양성: std={freq_std:.3f}, range={freq_range:.3f}")
    print(f"   {'✓ 매우 다양' if freq_std > 0.3 else '○ 적당' if freq_std > 0.1 else '⚠ 단조'}")

    print("\n4. Frequency bin별 분포:")
    for lo, hi in [(0.5, 1.5), (1.5, 2.5), (2.5, 3.5)]:
        count = ((demos["estimated_freqs"] >= lo) & (demos["estimated_freqs"] < hi)).sum()
        print(f"   freq [{lo:.1f}, {hi:.1f}): {count} eps {'✓' if count >= 5 else '⚠'}")

    mean_mono = demos["monotonicity"].mean()
    mean_sharp = demos["peak_sharpness"].mean()
    print(f"\n5. 평균 phase quality: mono={mean_mono:.2f}, sharp={mean_sharp:.2f}")
    print(f"   {'✓ 매우 깔끔' if mean_mono > 0.95 and mean_sharp > 5 else '○ 충분'}")


def _print_quality_distribution(enriched: Sequence[dict[str, Any]], n_processed: int) -> None:
    monos = [e["quality"]["monotonicity"] for e in enriched]
    sharps = [e["quality"]["peak_sharpness"] for e in enriched]
    stabs = [e["quality"]["freq_stability"] for e in enriched]
    freqs = [e["quality"]["estimated_freq"] for e in enriched]
    print(f"\nPhase 2: Quality 분포 (전체 {n_processed} eps)")
    print(f"  mono:      min={min(monos):.2f}, mean={np.mean(monos):.2f}, max={max(monos):.2f}")
    print(f"  sharp:     min={min(sharps):.2f}, mean={np.mean(sharps):.2f}, max={max(sharps):.2f}")
    print(f"  freq_stab: min={min(stabs):.2f}, mean={np.mean(stabs):.2f}, max={max(stabs):.2f}")
    print(f"  freq:      min={min(freqs):.2f}, mean={np.mean(freqs):.2f}, max={max(freqs):.2f}")


def _pack_selected_demos(selected: Sequence[dict[str, Any]], config: DemoExtractionConfig) -> dict[str, np.ndarray]:
    sel_freqs = np.array([e["quality"]["estimated_freq"] for e in selected])
    sel_monos = np.array([e["quality"]["monotonicity"] for e in selected])
    sel_sharps = np.array([e["quality"]["peak_sharpness"] for e in selected])

    f_mean, f_std = float(sel_freqs.mean()), float(sel_freqs.std())
    f_min, f_max = float(sel_freqs.min()), float(sel_freqs.max())
    print(f"\n=== 선택된 {len(selected)} demos 특성 ===")
    print(f"  freq:  mean={f_mean:.3f}, std={f_std:.3f}, range=[{f_min:.3f}, {f_max:.3f}] Hz")
    print(f"  mono:  mean={sel_monos.mean():.3f}, min={sel_monos.min():.3f}")
    print(f"  sharp: mean={sel_sharps.mean():.3f}, min={sel_sharps.min():.3f}")
    print(f"\n  In-distribution window (mean±2σ): [{f_mean - 2 * f_std:.3f}, {f_mean + 2 * f_std:.3f}] Hz")

    if len(selected) < 50:
        print(f"\n⚠ 선택된 demos {len(selected)}개 < 50. 임계값 완화 또는 max_episodes_to_process 증가 권장.")

    n_demos = len(selected)
    act_dim = selected[0]["act"].shape[-1]
    out = {
        "observations": np.zeros((n_demos, config.episode_length, config.expected_obs_dim), dtype=np.float32),
        "actions": np.zeros((n_demos, config.episode_length, act_dim), dtype=np.float32),
        "phases": np.zeros((n_demos, config.episode_length), dtype=np.float32),
        "episode_lengths": np.zeros((n_demos,), dtype=np.int32),
        "estimated_freqs": np.zeros((n_demos,), dtype=np.float32),
        "monotonicity": np.zeros((n_demos,), dtype=np.float32),
        "peak_sharpness": np.zeros((n_demos,), dtype=np.float32),
        "freq_stability": np.zeros((n_demos,), dtype=np.float32),
        "freq_window_mean": np.float32(f_mean),
        "freq_window_std": np.float32(f_std),
        "freq_window_min": np.float32(f_min),
        "freq_window_max": np.float32(f_max),
        "phase_joint_idx": np.int32(config.phase_joint_idx),
    }
    for idx, entry in enumerate(selected):
        length = entry["length"]
        out["observations"][idx, :length] = entry["obs"][:length]
        out["actions"][idx, :length] = entry["act"][:length]
        out["phases"][idx, :length] = entry["phases"][:length]
        out["episode_lengths"][idx] = length
        out["estimated_freqs"][idx] = entry["quality"]["estimated_freq"]
        out["monotonicity"][idx] = entry["quality"]["monotonicity"]
        out["peak_sharpness"][idx] = entry["quality"]["peak_sharpness"]
        out["freq_stability"][idx] = entry["quality"]["freq_stability"]
    return out
