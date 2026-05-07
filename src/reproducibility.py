"""Reproducibility helpers shared by training notebooks."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_global_seed(seed: int, *, deterministic: bool = False) -> int:
    """Seed Python, NumPy, and PyTorch RNGs and optionally request deterministic kernels."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    return seed


def resolve_device() -> str:
    """Return the preferred PyTorch device for notebooks."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def setup_reproducibility(seed: int, deterministic: bool = False) -> str:
    """Seed all RNGs, resolve the torch device, and print the legacy setup summary."""
    seed = set_global_seed(seed, deterministic=deterministic)
    device = resolve_device()
    print(f"PyTorch {torch.__version__}, device={device}, seed={seed}")
    return device


def print_data_summary(data: dict[str, Any]) -> None:
    """Print split and phase-frequency metadata used by all runs."""
    print(f"Train: {len(data['train_eps'])} eps | Val: {len(data['val_eps'])} eps")
    print(f"Freq window: {data['freq_window_mean']:.3f} ± {data['freq_window_std']:.3f} Hz")


def describe_project_paths(repo_root: Path, src_dir: Path, artifact_root: Path) -> None:
    """Print the resolved project paths to make notebook logs self-describing."""
    print(f"✓ repo root: {repo_root}")
    print(f"✓ src path:  {src_dir}")
    print(f"✓ artifact root: {artifact_root}")
    print(f"  src files: {[f.name for f in sorted(src_dir.glob('*.py'))]}")
