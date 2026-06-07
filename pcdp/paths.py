"""Project path utilities for reproducible artifact storage.

Generated artifacts default to the repository-local ``artifacts/`` directory so
Colab and local runs behave the same once the user has selected the repository
root as the current working directory.  Set ``PCDP_ARTIFACT_ROOT`` to override
that default, for example to point at a separate Google Drive folder.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


PROJECT_NAME = "phase_conditioned_diffusion_policy"
ARTIFACT_ROOT_ENV_VAR = "PCDP_ARTIFACT_ROOT"

# Repository locations.  This file lives in ``<repo>/pcdp/paths.py``.
REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"

# Artifact root candidates.
PROJECT_ARTIFACT_ROOT = REPO_ROOT / "artifacts"


def _expand_path(path: str | os.PathLike[str]) -> Path:
    """Return an absolute, user-expanded path without creating it."""
    return Path(path).expanduser().resolve()


def resolve_artifact_root(path: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the user artifact root.

    Priority order:
    1. Explicit ``path`` argument.
    2. ``PCDP_ARTIFACT_ROOT`` environment variable.
    3. Repository-local ``artifacts/`` directory.
    """
    if path is not None:
        return _expand_path(path)

    env_path = os.environ.get(ARTIFACT_ROOT_ENV_VAR)
    if env_path:
        return _expand_path(env_path)

    return PROJECT_ARTIFACT_ROOT


ARTIFACT_ROOT = resolve_artifact_root()
DATA_DIR = ARTIFACT_ROOT / "data"
CHECKPOINTS_DIR = ARTIFACT_ROOT / "checkpoints"
RESULTS_DIR = ARTIFACT_ROOT / "results"
FIGURES_DIR = ARTIFACT_ROOT / "figures"
VIDEOS_DIR = ARTIFACT_ROOT / "videos"

DEFAULT_ARTIFACT_DIRS = (
    ARTIFACT_ROOT,
    DATA_DIR,
    CHECKPOINTS_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    VIDEOS_DIR,
)


def ensure_artifact_dirs(extra_dirs: Iterable[Path] = ()) -> tuple[Path, ...]:
    """Create and return the default artifact directories.

    Directory creation is explicit so importing this module never mutates the
    filesystem.  ``extra_dirs`` can be used by notebooks or scripts that need
    task-specific subdirectories under the same artifact root.
    """
    dirs = (*DEFAULT_ARTIFACT_DIRS, *tuple(extra_dirs))
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)
    return dirs
