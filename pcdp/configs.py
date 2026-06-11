"""Central experiment configuration for phase-conditioned diffusion policy.

Notebook cells should import one named :class:`ExperimentConfig` instead of
hand-editing scattered constants.  The config objects are intentionally plain
Python dataclasses so they work in notebooks without an extra YAML/OmegaConf
runtime dependency, while still being easy to serialize into checkpoints.
"""

from __future__ import annotations

import random

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping


VariantName = str


def set_global_seed(seed: int, *, deterministic: bool = False) -> int:
    """Seed Python, NumPy, and PyTorch RNGs for a configured run."""
    import numpy as np
    import torch

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


@dataclass(frozen=True)
class DataConfig:
    """Dataset/DataLoader parameters shared by training notebooks."""

    batch_size: int = 256
    num_workers: int = 2


@dataclass(frozen=True)
class ModelConfig:
    """Architecture parameters that are common across model variants."""

    diffusion_step_embed_dim: int = 256
    down_dims: tuple[int, ...] = (256, 512, 1024)
    per_step_cond_dim: int = 2


@dataclass(frozen=True)
class DiffusionConfig:
    """DDPM/DDIM scheduler parameters."""

    num_train_timesteps: int = 100
    num_inference_steps: int = 16
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "epsilon"
    clip_sample: bool = True

    def scheduler_kwargs(self) -> dict[str, Any]:
        """Return kwargs for ``diffusers.DDPMScheduler``/``DDIMScheduler``."""
        return {
            "num_train_timesteps": self.num_train_timesteps,
            "beta_schedule": self.beta_schedule,
            "prediction_type": self.prediction_type,
            "clip_sample": self.clip_sample,
        }


@dataclass(frozen=True)
class TrainingConfig:
    """Optimization parameters consumed by ``train_diffusion_policy``."""

    num_epochs: int = 60
    lr: float = 1e-4
    weight_decay: float = 1e-6
    warmup_steps: int = 500
    val_every: int = 5
    val_n_batches: int = 8
    log_every_step: int = 100
    grad_clip: float | None = 1.0
    ema_power: float = 0.75

    def train_kwargs(self) -> dict[str, Any]:
        """Return only kwargs accepted by ``train_diffusion_policy``."""
        return {
            "num_epochs": self.num_epochs,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "val_every": self.val_every,
            "val_n_batches": self.val_n_batches,
            "log_every_step": self.log_every_step,
            "grad_clip": self.grad_clip,
        }


@dataclass(frozen=True)
class EvaluationConfig:
    """Rollout/evaluation defaults."""

    max_steps: int = 300
    n_seeds: int = 5
    deterministic_sampling: bool = True
    render: bool = False
    dt: float = 0.05


@dataclass(frozen=True)
class ArtifactConfig:
    """File names for outputs produced by one experiment variant."""

    checkpoint_name: str
    loss_plot_name: str

    def checkpoint_path(self, checkpoints_dir: str | Path | None = None) -> Path:
        """Resolve the checkpoint path under ``paths.CHECKPOINTS_DIR`` by default."""
        if checkpoints_dir is None:
            from .paths import CHECKPOINTS_DIR

            checkpoints_dir = CHECKPOINTS_DIR
        return Path(checkpoints_dir).expanduser() / self.checkpoint_name


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete config for one notebook experiment or ablation run."""

    name: VariantName
    display_name: str
    model_builder: str
    train_cond_fn: str
    sample_cond_fn: str
    data: DataConfig
    model: ModelConfig
    diffusion: DiffusionConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    artifacts: ArtifactConfig
    seed: int = 42
    deterministic: bool = False
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a checkpoint-friendly dict representation."""
        payload = asdict(self)
        payload["data"]["batch_size"] = int(payload["data"]["batch_size"])
        payload["data"]["num_workers"] = int(payload["data"]["num_workers"])
        return payload

    def checkpoint_path(self, checkpoints_dir: str | Path | None = None) -> Path:
        """Resolve this experiment's checkpoint path."""
        return self.artifacts.checkpoint_path(checkpoints_dir)

    def training_kwargs(self) -> dict[str, Any]:
        """Return kwargs for ``train_diffusion_policy``."""
        return self.training.train_kwargs()

    def noise_scheduler_config(self) -> dict[str, Any]:
        """Return a serializable scheduler config for sampling helpers."""
        return self.diffusion.scheduler_kwargs()

    def build_noise_scheduler(self):
        """Instantiate the DDPM training scheduler for this experiment."""
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

        return DDPMScheduler(**self.diffusion.scheduler_kwargs())

    def build_ema(self, model):
        """Instantiate EMA with this experiment's EMA setting."""
        from diffusers.training_utils import EMAModel

        return EMAModel(parameters=model.parameters(), power=self.training.ema_power)

    def build_model(self, data: Mapping[str, Any], device: str = "cuda"):
        """Build the configured model from a ``load_project_data`` dictionary."""
        builder = self.resolve_model_builder()
        kwargs: dict[str, Any] = {
            "obs_dim": int(data["OBS_DIM"]),
            "act_dim": int(data["ACT_DIM"]),
            "obs_horizon": int(data["OBS_HORIZON"]),
            "diffusion_step_embed_dim": self.model.diffusion_step_embed_dim,
            "down_dims": self.model.down_dims,
            "device": device,
        }
        if self.model_builder == "build_phase_trajectory_dp_model":
            kwargs["per_step_cond_dim"] = self.model.per_step_cond_dim
        return builder(**kwargs)

    def resolve_model_builder(self) -> Callable[..., Any]:
        """Resolve the model builder callable lazily."""
        from .models import (
            build_periodic_phase_dp_model,
            build_phase_trajectory_dp_model,
            build_vanilla_dp_model,
        )

        builders = {
            "build_vanilla_dp_model": build_vanilla_dp_model,
            "build_periodic_phase_dp_model": build_periodic_phase_dp_model,
            "build_phase_trajectory_dp_model": build_phase_trajectory_dp_model,
        }
        return builders[self.model_builder]

    def resolve_train_cond_fn(self) -> Callable[..., Any]:
        """Resolve the training conditioning function lazily."""
        from .training import (
            periodic_phase_cond_fn,
            trajectory_phase_cond_fn,
            trajectory_phase_continuation_cond_fn,
            vanilla_cond_fn,
        )

        cond_fns = {
            "vanilla_cond_fn": vanilla_cond_fn,
            "periodic_phase_cond_fn": periodic_phase_cond_fn,
            "trajectory_phase_cond_fn": trajectory_phase_cond_fn,
            "trajectory_phase_continuation_cond_fn": trajectory_phase_continuation_cond_fn,
        }
        return cond_fns[self.train_cond_fn]

    def resolve_sample_cond_fn(self) -> Callable[..., Any]:
        """Resolve the sampling conditioning function lazily."""
        from .sampling import (
            periodic_phase_sample_cond_fn,
            trajectory_phase_continuation_sample_cond_fn,
            trajectory_phase_sample_cond_fn,
            vanilla_sample_cond_fn,
        )

        cond_fns = {
            "vanilla_sample_cond_fn": vanilla_sample_cond_fn,
            "periodic_phase_sample_cond_fn": periodic_phase_sample_cond_fn,
            "trajectory_phase_sample_cond_fn": trajectory_phase_sample_cond_fn,
            "trajectory_phase_continuation_sample_cond_fn": trajectory_phase_continuation_sample_cond_fn,
        }
        return cond_fns[self.sample_cond_fn]


_BASE_DATA = DataConfig()
_BASE_MODEL = ModelConfig()
_PHASE_CONTINUATION_MODEL = replace(_BASE_MODEL, per_step_cond_dim=4)
_BASE_DIFFUSION = DiffusionConfig()
_BASE_TRAINING = TrainingConfig()
_BASE_EVALUATION = EvaluationConfig()

EXPERIMENT_CONFIGS: dict[VariantName, ExperimentConfig] = {
    "vanilla": ExperimentConfig(
        name="vanilla",
        display_name="Vanilla Diffusion Policy",
        model_builder="build_vanilla_dp_model",
        train_cond_fn="vanilla_cond_fn",
        sample_cond_fn="vanilla_sample_cond_fn",
        data=_BASE_DATA,
        model=_BASE_MODEL,
        diffusion=_BASE_DIFFUSION,
        training=_BASE_TRAINING,
        evaluation=_BASE_EVALUATION,
        artifacts=ArtifactConfig(
            checkpoint_name="vanilla_dp_ckpt.pt",
            loss_plot_name="vanilla_dp_loss.png",
        ),
        tags=("baseline",),
    ),
    "periodic_phase": ExperimentConfig(
        name="periodic_phase",
        display_name="Periodic Phase Conditioning",
        model_builder="build_periodic_phase_dp_model",
        train_cond_fn="periodic_phase_cond_fn",
        sample_cond_fn="periodic_phase_sample_cond_fn",
        data=_BASE_DATA,
        model=_BASE_MODEL,
        diffusion=_BASE_DIFFUSION,
        training=_BASE_TRAINING,
        evaluation=_BASE_EVALUATION,
        artifacts=ArtifactConfig(
            checkpoint_name="phase_periodic_ckpt.pt",
            loss_plot_name="phase_periodic_loss.png",
        ),
        tags=("phase", "global-cond"),
    ),
    "phase_trajectory": ExperimentConfig(
        name="phase_trajectory",
        display_name="Phase Trajectory Conditioning",
        model_builder="build_phase_trajectory_dp_model",
        train_cond_fn="trajectory_phase_cond_fn",
        sample_cond_fn="trajectory_phase_sample_cond_fn",
        data=_BASE_DATA,
        model=_BASE_MODEL,
        diffusion=_BASE_DIFFUSION,
        training=_BASE_TRAINING,
        evaluation=_BASE_EVALUATION,
        artifacts=ArtifactConfig(
            checkpoint_name="phase_trajectory_ckpt.pt",
            loss_plot_name="phase_trajectory_loss.png",
        ),
        tags=("phase", "per-step-cond", "main"),
    ),
    "phase_trajectory_sync": ExperimentConfig(
        name="phase_trajectory_sync",
        display_name="Phase Trajectory + Sync Loss (main)",
        model_builder="build_phase_trajectory_dp_model",
        train_cond_fn="trajectory_phase_cond_fn",
        sample_cond_fn="trajectory_phase_sample_cond_fn",
        data=_BASE_DATA,
        model=_BASE_MODEL,
        diffusion=_BASE_DIFFUSION,
        training=_BASE_TRAINING,
        evaluation=_BASE_EVALUATION,
        artifacts=ArtifactConfig(
            checkpoint_name="phase_trajectory_sync_lambda0.12.pt",
            loss_plot_name="phase_trajectory_sync_loss.png",
        ),
        tags=("phase", "per-step-cond", "sync", "ours"),
    ),
    "phase_trajectory_sync_v2": ExperimentConfig(
        name="phase_trajectory_sync_v2",
        display_name="Sync v2 (Velocity + SNR)",
        model_builder="build_phase_trajectory_dp_model",
        train_cond_fn="trajectory_phase_cond_fn",
        sample_cond_fn="trajectory_phase_sample_cond_fn",
        data=_BASE_DATA,
        model=_BASE_MODEL,
        diffusion=_BASE_DIFFUSION,
        training=_BASE_TRAINING,
        evaluation=_BASE_EVALUATION,
        artifacts=ArtifactConfig(
            checkpoint_name="phase_trajectory_sync_vel0.5_snr5_lambda0.12.pt",
            loss_plot_name="phase_trajectory_sync_v2_loss.png",
        ),
        tags=("phase", "per-step-cond", "sync-v2", "ablation"),
    ),
    "phase_continuation": ExperimentConfig(
        name="phase_continuation",
        display_name="Phase Continuation Conditioning",
        model_builder="build_phase_trajectory_dp_model",
        train_cond_fn="trajectory_phase_continuation_cond_fn",
        sample_cond_fn="trajectory_phase_continuation_sample_cond_fn",
        data=_BASE_DATA,
        model=_PHASE_CONTINUATION_MODEL,
        diffusion=_BASE_DIFFUSION,
        training=_BASE_TRAINING,
        evaluation=_BASE_EVALUATION,
        artifacts=ArtifactConfig(
            checkpoint_name="phase_continuation_ckpt.pt",
            loss_plot_name="phase_continuation_loss.png",
        ),
        tags=("phase", "phase-continuation", "per-step-cond", "ablation-base"),
    ),
    "phase_continuation_sync": ExperimentConfig(
        name="phase_continuation_sync",
        display_name="Phase Continuation + Sync Loss",
        model_builder="build_phase_trajectory_dp_model",
        train_cond_fn="trajectory_phase_continuation_cond_fn",
        sample_cond_fn="trajectory_phase_continuation_sample_cond_fn",
        data=_BASE_DATA,
        model=_PHASE_CONTINUATION_MODEL,
        diffusion=_BASE_DIFFUSION,
        training=_BASE_TRAINING,
        evaluation=_BASE_EVALUATION,
        artifacts=ArtifactConfig(
            checkpoint_name="phase_continuation_sync_lambda0.12.pt",
            loss_plot_name="phase_continuation_sync_loss.png",
        ),
        tags=("phase", "phase-continuation", "sync", "ablation"),
    ),
    "phase_continuation_sync_v2": ExperimentConfig(
        name="phase_continuation_sync_v2",
        display_name="Phase Continuation + Sync v2",
        model_builder="build_phase_trajectory_dp_model",
        train_cond_fn="trajectory_phase_continuation_cond_fn",
        sample_cond_fn="trajectory_phase_continuation_sample_cond_fn",
        data=_BASE_DATA,
        model=_PHASE_CONTINUATION_MODEL,
        diffusion=_BASE_DIFFUSION,
        training=_BASE_TRAINING,
        evaluation=_BASE_EVALUATION,
        artifacts=ArtifactConfig(
            checkpoint_name="phase_continuation_sync_v2_vel0.5_snr5_lambda0.12.pt",
            loss_plot_name="phase_continuation_sync_v2_loss.png",
        ),
        tags=("phase", "phase-continuation", "sync-v2", "ablation"),
    ),
    "phase_trajectory_sync_abs_l018": ExperimentConfig(
        name="phase_trajectory_sync_abs_l018",
        display_name="Strong Absolute Sync (lambda 0.18)",
        model_builder="build_phase_trajectory_dp_model",
        train_cond_fn="trajectory_phase_cond_fn",
        sample_cond_fn="trajectory_phase_sample_cond_fn",
        data=_BASE_DATA,
        model=_BASE_MODEL,
        diffusion=_BASE_DIFFUSION,
        training=_BASE_TRAINING,
        evaluation=_BASE_EVALUATION,
        artifacts=ArtifactConfig(
            checkpoint_name="phase_trajectory_sync_abs_l018.pt",
            loss_plot_name="phase_trajectory_sync_abs_l018_loss.png",
        ),
        tags=("phase", "per-step-cond", "sync", "hparam-sweep"),
    ),
    "phase_trajectory_sync_v2_l008": ExperimentConfig(
        name="phase_trajectory_sync_v2_l008",
        display_name="Soft Sync v2 (lambda 0.08)",
        model_builder="build_phase_trajectory_dp_model",
        train_cond_fn="trajectory_phase_cond_fn",
        sample_cond_fn="trajectory_phase_sample_cond_fn",
        data=_BASE_DATA,
        model=_BASE_MODEL,
        diffusion=_BASE_DIFFUSION,
        training=_BASE_TRAINING,
        evaluation=_BASE_EVALUATION,
        artifacts=ArtifactConfig(
            checkpoint_name="phase_trajectory_sync_v2_l008_vel0.5_snr5.pt",
            loss_plot_name="phase_trajectory_sync_v2_l008_loss.png",
        ),
        tags=("phase", "per-step-cond", "sync-v2", "hparam-sweep"),
    ),
    "phase_trajectory_sync_v2_vel025_snr10": ExperimentConfig(
        name="phase_trajectory_sync_v2_vel025_snr10",
        display_name="Mild Velocity/SNR Sync v2",
        model_builder="build_phase_trajectory_dp_model",
        train_cond_fn="trajectory_phase_cond_fn",
        sample_cond_fn="trajectory_phase_sample_cond_fn",
        data=_BASE_DATA,
        model=_BASE_MODEL,
        diffusion=_BASE_DIFFUSION,
        training=_BASE_TRAINING,
        evaluation=_BASE_EVALUATION,
        artifacts=ArtifactConfig(
            checkpoint_name="phase_trajectory_sync_v2_vel0.25_snr10_lambda0.12.pt",
            loss_plot_name="phase_trajectory_sync_v2_vel025_snr10_loss.png",
        ),
        tags=("phase", "per-step-cond", "sync-v2", "hparam-sweep"),
    ),
    "phase_continuation_sync_soft": ExperimentConfig(
        name="phase_continuation_sync_soft",
        display_name="Soft Phase Continuation Sync",
        model_builder="build_phase_trajectory_dp_model",
        train_cond_fn="trajectory_phase_continuation_cond_fn",
        sample_cond_fn="trajectory_phase_continuation_sample_cond_fn",
        data=_BASE_DATA,
        model=_PHASE_CONTINUATION_MODEL,
        diffusion=_BASE_DIFFUSION,
        training=_BASE_TRAINING,
        evaluation=_BASE_EVALUATION,
        artifacts=ArtifactConfig(
            checkpoint_name="phase_continuation_sync_soft_l008.pt",
            loss_plot_name="phase_continuation_sync_soft_loss.png",
        ),
        tags=("phase", "phase-continuation", "sync", "hparam-sweep"),
    ),
}


def available_experiments() -> tuple[VariantName, ...]:
    """Return supported experiment names."""
    return tuple(EXPERIMENT_CONFIGS.keys())


def _replace_nested(config: ExperimentConfig, section: str, values: Mapping[str, Any]) -> ExperimentConfig:
    section_obj = getattr(config, section)
    return replace(config, **{section: replace(section_obj, **dict(values))})


def get_experiment_config(
    name: VariantName,
    *,
    data: Mapping[str, Any] | None = None,
    model: Mapping[str, Any] | None = None,
    diffusion: Mapping[str, Any] | None = None,
    training: Mapping[str, Any] | None = None,
    evaluation: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, Any] | None = None,
    seed: int | None = None,
    deterministic: bool | None = None,
    tags: tuple[str, ...] | None = None,
) -> ExperimentConfig:
    """Return a named config with optional section-level overrides.

    Examples:
        ``get_experiment_config("phase_trajectory")``
        ``get_experiment_config("periodic_phase", training={"num_epochs": 30})``
        ``get_experiment_config("vanilla", model={"down_dims": (128, 256, 512)})``
        ``get_experiment_config("phase_trajectory", seed=123, deterministic=True)``
    """
    if name not in EXPERIMENT_CONFIGS:
        valid = ", ".join(available_experiments())
        raise ValueError(f"Unknown experiment config: {name!r}. Valid names: {valid}")

    config = EXPERIMENT_CONFIGS[name]
    overrides = {
        "data": data,
        "model": model,
        "diffusion": diffusion,
        "training": training,
        "evaluation": evaluation,
        "artifacts": artifacts,
    }
    for section, values in overrides.items():
        if values:
            config = _replace_nested(config, section, values)
    if seed is not None:
        config = replace(config, seed=int(seed))
    if deterministic is not None:
        config = replace(config, deterministic=bool(deterministic))
    if tags is not None:
        config = replace(config, tags=tags)
    return config


