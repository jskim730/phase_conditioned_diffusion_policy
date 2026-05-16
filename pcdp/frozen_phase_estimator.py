"""Frozen phase estimator utilities.

This module is intentionally separate from the existing diffusion training
helpers. It provides a supervised phase estimator and optional phase-sync
fine-tuning loop without changing the baseline notebooks or training module.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import encode_phase_cossin


@dataclass(frozen=True)
class PhaseEstimatorConfig:
    """Shape and optimization defaults for the frozen estimator."""

    obs_horizon: int
    pred_horizon: int
    obs_dim: int
    act_dim: int
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.05
    lr: float = 1e-3
    weight_decay: float = 1e-5
    num_epochs: int = 30
    grad_clip: float | None = 1.0


class PhaseEstimatorMLP(nn.Module):
    """Predict per-step phase sin/cos from obs window and action chunk.

    Inputs:
        obs_window: (B, obs_horizon, obs_dim)
        action_chunk: (B, pred_horizon, act_dim)
    Output:
        phase_sincos: (B, pred_horizon, 2), normalized along the last axis.
    """

    def __init__(
        self,
        obs_horizon: int,
        pred_horizon: int,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.obs_horizon = int(obs_horizon)
        self.pred_horizon = int(pred_horizon)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)

        input_dim = self.obs_horizon * self.obs_dim + self.pred_horizon * self.act_dim
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(int(num_layers)):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Mish(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, self.pred_horizon * 2))
        self.net = nn.Sequential(*layers)

    def forward(self, obs_window: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        obs_flat = obs_window.flatten(start_dim=1)
        act_flat = action_chunk.flatten(start_dim=1)
        out = self.net(torch.cat([obs_flat, act_flat], dim=-1))
        out = out.view(obs_window.shape[0], self.pred_horizon, 2)
        return F.normalize(out, dim=-1, eps=1e-8)


def build_phase_estimator_from_data(
    data: dict,
    *,
    hidden_dim: int = 256,
    num_layers: int = 3,
    dropout: float = 0.05,
    device: str = "cuda",
) -> PhaseEstimatorMLP:
    """Construct a phase estimator from a ``load_project_data`` dictionary."""

    return PhaseEstimatorMLP(
        obs_horizon=int(data["OBS_HORIZON"]),
        pred_horizon=int(data["PRED_HORIZON"]),
        obs_dim=int(data["OBS_DIM"]),
        act_dim=int(data["ACT_DIM"]),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)


def phase_target_sincos(phase: torch.Tensor) -> torch.Tensor:
    """Return target phase as normalized ``(cos, sin)`` vectors."""

    return encode_phase_cossin(phase)


def circular_sincos_loss(pred_sincos: torch.Tensor, target_sincos: torch.Tensor) -> torch.Tensor:
    """Mean circular loss ``1 - dot(target, pred)``."""

    pred_sincos = F.normalize(pred_sincos, dim=-1, eps=1e-8)
    target_sincos = F.normalize(target_sincos, dim=-1, eps=1e-8)
    dot = (pred_sincos * target_sincos).sum(dim=-1).clamp(-1.0, 1.0)
    return (1.0 - dot).mean()


@torch.no_grad()
def phase_mae_rad(pred_sincos: torch.Tensor, target_sincos: torch.Tensor) -> float:
    """Mean absolute circular phase error in radians."""

    pred = F.normalize(pred_sincos, dim=-1, eps=1e-8)
    target = F.normalize(target_sincos, dim=-1, eps=1e-8)
    dot = (pred * target).sum(dim=-1).clamp(-1.0, 1.0)
    return float(torch.acos(dot).mean().item())


@torch.no_grad()
def evaluate_phase_estimator(
    estimator: nn.Module,
    val_loader,
    *,
    device: str,
    n_batches: int | None = None,
) -> dict[str, float]:
    """Evaluate estimator loss and circular phase error."""

    estimator.eval()
    losses: list[float] = []
    maes: list[float] = []
    for i, batch in enumerate(val_loader):
        if n_batches is not None and i >= n_batches:
            break
        obs = batch["obs"].to(device, non_blocking=True)
        action = batch["action"].to(device, non_blocking=True)
        phase = batch["phase"].to(device, non_blocking=True)
        target = phase_target_sincos(phase)
        pred = estimator(obs, action)
        losses.append(float(circular_sincos_loss(pred, target).item()))
        maes.append(phase_mae_rad(pred, target))
    return {"loss": float(np.mean(losses)), "mae_rad": float(np.mean(maes))}


def train_phase_estimator(
    estimator: nn.Module,
    train_loader,
    val_loader,
    *,
    device: str,
    num_epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    grad_clip: float | None = 1.0,
    val_n_batches: int | None = None,
    log_every_step: int = 100,
) -> tuple[list[float], list[dict[str, float]], dict[str, torch.Tensor]]:
    """Supervised estimator training loop."""

    optimizer = torch.optim.AdamW(estimator.parameters(), lr=lr, weight_decay=weight_decay)
    train_losses: list[float] = []
    val_log: list[dict[str, float]] = []
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] = {}
    global_step = 0
    t0 = time.time()

    for epoch in range(num_epochs):
        estimator.train()
        ep_losses: list[float] = []
        for batch in train_loader:
            obs = batch["obs"].to(device, non_blocking=True)
            action = batch["action"].to(device, non_blocking=True)
            phase = batch["phase"].to(device, non_blocking=True)
            target = phase_target_sincos(phase)

            pred = estimator(obs, action)
            loss = circular_sincos_loss(pred, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(estimator.parameters(), max_norm=grad_clip)
            optimizer.step()

            ep_losses.append(float(loss.item()))
            global_step += 1
            if global_step % log_every_step == 0:
                recent = float(np.mean(ep_losses[-50:]))
                print(
                    f"  estimator step {global_step} | epoch {epoch + 1}/{num_epochs} | "
                    f"loss {recent:.5f} | elapsed {time.time() - t0:.0f}s"
                )

        train_losses.append(float(np.mean(ep_losses)))
        metrics = evaluate_phase_estimator(
            estimator, val_loader, device=device, n_batches=val_n_batches
        )
        metrics = {"epoch": float(epoch + 1), **metrics}
        val_log.append(metrics)
        print(
            f"  >> estimator epoch {epoch + 1}: train {train_losses[-1]:.5f} | "
            f"val {metrics['loss']:.5f} | phase MAE {metrics['mae_rad']:.4f} rad"
        )
        if metrics["loss"] < best_val:
            best_val = metrics["loss"]
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in estimator.state_dict().items()
                if torch.is_tensor(v)
            }
            print("     new best estimator")

    return train_losses, val_log, best_state


def save_phase_estimator_checkpoint(
    path: str | Path,
    estimator: nn.Module,
    *,
    config: PhaseEstimatorConfig | dict,
    train_losses: list[float],
    val_log: list[dict[str, float]],
    best_state: dict[str, torch.Tensor] | None = None,
) -> Path:
    """Save estimator weights and metadata."""

    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": estimator.state_dict(),
        "best_state_dict": best_state,
        "config": asdict(config) if hasattr(config, "__dataclass_fields__") else dict(config),
        "train_losses": train_losses,
        "val_log": val_log,
    }
    torch.save(payload, path)
    print(f"Saved phase estimator: {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def load_phase_estimator_checkpoint(
    path: str | Path,
    *,
    device: str = "cuda",
    use_best: bool = True,
) -> tuple[PhaseEstimatorMLP, dict]:
    """Load estimator checkpoint and return ``(estimator, metadata)``."""

    path = Path(path).expanduser()
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    estimator = PhaseEstimatorMLP(
        obs_horizon=int(cfg["obs_horizon"]),
        pred_horizon=int(cfg["pred_horizon"]),
        obs_dim=int(cfg["obs_dim"]),
        act_dim=int(cfg["act_dim"]),
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        num_layers=int(cfg.get("num_layers", 3)),
        dropout=float(cfg.get("dropout", 0.05)),
    ).to(device)
    state = ckpt.get("best_state_dict") if use_best else None
    estimator.load_state_dict(state or ckpt["model_state_dict"])
    print(f"Loaded phase estimator: {path}")
    return estimator, ckpt


def freeze_phase_estimator(estimator: nn.Module) -> nn.Module:
    """Freeze estimator parameters for use as a training-time regularizer."""

    estimator.eval()
    for param in estimator.parameters():
        param.requires_grad_(False)
    return estimator


def predict_x0_from_epsilon(
    noisy_action: torch.Tensor,
    noise_pred: torch.Tensor,
    timesteps: torch.Tensor,
    noise_scheduler,
) -> torch.Tensor:
    """Reconstruct predicted clean action ``x0_hat`` from epsilon prediction."""

    alpha_bar = noise_scheduler.alphas_cumprod.to(noisy_action.device)[timesteps]
    while alpha_bar.dim() < noisy_action.dim():
        alpha_bar = alpha_bar.unsqueeze(-1)
    return (noisy_action - torch.sqrt(1.0 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)


def phase_sync_loss(
    estimator: nn.Module,
    obs: torch.Tensor,
    x0_hat: torch.Tensor,
    phase: torch.Tensor,
) -> torch.Tensor:
    """Compute frozen-estimator phase synchronization loss."""

    target = phase_target_sincos(phase)
    pred = estimator(obs, x0_hat)
    return circular_sincos_loss(pred, target)


@torch.no_grad()
def evaluate_phase_sync_diffusion_policy(
    model,
    ema,
    noise_scheduler,
    estimator: nn.Module,
    val_loader,
    cond_fn: Callable,
    *,
    device: str,
    lambda_phase: float,
    n_batches: int | None = None,
) -> dict[str, float]:
    """Validation metrics for phase-sync fine-tuning."""

    ema.store(model.parameters())
    ema.copy_to(model.parameters())
    model.eval()
    estimator.eval()
    diffusion_losses: list[float] = []
    phase_losses: list[float] = []

    for i, batch in enumerate(val_loader):
        if n_batches is not None and i >= n_batches:
            break
        action = batch["action"].to(device, non_blocking=True)
        obs = batch["obs"].to(device, non_blocking=True)
        phase = batch["phase"].to(device, non_blocking=True)
        global_cond, per_step_cond = cond_fn(batch, device)

        noise = torch.randn_like(action)
        t = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (action.shape[0],), device=device
        ).long()
        noisy_action = noise_scheduler.add_noise(action, noise, t)
        noise_pred = model(noisy_action, t, global_cond, per_step_cond)
        diffusion_loss = F.mse_loss(noise_pred, noise)
        x0_hat = predict_x0_from_epsilon(noisy_action, noise_pred, t, noise_scheduler)
        p_loss = phase_sync_loss(estimator, obs, x0_hat, phase)

        diffusion_losses.append(float(diffusion_loss.item()))
        phase_losses.append(float(p_loss.item()))

    ema.restore(model.parameters())
    model.train()
    return {
        "diffusion_loss": float(np.mean(diffusion_losses)),
        "phase_loss": float(np.mean(phase_losses)),
        "total_loss": float(np.mean(diffusion_losses) + lambda_phase * np.mean(phase_losses)),
    }


def train_phase_sync_diffusion_policy(
    model,
    ema,
    noise_scheduler,
    estimator: nn.Module,
    train_loader,
    val_loader,
    cond_fn: Callable,
    *,
    device: str,
    num_epochs: int = 10,
    lr: float = 5e-5,
    weight_decay: float = 1e-6,
    lambda_phase: float = 0.05,
    phase_warmup_epochs: int = 1,
    val_every: int = 1,
    val_n_batches: int | None = 8,
    grad_clip: float | None = 1.0,
    log_every_step: int = 100,
) -> tuple[list[dict[str, float]], list[dict[str, float]], dict[str, torch.Tensor]]:
    """Fine-tune trajectory DP with frozen-estimator phase synchronization."""

    freeze_phase_estimator(estimator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_log: list[dict[str, float]] = []
    val_log: list[dict[str, float]] = []
    best_val = float("inf")
    best_ema_state: dict[str, torch.Tensor] = {}
    global_step = 0
    t0 = time.time()

    for epoch in range(num_epochs):
        model.train()
        diffusion_losses: list[float] = []
        phase_losses: list[float] = []
        use_phase = epoch >= phase_warmup_epochs

        for batch in train_loader:
            action = batch["action"].to(device, non_blocking=True)
            obs = batch["obs"].to(device, non_blocking=True)
            phase = batch["phase"].to(device, non_blocking=True)
            global_cond, per_step_cond = cond_fn(batch, device)

            noise = torch.randn_like(action)
            t = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (action.shape[0],), device=device
            ).long()
            noisy_action = noise_scheduler.add_noise(action, noise, t)
            noise_pred = model(noisy_action, t, global_cond, per_step_cond)

            diffusion_loss = F.mse_loss(noise_pred, noise)
            p_loss = torch.zeros((), device=device)
            if use_phase and lambda_phase > 0:
                x0_hat = predict_x0_from_epsilon(noisy_action, noise_pred, t, noise_scheduler)
                p_loss = phase_sync_loss(estimator, obs, x0_hat, phase)
            loss = diffusion_loss + lambda_phase * p_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            ema.step(model.parameters())

            diffusion_losses.append(float(diffusion_loss.item()))
            phase_losses.append(float(p_loss.item()))
            global_step += 1
            if global_step % log_every_step == 0:
                print(
                    f"  sync step {global_step} | epoch {epoch + 1}/{num_epochs} | "
                    f"diff {np.mean(diffusion_losses[-50:]):.5f} | "
                    f"phase {np.mean(phase_losses[-50:]):.5f} | "
                    f"elapsed {time.time() - t0:.0f}s"
                )

        train_metrics = {
            "epoch": float(epoch + 1),
            "diffusion_loss": float(np.mean(diffusion_losses)),
            "phase_loss": float(np.mean(phase_losses)),
            "total_loss": float(np.mean(diffusion_losses) + lambda_phase * np.mean(phase_losses)),
            "phase_enabled": float(use_phase),
        }
        train_log.append(train_metrics)

        if (epoch + 1) % val_every == 0 or epoch == num_epochs - 1:
            metrics = evaluate_phase_sync_diffusion_policy(
                model,
                ema,
                noise_scheduler,
                estimator,
                val_loader,
                cond_fn,
                device=device,
                lambda_phase=lambda_phase,
                n_batches=val_n_batches,
            )
            metrics = {"epoch": float(epoch + 1), **metrics}
            val_log.append(metrics)
            print(
                f"  >> sync epoch {epoch + 1}: train {train_metrics['total_loss']:.5f} | "
                f"val {metrics['total_loss']:.5f} | phase {metrics['phase_loss']:.5f}"
            )
            if metrics["total_loss"] < best_val:
                best_val = metrics["total_loss"]
                best_ema_state = {
                    k: v.detach().cpu().clone()
                    for k, v in ema.state_dict().items()
                    if torch.is_tensor(v)
                }
                print("     new best sync EMA")

    return train_log, val_log, best_ema_state


def load_phase_sync_checkpoint(
    path: str | Path,
    model,
    ema,
    *,
    device: str = "cuda",
    use_best_ema: bool = True,
) -> dict:
    """Load a phase-sync diffusion checkpoint.

    Phase-sync fine-tuning stores validation logs as dictionaries, unlike the
    baseline ``training.load_checkpoint`` helper which expects tuple entries.
    Keep this loader separate so existing checkpoint behavior remains
    unchanged.
    """

    path = Path(path).expanduser()
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    best = ckpt.get("best_ema_state")
    if use_best_ema and best is not None:
        ema_state = ckpt["ema_state_dict"]
        ema_state.update(best)
        ema.load_state_dict(ema_state)
        print(f"Loaded phase-sync checkpoint: {path} [best EMA]")
    else:
        ema.load_state_dict(ckpt["ema_state_dict"])
        print(f"Loaded phase-sync checkpoint: {path} [final EMA]")

    val_log = ckpt.get("val_log", [])
    if val_log:
        if isinstance(val_log[0], dict):
            key = "total_loss" if "total_loss" in val_log[0] else next(iter(val_log[0]))
            best_entry = min(val_log, key=lambda row: row.get(key, float("inf")))
            print(f"  Best val by {key}: {best_entry}")
        elif use_best_ema and best is not None:
            best_idx = min(range(len(val_log)), key=lambda i: val_log[i][1])
            print(f"  Best val: {val_log[best_idx]}")
        else:
            print(f"  Last val: {val_log[-1]}")

    return {
        "train_losses": ckpt.get("train_losses", []),
        "val_log": val_log,
        "best_ema_state": ckpt.get("best_ema_state"),
        "config": ckpt.get("config", {}),
    }
