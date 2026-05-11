"""Training and validation helpers for diffusion policy models."""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from typing import Callable, Optional

from .dataset import encode_phase_cossin


# =====================================================================
# Conditioning extractors — 각 단계에서 batch → (global_cond, per_step_cond) 뽑는 방식
# =====================================================================

def vanilla_cond_fn(batch: dict, device: str):
    """Vanilla DP: global_cond = flattened obs window. per_step_cond 없음."""
    obs = batch['obs'].to(device, non_blocking=True)
    global_cond = obs.flatten(start_dim=1)   # (B, OH * OD)
    return global_cond, None


def periodic_phase_cond_fn(batch: dict, device: str):
    """Step 3: global_cond = [flattened obs, (cosφ_0, sinφ_0)].

    Chunk 첫 step의 phase만 사용 (PAPL 스타일). phase는 batch['phase']의 [:, 0].
    """
    obs = batch['obs'].to(device, non_blocking=True)
    phase = batch['phase'].to(device, non_blocking=True)  # (B, PH)
    phi0 = phase[:, 0]                                    # (B,)
    phase_enc = encode_phase_cossin(phi0)                 # (B, 2)
    global_cond = torch.cat([obs.flatten(start_dim=1), phase_enc], dim=-1)
    return global_cond, None


def trajectory_phase_cond_fn(batch: dict, device: str):
    """Step 4: per-step phase trajectory conditioning.

    global_cond = flattened obs window (no phase).
    per_step_cond = (cos φ_t, sin φ_t) for each chunk step → (B, PH, 2).
    """
    obs = batch['obs'].to(device, non_blocking=True)
    phase = batch['phase'].to(device, non_blocking=True)   # (B, PH)

    global_cond = obs.flatten(start_dim=1)
    per_step_cond = encode_phase_cossin(phase)             # (B, PH, 2)
    return global_cond, per_step_cond


# =====================================================================
# Validation
# =====================================================================

@torch.no_grad()
def evaluate(model, ema, noise_scheduler, val_loader, cond_fn: Callable,
             device: str, n_batches: Optional[int] = None) -> float:
    """Validation MSE loss with EMA weights.

    cond_fn: batch → (global_cond, per_step_cond) callable.
    n_batches: None이면 전체. 학습 중 빠른 중간 체크엔 8 정도.
    """
    # EMA weights를 model에 임시 적용
    ema.store(model.parameters())
    ema.copy_to(model.parameters())

    model.eval()
    losses = []
    for i, batch in enumerate(val_loader):
        if n_batches is not None and i >= n_batches:
            break
        action = batch['action'].to(device, non_blocking=True)
        B = action.shape[0]

        global_cond, per_step_cond = cond_fn(batch, device)

        noise = torch.randn_like(action)
        t = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                          (B,), device=device).long()
        noisy_action = noise_scheduler.add_noise(action, noise, t)

        if per_step_cond is None:
            noise_pred = model(noisy_action, t, global_cond)
        else:
            noise_pred = model(noisy_action, t, global_cond, per_step_cond)

        loss = F.mse_loss(noise_pred, noise)
        losses.append(loss.item())

    ema.restore(model.parameters())
    model.train()
    return float(np.mean(losses))


# =====================================================================
# Training loop
# =====================================================================

def train_diffusion_policy(
    model,
    ema,
    noise_scheduler,
    train_loader,
    val_loader,
    cond_fn: Callable = vanilla_cond_fn,
    num_epochs: int = 100,
    lr: float = 1e-4,
    weight_decay: float = 1e-6,
    warmup_steps: int = 500,
    val_every: int = 5,
    val_n_batches: int = 8,
    log_every_step: int = 100,
    grad_clip: float = 1.0,
    device: str = 'cuda',
):
    """DDPM training. Returns (train_losses, val_log, best_ema_state).

    cond_fn: batch → (global_cond, per_step_cond)를 뽑는 callable.
        Vanilla DP → vanilla_cond_fn (default)
        Periodic Phase → periodic_phase_cond_fn
        Phase Trajectory → (Step 4에서 추가)
    """
    from diffusers.optimization import get_scheduler

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        'cosine', optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    train_losses = []          # per-epoch mean
    val_log = []               # [(epoch, val_loss), ...]
    best_val = float('inf')
    best_ema_state = None
    global_step = 0
    t0 = time.time()

    for epoch in range(num_epochs):
        model.train()
        ep_losses = []
        for batch in train_loader:
            action = batch['action'].to(device, non_blocking=True)
            B = action.shape[0]

            global_cond, per_step_cond = cond_fn(batch, device)

            noise = torch.randn_like(action)
            t = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                              (B,), device=device).long()
            noisy_action = noise_scheduler.add_noise(action, noise, t)

            if per_step_cond is None:
                noise_pred = model(noisy_action, t, global_cond)
            else:
                noise_pred = model(noisy_action, t, global_cond, per_step_cond)

            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            lr_scheduler.step()
            ema.step(model.parameters())

            ep_losses.append(loss.item())
            global_step += 1

            if global_step % log_every_step == 0:
                recent = float(np.mean(ep_losses[-50:]))
                print(f"  step {global_step}/{total_steps} | "
                      f"epoch {epoch+1}/{num_epochs} | "
                      f"loss {recent:.5f} | "
                      f"lr {lr_scheduler.get_last_lr()[0]:.2e} | "
                      f"elapsed {time.time()-t0:.0f}s")

        train_losses.append(float(np.mean(ep_losses)))

        if (epoch + 1) % val_every == 0 or epoch == num_epochs - 1:
            v = evaluate(model, ema, noise_scheduler, val_loader,
                         cond_fn=cond_fn, device=device, n_batches=val_n_batches)
            val_log.append((epoch + 1, v))
            print(f"  >> epoch {epoch+1}: train {train_losses[-1]:.5f} | val {v:.5f}")
            if v < best_val:
                best_val = v
                best_ema_state = {
                    k: v.detach().cpu().clone()
                    for k, v in ema.state_dict().items() if torch.is_tensor(v)
                }
                print(f"     ★ new best val")

    print(f"\n총 학습 시간: {(time.time()-t0)/60:.1f} min")
    return train_losses, val_log, best_ema_state


# =====================================================================
# Checkpoint helpers
# =====================================================================

def save_checkpoint(path: str | Path, model, ema, train_losses, val_log,
                    best_ema_state=None, config: Optional[dict] = None):
    """Save a training checkpoint under the user artifact workspace."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        'model_state_dict': model.state_dict(),
        'ema_state_dict':   ema.state_dict(),
        'best_ema_state':   best_ema_state,
        'train_losses':     train_losses,
        'val_log':          val_log,
        'config':           config or {},
    }
    torch.save(ckpt, path)
    print(f"✓ Saved: {path} ({path.stat().st_size / 1e6:.1f} MB)")


def load_checkpoint(path: str | Path, model, ema, device: str = 'cuda',
                    use_best_ema: bool = True):
    """저장된 ckpt를 model + ema에 로드. 학습 metadata 반환.

    use_best_ema=True (default): best_ema_state가 ckpt에 있으면 그걸 ema에 로드.
                                  Best validation 시점의 weights라 보통 final보다 좋음.
    use_best_ema=False: 최종 epoch의 ema를 로드.
    """
    path = Path(path).expanduser()
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    best = ckpt.get('best_ema_state')
    if use_best_ema and best is not None:
        # ema의 state_dict 구조 유지하면서 best 값 덮어쓰기
        ema_state = ckpt['ema_state_dict']
        ema_state.update(best)
        ema.load_state_dict(ema_state)
        print(f"✓ Loaded: {path}  [best EMA]")
    else:
        ema.load_state_dict(ckpt['ema_state_dict'])
        tag = "[final EMA]" if best is not None else "[no best — final EMA]"
        print(f"✓ Loaded: {path}  {tag}")

    if ckpt.get('val_log'):
        if use_best_ema and best is not None:
            best_idx = min(range(len(ckpt['val_log'])), key=lambda i: ckpt['val_log'][i][1])
            print(f"  Best val: {ckpt['val_log'][best_idx]}")
        else:
            print(f"  Last val: {ckpt['val_log'][-1]}")

    return {
        'train_losses':   ckpt.get('train_losses', []),
        'val_log':        ckpt.get('val_log', []),
        'best_ema_state': ckpt.get('best_ema_state'),
        'config':         ckpt.get('config', {}),
    }

