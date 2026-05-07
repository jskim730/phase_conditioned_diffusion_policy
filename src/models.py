"""1D U-Net architectures for phase-conditioned diffusion policies."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# Building blocks
# =====================================================================

class SinusoidalPosEmb(nn.Module):
    """Diffusion timestep용 sinusoidal embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """Conv1d → GroupNorm → Mish."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, n_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_ch),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """FiLM-conditioned 1D ResNet block.

    cond → MLP → (scale, bias) per channel → out = out × scale + bias.
    """

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int,
                 kernel_size: int = 5, n_groups: int = 8):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_ch, out_ch, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_ch, out_ch, kernel_size, n_groups=n_groups),
        ])
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_ch * 2),
            nn.Unflatten(-1, (-1, 1)),
        )
        self.out_ch = out_ch
        self.residual_conv = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T), cond: (B, cond_dim)
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)            # (B, 2*out_ch, 1)
        scale, bias = embed.chunk(2, dim=1)        # (B, out_ch, 1) each
        out = out * scale + bias                   # FiLM
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


# =====================================================================
# Main model
# =====================================================================

class ConditionalUnet1D(nn.Module):
    """Diffusion Policy의 표준 1D U-Net (Chi et al. 2023).

    Inputs:
        sample:      (B, T, input_dim)        — noisy action chunk
        timestep:    (B,) or scalar           — diffusion step
        global_cond: (B, global_cond_dim)     — observation features
                                                (Vanilla DP: flattened obs window;
                                                 Periodic phase encoding 추가시: cat with (cosφ, sinφ))
    Output:
        (B, T, input_dim) — predicted noise ε̂
    """

    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 256,
        down_dims: tuple = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        # Diffusion step encoder
        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        # Mid
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim,
                                       kernel_size=kernel_size, n_groups=n_groups),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim,
                                       kernel_size=kernel_size, n_groups=n_groups),
        ])

        # Down
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out, cond_dim,
                                           kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim,
                                           kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity(),
            ]))

        # Up
        self.up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim,
                                           kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(dim_in, dim_in, cond_dim,
                                           kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity(),
            ]))

        # Final
        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

    def forward(self, sample: torch.Tensor, timestep, global_cond: torch.Tensor) -> torch.Tensor:
        # (B, T, C) → (B, C, T) for conv1d along T
        x = sample.moveaxis(-1, -2)

        # Timestep handling
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=x.device)
        elif timestep.dim() == 0:
            timestep = timestep[None].to(x.device)
        timestep = timestep.expand(x.shape[0])

        ts_feat = self.diffusion_step_encoder(timestep)              # (B, dsed)
        global_feature = torch.cat([ts_feat, global_cond], axis=-1)  # (B, dsed + gcd)

        # Down
        h = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        # Mid
        for mid_block in self.mid_modules:
            x = mid_block(x, global_feature)

        # Up
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        x = x.moveaxis(-1, -2)  # (B, C, T) → (B, T, C)
        return x


# =====================================================================
# Convenience constructor
# =====================================================================

def build_vanilla_dp_model(
    obs_dim: int,
    act_dim: int,
    obs_horizon: int,
    diffusion_step_embed_dim: int = 256,
    down_dims: tuple = (256, 512, 1024),
    device: str = 'cuda',
) -> ConditionalUnet1D:
    """Vanilla DP용 모델 빌드. global_cond_dim = obs_horizon × obs_dim."""
    model = ConditionalUnet1D(
        input_dim=act_dim,
        global_cond_dim=obs_horizon * obs_dim,
        diffusion_step_embed_dim=diffusion_step_embed_dim,
        down_dims=down_dims,
    ).to(device)
    return model


def count_params(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable}



# ======================================================================

def build_periodic_phase_dp_model(
    obs_dim: int,
    act_dim: int,
    obs_horizon: int,
    diffusion_step_embed_dim: int = 256,
    down_dims: tuple = (256, 512, 1024),
    device: str = 'cuda',
) -> ConditionalUnet1D:
    """Build a global-condition U-Net with periodic phase features.

    Vanilla DP와 동일 architecture를 사용하며 global_cond_dim에
    chunk 첫 step의 (cos φ, sin φ) 2차원을 추가한다.
    """
    model = ConditionalUnet1D(
        input_dim=act_dim,
        global_cond_dim=obs_horizon * obs_dim + 2,   # +2 for (cos φ, sin φ)
        diffusion_step_embed_dim=diffusion_step_embed_dim,
        down_dims=down_dims,
    ).to(device)
    return model

#==========================================================================
# ============================================================
# Step 4: Per-step FiLM (Phase Trajectory Conditioning)
# ============================================================

class PhaseConditionalResidualBlock1D(nn.Module):
    """ResNet block with global FiLM (obs + timestep) AND per-step FiLM (phase).

    Per-step encoder는 zero-init이라 학습 시작 시 ConditionalResidualBlock1D와 동일.
    모델이 점진적으로 per-step phase 정보를 사용하도록 학습.
    """

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int,
                 per_step_cond_dim: int,
                 kernel_size: int = 5, n_groups: int = 8):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_ch, out_ch, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_ch, out_ch, kernel_size, n_groups=n_groups),
        ])
        # Global FiLM (T축 따라 일정)
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_ch * 2),
            nn.Unflatten(-1, (-1, 1)),
        )
        # Per-step FiLM (T축 따라 변함) — zero-init
        self.per_step_encoder = nn.Conv1d(per_step_cond_dim, out_ch * 2, 1)
        nn.init.zeros_(self.per_step_encoder.weight)
        nn.init.zeros_(self.per_step_encoder.bias)

        self.out_ch = out_ch
        self.residual_conv = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                per_step_cond: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T), cond: (B, cond_dim), per_step_cond: (B, ps_dim, T)
        out = self.blocks[0](x)

        # Global FiLM
        embed = self.cond_encoder(cond)
        scale, bias = embed.chunk(2, dim=1)
        out = out * scale + bias

        # Per-step FiLM with delta-scale (zero-init = identity 변환)
        ps_embed = self.per_step_encoder(per_step_cond)        # (B, 2*out_ch, T)
        delta_scale, ps_bias = ps_embed.chunk(2, dim=1)
        out = out * (1.0 + delta_scale) + ps_bias

        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class PhaseConditionedUnet1D(nn.Module):
    """1D U-Net with global cond + per-step phase trajectory cond.

    Inputs:
        sample:        (B, T, input_dim)         — noisy action chunk
        timestep:      (B,) or scalar            — diffusion step
        global_cond:   (B, global_cond_dim)      — flattened obs window
        per_step_cond: (B, T, per_step_cond_dim) — (B, 16, 2) for (cos φ_t, sin φ_t)
    Output:
        (B, T, input_dim) — predicted noise ε̂

    per_step_cond는 U-Net 각 level의 T에 맞춰 avg_pool로 downsample.
    """

    def __init__(self, input_dim: int, global_cond_dim: int,
                 per_step_cond_dim: int,
                 diffusion_step_embed_dim: int = 256,
                 down_dims: tuple = (256, 512, 1024),
                 kernel_size: int = 5, n_groups: int = 8):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim
        self.per_step_cond_dim = per_step_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            PhaseConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim, per_step_cond_dim,
                kernel_size=kernel_size, n_groups=n_groups),
            PhaseConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim, per_step_cond_dim,
                kernel_size=kernel_size, n_groups=n_groups),
        ])

        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(nn.ModuleList([
                PhaseConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim, per_step_cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                PhaseConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim, per_step_cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity(),
            ]))

        self.up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(nn.ModuleList([
                PhaseConditionalResidualBlock1D(
                    dim_out * 2, dim_in, cond_dim, per_step_cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                PhaseConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim, per_step_cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity(),
            ]))

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

    def forward(self, sample: torch.Tensor, timestep,
                global_cond: torch.Tensor,
                per_step_cond: torch.Tensor) -> torch.Tensor:
        # (B, T, C) → (B, C, T)
        x = sample.moveaxis(-1, -2)
        ps = per_step_cond.moveaxis(-1, -2)   # (B, ps_dim, T)

        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=x.device)
        elif timestep.dim() == 0:
            timestep = timestep[None].to(x.device)
        timestep = timestep.expand(x.shape[0])

        ts_feat = self.diffusion_step_encoder(timestep)
        global_feature = torch.cat([ts_feat, global_cond], axis=-1)

        # Per-step pyramid: T → T/2 → T/4 ...
        n_levels = len(self.down_modules)
        ps_pyramid = [ps]
        for _ in range(n_levels - 1):
            ps_pyramid.append(F.avg_pool1d(ps_pyramid[-1], kernel_size=2, stride=2))

        # Down
        h = []
        for i, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            ps_lvl = ps_pyramid[i]
            x = resnet(x, global_feature, ps_lvl)
            x = resnet2(x, global_feature, ps_lvl)
            h.append(x)
            x = downsample(x)

        # Mid (deepest)
        ps_mid = ps_pyramid[-1]
        for mid_block in self.mid_modules:
            x = mid_block(x, global_feature, ps_mid)

        # Up — i-th iter uses ps_pyramid[n_levels - 1 - i]
        for i, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            ps_lvl = ps_pyramid[n_levels - 1 - i]
            x = resnet(x, global_feature, ps_lvl)
            x = resnet2(x, global_feature, ps_lvl)
            x = upsample(x)

        x = self.final_conv(x)
        x = x.moveaxis(-1, -2)
        return x


def build_phase_trajectory_dp_model(
    obs_dim: int,
    act_dim: int,
    obs_horizon: int,
    per_step_cond_dim: int = 2,
    diffusion_step_embed_dim: int = 256,
    down_dims: tuple = (256, 512, 1024),
    device: str = 'cuda',
) -> PhaseConditionedUnet1D:
    """Step 4: Phase Trajectory Conditioning.

    global_cond = flattened obs (phase는 per-step으로만 전달).
    per_step_cond = (cos φ_t, sin φ_t) per chunk step.
    """
    model = PhaseConditionedUnet1D(
        input_dim=act_dim,
        global_cond_dim=obs_horizon * obs_dim,
        per_step_cond_dim=per_step_cond_dim,
        diffusion_step_embed_dim=diffusion_step_embed_dim,
        down_dims=down_dims,
    ).to(device)
    return model