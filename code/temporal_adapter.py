"""Small trainable latent residual module for temporal refinement."""

from __future__ import annotations

import torch
from torch import nn


class TemporalLatentAdapter(nn.Module):
    """Depth-aware temporal residual adapter.

    Input and output shape: [batch, frames, latent_channels, height, width].
    The last layer is zero-initialized so the untrained adapter is an identity.
    """

    def __init__(self, channels: int = 4, hidden: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=(3, 3, 3), padding=1),
            nn.GroupNorm(4, hidden),
            nn.SiLU(),
            nn.Conv3d(hidden, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        features = latents.permute(0, 2, 1, 3, 4)
        residual = self.net(features).permute(0, 2, 1, 3, 4)
        return latents + residual


class TimestepTemporalLatentAdapter(nn.Module):
    """Temporal residual adapter conditioned on the normalized diffusion timestep.

    ``latents`` has shape [batch, frames, channels, height, width].
    ``normalized_timesteps`` has shape [batch] or is a scalar in [0, 1].
    The last layer is zero-initialized so the untrained adapter is an identity.
    """

    def __init__(self, channels: int = 4, hidden: int = 16, time_hidden: int = 32) -> None:
        super().__init__()
        self.conv_in = nn.Conv3d(channels, hidden, kernel_size=(3, 3, 3), padding=1)
        self.norm = nn.GroupNorm(4, hidden)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_hidden),
            nn.SiLU(),
            nn.Linear(time_hidden, hidden * 2),
        )
        self.activation = nn.SiLU()
        self.conv_out = nn.Conv3d(hidden, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, latents: torch.Tensor, normalized_timesteps: torch.Tensor | float) -> torch.Tensor:
        batch = latents.shape[0]
        if not torch.is_tensor(normalized_timesteps):
            normalized_timesteps = torch.tensor(normalized_timesteps, device=latents.device, dtype=latents.dtype)
        normalized_timesteps = normalized_timesteps.to(device=latents.device, dtype=latents.dtype).reshape(-1, 1)
        if normalized_timesteps.shape[0] == 1 and batch != 1:
            normalized_timesteps = normalized_timesteps.expand(batch, -1)
        if normalized_timesteps.shape[0] != batch:
            raise ValueError(f"Expected one timestep per batch item, got {normalized_timesteps.shape[0]} for batch {batch}")
        features = latents.permute(0, 2, 1, 3, 4)
        features = self.norm(self.conv_in(features))
        scale, bias = self.time_mlp(normalized_timesteps.float()).to(features.dtype).chunk(2, dim=1)
        features = features * (1.0 + scale[:, :, None, None, None]) + bias[:, :, None, None, None]
        residual = self.conv_out(self.activation(features)).permute(0, 2, 1, 3, 4)
        return latents + residual
