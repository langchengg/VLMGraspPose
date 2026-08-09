from __future__ import annotations

import torch
from torch import nn


class RGBMapCropEncoder(nn.Module):
    def __init__(self, input_channels: int, output_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, 24, 3, padding=1), nn.GroupNorm(6,24), nn.GELU(),
            nn.MaxPool2d(2), nn.Conv2d(24,48,3,padding=1), nn.GroupNorm(8,48), nn.GELU(),
            nn.MaxPool2d(2), nn.Conv2d(48,64,3,padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64,output_dim), nn.GELU(),
        )

    def forward(self, crop: torch.Tensor) -> torch.Tensor:
        shape = crop.shape
        value = self.network(crop.reshape(-1,*shape[-3:]))
        return value.reshape(*shape[:-3],-1)

