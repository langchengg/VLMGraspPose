from __future__ import annotations

import torch
from torch import nn


class MultiScaleLatentAdapter(nn.Module):
    def __init__(self, input_dim: int = 224, layer_count: int = 12, per_layer_dim: int = 16, output_dim: int = 64):
        super().__init__()
        self.layer_count = int(layer_count)
        self.shared = nn.Sequential(nn.Linear(input_dim,64),nn.LayerNorm(64),nn.GELU(),nn.Linear(64,per_layer_dim),nn.GELU())
        self.output = nn.Sequential(nn.Linear(layer_count*per_layer_dim,output_dim),nn.GELU())

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-2] != self.layer_count:
            raise ValueError("latent layer count mismatch")
        encoded = self.shared(values)
        return self.output(encoded.flatten(-2))

