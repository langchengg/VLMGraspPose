"""Small, auditable architecture adapters for pinned official checkpoints.

``OfficialGGCNN2`` is adapted from ``dougsm/ggcnn`` commit
0c50aa7600e8a30d44c5c85cebd6e3394a81f30e, ``models/ggcnn2.py``.
The upstream source is BSD-3-Clause licensed; the architecture and parameter
names are intentionally unchanged so the official state_dict loads strictly.
"""

from __future__ import annotations

import torch
from torch import nn


class OfficialGGCNN2(nn.Module):
    """Official one-channel GG-CNN2 architecture with checkpoint-stable keys."""

    def __init__(
        self,
        input_channels: int = 1,
        filter_sizes: list[int] | None = None,
        l3_k_size: int = 5,
        dilations: list[int] | None = None,
    ) -> None:
        super().__init__()
        filters = [16, 16, 32, 16] if filter_sizes is None else list(filter_sizes)
        dilation_values = [2, 4] if dilations is None else list(dilations)
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, filters[0], 11, padding=5, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[0], filters[0], 5, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(filters[0], filters[1], 5, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[1], filters[1], 5, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(
                filters[1],
                filters[2],
                l3_k_size,
                dilation=dilation_values[0],
                padding=(l3_k_size // 2) * dilation_values[0],
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                filters[2],
                filters[2],
                l3_k_size,
                dilation=dilation_values[1],
                padding=(l3_k_size // 2) * dilation_values[1],
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(filters[2], filters[3], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(filters[3], filters[3], 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pos_output = nn.Conv2d(filters[3], 1, kernel_size=1)
        self.cos_output = nn.Conv2d(filters[3], 1, kernel_size=1)
        self.sin_output = nn.Conv2d(filters[3], 1, kernel_size=1)
        self.width_output = nn.Conv2d(filters[3], 1, kernel_size=1)

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(module.weight, gain=1)

    def forward(
        self, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.features(value)
        return (
            self.pos_output(features),
            self.cos_output(features),
            self.sin_output(features),
            self.width_output(features),
        )

