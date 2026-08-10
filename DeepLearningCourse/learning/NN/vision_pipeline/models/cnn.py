"""Plain conv stack — the v1.0 CIFAR-10 baseline from the notebook.

    input -> [conv -> gelu -> (dropout) -> maxpool] x N -> fc -> fc (logits)

Kept as the reference point the ResNet is measured against: v1.0 reached
Train 0.8922 / Test 0.7565.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class CNN(nn.Module):
    """Configurable 4-block conv net.

    Args:
        input_channels: channels in the input image (3 for CIFAR-10).
        num_classes: number of classes to predict.
        widths: output channels per conv block.
        dropout: dropout probability after each block (0 disables).
        hidden_dim: width of the single hidden FC layer.
        batch_norm: insert BatchNorm after each conv.
        image_size: spatial size of the input, used to size the FC input.
    """

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 10,
        widths: Sequence[int] = (16, 32, 64, 64),
        dropout: Sequence[float] = (0.0, 0.2, 0.3, 0.0),
        hidden_dim: int = 128,
        batch_norm: bool = False,
        image_size: int = 32,
    ):
        super().__init__()

        if len(dropout) != len(widths):
            raise ValueError("`dropout` must have one entry per entry in `widths`.")

        layers: list[nn.Module] = []
        in_channels = input_channels

        for index, (width, p) in enumerate(zip(widths, dropout)):
            layers.append(
                nn.Conv2d(in_channels, width, kernel_size=3, stride=1, padding=1)
            )
            if batch_norm:
                layers.append(nn.BatchNorm2d(width))
            layers.append(nn.GELU())
            if p > 0:
                layers.append(nn.Dropout(p))
            # No pool after the last block — it feeds the FC head directly.
            if index < len(widths) - 1:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = width

        self.features = nn.Sequential(*layers)

        # Derive the flattened width from a dry forward so changing `widths`
        # or `image_size` cannot silently desync the FC head.
        with torch.no_grad():
            probe = torch.zeros(1, input_channels, image_size, image_size)
            flat_dim = self.features(probe).flatten(1).shape[1]

        self.fc1 = nn.Linear(flat_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)  # output layer

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = F.gelu(self.fc1(x))
        return self.fc2(x)


def build_cnn(
    num_classes: int = 10,
    input_channels: int = 3,
    widths: Sequence[int] = (16, 32, 64, 64),
    dropout: Sequence[float] = (0.0, 0.2, 0.3, 0.0),
    hidden_dim: int = 128,
    batch_norm: bool = False,
    image_size: int = 32,
    **_ignored,
) -> CNN:
    return CNN(
        input_channels=input_channels,
        num_classes=num_classes,
        widths=widths,
        dropout=dropout,
        hidden_dim=hidden_dim,
        batch_norm=batch_norm,
        image_size=image_size,
    )
