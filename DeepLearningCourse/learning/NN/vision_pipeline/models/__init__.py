"""Model registry.

Add a new architecture family as its own module (e.g. `vgg.py`, `vit.py`) and
register its builder here; `build_model` then reaches it from any ModelConfig.
"""
from __future__ import annotations

import torch
from torch import nn

from ..config import ModelConfig
from .cnn import CNN, build_cnn
from .resnet import (
    BasicResidualBlock,
    BottleneckResidualBlock,
    ResNet,
    build_resnet,
)

# arch name -> builder; builders accept the full kwarg set and ignore the rest.
REGISTRY = {
    "resnet18": build_resnet,
    "resnet34": build_resnet,
    "resnet50": build_resnet,
    "resnet101": build_resnet,
    "resnet152": build_resnet,
    "resnet_custom": build_resnet,
    "cnn": build_cnn,
}


def build_model(cfg: ModelConfig | None = None, device: torch.device | None = None) -> nn.Module:
    """Instantiate the architecture named by `cfg.arch` and move it to `device`."""
    cfg = cfg or ModelConfig()

    if cfg.arch not in REGISTRY:
        raise ValueError(f"Unknown arch {cfg.arch!r}. Known: {sorted(REGISTRY)}")

    if cfg.arch == "resnet_custom" and cfg.layers is None:
        raise ValueError("arch='resnet_custom' requires ModelConfig.layers")

    model = REGISTRY[cfg.arch](
        arch=cfg.arch,
        num_classes=cfg.num_classes,
        input_channels=cfg.input_channels,
        stem=cfg.stem,
        layers=cfg.layers,
        block=cfg.block,
        zero_init_residual=cfg.zero_init_residual,
        image_size=cfg.image_size,
        widths=cfg.widths,
        dropout=cfg.dropout,
        hidden_dim=cfg.hidden_dim,
        batch_norm=cfg.batch_norm,
    )

    if device is not None:
        model = model.to(device)

    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if cfg.compile:
        model = torch.compile(model, fullgraph=True)

    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


__all__ = [
    "REGISTRY",
    "BasicResidualBlock",
    "BottleneckResidualBlock",
    "CNN",
    "ResNet",
    "build_cnn",
    "build_model",
    "build_resnet",
    "count_parameters",
]
