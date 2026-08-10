"""ResNet (He et al. 2015), with a switchable stem.

    stem="imagenet": 7×7 s2 conv + 3×3 s2 maxpool — 224 -> 56, stages 56/28/14/7
    stem="cifar":    3×3 s1 conv, no maxpool      —  32 -> 32, stages 32/16/8/4

The stem is the only architectural difference between the two datasets. Using
the ImageNet stem on 32×32 inputs shrinks them to 8×8 before the first residual
block, leaving the deep stages with 2×2/1×1 maps; using the CIFAR stem on 224×224
inputs makes every stage 7× larger than it needs to be.

One file per architecture family; register new families in models/__init__.py.
"""
from __future__ import annotations

from typing import Sequence, Type

import torch
import torch.nn.functional as F
from torch import nn


class BasicResidualBlock(nn.Module):
    """Canonical basic block (resnet18 / resnet34), He et al. 2015:

    input > 3×3 Conv (stride) > BN > ReLU > 3×3 Conv (1) > BN > Add(identity) > ReLU

    Convs carry no bias: the BatchNorm right after each conv has its own shift,
    so a conv bias would be a dead parameter.

    CIFAR-10 is not complicated enough for the bottleneck block to make a
    difference — basic is the right default there.
    """

    expansion = 1

    def __init__(
        self,
        input_dim: int,  # number of channels
        planes: int,  # internal width per stage in net
        stride: int = 1,  # initial stride
        downsample: nn.Module | None = None,  # block of shortcut
    ):
        super().__init__()

        output_channels = planes * self.expansion

        self.residual = nn.Sequential(
            nn.Conv2d(input_dim, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),

            nn.Conv2d(planes, output_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
        )

        self.downsample = downsample if downsample else nn.Identity()

    @property
    def last_bn(self) -> nn.BatchNorm2d:
        """The BN whose gamma gets zero-initialised for identity-at-init."""
        return self.residual[4]

    def forward(self, x):
        identity = self.downsample(x)
        residual = self.residual(x)
        # Post-add activation: out = relu(F(x) + x). The nonlinearity must sit
        # AFTER the addition, otherwise blocks can only ever add non-negative values.
        return F.relu(identity + residual)


class BottleneckResidualBlock(nn.Module):
    """1×1 > 3×3 > 1×1 block (resnet50/101/152). Stride goes on the 3×3 (v1.5)."""

    expansion = 4

    def __init__(
        self,
        input_dim: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ):
        super().__init__()

        output_channels = planes * self.expansion

        self.residual = nn.Sequential(
            nn.Conv2d(input_dim, planes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),

            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),

            nn.Conv2d(planes, output_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(output_channels),
        )

        self.downsample = downsample if downsample else nn.Identity()

    @property
    def last_bn(self) -> nn.BatchNorm2d:
        return self.residual[7]

    def forward(self, x):
        identity = self.downsample(x)
        residual = self.residual(x)
        return F.relu(identity + residual)


BLOCKS: dict[str, Type[nn.Module]] = {
    "basic": BasicResidualBlock,
    "bottleneck": BottleneckResidualBlock,
}

STEMS = ("cifar", "imagenet")


def build_stem(stem: str, input_channels: int, planes: int = 64) -> tuple[nn.Module, nn.Module]:
    """Returns (conv1, pool) for the named stem."""
    if stem == "imagenet":
        conv1 = nn.Sequential(
            nn.Conv2d(input_channels, planes, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        )
        return conv1, nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    if stem == "cifar":
        # He et al. sec 4.2.
        conv1 = nn.Sequential(
            nn.Conv2d(input_channels, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        )
        return conv1, nn.Identity()

    raise ValueError(f"Unknown stem {stem!r}. Known: {STEMS}")


class ResNet(nn.Module):
    def __init__(
        self,
        Block: Type[nn.Module],
        layers: Sequence[int],
        num_classes: int = 10,
        input_channels: int = 3,
        stem: str = "cifar",
        zero_init_residual: bool = True,
    ):
        super().__init__()

        self.in_channels = 64
        self.stem = stem

        self.conv1, self.maxpool = build_stem(stem, input_channels, planes=64)

        # Stage strides are stem-independent: 1, 2, 2, 2.
        self.big_layers = nn.Sequential(
            self._make_layer(Block, planes=64, number_of_blocks=layers[0], stride=1),
            self._make_layer(Block, planes=128, number_of_blocks=layers[1], stride=2),
            self._make_layer(Block, planes=256, number_of_blocks=layers[2], stride=2),
            self._make_layer(Block, planes=512, number_of_blocks=layers[3], stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Single linear head: the pooled vector is already a summary,
        # a deep MLP here is pure memorization capacity.
        self.classification_head = nn.Linear(512 * Block.expansion, num_classes)

        # Kaiming fan-out init for ReLU conv stacks (PyTorch default is fan-in
        # with a=sqrt(5), which under-scales). BN starts at gamma=1, beta=0.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Zero-init the LAST BN gamma of each block so every block starts close to
        # an identity mapping (Goyal et al. 2017) — this makes early optimization
        # more stable at high LR and sometimes slightly improves accuracy.
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, (BasicResidualBlock, BottleneckResidualBlock)):
                    nn.init.zeros_(m.last_bn.weight)

    def _make_layer(
        self,
        Block: Type[nn.Module],
        planes: int,
        number_of_blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        output_channels = planes * Block.expansion

        layers = []
        downsample = None

        if stride != 1 or self.in_channels != output_channels:
            # Shortcut projection uses the SAME normalization as the residual
            # branch (BN), so both paths stay in one statistics regime.
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, output_channels,
                    kernel_size=1, stride=stride, padding=0, bias=False,
                ),
                nn.BatchNorm2d(output_channels),
            )

        layers.append(
            Block(
                input_dim=self.in_channels,
                planes=planes,
                downsample=downsample,
                stride=stride,
            )
        )

        self.in_channels = output_channels

        for _ in range(1, number_of_blocks):
            layers.append(Block(self.in_channels, planes=planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.big_layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # Flatten the tensor
        return self.classification_head(x)


# Depth presets
_DEPTHS: dict[str, tuple[str, list[int]]] = {
    "resnet18": ("basic", [2, 2, 2, 2]),
    "resnet34": ("basic", [3, 4, 6, 3]),
    "resnet50": ("bottleneck", [3, 4, 6, 3]),
    "resnet101": ("bottleneck", [3, 4, 23, 3]),
    "resnet152": ("bottleneck", [3, 8, 36, 3]),
}


def build_resnet(
    arch: str = "resnet18",
    num_classes: int = 10,
    input_channels: int = 3,
    stem: str = "cifar",
    layers: Sequence[int] | None = None,
    block: str = "basic",
    zero_init_residual: bool = True,
    **_ignored,
) -> ResNet:
    """Build a ResNet by preset name, or a custom depth via `layers`."""
    if layers is not None:
        block_name = block
    elif arch in _DEPTHS:
        block_name, layers = _DEPTHS[arch]
    else:
        raise ValueError(
            f"Unknown resnet arch {arch!r}. Known: {sorted(_DEPTHS)}, "
            "or pass explicit `layers`."
        )

    if block_name not in BLOCKS:
        raise ValueError(f"Unknown block {block_name!r}. Known: {sorted(BLOCKS)}")

    return ResNet(
        Block=BLOCKS[block_name],
        layers=layers,
        num_classes=num_classes,
        input_channels=input_channels,
        stem=stem,
        zero_init_residual=zero_init_residual,
    )
