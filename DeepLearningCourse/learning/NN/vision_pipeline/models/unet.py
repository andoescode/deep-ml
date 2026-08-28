"""U-Net (Ronneberger et al. 2015) — encoder/decoder with skip connections.

Two deliberate departures from the paper, both needed to make it trainable here:

  * "same" padding instead of the paper's valid convs. The 1×1-crop U-Net shrinks
    a 128px input to nothing by depth 4 (128 -> 124 -> 62 -> 58 -> 29 -> 25 -> 12
    -> 8 -> 4 -> the bottleneck needs 4-4=0), so padding=1 it is.
  * BatchNorm after every conv. Not in the 2015 paper, but it is what makes the
    deeper stacks trainable at lr=1e-3.

`up_mode` picks the upsampling operator:

    "transpose": ConvTranspose2d(k=2, s=2) — the paper's "up-conv 2×2".
    "bilinear":  Upsample + 3×3 conv — fixed interpolation instead of a learned
                 stride-2 kernel, so no checkerboard artifacts in the mask. Costs
                 ~11% MORE parameters (a 3×3 conv is 2.25x a 2×2 transposed one):
                 31.04M -> 34.52M at base 64, depth 4. Worth it for smooth masks.

One file per architecture family; register new families in models/__init__.py.
"""
from __future__ import annotations

from torch import nn
import torch
import torchvision.transforms.v2.functional as TF


class Block(nn.Module):
    """(conv 3×3 -> BN -> ReLU) × 2 — the repeated unit of both halves.

    Convs carry no bias: the BatchNorm right after each one has its own shift,
    so a conv bias would be a dead parameter.
    """

    def __init__(self, in_channels: int, out_channels: int, drop_rate: float = 0.0):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]

        if drop_rate > 0:
            # Dropout2d drops whole channels: neighbouring pixels in a feature map
            # are correlated, so element-wise dropout leaks most of what it masks.
            layers.append(nn.Dropout2d(p=drop_rate))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    """Contracting path. Returns the bottom features plus one skip per stage."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        depth: int = 4,
        drop_rate: float = 0.0,
    ):
        super().__init__()

        self.hidden_channels = tuple(base_channels * (2 ** i) for i in range(depth))
        channels = (in_channels, *self.hidden_channels)  # (in, 64, 128, 256, 512)

        self.blocks = nn.ModuleList([
            Block(channels[i], channels[i + 1], drop_rate=drop_rate)
            for i in range(depth)
        ])

        self.down_scale = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip_features = []  # gather skipped features for the decoder

        for block in self.blocks:
            x = block(x)
            skip_features.append(x)  # pre-pool, i.e. full resolution for this stage
            x = self.down_scale(x)

        return x, skip_features

class Decoder(nn.Module):
    """Expansive path. Upsamples, concatenates the matching skip, then convolves."""

    def __init__(
        self,
        bottleneck_channels: int = 1024,
        base_channels: int = 64,
        depth: int = 4,
        up_mode: str = "transpose",
        drop_rate: float = 0.0,
    ):
        super().__init__()

        skip_channels = tuple(
            reversed(tuple(base_channels * (2 ** i) for i in range(depth)))
        )
        channels = (bottleneck_channels, *skip_channels)  # (1024, 512, 256, 128, 64)

        # in = channels[i + 1] * 2 because the upsampled output is concatenated
        # with the skip, and both carry channels[i + 1] channels.
        self.blocks = nn.ModuleList([
            Block(channels[i + 1] * 2, channels[i + 1], drop_rate=drop_rate)
            for i in range(depth)
        ])

        self.up_convs = nn.ModuleList([
            _make_up(channels[i], channels[i + 1], up_mode) for i in range(depth)
        ])

    def forward(self, x, skip_features):
        for upconv, block, skip in zip(
            self.up_convs, self.blocks, reversed(skip_features)
        ):
            x = upconv(x)
            encoder_feature = self.copy_and_crop(skip, x)
            x = torch.cat([x, encoder_feature], dim=1)
            x = block(x)

        return x

    def copy_and_crop(self, skip_feature, decoder_feature):
        # With "same" convs and an input divisible by 2 ** depth the sizes already
        # agree; the crop is the fallback for odd input sizes.
        target_size = decoder_feature.shape[-2:]

        if skip_feature.shape[-2:] == target_size:
            return skip_feature

        return TF.center_crop(skip_feature, list(target_size))


def _make_up(in_channels: int, out_channels: int, up_mode: str) -> nn.Module:
    if up_mode == "transpose":
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    if up_mode == "bilinear":
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    raise ValueError(f"Unknown up_mode {up_mode!r}. Use 'transpose' or 'bilinear'.")


class Unet(nn.Module):
    """U-Net for dense prediction.

    `num_classes=1` emits a single logit per pixel — binary segmentation via
    sigmoid + threshold, paired with BCE/Dice. `num_classes>=2` emits one logit
    per class per pixel, paired with CrossEntropyLoss.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 64,
        depth: int = 4,
        up_mode: str = "transpose",
        drop_rate: float = 0.0,
    ):
        super().__init__()

        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")

        self.depth = depth
        self.num_classes = num_classes
        # Inputs must be divisible by this or the skips misalign (the centre-crop
        # fallback papers over it, but the output is then smaller than the input).
        self.size_divisor = 2 ** depth

        self.encoder = Encoder(in_channels, base_channels, depth, drop_rate=drop_rate)

        encoder_channels = tuple(base_channels * (2 ** i) for i in range(depth))
        bottleneck_channels = base_channels * (2 ** depth)  # 1024 at base 64, depth 4

        self.bottle_neck = Block(
            encoder_channels[-1], bottleneck_channels, drop_rate=drop_rate
        )

        self.decoder = Decoder(
            bottleneck_channels, base_channels, depth,
            up_mode=up_mode, drop_rate=drop_rate,
        )

        # 1×1 conv, not a Linear: the output is a map, not a vector.
        self.fc = nn.Conv2d(base_channels, num_classes, kernel_size=1, stride=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        # The encoder returns (downsampled features, skips) — both are needed.
        x, skip_features = self.encoder(x)
        x = self.bottle_neck(x)
        x = self.decoder(x, skip_features)
        return self.fc(x)


# Width/depth presets
_VARIANTS: dict[str, dict] = {
    # The 2015 paper's widths — 31.04M params at num_classes=1.
    "unet": {"base_channels": 64, "depth": 4},
    # Half width, ~7.8M params. Within ~0.5 Dice of full width on Oxford-Pet and
    # 3x faster, which makes it the better choice for sweeps.
    "unet_small": {"base_channels": 32, "depth": 4},
    # 3 stages, quarter width — ~1.0M params, for smoke tests and CPU runs.
    "unet_tiny": {"base_channels": 16, "depth": 3},
}


def build_unet(
    arch: str = "unet",
    num_classes: int = 1,
    input_channels: int = 3,
    base_channels: int | None = None,
    depth: int | None = None,
    up_mode: str = "transpose",
    drop_rate: float = 0.0,
    image_size: int | None = None,
    **_ignored,
) -> Unet:
    """Build a U-Net by preset name; explicit args override the preset.

    `arch="unet_custom"` requires both `base_channels` and `depth`.
    """
    if arch == "unet_custom":
        if base_channels is None or depth is None:
            raise ValueError(
                "arch='unet_custom' requires both base_channels and depth"
            )
        preset = {}
    elif arch in _VARIANTS:
        preset = _VARIANTS[arch]
    else:
        raise ValueError(
            f"Unknown unet arch {arch!r}. Known: {sorted(_VARIANTS)}, 'unet_custom'."
        )

    base_channels = base_channels if base_channels is not None else preset["base_channels"]
    depth = depth if depth is not None else preset["depth"]

    # Caught here rather than as a confusing shape error 40 layers in.
    if image_size is not None and image_size % (2 ** depth):
        raise ValueError(
            f"image_size {image_size} is not divisible by 2**depth = {2 ** depth}; "
            f"use a multiple of {2 ** depth} or reduce depth."
        )

    return Unet(
        in_channels=input_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        depth=depth,
        up_mode=up_mode,
        drop_rate=drop_rate,
    )


__all__ = ["Block", "Decoder", "Encoder", "Unet", "build_unet"]
