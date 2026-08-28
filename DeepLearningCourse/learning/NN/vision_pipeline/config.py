"""Central configuration + reproducibility helpers.

Everything tunable lives here as a dataclass so a notebook can build variants
(`replace(cfg.train, lr=0.05)`) without editing module source.

Start from a preset rather than the bare defaults:

    cfg = Config.preset("cifar10")     # v2.0 recipe: resnet18 + SGD 0.1
    cfg = Config.preset("imagenet")    # resnet34 + AdamW 1e-3
    cfg = Config.preset("oxford_pet")  # U-Net segmentation + BCE/Dice
"""
from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass, field, replace
from typing import Sequence

import numpy as np
import torch

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed every RNG so runs are comparable.

    CIFAR-10 seed variance (~±0.3-0.7%) is the same magnitude as many
    single-change effects, so call this before dataset construction *and* right
    before weight init.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class DataConfig:
    """Union of the knobs both datasets use.

    Each dataset module reads the fields that apply to it and ignores the rest,
    so one config type covers every dataset in the registry.
    """

    dataset: str = "cifar10"  # any key in data.REGISTRY
    root: str = "dataset/"
    download: bool = True  # cifar10 only

    image_size: int = 32
    batch_size: int = 128
    eval_batch_size: int = 256

    # Augmentation — shared.
    hflip_p: float = 0.5
    random_erasing_p: float = 0.0  # cutout-style occlusion; 0 disables

    # cifar10: RandomCrop with reflect padding.
    crop_padding: int = 4
    crop_padding_mode: str = "reflect"

    # imagenet, oxford_pet: RandomResizedCrop bounds + val Resize -> CenterCrop.
    crop_scale: tuple[float, float] = (0.08, 1.0)
    crop_ratio: tuple[float, float] = (3 / 4, 4 / 3)
    resize_size: int = 256

    # oxford_pet: Resize((presize, presize)) -> RandomCrop(image_size) instead of
    # RandomResizedCrop. Reproduces the v1.0 geometry; None uses RandomResizedCrop.
    presize: int | None = None
    normalize: bool = True
    rotation_degrees: float = 0.0
    color_jitter: float = 0.0  # brightness/contrast/saturation amount; hue gets half

    # oxford_pet: what to do with trimap class 3, the annotators' "not classified"
    # band (13% of all pixels). See data/oxford_pet.py — this changes the task.
    boundary: str = "ignore"  # "ignore" | "foreground" | "background" | "class"

    # Clean fixed subset of the train split used for train-accuracy readings.
    # cifar10 takes the first N samples; imagenet takes N per class.
    eval_train_size: int = 10_000
    eval_samples_per_class: int = 10

    num_workers: int = field(default_factory=lambda: min(8, os.cpu_count() or 1))
    eval_num_workers: int = 4
    pin_memory: bool = field(default_factory=torch.cuda.is_available)
    persistent_workers: bool = True
    prefetch_factor: int = 2

    seed: int = SEED


@dataclass
class ModelConfig:
    arch: str = "resnet18"  # any key in models.REGISTRY
    num_classes: int = 10
    input_channels: int = 3

    # ResNet-only knobs.
    # stem="cifar": 3×3 s1, no maxpool (32×32 inputs).
    # stem="imagenet": 7×7 s2 + 3×3 s2 maxpool (224×224 inputs).
    stem: str = "cifar"
    layers: Sequence[int] | None = None  # required for arch="resnet_custom"
    block: str = "basic"
    zero_init_residual: bool = True

    # CNN-only knobs. `image_size` sizes the FC head and must match
    # DataConfig.image_size; the ResNet is resolution-agnostic (adaptive pool).
    image_size: int = 32
    widths: Sequence[int] = (16, 32, 64, 64)
    dropout: Sequence[float] = (0.0, 0.2, 0.3, 0.0)
    hidden_dim: int = 128
    batch_norm: bool = False

    # U-Net-only knobs. `depth` below is shared with the ViT (it means "number of
    # downsampling stages" here, "number of encoder blocks" there); base_channels
    # None = "use the arch preset's value" (see models/unet.py _VARIANTS).
    # `image_size` must be divisible by 2 ** depth or the skips misalign.
    base_channels: int | None = None
    up_mode: str = "transpose"  # "transpose" (paper up-conv) | "bilinear" (no checkerboard)

    # ViT-only knobs. None = "use the arch preset's value" (see models/vit.py
    # _VARIANTS); `image_size` must match DataConfig.image_size, since pos_embed
    # has one row per patch and cannot be resized after init.
    patch_size: int | None = None
    embed_dim: int | None = None
    depth: int | None = None
    num_heads: int | None = None
    mlp_dim: int | None = None  # None -> embed_dim * mlp_ratio
    mlp_ratio: float = 4.0
    drop_rate: float = 0.1
    attention: str = "torch"  # "torch" (fused) | "custom" (inspectable)

    channels_last: bool = True
    compile: bool = False


@dataclass
class TrainConfig:
    epochs: int = 100
    optimizer: str = "adamw"  # "adamw" | "sgd"
    lr: float = 1e-3
    weight_decay: float = 1e-2
    momentum: float = 0.9  # sgd only
    nesterov: bool = True  # sgd only

    # "auto" picks CrossEntropyLoss for classification and BCE+Dice for
    # segmentation, from the dataset module's TASK.
    criterion: str = "auto"  # "auto" | "cross_entropy" | "bce" | "bce_dice"
    label_smoothing: float = 0.1
    dice_weight: float = 0.5  # segmentation only: BCE + dice_weight * soft Dice

    scheduler: str = "warmup_cosine"  # "warmup_cosine" | "cosine" | "none"
    warmup_epochs: int = 5
    warmup_start_factor: float = 0.1
    eta_min: float = 0.0

    amp: bool = True
    grad_clip: float | None = None

    log_every: int = 50
    log_images_every: int = 5  # segmentation only: epochs between mask previews
    run_dir: str = "runs"  # per-dataset subdir is appended by train()
    checkpoint_dir: str = "checkpoints"
    run_name: str | None = None  # auto-timestamped when None

    seed: int = SEED


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def preset(cls, name: str, **overrides) -> "Config":
        """A known-good starting config for `name` (see PRESETS)."""
        if name not in PRESETS:
            raise ValueError(f"Unknown preset {name!r}. Known: {sorted(PRESETS)}")
        cfg = PRESETS[name]()
        return replace(cfg, **overrides) if overrides else cfg


def _cifar10_preset() -> Config:
    """v2.0 SGD recipe — verified 95.61% test acc on an RTX 5080, ~6 min.

    SGD wins here: the AdamW run of the same recipe reached 94.14%, the classic
    Adam generalization deficit. ImageNet keeps AdamW (see _imagenet_preset).
    """
    return Config(
        data=DataConfig(
            dataset="cifar10",
            root="dataset/",
            image_size=32,
            batch_size=128,  # lr=0.1 below is calibrated to this batch size
            random_erasing_p=0.5,
            eval_train_size=10_000,
        ),
        model=ModelConfig(arch="resnet18", stem="cifar", num_classes=10, image_size=32),
        train=TrainConfig(
            epochs=100,
            optimizer="sgd",
            lr=0.1,
            weight_decay=5e-4,
            momentum=0.9,
            nesterov=True,
            label_smoothing=0.1,
            scheduler="warmup_cosine",
            warmup_epochs=5,
        ),
    )


def _imagenet_preset() -> Config:
    """AdamW recipe. SGD is only the better-performing choice on CIFAR-10."""
    return Config(
        data=DataConfig(
            dataset="imagenet",
            root="./dataset/Imagenet_1k_extract",
            image_size=224,
            resize_size=256,
            batch_size=128,
            eval_batch_size=128,
            eval_samples_per_class=10,
        ),
        model=ModelConfig(arch="resnet34", stem="imagenet", num_classes=1000, image_size=224),
        train=TrainConfig(
            epochs=100,
            optimizer="adamw",
            lr=1e-3,
            weight_decay=1e-2,
            label_smoothing=0.1,
            scheduler="warmup_cosine",
            warmup_epochs=5,
        ),
    )


def _imagenet_vit_preset() -> Config:
    """ViT-Small/16 on 224x224 — the from-scratch ViT recipe (Dosovitskiy et al.).

    Differs from the ResNet recipe in the ways ViTs need: lower lr, much higher
    weight decay (0.1), a longer warmup, and gradient clipping at 1.0 — an
    unclipped from-scratch ViT diverges in the first few hundred steps often
    enough to be worth the default.

    ViTs have no convolutional prior, so the augmentation column matters more
    here than it does for a ResNet; RandomErasing is on by default for that
    reason (Mixup/CutMix would be the next lever, and are not implemented yet).
    """
    return Config(
        data=DataConfig(
            dataset="imagenet",
            root="./dataset/Imagenet_1k_extract",
            image_size=224,
            resize_size=256,
            batch_size=128,
            eval_batch_size=128,
            random_erasing_p=0.25,
            eval_samples_per_class=10,
        ),
        model=ModelConfig(
            arch="vit_small",
            num_classes=1000,
            image_size=224,  # must equal data.image_size — pos_embed is sized here
            patch_size=16,  # -> 196 patches + 1 CLS token
            drop_rate=0.1,
        ),
        train=TrainConfig(
            epochs=100,
            optimizer="adamw",
            lr=3e-4,
            weight_decay=0.1,
            label_smoothing=0.1,
            scheduler="warmup_cosine",
            warmup_epochs=10,
            grad_clip=1.0,
        ),
    )


def _oxford_pet_preset() -> Config:
    """U-Net on Oxford-IIIT Pet — binary segmentation, the recommended recipe.

    Differs from the notebook's v1.0 run (see _oxford_pet_v1_preset) in the four
    ways the v1.0 post-mortem called for, in payoff order:

      * boundary="ignore" — trimap class 3 is excluded from loss and metric
        instead of counted as pet, matching how VOC/Cityscapes treat void borders.
        (Note: this RAISES the reported Dice by ~0.5 points, not lowers it — the
        void band is the hard region. The reason to switch is comparability, not
        conservatism. See oxford_pet_analysis.ipynb S5a.)
      * BCE + 0.5 x soft Dice, so the loss optimises the region overlap the metric
        scores rather than per-pixel likelihood alone. (Weaker justification than
        it first appeared: the analysis notebook S5f finds the boundary carries
        only ~13% of the error mass, most of it being whole-object failure.)
      * RandomResizedCrop + rotation + colour jitter, replacing v1.0's
        Resize(144) -> RandomCrop(128), which zoomed train but not test.
      * 60 epochs, not 100. v1.0's test Dice moved 0.2 points over its last 40
        epochs; the compute is better spent on image_size.

    up_mode="bilinear" avoids the checkerboard artifacts ConvTranspose2d leaves in
    the mask. It costs ~11% more parameters (31.04M -> 34.52M), not fewer — the
    3x3 conv that follows the upsample is bigger than the 2x2 transposed conv it
    replaces.

    image_size stays 128 for comparability with v1.0. Raising it to 224 (a
    multiple of 2**4) is the next single-line experiment worth running.
    """
    return Config(
        data=DataConfig(
            dataset="oxford_pet",
            root="dataset/",
            image_size=128,
            batch_size=32,
            eval_batch_size=64,
            boundary="ignore",
            crop_scale=(0.7, 1.0),
            crop_ratio=(3 / 4, 4 / 3),
            rotation_degrees=10.0,
            color_jitter=0.2,
            eval_train_size=1_000,
        ),
        model=ModelConfig(
            arch="unet",
            num_classes=1,  # one logit per pixel; boundary="class" needs 3
            image_size=128,
            up_mode="bilinear",
            drop_rate=0.0,  # NOT the lever here: v1.0's gap was 3 points
        ),
        train=TrainConfig(
            epochs=60,
            optimizer="adamw",
            lr=1e-3,
            weight_decay=1e-4,
            criterion="bce_dice",
            dice_weight=0.5,
            scheduler="warmup_cosine",
            warmup_epochs=5,
        ),
    )


def _oxford_pet_v1_preset() -> Config:
    """Exact reproduction of the notebook's v1.0 run.

    Verified: best test Dice 0.9367 (epoch 86), IoU 0.8964, PixAcc 0.9538 on an
    RTX 5080, ~6.1 s/epoch. Kept so that number stays reproducible after the
    default recipe moves on — but read the caveats in the notebook before
    comparing it to anything: boundary="foreground" and normalize=False are both
    non-standard, and it is scored at 128px rather than native resolution.
    """
    return Config(
        data=DataConfig(
            dataset="oxford_pet",
            root="dataset/",
            image_size=128,
            batch_size=32,
            eval_batch_size=64,
            boundary="foreground",  # trimap class 3 counted as pet
            normalize=False,  # v1.0 fed [0, 1] straight in
            presize=144,  # Resize(144) -> RandomCrop(128)
            eval_train_size=1_000,
        ),
        model=ModelConfig(
            arch="unet",
            num_classes=1,
            image_size=128,
            up_mode="transpose",
            drop_rate=0.0,
        ),
        train=TrainConfig(
            epochs=100,
            optimizer="adamw",
            lr=1e-3,
            weight_decay=1e-4,
            criterion="bce",  # dice_weight ignored
            scheduler="warmup_cosine",
            warmup_epochs=5,
        ),
    )


PRESETS = {
    "cifar10": _cifar10_preset,
    "imagenet": _imagenet_preset,
    "imagenet_vit": _imagenet_vit_preset,
    "oxford_pet": _oxford_pet_preset,
    "oxford_pet_v1": _oxford_pet_v1_preset,
}
