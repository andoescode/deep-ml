"""Central configuration + reproducibility helpers.

Everything tunable lives here as a dataclass so a notebook can build variants
(`replace(cfg.train, lr=0.05)`) without editing module source.

Start from a preset rather than the bare defaults:

    cfg = Config.preset("cifar10")    # v2.0 recipe: resnet18 + SGD 0.1
    cfg = Config.preset("imagenet")   # resnet34 + AdamW 1e-3
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

    # imagenet: RandomResizedCrop bounds + val Resize -> CenterCrop.
    crop_scale: tuple[float, float] = (0.08, 1.0)
    crop_ratio: tuple[float, float] = (3 / 4, 4 / 3)
    resize_size: int = 256

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

    label_smoothing: float = 0.1

    scheduler: str = "warmup_cosine"  # "warmup_cosine" | "cosine" | "none"
    warmup_epochs: int = 5
    warmup_start_factor: float = 0.1
    eta_min: float = 0.0

    amp: bool = True
    grad_clip: float | None = None

    log_every: int = 50
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


PRESETS = {
    "cifar10": _cifar10_preset,
    "imagenet": _imagenet_preset,
}
