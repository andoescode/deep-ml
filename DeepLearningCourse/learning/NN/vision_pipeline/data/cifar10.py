"""CIFAR-10 preprocessing: transforms and datasets.

One file per dataset; register new datasets in data/__init__.py.
"""
from __future__ import annotations

from torch.utils.data import Subset
from torchvision import datasets, transforms

from ..config import DataConfig

# Per-channel statistics of the CIFAR-10 train split (published values).
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

NUM_CLASSES = 10
IMAGE_SIZE = 32

CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def build_train_transform(cfg: DataConfig) -> transforms.Compose:
    steps = [
        # Geometric augmentation happens in PIL space, BEFORE ToTensor.
        transforms.RandomCrop(
            cfg.image_size,
            padding=cfg.crop_padding,
            padding_mode=cfg.crop_padding_mode,
        ),
        transforms.RandomHorizontalFlip(p=cfg.hflip_p),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]

    if cfg.random_erasing_p > 0:
        # Cutout-style occlusion on the normalized tensor — attacks memorization.
        steps.append(
            transforms.RandomErasing(
                p=cfg.random_erasing_p, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0
            )
        )

    return transforms.Compose(steps)


def build_eval_transform(cfg: DataConfig) -> transforms.Compose:
    """Deterministic transform — used for test *and* for clean train accuracy."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def build_datasets(cfg: DataConfig):
    """Returns (train_dataset, val_dataset, eval_train_subset, classes)."""
    train_transform = build_train_transform(cfg)
    eval_transform = build_eval_transform(cfg)

    train_dataset = datasets.CIFAR10(
        root=cfg.root, train=True, transform=train_transform, download=cfg.download
    )
    val_dataset = datasets.CIFAR10(
        root=cfg.root, train=False, transform=eval_transform, download=cfg.download
    )
    # Same files as train_dataset but with the deterministic transform.
    eval_train_dataset = datasets.CIFAR10(
        root=cfg.root, train=True, transform=eval_transform, download=cfg.download
    )

    # Fixed CLEAN subset of the train split for measuring train accuracy:
    # evaluating on the augmented train loader understates it (and a full 50k
    # pass every epoch is 5x the eval cost for no extra signal).
    size = min(cfg.eval_train_size, len(eval_train_dataset))
    eval_train_subset = Subset(eval_train_dataset, range(size))

    classes = list(train_dataset.classes or CLASSES)

    return train_dataset, val_dataset, eval_train_subset, classes
