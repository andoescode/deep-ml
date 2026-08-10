"""ImageNet-1K preprocessing: transforms, NumericImageFolder, balanced subsets.

Download notes:
  https://velog.io/@jasonlee1995/Linux-Server-Download-ImageNet-1K
"""
from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

from ..config import DataConfig

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

NUM_CLASSES = 1000
IMAGE_SIZE = 224

TRAIN_SUBDIR = "train"
VAL_SUBDIR = "val"


def build_train_transform(cfg: DataConfig) -> v2.Compose:
    steps = [
        v2.ToImage(),
        v2.RandomResizedCrop(
            size=(cfg.image_size, cfg.image_size),
            scale=cfg.crop_scale,
            ratio=cfg.crop_ratio,
            antialias=True,
        ),
        v2.RandomHorizontalFlip(p=cfg.hflip_p),
        # uint8 [0, 255] -> float32 [0, 1].
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=MEAN, std=STD),
    ]

    if cfg.random_erasing_p > 0:
        # Value 0 corresponds approximately to the normalized mean.
        steps.append(
            v2.RandomErasing(
                p=cfg.random_erasing_p, scale=(0.02, 0.20), ratio=(0.3, 3.3), value=0
            )
        )

    return v2.Compose(steps)


def build_eval_transform(cfg: DataConfig) -> v2.Compose:
    """Deterministic transform — used for val *and* for clean train accuracy."""
    return v2.Compose([
        v2.ToImage(),
        v2.Resize(cfg.resize_size, antialias=True),
        v2.CenterCrop((cfg.image_size, cfg.image_size)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=MEAN, std=STD),
    ])


class NumericImageFolder(ImageFolder):
    """ImageFolder that interprets numeric folder names numerically.

    Default ImageFolder sorts lexicographically, which maps '10' before '2' and
    silently scrambles the label space for '0'..'999' directories.
    """

    def find_classes(
        self,
        directory: str | Path,
    ) -> tuple[list[str], dict[str, int]]:
        classes = [entry.name for entry in os.scandir(directory) if entry.is_dir()]

        if not classes:
            raise FileNotFoundError(f"No class directories found in {directory}")

        try:
            classes = sorted(classes, key=int)
        except ValueError as exc:
            raise ValueError(
                "NumericImageFolder requires class directories named "
                "'0', '1', ..., '999'."
            ) from exc

        class_to_idx = {class_name: int(class_name) for class_name in classes}

        expected_classes = list(range(len(classes)))
        actual_classes = [int(name) for name in classes]

        if actual_classes != expected_classes:
            raise ValueError(
                "Class directories must form a continuous numeric range. "
                f"Expected 0..{len(classes) - 1}."
            )

        return classes, class_to_idx


def make_balanced_subset(
    dataset: ImageFolder,
    samples_per_class: int = 10,
    seed: int = 42,
) -> Subset:
    """Reproducible subset with the same number of samples from every class.

    For ImageNet-1K: 10 samples/class × 1,000 classes = 10,000 samples.
    """
    indices_by_class: dict[int, list[int]] = defaultdict(list)

    for sample_index, target in enumerate(dataset.targets):
        indices_by_class[target].append(sample_index)

    generator = torch.Generator().manual_seed(seed)
    selected_indices: list[int] = []

    for class_index in range(len(dataset.classes)):
        class_indices = indices_by_class[class_index]

        if len(class_indices) < samples_per_class:
            raise ValueError(
                f"Class {class_index} only contains {len(class_indices)} "
                f"samples, but {samples_per_class} were requested."
            )

        permutation = torch.randperm(len(class_indices), generator=generator)
        selected_indices.extend(
            class_indices[index]
            for index in permutation[:samples_per_class].tolist()
        )

    return Subset(dataset=dataset, indices=selected_indices)


def build_datasets(cfg: DataConfig):
    """Returns (train_dataset, val_dataset, eval_train_subset, classes)."""
    train_dir = os.path.join(cfg.root, TRAIN_SUBDIR)
    val_dir = os.path.join(cfg.root, VAL_SUBDIR)

    train_transform = build_train_transform(cfg)
    eval_transform = build_eval_transform(cfg)

    train_dataset = NumericImageFolder(root=train_dir, transform=train_transform)
    val_dataset = NumericImageFolder(root=val_dir, transform=eval_transform)
    # Same files as train_dataset but with the deterministic transform.
    eval_train_dataset = NumericImageFolder(root=train_dir, transform=eval_transform)

    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError("Training and validation class mappings do not match.")
    if train_dataset.class_to_idx != eval_train_dataset.class_to_idx:
        raise ValueError("Training and evaluation class mappings do not match.")

    eval_train_subset = make_balanced_subset(
        dataset=eval_train_dataset,
        samples_per_class=cfg.eval_samples_per_class,
        seed=cfg.seed,
    )

    return train_dataset, val_dataset, eval_train_subset, list(train_dataset.classes)
