"""Dataset registry + the shared DataLoader wiring.

Add a new dataset as its own module exposing `build_datasets(cfg)`,
`build_train_transform(cfg)`, `build_eval_transform(cfg)`, `MEAN` and `STD`,
then register it here.

Optional module attributes:
    TASK                     — "classification" (assumed) or "segmentation"
    build_image_transform    — image-only eval transform, for inference on files
                               when the dataset's own transform is a paired one
    num_classes(cfg)         — when the head width depends on a data setting
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from ..config import DataConfig
from . import cifar10, imagenet, oxford_pet

REGISTRY = {
    "cifar10": cifar10,
    "imagenet": imagenet,
    "oxford_pet": oxford_pet,
}


def get_dataset_module(name: str):
    if name not in REGISTRY:
        raise ValueError(f"Unknown dataset {name!r}. Known: {sorted(REGISTRY)}")
    return REGISTRY[name]


def normalization(cfg: DataConfig) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """(mean, std) of the configured dataset — for un-normalizing previews."""
    module = get_dataset_module(cfg.dataset)
    return module.MEAN, module.STD


def task(cfg: DataConfig) -> str:
    """"classification" or "segmentation" — what the loop should optimize/score.

    Datasets that predate the distinction do not declare TASK, so the default
    keeps them classification.
    """
    return getattr(get_dataset_module(cfg.dataset), "TASK", "classification")


def num_classes(cfg: DataConfig) -> int | None:
    """Head width the dataset requires, when it depends on a data setting.

    Oxford-Pet needs this: `boundary="class"` turns a 1-logit binary head into a
    3-class one. None means "the dataset does not care", i.e. trust ModelConfig.
    """
    module = get_dataset_module(cfg.dataset)
    resolver = getattr(module, "num_classes", None)
    if callable(resolver):
        return resolver(cfg)
    return getattr(module, "NUM_CLASSES", None)


def build_train_transform(cfg: DataConfig):
    return get_dataset_module(cfg.dataset).build_train_transform(cfg)


def build_eval_transform(cfg: DataConfig):
    return get_dataset_module(cfg.dataset).build_eval_transform(cfg)


def build_image_transform(cfg: DataConfig):
    """Image-only eval transform for inference on bare files.

    Segmentation datasets use *paired* (image, mask) transforms that cannot be
    called with an image alone, so they publish a separate image-only version;
    for classification datasets the eval transform already is one.
    """
    module = get_dataset_module(cfg.dataset)
    builder = getattr(module, "build_image_transform", None)
    return builder(cfg) if builder else module.build_eval_transform(cfg)


def build_datasets(cfg: DataConfig):
    return get_dataset_module(cfg.dataset).build_datasets(cfg)


@dataclass
class Loaders:
    """The three loaders a run needs, plus the class names for reporting.

    `val` is the held-out split whatever the dataset calls it — CIFAR-10's test
    split is exposed as both `val` and `test`.
    """

    train: DataLoader
    val: DataLoader
    eval_train: DataLoader
    classes: list[str]

    @property
    def test(self) -> DataLoader:
        return self.val

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def summary(self) -> str:
        return (
            f"Training images:          {len(self.train.dataset):,}\n"
            f"Held-out images:          {len(self.val.dataset):,}\n"
            f"Evaluation subset images: {len(self.eval_train.dataset):,}\n"
            f"Number of classes:        {self.num_classes:,}"
        )


def build_loaders(cfg: DataConfig | None = None) -> Loaders:
    cfg = cfg or DataConfig()
    train_dataset, val_dataset, eval_train_subset, classes = build_datasets(cfg)

    multiproc = cfg.num_workers > 0
    eval_multiproc = cfg.eval_num_workers > 0

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers and multiproc,
        prefetch_factor=cfg.prefetch_factor if multiproc else None,
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.eval_num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers and eval_multiproc,
        prefetch_factor=cfg.prefetch_factor if eval_multiproc else None,
    )
    # No persistent workers here: this loader is short-lived per epoch.
    eval_train_loader = DataLoader(
        dataset=eval_train_subset,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.eval_num_workers,
        pin_memory=cfg.pin_memory,
    )

    return Loaders(
        train=train_loader,
        val=val_loader,
        eval_train=eval_train_loader,
        classes=classes,
    )


__all__ = [
    "REGISTRY",
    "Loaders",
    "build_datasets",
    "build_eval_transform",
    "build_image_transform",
    "build_loaders",
    "build_train_transform",
    "get_dataset_module",
    "normalization",
    "num_classes",
    "task",
]
