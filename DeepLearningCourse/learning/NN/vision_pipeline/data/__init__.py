"""Dataset registry + the shared DataLoader wiring.

Add a new dataset as its own module exposing `build_datasets(cfg)`,
`build_train_transform(cfg)`, `build_eval_transform(cfg)`, `MEAN` and `STD`,
then register it here.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from ..config import DataConfig
from . import cifar10, imagenet

REGISTRY = {
    "cifar10": cifar10,
    "imagenet": imagenet,
}


def get_dataset_module(name: str):
    if name not in REGISTRY:
        raise ValueError(f"Unknown dataset {name!r}. Known: {sorted(REGISTRY)}")
    return REGISTRY[name]


def normalization(cfg: DataConfig) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """(mean, std) of the configured dataset — for un-normalizing previews."""
    module = get_dataset_module(cfg.dataset)
    return module.MEAN, module.STD


def build_train_transform(cfg: DataConfig):
    return get_dataset_module(cfg.dataset).build_train_transform(cfg)


def build_eval_transform(cfg: DataConfig):
    return get_dataset_module(cfg.dataset).build_eval_transform(cfg)


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
    "build_loaders",
    "build_train_transform",
    "get_dataset_module",
    "normalization",
]
