"""Image-classification pipeline — CIFAR-10 and ImageNet-1K share one codebase.

Modules:
    config.py     — dataclass configs, `Config.preset("cifar10"|"imagenet")`, seeding
    data/         — one file per dataset (cifar10.py, imagenet.py) + loader wiring
    models/       — one file per architecture family (resnet.py, cnn.py, vit.py)
    train.py      — optimizer/scheduler/criterion, epoch loop, checkpoints
    inference.py  — Predictor for deployment, TorchScript/ONNX export
    cli.py        — `python -m vision_pipeline.cli train|eval|predict --dataset ...`

The dataset only changes the data module and the ResNet stem
(`ModelConfig.stem`); everything downstream is shared.
"""
from .config import (
    PRESETS,
    Config,
    DataConfig,
    ModelConfig,
    TrainConfig,
    get_device,
    set_seed,
)
from .data import Loaders, build_loaders, normalization
from .inference import Predictor, evaluate_checkpoint
from .models import ViT, build_model, build_vit, count_parameters
from .train import (
    History,
    build_criterion,
    build_optimizer,
    build_scheduler,
    check_accuracy,
    confusion_matrix,
    load_checkpoint,
    setup,
    top1,
    train,
)

__all__ = [
    "PRESETS", "Config", "DataConfig", "ModelConfig", "TrainConfig",
    "History", "Loaders", "Predictor", "ViT",
    "build_criterion", "build_loaders", "build_model", "build_optimizer",
    "build_scheduler", "build_vit", "check_accuracy", "confusion_matrix",
    "count_parameters",
    "evaluate_checkpoint", "get_device", "load_checkpoint", "normalization",
    "set_seed", "setup", "top1", "train",
]
