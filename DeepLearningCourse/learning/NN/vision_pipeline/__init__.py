"""Vision pipeline — CIFAR-10, ImageNet-1K and Oxford-IIIT Pet share one codebase.

Modules:
    config.py        — dataclass configs, `Config.preset(...)`, seeding
    data/            — one file per dataset (cifar10, imagenet, oxford_pet) + loaders
    models/          — one file per architecture family (resnet, cnn, vit, unet)
    segmentation.py  — dense-prediction losses (BCE+Dice), Dice/IoU, mask previews
    train.py         — optimizer/scheduler/criterion, epoch loop, checkpoints
    inference.py     — Predictor for deployment, TorchScript/ONNX export
    cli.py           — `python -m vision_pipeline.cli train|eval|predict|segment`

Two tasks share the loop. The dataset module declares which one it is
(`TASK = "segmentation"`), and that selects the criterion, the target shape, and
the score used for checkpoint selection — everything else (warmup/cosine, AMP,
channels_last, resume, TensorBoard) is common. Otherwise the dataset only
changes the data module and the ResNet stem (`ModelConfig.stem`).

    python -m vision_pipeline.cli train --dataset cifar10       # 95.6% test acc
    python -m vision_pipeline.cli train --preset oxford_pet     # U-Net segmentation
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
from .data import Loaders, build_loaders, normalization, task
from .inference import Predictor, Segmentation, evaluate_checkpoint
from .models import Unet, ViT, build_model, build_unet, build_vit, count_parameters
from .segmentation import BCEDiceLoss, SoftDiceLoss, dice
from .train import (
    History,
    build_criterion,
    build_optimizer,
    build_scheduler,
    check_accuracy,
    confusion_matrix,
    evaluate,
    load_checkpoint,
    setup,
    top1,
    train,
)

__all__ = [
    "PRESETS", "Config", "DataConfig", "ModelConfig", "TrainConfig",
    "BCEDiceLoss", "History", "Loaders", "Predictor", "Segmentation",
    "SoftDiceLoss", "Unet", "ViT",
    "build_criterion", "build_loaders", "build_model", "build_optimizer",
    "build_scheduler", "build_unet", "build_vit", "check_accuracy",
    "confusion_matrix", "count_parameters", "dice", "evaluate",
    "evaluate_checkpoint", "get_device", "load_checkpoint", "normalization",
    "set_seed", "setup", "task", "top1", "train",
]
