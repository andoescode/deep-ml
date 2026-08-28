"""Training: optimizer/scheduler construction, the epoch loop, checkpointing.

`train()` is the one entry point the notebook and the CLI both call. It returns
the per-epoch history so the notebook can plot/compare parameter tweaks without
re-reading TensorBoard.

The loop is task-agnostic. It asks the dataset module whether this is
classification or segmentation (`data.task`) and then routes three things:

    criterion — CrossEntropyLoss vs BCE+Dice (see segmentation.py)
    targets   — class indices vs (B, 1, H, W) label maps
    scoring   — top-1 accuracy vs Dice

Everything else — warmup/cosine, AMP, channels_last, checkpoint-on-best,
TensorBoard, resume — is shared, and `History.test_acc` holds whichever primary
score the task uses (`History.metric_name` says which).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader

from . import segmentation
from .config import Config, TrainConfig, get_device, set_seed
from .data import Loaders

CLASSIFICATION = "classification"
SEGMENTATION = "segmentation"


# Optimizer / scheduler / criterion
def split_decay_params(model: nn.Module) -> tuple[list, list]:
    """Split params into (weight-decayed, not-decayed).

    Weight decay goes on conv/linear weights ONLY: decaying BN/LayerNorm affine
    params and biases (everything with ndim == 1) fights the normalization
    instead of regularizing.

    A model can exempt extra params by name via a `no_weight_decay()` method —
    the ViT uses it for `cls_token` and `pos_embed`, which are ndim 3 and so
    would otherwise land in the decayed group.
    """
    exempt = model.no_weight_decay() if hasattr(model, "no_weight_decay") else set()

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        # torch.compile wraps the module, prefixing every param name.
        name = name.removeprefix("_orig_mod.")
        (no_decay if p.ndim == 1 or name in exempt else decay).append(p)
    return decay, no_decay


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    decay, no_decay = split_decay_params(model)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    name = cfg.optimizer.lower()
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=cfg.lr)
    if name == "sgd":
        # SGD + momentum is the canonical CIFAR ResNet recipe; the AdamW runs
        # showed the classic Adam generalization deficit (train 92.5 / test 82.5).
        return torch.optim.SGD(
            groups, lr=cfg.lr, momentum=cfg.momentum, nesterov=cfg.nesterov
        )
    raise ValueError(f"Unknown optimizer {cfg.optimizer!r}. Use 'sgd' or 'adamw'.")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainConfig):
    """Per-epoch LR schedule (stepped once per epoch in the loop)."""
    name = cfg.scheduler.lower()

    if name in ("none", ""):
        return None

    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=cfg.eta_min
        )

    if name == "warmup_cosine":
        if cfg.warmup_epochs <= 0:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.epochs, eta_min=cfg.eta_min
            )
        # Linear warmup (e.g. 0.01 -> 0.1) guards against early divergence at
        # lr=0.1, then cosine decay to eta_min.
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=cfg.warmup_start_factor,
            total_iters=cfg.warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, cfg.epochs - cfg.warmup_epochs), eta_min=cfg.eta_min
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[cfg.warmup_epochs]
        )

    raise ValueError(f"Unknown scheduler {cfg.scheduler!r}.")


def build_criterion(
    cfg: TrainConfig,
    task: str = CLASSIFICATION,
    num_classes: int = 10,
) -> nn.Module:
    """Loss for `task`, or for `cfg.criterion` when that is not "auto"."""
    name = cfg.criterion.lower()

    if name == "auto":
        name = "bce_dice" if task == SEGMENTATION else "cross_entropy"

    if name == "cross_entropy":
        if task == SEGMENTATION:
            # Dense CE: ignore_index skips the void band, and the loop squeezes
            # the channel dim off the mask before calling it.
            return segmentation.build_criterion(
                num_classes=max(2, num_classes),
                label_smoothing=cfg.label_smoothing,
            )
        # Label smoothing caps logit over-confidence; the printed loss floors near
        # ~0.5 instead of 0 — that is expected, not a bug.
        return nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    if name in ("bce", "bce_dice"):
        if task != SEGMENTATION:
            raise ValueError(f"criterion {name!r} is segmentation-only")
        # "bce" is the v1.0 recipe: no region term at all.
        return segmentation.build_criterion(
            num_classes=num_classes,
            dice_weight=cfg.dice_weight if name == "bce_dice" else 0.0,
        )

    raise ValueError(
        f"Unknown criterion {cfg.criterion!r}. Use 'auto', 'cross_entropy', "
        "'bce' or 'bce_dice'."
    )


# Evaluation
@torch.inference_mode()
def check_accuracy(
    loader: DataLoader,
    model: nn.Module,
    device: torch.device | None = None,
    amp: bool = True,
    topk: tuple[int, ...] = (1,),
) -> dict[int, float]:
    """Top-k accuracy over `loader`. Returns {k: accuracy}."""
    device = device or get_device()
    model.eval()

    correct = {k: 0 for k in topk}
    num_samples = 0
    max_k = max(topk)

    for x, y in loader:
        x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            scores = model(x)

        _, pred = scores.topk(max_k, dim=1)
        hits = pred.eq(y.unsqueeze(1))

        for k in topk:
            correct[k] += hits[:, :k].any(dim=1).sum().item()
        num_samples += y.size(0)

    return {k: correct[k] / max(1, num_samples) for k in topk}


def top1(loader: DataLoader, model: nn.Module, **kw) -> float:
    return check_accuracy(loader, model, topk=(1,), **kw)[1]


# The score the loop selects checkpoints on, per task.
PRIMARY_METRIC = {CLASSIFICATION: "acc", SEGMENTATION: "dice"}


def evaluate(
    loader: DataLoader,
    model: nn.Module,
    task: str = CLASSIFICATION,
    device: torch.device | None = None,
    amp: bool = True,
) -> dict[str, float]:
    """Metrics for `task`. Always contains PRIMARY_METRIC[task]."""
    if task == SEGMENTATION:
        return segmentation.evaluate(loader, model, device=device, amp=amp)

    return {"acc": top1(loader, model, device=device, amp=amp)}


@torch.inference_mode()
def confusion_matrix(
    loader: DataLoader,
    model: nn.Module,
    num_classes: int = 10,
    device: torch.device | None = None,
) -> torch.Tensor:
    """(num_classes, num_classes) counts, rows = true label, cols = prediction.

    Accumulated with bincount rather than a Python loop — at 1,000 ImageNet
    classes the per-sample loop dominates the forward pass.
    """
    device = device or get_device()
    model.eval()

    matrix = torch.zeros(num_classes * num_classes, dtype=torch.long, device=device)

    for x, y in loader:
        x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        preds = model(x).argmax(1)
        matrix += torch.bincount(
            y * num_classes + preds, minlength=num_classes * num_classes
        )

    return matrix.reshape(num_classes, num_classes).cpu()


# Checkpoints
def save_checkpoint(path: str, model, optimizer=None, scheduler=None, **extra) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {"model_state": model.state_dict(), **extra}
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer=None,
    scheduler=None,
    device: torch.device | None = None,
    fresh_schedule: bool = False,
) -> dict:
    """Restore weights (and optionally optimizer/scheduler/epoch) in place.

    Returns {"start_epoch": int, "best_acc": float} so training can continue.
    """
    device = device or get_device()
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if not (isinstance(ckpt, dict) and "model_state" in ckpt):
        # Raw state_dict (weights only).
        model.load_state_dict(ckpt)
        return {"start_epoch": 0, "best_acc": 0.0}

    model.load_state_dict(ckpt["model_state"])

    best_acc = float(ckpt.get("test_acc", 0.0))
    start_epoch = 0

    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    if not fresh_schedule:
        if scheduler is not None and "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1  # continue after saved epoch

    return {"start_epoch": start_epoch, "best_acc": best_acc}


# The loop
@dataclass
class History:
    """Per-epoch metrics — the return value the notebook plots.

    `train_acc`/`test_acc` hold the task's PRIMARY score, which is Dice for
    segmentation runs; `metric_name` says which. The names are kept as-is so
    existing classification notebooks keep working, and `test_metrics` carries
    the full per-epoch dict (pixel_acc, dice, iou, mean_iou) when there is one.
    """

    run_name: str
    best_acc: float = 0.0
    best_epoch: int = -1
    metric_name: str = "acc"
    epoch: list[int] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    test_acc: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    epoch_time: list[float] = field(default_factory=list)
    train_metrics: list[dict[str, float]] = field(default_factory=list)
    test_metrics: list[dict[str, float]] = field(default_factory=list)

    @property
    def gap(self) -> list[float]:
        """train - test primary score per epoch; the overfitting readout."""
        return [tr - te for tr, te in zip(self.train_acc, self.test_acc)]

    @property
    def best_score(self) -> float:
        """`best_acc` under a task-neutral name."""
        return self.best_acc

    def series(self, key: str, split: str = "test") -> list[float]:
        """One named metric's curve, e.g. history.series("iou")."""
        rows = self.test_metrics if split == "test" else self.train_metrics
        return [row[key] for row in rows if key in row]

    def to_frame(self):
        """pandas DataFrame of the per-epoch curves (pandas imported lazily)."""
        import pandas as pd

        frame = pd.DataFrame({
            "epoch": self.epoch,
            "loss": self.loss,
            f"train_{self.metric_name}": self.train_acc,
            f"test_{self.metric_name}": self.test_acc,
            "gap": self.gap,
            "lr": self.lr,
            "epoch_time": self.epoch_time,
        })

        # Secondary segmentation metrics, when the task produced any.
        for key in ("pixel_acc", "iou", "mean_iou"):
            values = self.series(key)
            if len(values) == len(self.epoch):
                frame[f"test_{key}"] = values

        return frame


def default_run_name(cfg: Config) -> str:
    # One directory per run so TensorBoard curves never overlap between runs.
    return cfg.train.run_name or f"{datetime.now():%Y%m%d-%H%M%S}_{cfg.model.arch}"


def run_dir_for(cfg: Config, run_name: str) -> str:
    """runs/<dataset>/<run_name> — keeps datasets from colliding in TensorBoard."""
    return os.path.join(cfg.train.run_dir, cfg.data.dataset, run_name)


def train(
    model: nn.Module,
    loaders: Loaders,
    cfg: Config | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    criterion: nn.Module | None = None,
    device: torch.device | None = None,
    start_epoch: int = 0,
    best_acc: float = 0.0,
    writer=None,
    verbose: bool = True,
) -> History:
    """Train `model` and return the per-epoch History.

    Any of optimizer/scheduler/criterion/writer may be supplied to override the
    ones built from `cfg` — handy for notebook experiments.
    """
    from .data import normalization, task as task_of

    cfg = cfg or Config()
    tcfg = cfg.train
    device = device or get_device()

    task = task_of(cfg.data)
    metric_name = PRIMARY_METRIC[task]
    segmenting = task == SEGMENTATION
    # Dense CE wants (B, H, W) targets; BCE wants the channel dim kept.
    squeeze_target = segmenting and cfg.model.num_classes > 1

    optimizer = optimizer or build_optimizer(model, tcfg)
    if scheduler is None:
        scheduler = build_scheduler(optimizer, tcfg)
    criterion = criterion or build_criterion(
        tcfg, task=task, num_classes=cfg.model.num_classes
    )

    run_name = default_run_name(cfg)
    owns_writer = writer is None
    if owns_writer:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(run_dir_for(cfg, run_name))
        writer.add_text("config", json.dumps(cfg.to_dict(), default=str))

    amp = tcfg.amp and device.type == "cuda"
    # AMP: forward in reduced precision, gradients rescaled to avoid underflow.
    # enabled= flags keep the whole loop runnable on CPU too.
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    os.makedirs(tcfg.checkpoint_dir, exist_ok=True)
    ckpt_best = os.path.join(tcfg.checkpoint_dir, f"{run_name}_best.pt")
    ckpt_final = os.path.join(tcfg.checkpoint_dir, f"{run_name}_final.pt")

    history = History(run_name=run_name, best_acc=best_acc, metric_name=metric_name)
    step = start_epoch * len(loaders.train)

    # Previews are un-normalized before display, so the mask rows are readable.
    preview_stats = normalization(cfg.data) if segmenting and cfg.data.normalize else (None, None)

    try:
        for epoch in range(start_epoch, tcfg.epochs):
            t0 = time.perf_counter()
            running_loss = 0.0
            model.train()

            for data, targets in loaders.train:
                data = data.to(device, non_blocking=True, memory_format=torch.channels_last)
                targets = targets.to(device, non_blocking=True)
                if squeeze_target:
                    targets = targets.squeeze(1)

                # Forward propagation
                with torch.amp.autocast("cuda", enabled=amp):
                    scores = model(data)
                    loss = criterion(scores, targets)

                # Backward propagation
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if tcfg.grad_clip:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()

                if step % tcfg.log_every == 0:
                    writer.add_scalar("Loss/train_batch", loss.item(), step)
                step += 1

            avg_epoch_loss = running_loss / max(1, len(loaders.train))

            # Train score on the clean fixed subset, test on the full test set.
            train_metrics = evaluate(loaders.eval_train, model, task, device=device, amp=amp)
            test_metrics = evaluate(loaders.val, model, task, device=device, amp=amp)
            train_acc = train_metrics[metric_name]
            test_acc = test_metrics[metric_name]

            if scheduler is not None:
                scheduler.step()
            # LR that the *next* epoch will use.
            current_lr = optimizer.param_groups[0]["lr"]

            epoch_time = time.perf_counter() - t0

            writer.add_scalar("Loss/train_epoch", avg_epoch_loss, epoch)
            # One scalar group per metric so TensorBoard overlays train vs test.
            for key, value in train_metrics.items():
                writer.add_scalar(f"{key}/train", value, epoch)
            for key, value in test_metrics.items():
                writer.add_scalar(f"{key}/test", value, epoch)
            writer.add_scalar(f"{metric_name}/gap", train_acc - test_acc, epoch)
            writer.add_scalar("LR", current_lr, epoch)
            writer.add_scalar("Time/epoch_sec", epoch_time, epoch)

            if segmenting and (
                epoch % max(1, tcfg.log_images_every) == 0 or epoch == tcfg.epochs - 1
            ):
                # Scalars do not show whether the masks look right.
                segmentation.log_predictions(
                    writer, model, loaders.val, epoch,
                    device=device, amp=amp,
                    mean=preview_stats[0], std=preview_stats[1],
                )

            history.epoch.append(epoch)
            history.loss.append(avg_epoch_loss)
            history.train_acc.append(train_acc)
            history.test_acc.append(test_acc)
            history.lr.append(current_lr)
            history.epoch_time.append(epoch_time)
            history.train_metrics.append(train_metrics)
            history.test_metrics.append(test_metrics)

            # Keep the best weights — cosine-to-zero means the last epoch is
            # usually the best, but this is insurance against surprises.
            if test_acc > history.best_acc:
                history.best_acc = test_acc
                history.best_epoch = epoch
                save_checkpoint(
                    ckpt_best, model, optimizer, scheduler,
                    epoch=epoch, test_acc=test_acc, test_metrics=test_metrics,
                    metric_name=metric_name, config=cfg.to_dict(),
                )

            if verbose:
                label = metric_name.capitalize()
                extra = (
                    f"Test IoU: {test_metrics['iou']:.4f}, " if segmenting else ""
                )
                print(
                    f"Epoch [{epoch + 1}/{tcfg.epochs}], "
                    f"Loss: {avg_epoch_loss:.4f}, "
                    f"Train {label}: {train_acc:.4f}, "
                    f"Test {label}: {test_acc:.4f}, "
                    f"{extra}"
                    f"LR: {current_lr:.6f}, "
                    f"{epoch_time:.1f}s",
                    flush=True,
                )
    finally:
        torch.save(model.state_dict(), ckpt_final)
        if owns_writer:
            writer.close()

    if verbose:
        print(f"Best test {metric_name}: {history.best_acc:.4f} ({ckpt_best})")

    return history


def setup(cfg: Config | None = None, resume: str | None = None, fresh_schedule: bool = False):
    """Build everything a run needs: (model, loaders, optimizer, scheduler,
    criterion, device, resume_state).

    The notebook calls this to get handles it can inspect before training.
    """
    from .data import build_loaders, num_classes as data_num_classes, task as task_of
    from .models import build_model, count_parameters

    cfg = cfg or Config()
    device = get_device()

    task = task_of(cfg.data)

    # Catch a head/target mismatch here rather than as a shape error 40 layers in:
    # oxford_pet's boundary="class" mode needs a 3-channel head, every other mode
    # needs 1. (Classification datasets have a fixed class count, so a mismatch
    # there is a config typo just the same.)
    required = data_num_classes(cfg.data)
    if required is not None and required != cfg.model.num_classes:
        raise ValueError(
            f"dataset {cfg.data.dataset!r} needs model.num_classes={required}, "
            f"got {cfg.model.num_classes}"
        )

    set_seed(cfg.data.seed)
    loaders = build_loaders(cfg.data)

    torch.backends.cudnn.benchmark = True  # fixed input size -> fastest kernels

    set_seed(cfg.train.seed)  # seed right before weight init
    model = build_model(cfg.model, device=device)

    optimizer = build_optimizer(model, cfg.train)
    scheduler = build_scheduler(optimizer, cfg.train)
    criterion = build_criterion(cfg.train, task=task, num_classes=cfg.model.num_classes)

    resume_state = {"start_epoch": 0, "best_acc": 0.0}
    if resume:
        resume_state = load_checkpoint(
            resume, model, optimizer, scheduler,
            device=device, fresh_schedule=fresh_schedule,
        )

    print(f"{device}{' ' + torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}")
    print(f"{count_parameters(model) / 1e6:.2f}M parameters")

    return model, loaders, optimizer, scheduler, criterion, device, resume_state
