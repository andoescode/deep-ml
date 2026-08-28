"""Segmentation-specific losses, metrics, and TensorBoard previews.

Kept out of train.py so the epoch loop stays task-agnostic: it asks this module
for a criterion and an `evaluate` function and otherwise does not know whether
it is predicting labels or masks.

Targets arrive as int64 masks of shape (B, 1, H, W) holding class indices, with
`ignore_index` (default 255) marking pixels excluded from BOTH loss and metric.
Oxford-IIIT Pet needs that: trimap class 3 is the annotators' "not classified"
band, and it is 13% of all pixels.

Metrics — pixel accuracy alone is misleading (on Oxford-Pet, predicting
all-background already scores 0.577), so Dice and IoU are the numbers to read.
The same four keys come back for both head shapes, but the axis they average
over differs, because that is what each regime's literature reports:

                 binary (1 logit/pixel)          multi-class (C logits/pixel)
    pixel_acc    correct / non-ignored pixels     same
    dice         positive-class Dice, per image   per-class Dice, macro mean
    iou          positive-class IoU, dataset-     frequency-weighted mean IoU
                 aggregated (micro)
    mean_iou     positive-class IoU, per image    per-class IoU, macro mean
                 then averaged (macro)            = the standard mIoU

For binary, `dice`/`mean_iou` and `iou` disagree by a few points because micro
IoU is dominated by large objects; papers usually report the per-image mean. All
four are logged so a comparison cannot silently mix conventions.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

IGNORE_INDEX = 255


# Losses
def _valid_mask(target: torch.Tensor, ignore_index: int) -> torch.Tensor:
    return target != ignore_index


class SoftDiceLoss(nn.Module):
    """1 - soft Dice on sigmoid probabilities (binary, one logit per pixel).

    BCE optimises per-pixel likelihood on the easy interior; Dice optimises the
    region overlap that Dice/IoU actually score. On Oxford-Pet all the remaining
    error is at the boundary, which is exactly where the two disagree.

    Computed per image and averaged, matching how the Dice metric is aggregated —
    a batch-summed variant would let large objects dominate the gradient.
    """

    def __init__(self, ignore_index: int = IGNORE_INDEX, eps: float = 1e-6):
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = _valid_mask(target, self.ignore_index).float()
        # Ignored pixels are zeroed on BOTH sides, so they contribute to neither
        # the intersection nor either cardinality.
        probs = torch.sigmoid(logits.float()) * valid
        truth = (target * valid.long()).clamp(max=1).float() * valid

        dims = tuple(range(1, probs.ndim))
        intersection = (probs * truth).sum(dims)
        cardinality = probs.sum(dims) + truth.sum(dims)

        dice = (2 * intersection + self.eps) / (cardinality + self.eps)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """BCE-with-logits + `dice_weight` × soft Dice, both ignore-aware.

    `dice_weight=0` is plain BCE (the v1.0 recipe); 0.5 is a good default.
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        pos_weight: torch.Tensor | None = None,
        ignore_index: int = IGNORE_INDEX,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        self.dice = SoftDiceLoss(ignore_index=ignore_index)
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = _valid_mask(target, self.ignore_index)
        truth = (target * valid.long()).clamp(max=1).float()

        # reduction="none" then a masked mean: reducing first would average the
        # ignored pixels' (meaningless) losses into the result.
        per_pixel = F.binary_cross_entropy_with_logits(
            logits.float(), truth, pos_weight=self.pos_weight, reduction="none"
        )
        denom = valid.sum().clamp(min=1)
        loss = (per_pixel * valid.float()).sum() / denom

        if self.dice_weight > 0:
            loss = loss + self.dice_weight * self.dice(logits, target)

        return loss


def build_criterion(
    num_classes: int = 1,
    dice_weight: float = 0.5,
    ignore_index: int = IGNORE_INDEX,
    label_smoothing: float = 0.0,
) -> nn.Module:
    """Criterion for a segmentation head with `num_classes` output channels."""
    if num_classes == 1:
        return BCEDiceLoss(dice_weight=dice_weight, ignore_index=ignore_index)

    # Multi-class: CrossEntropyLoss handles ignore_index natively. It wants
    # (B, C, H, W) logits against (B, H, W) int64 targets, so the loop squeezes
    # the channel dim off the mask before calling this.
    return nn.CrossEntropyLoss(
        ignore_index=ignore_index, label_smoothing=label_smoothing
    )


# Metrics
def predict_masks(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Logits -> int64 class-index mask of shape (B, 1, H, W).

    Single logit per pixel -> sigmoid + threshold; C >= 2 -> argmax over channels.
    """
    if logits.shape[1] == 1:
        return (torch.sigmoid(logits.float()) > threshold).long()
    return logits.float().argmax(dim=1, keepdim=True)


@torch.inference_mode()
def evaluate(
    loader: DataLoader,
    model: nn.Module,
    device: torch.device | None = None,
    amp: bool = True,
    threshold: float = 0.5,
    ignore_index: int = IGNORE_INDEX,
    eps: float = 1e-7,
) -> dict[str, float]:
    """Segmentation metrics over `loader`. See the module docstring.

    Binary heads are scored per image against the positive class; multi-class
    heads are scored from a confusion matrix, since "the positive class" is not
    defined once there are three of them.
    """
    from .config import get_device

    device = device or get_device()
    model.eval()

    num_correct = 0
    num_pixels = 0
    dice_sum = 0.0
    image_iou_sum = 0.0
    inter_sum = 0.0
    union_sum = 0.0
    num_images = 0
    confusion: torch.Tensor | None = None  # multi-class only
    channels = 1

    for x, y in loader:
        x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            logits = model(x)

        channels = logits.shape[1]
        preds = predict_masks(logits, threshold=threshold)
        valid = _valid_mask(y, ignore_index)

        if channels > 1:
            # (C, C) counts, rows = truth, cols = prediction, ignored pixels
            # dropped entirely rather than folded into any class.
            if confusion is None:
                confusion = torch.zeros(
                    channels * channels, dtype=torch.long, device=device
                )
            truth_flat = y[valid].reshape(-1)
            pred_flat = preds[valid].reshape(-1)
            confusion += torch.bincount(
                truth_flat * channels + pred_flat, minlength=channels * channels
            )
            num_correct += (pred_flat == truth_flat).sum().item()
            num_pixels += truth_flat.numel()
            num_images += y.size(0)
            continue

        truth = (y * valid.long()).clamp(max=1)
        preds = preds * valid.long()  # never credit or blame an ignored pixel

        num_correct += ((preds == truth) & valid).sum().item()
        num_pixels += valid.sum().item()

        dims = tuple(range(1, preds.ndim))
        intersection = (preds * truth).sum(dims)
        pred_sum = preds.sum(dims)
        target_sum = truth.sum(dims)
        union = pred_sum + target_sum - intersection

        dice_sum += ((2 * intersection + eps) / (pred_sum + target_sum + eps)).sum().item()
        image_iou_sum += ((intersection + eps) / (union + eps)).sum().item()
        inter_sum += intersection.sum().item()
        union_sum += union.sum().item()
        num_images += y.size(0)

    if confusion is not None:
        return _confusion_metrics(confusion.reshape(channels, channels).float(), eps)

    images = max(1, num_images)
    return {
        "pixel_acc": num_correct / max(1, num_pixels),
        "dice": dice_sum / images,
        "iou": inter_sum / max(union_sum, eps),
        "mean_iou": image_iou_sum / images,
    }


def _confusion_metrics(confusion: torch.Tensor, eps: float = 1e-7) -> dict[str, float]:
    """Per-class Dice/IoU from a (C, C) truth-by-prediction count matrix.

    Classes absent from the ground truth are dropped from the macro means rather
    than scored 0 — a class with no pixels in this split would otherwise drag
    mIoU down by 1/C for reasons that have nothing to do with the model.
    """
    true_positive = confusion.diag()
    truth_total = confusion.sum(dim=1)  # pixels of each class in the labels
    pred_total = confusion.sum(dim=0)  # pixels predicted as each class
    union = truth_total + pred_total - true_positive

    per_class_iou = true_positive / union.clamp(min=eps)
    per_class_dice = 2 * true_positive / (truth_total + pred_total).clamp(min=eps)

    present = truth_total > 0
    total = confusion.sum().clamp(min=eps)

    return {
        "pixel_acc": (true_positive.sum() / total).item(),
        "dice": per_class_dice[present].mean().item(),
        # Frequency-weighted, so it stays the "micro" counterpart of mean_iou.
        "iou": ((truth_total * per_class_iou)[present].sum() / total).item(),
        "mean_iou": per_class_iou[present].mean().item(),
    }


def dice(loader: DataLoader, model: nn.Module, **kw) -> float:
    """The primary score for model selection — mirrors train.top1."""
    return evaluate(loader, model, **kw)["dice"]


# Previews
@torch.inference_mode()
def log_predictions(
    writer,
    model: nn.Module,
    loader: DataLoader,
    epoch: int,
    device: torch.device | None = None,
    amp: bool = True,
    max_images: int = 8,
    mean: tuple[float, ...] | None = None,
    std: tuple[float, ...] | None = None,
    threshold: float = 0.5,
    ignore_index: int = IGNORE_INDEX,
) -> None:
    """Log image/ground-truth/prediction triplets to TensorBoard's IMAGES tab.

    Scalars alone do not show whether the masks look *right* — a model can hold a
    flat Dice while the failure mode shifts from missing legs to bleeding into the
    background. Row 1 = input, row 2 = ground truth, row 3 = prediction.
    """
    from .config import get_device

    device = device or get_device()
    model.eval()

    x, y = next(iter(loader))
    x = x[:max_images].to(device, non_blocking=True, memory_format=torch.channels_last)
    y = y[:max_images].to(device, non_blocking=True)

    with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
        logits = model(x)

    valid = _valid_mask(y, ignore_index).long()
    preds = predict_masks(logits, threshold=threshold)
    truth = y * valid  # ignored pixels render as class 0

    # Scale class indices into [0, 1] so a 3-class trimap shows as three greys
    # instead of two saturated ones.
    levels = max(1, logits.shape[1] - 1) if logits.shape[1] > 1 else 1
    preds = preds.float() / levels
    truth = truth.float() / levels

    images = x.float().cpu()
    if mean is not None and std is not None:
        # Undo Normalize, otherwise the preview row is grey soup.
        mean_t = torch.tensor(mean).view(1, -1, 1, 1)
        std_t = torch.tensor(std).view(1, -1, 1, 1)
        images = images * std_t + mean_t

    panel = torch.cat([
        images,
        truth.cpu().repeat(1, 3, 1, 1),
        preds.cpu().repeat(1, 3, 1, 1),
    ], dim=0)

    writer.add_images("Predictions/image_gt_pred", panel.clamp(0, 1), epoch)


__all__ = [
    "BCEDiceLoss",
    "IGNORE_INDEX",
    "SoftDiceLoss",
    "build_criterion",
    "dice",
    "evaluate",
    "log_predictions",
    "predict_masks",
]
