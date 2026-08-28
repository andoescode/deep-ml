"""Oxford-IIIT Pet segmentation preprocessing: paired transforms and datasets.

The targets are *trimaps*, not labels, so every geometric transform has to be
applied identically to the image and the mask — hence torchvision v2 transforms
over `tv_tensors.Image` / `tv_tensors.Mask` pairs rather than the image-only
`transforms` used by cifar10.py and imagenet.py.

Trimap encoding (the raw PNG values): 1 = pet, 2 = background, 3 = boundary.

Class 3 is the annotators' "not classified" band, and it is **13.0% of all
pixels** — 29.1% of the pet-plus-boundary area. What you do with it changes the
task, so it is an explicit knob (`DataConfig.boundary`):

    "ignore"      -> 255, excluded from loss AND metric. The honest default;
                     comparable to how VOC/Cityscapes treat their void borders.
    "foreground"  -> pet. Reproduces the notebook's v1.0 run.
    "background"  -> background. Punishes any bleed past the outline.
    "class"       -> its own class in a 3-way task (num_classes=3).

Measured on the v1.0 checkpoint (see oxford_pet_analysis.ipynb S5a), the same
weights score: 0.9418 Dice under "ignore", 0.9367 under "foreground", and 0.7801
under "background". Two things follow. Excluding the void band *raises* the score
by ~0.5 points rather than lowering it — the band is the hard region, so dropping
it removes the pixels the model is worst at. And a model trained with
"foreground" predicts pet-plus-band, so scoring it against the animal's actual
outline costs 15.6 points. Pick a mode, then report the number it produces; these
three are not interchangeable.

One file per dataset; register new datasets in data/__init__.py.
"""
from __future__ import annotations

import torch
from torch.utils.data import Subset
from torchvision import datasets, tv_tensors
from torchvision.transforms import v2

from ..config import DataConfig
from ..segmentation import IGNORE_INDEX

# ImageNet statistics, not Oxford-Pet's own: these photos are ImageNet-like, and
# using the standard values keeps a pretrained encoder (the biggest lever on a
# 3,680-image train split) a drop-in change rather than a recalibration.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

TASK = "segmentation"
IMAGE_SIZE = 128

# Raw trimap values.
TRIMAP_PET = 1
TRIMAP_BACKGROUND = 2
TRIMAP_BOUNDARY = 3

BOUNDARY_MODES = ("ignore", "foreground", "background", "class")

BINARY_CLASSES = ("background", "pet")
TRIMAP_CLASSES = ("pet", "background", "boundary")


def num_classes(cfg: DataConfig) -> int:
    """Output channels the head needs: 1 logit for binary, 3 for the trimap task."""
    return 3 if cfg.boundary == "class" else 1


def classes(cfg: DataConfig) -> list[str]:
    return list(TRIMAP_CLASSES if cfg.boundary == "class" else BINARY_CLASSES)


class PreparePetSample:
    """Wrap the PIL pair as tv_tensors so v2 knows which is which.

    Without this the mask would be resampled bilinearly and normalized like an
    image; `tv_tensors.Mask` is what routes it to nearest-neighbour and exempts
    it from Normalize.
    """

    def __call__(self, image, mask):
        return tv_tensors.Image(image), tv_tensors.Mask(mask)


class RemapTrimap:
    """Trimap {1, 2, 3} -> training target, per `boundary` mode.

    Runs LAST, after every geometric op: remapping first would let a resize
    interpolate across label boundaries, and `RandomRotation`'s fill value is
    expressed in trimap space (see `_rotation_fill`).
    """

    def __init__(self, boundary: str = "ignore", ignore_index: int = IGNORE_INDEX):
        if boundary not in BOUNDARY_MODES:
            raise ValueError(
                f"Unknown boundary mode {boundary!r}. Known: {list(BOUNDARY_MODES)}"
            )
        self.boundary = boundary
        self.ignore_index = ignore_index

    def __call__(self, image, mask):
        # tv_tensors.Mask is already (1, H, W) — no unsqueeze. Adding one here is
        # what produced the (B, 1, 1, H, W) batches.
        if self.boundary == "class":
            # 1,2,3 -> 0,1,2 (pet, background, boundary).
            return image, (mask.long() - 1)

        target = (mask == TRIMAP_PET).long()

        if self.boundary == "foreground":
            target = (mask != TRIMAP_BACKGROUND).long()
        elif self.boundary == "ignore":
            target = torch.where(
                mask == TRIMAP_BOUNDARY,
                torch.full_like(target, self.ignore_index),
                target,
            )
        # "background" needs nothing further: pet==1, everything else 0.

        return image, tv_tensors.Mask(target)


def _rotation_fill() -> dict:
    """Fill for pixels rotated in from outside the frame.

    Type-keyed because the two halves need different values: 0 is black for the
    image, but on a raw trimap 0 is not a class at all — left as 0 it would remap
    to *pet* under `boundary="foreground"`, painting phantom animal into every
    corner. Background (2) is the honest fill.
    """
    return {tv_tensors.Image: 0, tv_tensors.Mask: TRIMAP_BACKGROUND}


def build_train_transform(cfg: DataConfig) -> v2.Compose:
    steps: list = [PreparePetSample()]

    if cfg.presize:
        # v1.0 geometry: resize to a fixed square, then crop. Destroys the aspect
        # ratio and applies a fixed presize/image_size zoom that the eval
        # transform does not, so train and test see different scale statistics.
        steps += [
            v2.Resize((cfg.presize, cfg.presize), antialias=True),
            v2.RandomCrop((cfg.image_size, cfg.image_size)),
        ]
    else:
        # Scale + aspect jitter in one op, and the eval transform's geometry is
        # inside its range — no systematic train/test scale mismatch.
        steps.append(
            v2.RandomResizedCrop(
                size=(cfg.image_size, cfg.image_size),
                scale=cfg.crop_scale,
                ratio=cfg.crop_ratio,
                antialias=True,
            )
        )

    steps.append(v2.RandomHorizontalFlip(p=cfg.hflip_p))

    if cfg.rotation_degrees:
        steps.append(
            v2.RandomRotation(degrees=cfg.rotation_degrees, fill=_rotation_fill())
        )

    if cfg.color_jitter:
        # Image-only by construction: v2.ColorJitter leaves Mask inputs alone.
        jitter = cfg.color_jitter
        steps.append(
            v2.ColorJitter(
                brightness=jitter, contrast=jitter, saturation=jitter, hue=jitter / 2
            )
        )

    steps += _finalize(cfg)
    return v2.Compose(steps)


def build_eval_transform(cfg: DataConfig) -> v2.Compose:
    """Deterministic paired transform — used for val *and* for clean train Dice."""
    return v2.Compose([
        PreparePetSample(),
        v2.Resize((cfg.image_size, cfg.image_size), antialias=True),
        *_finalize(cfg),
    ])


def _finalize(cfg: DataConfig) -> list:
    """Dtype conversion, optional normalization, then the trimap remap."""
    steps: list = [
        # Image -> float32 [0, 1]; the mask stays an integer label map (scaling
        # only applies to float conversions, so 1/2/3 survive intact).
        v2.ToDtype(
            {tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.long},
            scale=True,
        ),
    ]

    if cfg.normalize:
        steps.append(v2.Normalize(mean=MEAN, std=STD))

    steps.append(RemapTrimap(boundary=cfg.boundary))
    return steps


def build_image_transform(cfg: DataConfig) -> v2.Compose:
    """Image-only eval transform, for inference on files with no mask."""
    steps = [
        v2.ToImage(),
        v2.Resize((cfg.image_size, cfg.image_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
    ]
    if cfg.normalize:
        steps.append(v2.Normalize(mean=MEAN, std=STD))
    return v2.Compose(steps)


def build_datasets(cfg: DataConfig):
    """Returns (train_dataset, val_dataset, eval_train_subset, classes).

    Splits are the dataset's own: trainval (3,680) and test (3,669).
    """
    train_transform = build_train_transform(cfg)
    eval_transform = build_eval_transform(cfg)

    def pets(split: str, transforms):
        return datasets.OxfordIIITPet(
            root=cfg.root,
            split=split,
            target_types="segmentation",
            transforms=transforms,
            download=cfg.download,
        )

    train_dataset = pets("trainval", train_transform)
    val_dataset = pets("test", eval_transform)
    # Same files as train_dataset but with the deterministic transform.
    eval_train_dataset = pets("trainval", eval_transform)

    # Fixed CLEAN subset of the train split for measuring train Dice: evaluating
    # on the augmented train loader understates it, and a full pass every epoch
    # doubles the eval cost for no extra signal.
    size = min(cfg.eval_train_size, len(eval_train_dataset))
    eval_train_subset = Subset(eval_train_dataset, range(size))

    return train_dataset, val_dataset, eval_train_subset, classes(cfg)


__all__ = [
    "BOUNDARY_MODES",
    "MEAN",
    "STD",
    "TASK",
    "PreparePetSample",
    "RemapTrimap",
    "build_datasets",
    "build_eval_transform",
    "build_image_transform",
    "build_train_transform",
    "classes",
    "num_classes",
]
