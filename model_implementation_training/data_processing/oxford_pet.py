"""
Oxford-IIIT Pet data pipeline for three-class segmentation.

Run inside your uv project (requires numpy, torch, torchvision):
```
    uv run data_processing.py
    uv run data_processing.py --image-size 256 --batch-size 16
    uv run data_processing.py --root dataset/ --no-download --num-workers 0
```

Import without downloading data or starting workers:
```
    from data_processing import build_data_loaders
    data = build_data_loaders()
    train_loader = data.train_loader
```
"""

import argparse
import os
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, tv_tensors
from torchvision.transforms import v2

from model_implementation_training.helpers.utils import set_seed, seed_worker

SEED = 42
IMAGE_SIZE = 128
RESIZE_SIZE = round(IMAGE_SIZE * 1.125)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CHANNELS = 3
NUM_CLASSES = 3
CLASS_NAMES = ("pet", "background", "boundary")
BATCH_SIZE = 32
VAL_FRACTION = 0.2
NUM_WORKERS = min(8, os.cpu_count() or 1)


class PreparePetSample:
    """Wrap images and masks so spatial transforms operate on both together."""

    def __call__(self, image, mask):
        image = tv_tensors.Image(image)
        mask = tv_tensors.Mask(mask)
        return image, mask


class ConvertToClassMask:
    """Map trimap IDs 1/2/3 to 0/1/2 and return an int64 [H,W] mask."""

    def __call__(self, image, mask):
        # 0 = pet, 1 = background, 2 = boundary.
        mask = mask.to(torch.long) - 1
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)
        if mask.ndim != 2:
            raise ValueError(f"Expected a 2D mask, got {tuple(mask.shape)}")
        if mask.min().item() < 0 or mask.max().item() >= NUM_CLASSES:
            raise ValueError("Mask contains unexpected class IDs.")
        return image, mask


def build_transforms(image_size=IMAGE_SIZE, resize_size=None):
    if resize_size is None:
        resize_size = round(image_size * 1.125)
    if image_size < 1 or resize_size < image_size:
        raise ValueError("Require image_size > 0 and resize_size >= image_size.")

    train_transform = v2.Compose([
        PreparePetSample(),
        v2.Resize((resize_size, resize_size), antialias=True),
        v2.RandomCrop((image_size, image_size)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype({
            tv_tensors.Image: torch.float32,
            tv_tensors.Mask: torch.long,
        }, scale=True),
        # Normalize images only; masks retain integer class labels.
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ConvertToClassMask(),
    ])
    eval_transform = v2.Compose([
        PreparePetSample(),
        v2.Resize((image_size, image_size), antialias=True),
        v2.ToDtype({
            tv_tensors.Image: torch.float32,
            tv_tensors.Mask: torch.long,
        }, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ConvertToClassMask(),
    ])
    return train_transform, eval_transform


def make_loader(dataset, shuffle=False, *, batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS, pin_memory=None, seed=SEED):
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
        drop_last=False,
    )


@dataclass
class PetData:
    """Datasets, loaders, and split indices for reuse in your training script."""

    device: torch.device
    train_dataset: Subset
    val_dataset: Subset
    test_dataset: datasets.OxfordIIITPet
    eval_train_dataset: Subset
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    eval_train_loader: DataLoader
    train_indices: list[int]
    val_indices: list[int]


def build_data_loaders(*, root="dataset/", image_size=IMAGE_SIZE,
                       resize_size=None, batch_size=BATCH_SIZE,
                       val_fraction=VAL_FRACTION, seed=SEED,
                       num_workers=NUM_WORKERS, download=True):
    """Build the original pipeline; defaults reproduce the supplied code.

    val_fraction applies only to the official trainval split. The official
    test split stays intact. Calling this function seeds the global RNGs
    and enables cuDNN benchmarking, matching the original setup.
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1.")
    if batch_size < 1 or num_workers < 0:
        raise ValueError("Require batch_size > 0 and num_workers >= 0.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    set_seed(seed)
    pin_memory = device.type == "cuda"
    train_transform, eval_transform = build_transforms(image_size, resize_size)

    train_base = datasets.OxfordIIITPet(
        root=root, split="trainval", target_types="segmentation",
        transforms=train_transform, download=download,
    )
    eval_base = datasets.OxfordIIITPet(
        root=root, split="trainval", target_types="segmentation",
        transforms=eval_transform, download=download,
    )
    test_dataset = datasets.OxfordIIITPet(
        root=root, split="test", target_types="segmentation",
        transforms=eval_transform, download=download,
    )

    split_generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_base), generator=split_generator).tolist()
    num_val = round(val_fraction * len(indices))
    if not 0 < num_val < len(indices):
        raise ValueError("The chosen val_fraction creates an empty train or validation split.")
    val_indices = indices[:num_val]
    train_indices = indices[num_val:]
    train_dataset = Subset(train_base, train_indices)
    val_dataset = Subset(eval_base, val_indices)
    eval_train_dataset = Subset(
        eval_base, train_indices[:min(1000, len(train_indices))],
    )

    loader_options = dict(batch_size=batch_size, num_workers=num_workers,
                          pin_memory=pin_memory, seed=seed)
    train_loader = make_loader(train_dataset, shuffle=True, **loader_options)
    val_loader = make_loader(val_dataset, **loader_options)
    test_loader = make_loader(test_dataset, **loader_options)
    eval_train_loader = make_loader(eval_train_dataset, **loader_options)
    return PetData(
        device=device,
        train_dataset=train_dataset, val_dataset=val_dataset,
        test_dataset=test_dataset, eval_train_dataset=eval_train_dataset,
        train_loader=train_loader, val_loader=val_loader,
        test_loader=test_loader, eval_train_loader=eval_train_loader,
        train_indices=train_indices, val_indices=val_indices,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="dataset/", help="Dataset directory (default: dataset/).")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--resize-size", type=int, default=None,
                        help="Training resize before crop; default: round(image_size * 1.125).")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION,
                        help="Validation fraction of trainval only (default: 0.2).")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--no-download", action="store_false", dest="download",
                        help="Require Oxford Pet to already exist in the dataset directory.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    data = build_data_loaders(**vars(args))
    print(data.device, torch.cuda.get_device_name(0) if data.device.type == "cuda" else "")
    print("Training images:", len(data.train_dataset))
    print("Validation images:", len(data.val_dataset))
    print("Test images:", len(data.test_dataset))
    print("Training evaluation subset:", len(data.eval_train_dataset))

    images, masks = next(iter(data.val_loader))
    print("Image shape:", images.shape)
    print("Mask shape:", masks.shape)
    print("Image dtype:", images.dtype)
    print("Mask dtype:", masks.dtype)
    print("Mask class IDs:", torch.unique(masks))


if __name__ == "__main__":
    main()
