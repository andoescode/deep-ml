"""Inference / deployment: load a checkpoint once, predict on images or tensors.

    from vision_pipeline.inference import Predictor

    p = Predictor.from_checkpoint("checkpoints/..._best.pt")
    p.predict_paths(["cat.png"], topk=3)

Segmentation checkpoints use the same class:

    p = Predictor.from_checkpoint("checkpoints/..._best.pt")
    p.segment_paths(["cat.jpg"])[0].save_mask("cat_mask.png")

The checkpoint carries its own Config, so the predictor rebuilds the right
architecture *and* the right eval transform for whichever dataset it was
trained on.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from . import segmentation
from .config import Config, DataConfig, ModelConfig, get_device
from .data import build_image_transform, get_dataset_module, task as task_of
from .models import build_model


def default_class_names(cfg: DataConfig) -> list[str] | None:
    """Human-readable names when the dataset module publishes them."""
    names = getattr(get_dataset_module(cfg.dataset), "CLASSES", None)
    return list(names) if names else None


@dataclass
class Segmentation:
    """One image's predicted mask, at the ORIGINAL image resolution.

    Logits are computed at the model's training resolution and then bilinearly
    upsampled back to the source size before thresholding — the same protocol
    published segmentation numbers use. Thresholding first and resizing the mask
    would step-quantize exactly the boundary that carries all the error.
    """

    mask: torch.Tensor  # (H, W) int64 class indices at the source resolution
    size: tuple[int, int]  # (width, height) of the source image
    class_names: list[str] | None = None

    @property
    def foreground_fraction(self) -> float:
        return (self.mask > 0).float().mean().item()

    def to_pil(self, palette: Sequence[tuple[int, int, int]] | None = None) -> Image.Image:
        """Colourize the mask for viewing (class 0 stays black)."""
        colors = palette or _DEFAULT_PALETTE
        rgb = torch.zeros(*self.mask.shape, 3, dtype=torch.uint8)
        for index in self.mask.unique().tolist():
            rgb[self.mask == index] = torch.tensor(
                colors[index % len(colors)], dtype=torch.uint8
            )
        return Image.fromarray(rgb.numpy(), mode="RGB")

    def save_mask(self, path: str | os.PathLike) -> str:
        """Write the raw class-index mask as an 8-bit PNG (0, 1, 2, ...)."""
        Image.fromarray(self.mask.to(torch.uint8).numpy(), mode="L").save(path)
        return str(path)

    def overlay(self, image: Image.Image, alpha: float = 0.5) -> Image.Image:
        """Blend the colourized mask over the source image."""
        base = image.convert("RGB")
        if base.size != self.size:
            base = base.resize(self.size)
        return Image.blend(base, self.to_pil().resize(base.size), alpha)

    def __repr__(self) -> str:
        name = (self.class_names or ["", "foreground"])[-1]
        return (
            f"Segmentation({self.size[0]}x{self.size[1]}, "
            f"{name}={self.foreground_fraction:.1%})"
        )


# Distinct, colour-blind-safe enough for a handful of classes.
_DEFAULT_PALETTE = (
    (0, 0, 0), (220, 50, 47), (38, 139, 210), (133, 153, 0),
    (181, 137, 0), (108, 113, 196), (42, 161, 152), (203, 75, 22),
)


@dataclass
class Prediction:
    """One image's top-k result."""

    labels: list[int]  # class indices, best first
    scores: list[float]  # softmax probabilities, aligned with labels
    names: list[str] | None = None  # human-readable names when available

    @property
    def top1(self) -> int:
        return self.labels[0]

    @property
    def top1_name(self) -> str:
        return self.names[0] if self.names else str(self.labels[0])

    def __repr__(self) -> str:
        head = self.names or [str(label) for label in self.labels]
        pairs = ", ".join(f"{n}={s:.3f}" for n, s in zip(head, self.scores))
        return f"Prediction({pairs})"


class Predictor:
    """Wrapper around a trained model held in eval mode."""

    def __init__(
        self,
        model: nn.Module,
        data_cfg: DataConfig | None = None,
        device: torch.device | None = None,
        class_names: Sequence[str] | None = None,
        amp: bool = True,
    ):
        self.device = device or get_device()
        self.data_cfg = data_cfg or DataConfig()
        # Image-only: a segmentation dataset's eval transform expects a pair.
        self.transform = build_image_transform(self.data_cfg)
        self.class_names = (
            list(class_names) if class_names else default_class_names(self.data_cfg)
        )
        self.task = task_of(self.data_cfg)
        self.amp = amp

        self.model = model.to(self.device).eval()
        self.model = self.model.to(memory_format=torch.channels_last)

    # Construction
    @classmethod
    def from_checkpoint(
        cls,
        path: str | os.PathLike,
        model_cfg: ModelConfig | None = None,
        data_cfg: DataConfig | None = None,
        device: torch.device | None = None,
        class_names: Sequence[str] | None = None,
        prefer_embedded: bool = True,
    ) -> "Predictor":
        """Rebuild the architecture and load weights.

        Checkpoints written by `train()` embed the config they were trained with,
        and by default that wins: `model_cfg` is the FALLBACK for raw state_dicts
        (the `_final.pt` files are bare `state_dict`s and carry nothing).

        Pass `prefer_embedded=False` to force `model_cfg` instead — but note the
        weights have to match whatever you force, so this is for deliberate
        surgery, not for correcting a guess. Deriving the architecture from the
        checkpoint is what lets `eval` work without re-specifying --arch,
        --base-channels and --up-mode exactly as the training run had them.
        """
        device = device or get_device()
        ckpt = torch.load(path, map_location=device, weights_only=False)

        state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        embedded = ckpt.get("config") if isinstance(ckpt, dict) else None
        has_embedded = bool(embedded and "model" in embedded)

        if has_embedded and (prefer_embedded or model_cfg is None):
            model_cfg = ModelConfig(**embedded["model"])
        elif model_cfg is None:
            raise ValueError(
                f"{path} carries no embedded config; pass model_cfg explicitly."
            )

        if data_cfg is None and embedded and "data" in embedded:
            data_cfg = DataConfig(**embedded["data"])

        model = build_model(model_cfg, device=device)
        # Strip torch.compile / DataParallel prefixes so compiled-run
        # checkpoints load into an uncompiled model.
        state = {
            k.replace("_orig_mod.", "").replace("module.", ""): v
            for k, v in state.items()
        }
        model.load_state_dict(state)

        return cls(model, data_cfg=data_cfg, device=device, class_names=class_names)

    # Prediction
    def _to_batch(self, images: Iterable[Image.Image]) -> torch.Tensor:
        # The CIFAR eval transform has no resize step of its own, so arbitrary
        # input images are squared off here before normalization. (Transforms
        # that do resize just see an already-correct size and no-op.)
        size = self.data_cfg.image_size
        tensors = []
        for img in images:
            img = img.convert("RGB")
            if min(img.size) != size or img.size[0] != img.size[1]:
                img = img.resize((size, size))
            tensors.append(self.transform(img))
        return torch.stack(tensors)

    @torch.inference_mode()
    def predict_batch(self, batch: torch.Tensor, topk: int = 3) -> list[Prediction]:
        """Predict on an already-transformed float batch of shape (N, C, H, W)."""
        batch = batch.to(self.device, non_blocking=True).contiguous(
            memory_format=torch.channels_last
        )

        with torch.amp.autocast("cuda", enabled=self.amp and self.device.type == "cuda"):
            logits = self.model(batch)

        probs = logits.float().softmax(dim=1)
        k = min(topk, probs.size(1))
        scores, labels = probs.topk(k, dim=1)

        results = []
        for row_scores, row_labels in zip(scores.tolist(), labels.tolist()):
            names = (
                [self.class_names[i] for i in row_labels] if self.class_names else None
            )
            results.append(Prediction(labels=row_labels, scores=row_scores, names=names))
        return results

    def predict_images(
        self, images: Iterable[Image.Image], topk: int = 3
    ) -> list[Prediction]:
        return self.predict_batch(self._to_batch(images), topk=topk)

    def predict_paths(
        self, paths: Sequence[str | os.PathLike], topk: int = 3, batch_size: int = 64
    ) -> list[Prediction]:
        """Predict on image files, chunked so large lists stay within memory."""
        results: list[Prediction] = []
        for start in range(0, len(paths), batch_size):
            chunk = paths[start : start + batch_size]
            images = [Image.open(Path(p)) for p in chunk]
            try:
                results.extend(self.predict_images(images, topk=topk))
            finally:
                for img in images:
                    img.close()
        return results

    # Segmentation
    @torch.inference_mode()
    def segment_batch(
        self,
        batch: torch.Tensor,
        sizes: Sequence[tuple[int, int]] | None = None,
        threshold: float = 0.5,
        hflip_tta: bool = False,
    ) -> list[Segmentation]:
        """Masks for an already-transformed batch, one per row.

        `sizes` are the source (width, height) pairs to upsample the logits back
        to; omit them to keep the model's own resolution. `hflip_tta` averages the
        logits with those of the mirrored input — a free ~0.2-0.5 Dice.
        """
        batch = batch.to(self.device, non_blocking=True).contiguous(
            memory_format=torch.channels_last
        )

        with torch.amp.autocast("cuda", enabled=self.amp and self.device.type == "cuda"):
            logits = self.model(batch)
            if hflip_tta:
                logits = logits + torch.flip(self.model(torch.flip(batch, dims=[-1])), dims=[-1])

        logits = logits.float()
        if hflip_tta:
            logits = logits / 2

        results = []
        for index in range(logits.size(0)):
            row = logits[index : index + 1]

            if sizes is not None:
                width, height = sizes[index]
                # Upsample the LOGITS, then threshold — see Segmentation's docstring.
                row = F.interpolate(
                    row, size=(height, width), mode="bilinear", align_corners=False
                )

            mask = segmentation.predict_masks(row, threshold=threshold)[0, 0].cpu()
            height, width = mask.shape
            results.append(
                Segmentation(
                    mask=mask, size=(width, height), class_names=self.class_names
                )
            )

        return results

    def segment_images(
        self, images: Iterable[Image.Image], native: bool = True, **kw
    ) -> list[Segmentation]:
        """Masks for PIL images; `native=False` returns them at model resolution."""
        images = [img.convert("RGB") for img in images]
        sizes = [img.size for img in images] if native else None
        batch = torch.stack([self.transform(img) for img in images])
        return self.segment_batch(batch, sizes=sizes, **kw)

    def segment_paths(
        self,
        paths: Sequence[str | os.PathLike],
        batch_size: int = 16,
        **kw,
    ) -> list[Segmentation]:
        """Masks for image files, chunked so large lists stay within memory."""
        results: list[Segmentation] = []
        for start in range(0, len(paths), batch_size):
            images = [Image.open(Path(p)) for p in paths[start : start + batch_size]]
            try:
                results.extend(self.segment_images(images, **kw))
            finally:
                for img in images:
                    img.close()
        return results

    # Export
    def export_torchscript(self, path: str | os.PathLike) -> str:
        """Trace to TorchScript for serving without the Python model code."""
        size = self.data_cfg.image_size
        example = torch.randn(1, 3, size, size, device=self.device).contiguous(
            memory_format=torch.channels_last
        )
        with torch.inference_mode():
            scripted = torch.jit.trace(self.model, example)
        scripted.save(str(path))
        return str(path)

    def export_onnx(self, path: str | os.PathLike, opset: int = 17) -> str:
        size = self.data_cfg.image_size
        example = torch.randn(1, 3, size, size, device=self.device)
        torch.onnx.export(
            self.model,
            example,
            str(path),
            opset_version=opset,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        )
        return str(path)


def evaluate_checkpoint(
    path: str | os.PathLike,
    cfg: Config | None = None,
    topk: tuple[int, ...] = (1,),
    prefer_embedded: bool = True,
) -> dict:
    """Metrics for a saved checkpoint on the held-out split.

    Returns {k: top-k accuracy} for classification, or the segmentation metric
    dict (pixel_acc, dice, iou, mean_iou) for segmentation.

    Note this scores at `cfg.data.image_size`, matching how training measured it.
    `Predictor.segment_paths` is the one that upsamples to native resolution.
    """
    from .data import build_loaders
    from .train import check_accuracy

    cfg = cfg or Config()
    predictor = Predictor.from_checkpoint(
        path, model_cfg=cfg.model, data_cfg=cfg.data, prefer_embedded=prefer_embedded
    )
    loaders = build_loaders(cfg.data)

    if predictor.task == "segmentation":
        return segmentation.evaluate(
            loaders.val, predictor.model, device=predictor.device
        )

    return check_accuracy(
        loaders.val, predictor.model, device=predictor.device, topk=topk
    )
