"""Inference / deployment: load a checkpoint once, predict on images or tensors.

    from vision_pipeline.inference import Predictor

    p = Predictor.from_checkpoint("checkpoints/..._best.pt")
    p.predict_paths(["cat.png"], topk=3)

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
from PIL import Image
from torch import nn

from .config import Config, DataConfig, ModelConfig, get_device
from .data import build_eval_transform, get_dataset_module
from .models import build_model


def default_class_names(cfg: DataConfig) -> list[str] | None:
    """Human-readable names when the dataset module publishes them."""
    names = getattr(get_dataset_module(cfg.dataset), "CLASSES", None)
    return list(names) if names else None


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
        self.transform = build_eval_transform(self.data_cfg)
        self.class_names = (
            list(class_names) if class_names else default_class_names(self.data_cfg)
        )
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
    ) -> "Predictor":
        """Rebuild the architecture and load weights.

        Checkpoints written by `train()` embed their config, so `model_cfg` is
        only needed for raw state_dicts or when overriding.
        """
        device = device or get_device()
        ckpt = torch.load(path, map_location=device, weights_only=False)

        state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        embedded = ckpt.get("config") if isinstance(ckpt, dict) else None

        if model_cfg is None:
            if embedded and "model" in embedded:
                model_cfg = ModelConfig(**embedded["model"])
            else:
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
        # input images are squared off here before normalization.
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
) -> dict[int, float]:
    """Top-k accuracy of a saved checkpoint on the held-out split."""
    from .data import build_loaders
    from .train import check_accuracy

    cfg = cfg or Config()
    predictor = Predictor.from_checkpoint(path, model_cfg=cfg.model, data_cfg=cfg.data)
    loaders = build_loaders(cfg.data)
    return check_accuracy(
        loaders.val, predictor.model, device=predictor.device, topk=topk
    )
