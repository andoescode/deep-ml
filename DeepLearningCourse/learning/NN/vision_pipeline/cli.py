"""CLI entry point — runs as a standalone process so training survives the
notebook kernel and DataLoader workers shut down cleanly on exit.

    python -m vision_pipeline.cli train --dataset cifar10
    python -m vision_pipeline.cli train --dataset imagenet --epochs 100
    python -m vision_pipeline.cli train --dataset cifar10 --arch cnn --lr 3e-4
    python -m vision_pipeline.cli train --preset imagenet_vit
    python -m vision_pipeline.cli train --preset imagenet_vit --arch vit_base --patch-size 32
    python -m vision_pipeline.cli train --dataset imagenet --resume checkpoints/<run>_best.pt
    python -m vision_pipeline.cli eval  --dataset cifar10 --checkpoint checkpoints/<run>_best.pt
    python -m vision_pipeline.cli predict --dataset cifar10 --checkpoint <ckpt> img.png

Segmentation works the same way — the dataset picks the task:

    python -m vision_pipeline.cli train --preset oxford_pet
    python -m vision_pipeline.cli train --preset oxford_pet_v1        # reproduce v1.0
    python -m vision_pipeline.cli train --preset oxford_pet --image-size 224 --epochs 60
    python -m vision_pipeline.cli train --preset oxford_pet --boundary class --num-classes 3
    python -m vision_pipeline.cli eval --dataset oxford_pet --checkpoint <ckpt>
    python -m vision_pipeline.cli segment --dataset oxford_pet --checkpoint <ckpt> \
        --out-dir masks/ --overlay cat.jpg

Every flag defaults to the dataset preset in config.PRESETS; only flags you
actually pass override it.
"""
from __future__ import annotations

import argparse
from dataclasses import replace

from .config import PRESETS, Config
from .data import REGISTRY as DATASETS
from .data.oxford_pet import BOUNDARY_MODES

# CLI flag -> which sub-config it belongs to.
DATA_FIELDS = (
    "root", "batch_size", "image_size", "num_workers", "random_erasing_p",
    "boundary", "presize", "normalize", "rotation_degrees", "color_jitter",
)
# image_size lands in both: it selects the crop size AND sizes the CNN's FC head
# / the ViT's patch grid + position table / the U-Net's divisibility check.
MODEL_FIELDS = (
    "arch", "num_classes", "stem", "image_size",
    "patch_size", "embed_dim", "depth", "num_heads", "mlp_dim", "drop_rate",
    "base_channels", "up_mode",
)
TRAIN_FIELDS = (
    "epochs", "optimizer", "lr", "weight_decay", "label_smoothing",
    "warmup_epochs", "grad_clip", "run_name", "criterion", "dice_weight",
)


def build_config(args) -> Config:
    """Start from the named preset, then apply only the flags that were given.

    `--preset` defaults to `--dataset`, so `--dataset imagenet` still means the
    imagenet preset; presets that are not dataset names (`imagenet_vit`) carry
    their own `data.dataset`, which `--dataset` can still override.
    """
    preset = args.preset or args.dataset or "cifar10"
    cfg = Config.preset(preset)

    def given(fields):
        return {f: getattr(args, f) for f in fields if getattr(args, f, None) is not None}

    cfg = replace(
        cfg,
        data=replace(cfg.data, dataset=args.dataset or cfg.data.dataset, **given(DATA_FIELDS)),
        model=replace(cfg.model, **given(MODEL_FIELDS)),
        train=replace(cfg.train, **given(TRAIN_FIELDS)),
    )

    if args.seed is not None:
        cfg = replace(
            cfg,
            data=replace(cfg.data, seed=args.seed),
            train=replace(cfg.train, seed=args.seed),
        )

    return cfg


def gave_model_flags(args) -> bool:
    """Did the user pass any architecture flag on this invocation?

    Decides whether eval/predict/segment rebuild from the checkpoint's own
    embedded config (the default, and almost always what you want) or from the
    preset plus flags. Without this, `eval --dataset oxford_pet` would rebuild
    the *preset* architecture and fail to load any run that changed --arch,
    --base-channels or --up-mode.
    """
    return any(getattr(args, field, None) is not None for field in MODEL_FIELDS)


def add_common(parser):
    """Flags shared by every subcommand. Defaults are None = 'use the preset'."""
    parser.add_argument("--dataset", default=None, choices=sorted(DATASETS))
    parser.add_argument(
        "--preset",
        default=None,
        choices=sorted(PRESETS),
        help="config preset to start from (default: same name as --dataset)",
    )
    parser.add_argument("--root", default=None, help="dataset root directory")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--random-erasing", dest="random_erasing_p", type=float, default=None)
    parser.add_argument("--arch", default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--stem", default=None, choices=["cifar", "imagenet"])
    # ViT knobs — unset means "use the arch preset's value".
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None, help="ViT blocks / U-Net stages")
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--mlp-dim", type=int, default=None)
    parser.add_argument("--drop-rate", type=float, default=None)
    # U-Net knobs.
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--up-mode", default=None, choices=["transpose", "bilinear"])
    # Segmentation data knobs.
    parser.add_argument(
        "--boundary",
        default=None,
        choices=list(BOUNDARY_MODES),
        help="oxford_pet: what trimap class 3 (the void band) becomes",
    )
    parser.add_argument(
        "--presize", type=int, default=None,
        help="Resize to this square, then RandomCrop (v1.0 geometry)",
    )
    parser.add_argument(
        "--no-normalize", dest="normalize", action="store_const", const=False,
        default=None, help="feed [0, 1] instead of mean/std-normalized input",
    )
    parser.add_argument("--rotation", dest="rotation_degrees", type=float, default=None)
    parser.add_argument("--color-jitter", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    # Present but unset on non-train subcommands so build_config can read them.
    parser.set_defaults(**{f: None for f in TRAIN_FIELDS})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vision_pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train or resume training")
    add_common(p_train)
    p_train.add_argument("--epochs", type=int, default=None)
    p_train.add_argument("--optimizer", default=None, choices=["adamw", "sgd"])
    p_train.add_argument("--lr", type=float, default=None)
    p_train.add_argument("--weight-decay", type=float, default=None)
    p_train.add_argument("--label-smoothing", type=float, default=None)
    p_train.add_argument(
        "--criterion", default=None,
        choices=["auto", "cross_entropy", "bce", "bce_dice"],
    )
    p_train.add_argument("--dice-weight", type=float, default=None)
    p_train.add_argument("--warmup-epochs", type=int, default=None)
    p_train.add_argument("--grad-clip", type=float, default=None)
    p_train.add_argument("--run-name", default=None)
    p_train.add_argument("--resume", default=None, help="checkpoint to continue from")
    p_train.add_argument(
        "--fresh-schedule",
        action="store_true",
        help="reload weights but restart the LR schedule from epoch 0",
    )

    p_eval = sub.add_parser("eval", help="score a checkpoint on the held-out split")
    add_common(p_eval)
    p_eval.add_argument("--checkpoint", required=True)
    p_eval.add_argument("--topk", type=int, nargs="+", default=[1, 5])

    p_pred = sub.add_parser("predict", help="predict on image files")
    add_common(p_pred)
    p_pred.add_argument("--checkpoint", required=True)
    p_pred.add_argument("--topk", type=int, default=3)
    p_pred.add_argument("images", nargs="+")

    p_seg = sub.add_parser("segment", help="write predicted masks for image files")
    add_common(p_seg)
    p_seg.add_argument("--checkpoint", required=True)
    p_seg.add_argument("--out-dir", default="masks", help="where to write the PNGs")
    p_seg.add_argument(
        "--overlay", action="store_true", help="also write a mask-over-image blend"
    )
    p_seg.add_argument(
        "--tta", action="store_true", help="average logits with the hflipped input"
    )
    p_seg.add_argument(
        "--model-resolution",
        action="store_true",
        help="keep masks at image_size instead of upsampling to the source size",
    )
    p_seg.add_argument("images", nargs="+")

    args = parser.parse_args(argv)
    cfg = build_config(args)

    if args.command == "train":
        from .train import setup, train

        model, loaders, optimizer, scheduler, criterion, device, resume_state = setup(
            cfg, resume=args.resume, fresh_schedule=args.fresh_schedule
        )
        print(loaders.summary(), flush=True)
        if args.resume:
            print(
                f"Resumed {args.resume}: start_epoch {resume_state['start_epoch']}, "
                f"best_acc {resume_state['best_acc']:.4f} "
                f"({'fresh' if args.fresh_schedule else 'resumed'} schedule)",
                flush=True,
            )
        train(
            model, loaders, cfg,
            optimizer=optimizer, scheduler=scheduler, criterion=criterion,
            device=device, **resume_state,
        )
        return 0

    if args.command == "eval":
        from .inference import evaluate_checkpoint

        metrics = evaluate_checkpoint(
            args.checkpoint, cfg, topk=tuple(args.topk),
            prefer_embedded=not gave_model_flags(args),
        )
        for key, value in sorted(metrics.items(), key=lambda kv: str(kv[0])):
            # Classification returns {k: acc}; segmentation returns named metrics.
            label = f"top-{key}" if isinstance(key, int) else key
            print(f"{label}: {value:.4f}")
        return 0

    if args.command == "predict":
        from .inference import Predictor

        predictor = Predictor.from_checkpoint(
            args.checkpoint, model_cfg=cfg.model, data_cfg=cfg.data,
            prefer_embedded=not gave_model_flags(args),
        )
        for path, pred in zip(args.images, predictor.predict_paths(args.images, topk=args.topk)):
            print(f"{path}: {pred}")
        return 0

    if args.command == "segment":
        import os
        from pathlib import Path

        from PIL import Image

        from .inference import Predictor

        predictor = Predictor.from_checkpoint(
            args.checkpoint, model_cfg=cfg.model, data_cfg=cfg.data,
            prefer_embedded=not gave_model_flags(args),
        )
        if predictor.task != "segmentation":
            raise SystemExit(
                f"{args.checkpoint} is a {predictor.task} model; use 'predict'."
            )

        os.makedirs(args.out_dir, exist_ok=True)
        masks = predictor.segment_paths(
            args.images, native=not args.model_resolution, hflip_tta=args.tta
        )

        for path, mask in zip(args.images, masks):
            stem = Path(path).stem
            out = os.path.join(args.out_dir, f"{stem}_mask.png")
            mask.save_mask(out)
            print(f"{path}: {mask} -> {out}")

            if args.overlay:
                blended = os.path.join(args.out_dir, f"{stem}_overlay.png")
                with Image.open(path) as source:
                    mask.overlay(source).save(blended)
                print(f"{path}: overlay -> {blended}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
