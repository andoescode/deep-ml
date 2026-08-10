"""CLI entry point — runs as a standalone process so training survives the
notebook kernel and DataLoader workers shut down cleanly on exit.

    python -m vision_pipeline.cli train --dataset cifar10
    python -m vision_pipeline.cli train --dataset imagenet --epochs 100
    python -m vision_pipeline.cli train --dataset cifar10 --arch cnn --lr 3e-4
    python -m vision_pipeline.cli train --dataset imagenet --resume checkpoints/<run>_best.pt
    python -m vision_pipeline.cli eval  --dataset cifar10 --checkpoint checkpoints/<run>_best.pt
    python -m vision_pipeline.cli predict --dataset cifar10 --checkpoint <ckpt> img.png

Every flag defaults to the dataset preset in config.PRESETS; only flags you
actually pass override it.
"""
from __future__ import annotations

import argparse
from dataclasses import replace

from .config import PRESETS, Config

# CLI flag -> which sub-config it belongs to.
DATA_FIELDS = ("root", "batch_size", "image_size", "num_workers", "random_erasing_p")
# image_size lands in both: it selects the crop size AND sizes the CNN's FC head.
MODEL_FIELDS = ("arch", "num_classes", "stem", "image_size")
TRAIN_FIELDS = (
    "epochs", "optimizer", "lr", "weight_decay", "label_smoothing",
    "warmup_epochs", "run_name",
)


def build_config(args) -> Config:
    """Start from the dataset preset, then apply only the flags that were given."""
    cfg = Config.preset(args.dataset)

    def given(fields):
        return {f: getattr(args, f) for f in fields if getattr(args, f, None) is not None}

    cfg = replace(
        cfg,
        data=replace(cfg.data, dataset=args.dataset, **given(DATA_FIELDS)),
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


def add_common(parser):
    """Flags shared by every subcommand. Defaults are None = 'use the preset'."""
    parser.add_argument("--dataset", default="cifar10", choices=sorted(PRESETS))
    parser.add_argument("--root", default=None, help="dataset root directory")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--random-erasing", dest="random_erasing_p", type=float, default=None)
    parser.add_argument("--arch", default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--stem", default=None, choices=["cifar", "imagenet"])
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
    p_train.add_argument("--warmup-epochs", type=int, default=None)
    p_train.add_argument("--run-name", default=None)
    p_train.add_argument("--resume", default=None, help="checkpoint to continue from")
    p_train.add_argument(
        "--fresh-schedule",
        action="store_true",
        help="reload weights but restart the LR schedule from epoch 0",
    )

    p_eval = sub.add_parser("eval", help="accuracy of a checkpoint on the held-out split")
    add_common(p_eval)
    p_eval.add_argument("--checkpoint", required=True)
    p_eval.add_argument("--topk", type=int, nargs="+", default=[1, 5])

    p_pred = sub.add_parser("predict", help="predict on image files")
    add_common(p_pred)
    p_pred.add_argument("--checkpoint", required=True)
    p_pred.add_argument("--topk", type=int, default=3)
    p_pred.add_argument("images", nargs="+")

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

        accs = evaluate_checkpoint(args.checkpoint, cfg, topk=tuple(args.topk))
        for k, v in sorted(accs.items()):
            print(f"top-{k}: {v:.4f}")
        return 0

    if args.command == "predict":
        from .inference import Predictor

        predictor = Predictor.from_checkpoint(
            args.checkpoint, model_cfg=cfg.model, data_cfg=cfg.data
        )
        for path, pred in zip(args.images, predictor.predict_paths(args.images, topk=args.topk)):
            print(f"{path}: {pred}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
