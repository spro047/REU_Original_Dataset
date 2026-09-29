"""Command-line interface: score a directory of suturing images to a CSV."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from .data import SCORE_KEYS, list_images, load_annotations
from .dataset import split_annotations
from .model import ToyScorer
from .preprocess import DEFAULT_SIZE, preprocess_image
from .train import train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="suture_scoring", description="Score suturing images for ISD, Slack, Position, Angulation, Width and Overall"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    score = sub.add_parser("score", help="score a directory of images to a CSV")
    score.add_argument("images_dir", type=Path, help="directory of suturing images")
    score.add_argument("--out", type=Path, required=True, help="output CSV path")
    score.add_argument("--size", type=int, default=DEFAULT_SIZE, help="preprocessing input size")

    train = sub.add_parser("train", help="train the ordinal suture model on the Train cohort")
    train.add_argument("--annotations", type=Path, required=True, help="Train annotations xlsx")
    train.add_argument("--images", type=Path, required=True, help="Train images directory")
    train.add_argument("--out", type=Path, required=True, help="checkpoint output path")
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--size", type=int, default=224, help="preprocessing input size")
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--backbone", choices=["tiny", "resnet18"], default="tiny")
    train.add_argument("--pretrained", action="store_true", help="use ImageNet weights (paper run)")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--val-frac", type=float, default=0.12, help="held-out slice of Train for model selection")
    return parser


def cmd_score(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if not args.images_dir.is_dir():
        parser.error(f"images directory not found: {args.images_dir}")
    if args.size <= 0:
        parser.error(f"invalid --size {args.size}: must be positive")
    image_paths = list_images(args.images_dir)
    if not image_paths:
        parser.error(f"no images found in {args.images_dir}")

    scorer = ToyScorer()
    header = ["filename", *SCORE_KEYS, *(f"conf_{key}" for key in SCORE_KEYS)]
    failures: list[str] = []
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for path in image_paths:
            try:
                image = preprocess_image(path, size=args.size)
            except Exception as exc:  # one bad file must not abort the whole batch
                failures.append(f"{path.name}: {exc}")
                continue
            row = {"filename": path.name, **scorer.score(image)}
            writer.writerow(row)

    for failure in failures:
        print(f"skipped {failure}", file=sys.stderr)
    print(f"scored {len(image_paths) - len(failures)} images -> {args.out}")
    if len(failures) == len(image_paths):
        return 1
    return 0


def cmd_train(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if not args.images.is_dir():
        parser.error(f"images directory not found: {args.images}")
    if not (0.0 < args.val_frac < 1.0):
        parser.error("--val-frac must be between 0 and 1")
    annotations = load_annotations(args.annotations)
    train_anns, val_anns = split_annotations(annotations, frac=args.val_frac, seed=args.seed)
    print(f"{len(train_anns)} train / {len(val_anns)} validation images (official Val untouched)")
    train_model(
        args.images,
        train_anns,
        val_anns,
        epochs=args.epochs,
        batch_size=args.batch_size,
        size=args.size,
        lr=args.lr,
        backbone=args.backbone,
        pretrained=args.pretrained,
        seed=args.seed,
        checkpoint=args.out,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "score":
        return cmd_score(parser, args)
    if args.command == "train":
        return cmd_train(parser, args)
    return 1  # unreachable: subparsers are required


if __name__ == "__main__":
    raise SystemExit(main())