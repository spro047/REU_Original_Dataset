"""Command-line interface: score a directory of suturing images to a CSV."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from .artifacts import generate_all
from .data import SCORE_KEYS, list_images, load_annotations
from .dataset import split_annotations
from .model import ToyScorer
from .preprocess import DEFAULT_SIZE, preprocess_image
from .score import fit_aggregation, score_cohort, write_csv
from .train import load_model, train_model
from .validate import agreement, alignment_report, load_expert_labels, read_scores_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="suture_scoring", description="Score suturing images for ISD, Slack, Position, Angulation, Width and Overall"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    score = sub.add_parser("score", help="score a directory of images to a CSV")
    score.add_argument("images_dir", type=Path, help="directory of suturing images")
    score.add_argument("--out", type=Path, required=True, help="output CSV path")
    score.add_argument("--size", type=int, default=DEFAULT_SIZE, help="preprocessing input size")
    score.add_argument("--checkpoint", type=Path, help="trained model checkpoint (uses real model + fitted Overall aggregation)")
    score.add_argument("--annotations", type=Path, help="Train annotations xlsx (required with --checkpoint, to fit the aggregation)")
    score.add_argument("--train-images", type=Path, help="Train images directory (required with --checkpoint)")

    validate = sub.add_parser("validate", help="AI-vs-expert agreement (ICC, weighted kappa) for a scored cohort")
    validate.add_argument("--scores", type=Path, required=True, help="scored cohort CSV from the score command")
    validate.add_argument("--labels", type=Path, required=True, help="expert annotations xlsx (application cohort naming)")
    validate.add_argument("--level", type=str, default="", help="report label only; per-level agreement is computed by running validate per cohort CSV")

    artifacts = sub.add_parser("artifacts", help="generate paper tables and figures from the scored cohort CSVs")
    artifacts.add_argument("--scores-dir", type=Path, required=True, help="directory with scores-<Cohort>.csv files")
    artifacts.add_argument("--val-annotations", type=Path, required=True, help="Val annotations xlsx (final numbers)")
    artifacts.add_argument("--output-dir", type=Path, required=True, help="directory to write tables and figures into")
    artifacts.add_argument("--expert-labels", type=Path, default=None, help="expert annotations xlsx for the application cohort (optional)")

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
    if args.checkpoint is not None:
        if args.annotations is None or args.train_images is None:
            parser.error("--checkpoint requires --annotations and --train-images to fit the Overall aggregation")
        if not args.train_images.is_dir():
            parser.error(f"train images directory not found: {args.train_images}")
        model = load_model(args.checkpoint)
        annotations = load_annotations(args.annotations)
        aggregation = fit_aggregation(model, args.train_images, annotations, size=args.size)
        rows = score_cohort(model, aggregation, args.images_dir, size=args.size)
        write_csv(rows, args.out)
        print(f"scored {len(rows)} images -> {args.out} (trained model)")
        return 0

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


def cmd_validate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if not args.scores.exists():
        parser.error(f"scores CSV not found: {args.scores}")
    if not args.labels.exists():
        parser.error(f"expert labels not found: {args.labels}")
    scores = read_scores_csv(args.scores)
    labels = load_expert_labels(args.labels)
    alignment = alignment_report(scores, labels)
    label = f" ({args.level})" if args.level else ""
    print(f"AI-vs-expert agreement{label} on {len(set(scores) & set(labels))} images")
    for kind, names in alignment.items():
        if names:
            print(f"note: {len(names)} {kind.replace('_', ' ')}")
    result = agreement(scores, labels)
    print(f"{'output':12s} {'icc':>8s} {'kappa':>8s}")
    for key in SCORE_KEYS:
        print(f"{key:12s} {result[key]['icc']:8.3f} {result[key]['kappa']:8.3f}")
    return 0


def cmd_artifacts(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if not args.scores_dir.is_dir():
        parser.error(f"scores directory not found: {args.scores_dir}")
    if not args.val_annotations.exists():
        parser.error(f"val annotations not found: {args.val_annotations}")
    produced = generate_all(args.scores_dir, args.val_annotations, args.output_dir, args.expert_labels)
    print(f"wrote {len(produced)} artifacts to {args.output_dir}:")
    for path in produced.values():
        print(f"  {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "score":
        return cmd_score(parser, args)
    if args.command == "train":
        return cmd_train(parser, args)
    if args.command == "validate":
        return cmd_validate(parser, args)
    if args.command == "artifacts":
        return cmd_artifacts(parser, args)
    return 1  # unreachable: subparsers are required


if __name__ == "__main__":
    raise SystemExit(main())