"""Command-line interface: score a directory of suturing images to a CSV."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from .data import SCORE_KEYS, list_images
from .model import ToyScorer
from .preprocess import DEFAULT_SIZE, preprocess_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="suture_scoring", description="Score suturing images for ISD, Slack, Position, Angulation, Width and Overall"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    score = sub.add_parser("score", help="score a directory of images to a CSV")
    score.add_argument("images_dir", type=Path, help="directory of suturing images")
    score.add_argument("--out", type=Path, required=True, help="output CSV path")
    score.add_argument("--size", type=int, default=DEFAULT_SIZE, help="preprocessing input size")
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "score":
        return cmd_score(parser, args)
    return 1  # unreachable: subparsers are required


if __name__ == "__main__":
    raise SystemExit(main())