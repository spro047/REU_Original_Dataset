"""Annotations loading and image-to-label joining.

The annotation files store every score cell as a text string (verified in the
real dataset), so all scores are int-cast here and never compared as strings
(lexicographic ordering misreports the range, hiding '10' behind '9').
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import openpyxl

SCORE_KEYS = ("Overall", "ISD", "Slack", "Position", "Angulation", "Width")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
_IMAGE_ID_RE = re.compile(r"Image_(\d+)_")


def list_images(image_dir: str | Path) -> List[Path]:
    """Sorted list of image files in a directory; non-image files are ignored."""
    return sorted(
        p for p in Path(image_dir).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


@dataclass(frozen=True)
class Annotation:
    """One row of expert annotation: filename plus integer scores on 0-10."""

    name: str
    scores: Dict[str, int]
    image_id: int = field(init=False)

    def __post_init__(self) -> None:
        match = _IMAGE_ID_RE.match(self.name)
        if match is None:
            raise ValueError(f"filename {self.name!r} does not encode an image id (Image_NNNN_...)")
        object.__setattr__(self, "image_id", int(match.group(1)))


@dataclass(frozen=True)
class JoinResult:
    """Outcome of joining an image directory against a set of annotations."""

    matched: Dict[str, Annotation]
    missing_images: List[str]  # annotations whose image file is absent
    missing_labels: List[str]  # image files with no annotation


def load_annotations(xlsx_path: str | Path) -> List[Annotation]:
    """Read the Train/Val annotations workbook (sheet 'Sheet1').

    Columns: index (0-based, equals the filename id), Name, then the six score
    columns. Score cells are stored as text and cast to int. Raises ValueError
    if the index column disagrees with the id encoded in the filename.
    """
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    try:
        ws = wb["Sheet1"]
        rows = ws.iter_rows(values_only=True)
        header = next(rows)
        name_col = header.index("Name")
        score_cols = {key: header.index(key) for key in SCORE_KEYS}

        annotations: List[Annotation] = []
        for row in rows:
            if row is None or row[name_col] is None:
                continue
            name = str(row[name_col]).strip()
            scores = {key: int(row[score_cols[key]]) for key in SCORE_KEYS}
            annotation = Annotation(name=name, scores=scores)
            if row[0] is None or int(row[0]) != annotation.image_id:
                raise ValueError(
                    f"index column {row[0]!r} does not match filename id "
                    f"{annotation.image_id} for {name}"
                )
            annotations.append(annotation)
        return annotations
    finally:
        wb.close()


def join_images_to_labels(image_dir: str | Path, annotations: List[Annotation]) -> JoinResult:
    """Match image files in a directory to annotations by filename.

    Only image suffixes are considered; non-image files (e.g. .DS_Store) are
    ignored. Missing files on either side are reported, never silently dropped.
    """
    image_names = {p.name for p in list_images(image_dir)}
    by_name = {a.name: a for a in annotations}
    matched = {name: ann for name, ann in by_name.items() if name in image_names}
    missing_images = sorted(set(by_name) - image_names)
    missing_labels = sorted(image_names - set(by_name))
    return JoinResult(matched=matched, missing_images=missing_images, missing_labels=missing_labels)