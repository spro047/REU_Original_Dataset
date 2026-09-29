"""Integration checks against the real annotation files.

These lock the dataset invariants (row counts, text-typed scores, index == filename id)
so a future loader change cannot silently break them. Skipped if the data is absent.
"""
from pathlib import Path

import pytest

from suture_scoring.data import join_images_to_labels, load_annotations

REAL = {
    "train": Path("Dataset/Train/Train_cohort/Train_annotations.xlsx"),
    "val": Path("Dataset/Val/Validation_cohort/Validation_annotations.xlsx"),
}
REAL_DIRS = {
    "train": Path("Dataset/Train/Train_cohort/Train"),
    "val": Path("Dataset/Val/Validation_cohort/Validation"),
}

pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in REAL.values()),
    reason="real annotation files not present",
)


@pytest.mark.parametrize("name", ["train", "val"])
def test_real_annotation_invariants(name):
    annotations = load_annotations(REAL[name])
    assert len(annotations) == (1010 if name == "train" else 206)
    for a in annotations:
        assert all(0 <= v <= 10 for v in a.scores.values())
        assert all(isinstance(v, int) for v in a.scores.values())


@pytest.mark.parametrize("name", ["train", "val"])
def test_real_join_zero_mismatches(name):
    annotations = load_annotations(REAL[name])
    result = join_images_to_labels(REAL_DIRS[name], annotations)
    assert result.missing_images == []
    assert result.missing_labels == []
    assert len(result.matched) == (1010 if name == "train" else 206)