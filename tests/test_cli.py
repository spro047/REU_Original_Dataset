import csv
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCORE_COLS = ["Overall", "ISD", "Slack", "Position", "Angulation", "Width"]
CONF_COLS = [f"conf_{c}" for c in SCORE_COLS]


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "suture_scoring", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_cli_scores_directory_to_csv(fixture_images_dir, tmp_path):
    out = tmp_path / "scores.csv"
    result = run_cli("score", str(fixture_images_dir), "--out", str(out))
    assert result.returncode == 0, result.stderr

    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))

    expected_names = {p.name for p in fixture_images_dir.glob("*.png")}
    assert len(rows) == 3
    for row in rows:
        assert row["filename"] in expected_names
        for col in SCORE_COLS + CONF_COLS:
            assert col in row, f"missing column {col}"
        for col in SCORE_COLS:
            assert 0.0 <= float(row[col]) <= 10.0
        for col in CONF_COLS:
            assert 0.0 <= float(row[col]) <= 1.0


def test_cli_reports_missing_image_directory(tmp_path):
    out = tmp_path / "scores.csv"
    result = run_cli("score", str(tmp_path / "does_not_exist"), "--out", str(out))
    assert result.returncode != 0


def test_cli_skips_corrupt_image_and_keeps_good_ones(fixture_images_dir, tmp_path):
    images = tmp_path / "images"
    shutil.copytree(fixture_images_dir, images)
    (images / "corrupt.png").write_bytes(b"this is not a png")
    out = tmp_path / "scores.csv"
    result = run_cli("score", str(images), "--out", str(out))
    assert result.returncode == 0
    assert "corrupt.png" in result.stderr
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3  # good images only


def test_cli_rejects_non_positive_size(fixture_images_dir, tmp_path):
    out = tmp_path / "scores.csv"
    result = run_cli("score", str(fixture_images_dir), "--out", str(out), "--size", "0")
    assert result.returncode != 0