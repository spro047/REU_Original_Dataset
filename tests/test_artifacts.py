import csv

import numpy as np

from suture_scoring.artifacts import (
    calibration_grid,
    generate_all,
    scatter_grid,
    trainee_violin,
    val_metrics_table,
)
from suture_scoring.data import SCORE_KEYS, load_annotations


def _write_scores_csv(path, rows):
    header = ["filename", *SCORE_KEYS, *(f"conf_{k}" for k in SCORE_KEYS)]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _perfect_rows(annotations, conf=1.0):
    rows = []
    for a in annotations:
        rows.append(
            {
                "filename": a.name,
                **{k: float(v) for k, v in a.scores.items()},
                **{f"conf_{k}": conf for k in SCORE_KEYS},
            }
        )
    return rows


def test_val_metrics_table_perfect(fixture_annotations, tmp_path):
    annotations = load_annotations(fixture_annotations)
    scores_csv = tmp_path / "scores-Val.csv"
    _write_scores_csv(scores_csv, _perfect_rows(annotations))

    table = val_metrics_table(scores_csv, fixture_annotations)
    assert set(table) == set(SCORE_KEYS)
    for key in SCORE_KEYS:
        assert table[key]["mae"] == 0.0
        assert table[key]["exact"] == 1.0
        assert table[key]["tol1"] == 1.0
        assert table[key]["ece"] == 0.0


def test_val_metrics_table_rounds_fractional_predictions_for_exact(fixture_annotations, tmp_path):
    # fractional prediction that rounds to the expert score must count as exact
    annotations = load_annotations(fixture_annotations)
    rows = []
    for a in annotations:
        scores = {k: float(v) + 0.4 for k, v in a.scores.items()}  # rounds back to v
        rows.append({"filename": a.name, **scores, **{f"conf_{k}": 1.0 for k in SCORE_KEYS}})
    scores_csv = tmp_path / "scores-Val.csv"
    _write_scores_csv(scores_csv, rows)

    table = val_metrics_table(scores_csv, fixture_annotations)
    for key in SCORE_KEYS:
        assert table[key]["exact"] == 1.0


def test_figures_write_files(tmp_path):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 11, 40)
    p = np.clip(y + rng.normal(0, 1), 0, 10)
    scatter_grid({key: (y, p) for key in SCORE_KEYS}, tmp_path / "scatter-grid.png")
    calibration_grid({key: (np.full(40, 0.5), (y == np.round(p)).astype(float)) for key in SCORE_KEYS}, tmp_path / "calibration-grid.png")
    trainee_violin({"2week": p, "Resident": p + 1}, tmp_path / "violin.png")
    for name in ["scatter-grid.png", "calibration-grid.png", "violin.png"]:
        assert (tmp_path / name).exists()
        assert (tmp_path / name).stat().st_size > 0


def test_generate_all_fixture(fixture_annotations, tmp_path):
    annotations = load_annotations(fixture_annotations)
    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()
    _write_scores_csv(scores_dir / "scores-Val.csv", _perfect_rows(annotations))
    for level in ["2week", "4week", "Resident"]:
        _write_scores_csv(scores_dir / f"scores-{level}.csv", _perfect_rows(annotations, conf=0.8))

    output_dir = tmp_path / "artifacts"
    produced = generate_all(scores_dir, fixture_annotations, output_dir)

    expected = {"val-metrics.md", "scatter-grid.png", "calibration-grid.png", "trainee-levels.png", "agreement.md", "manifest.md"}
    assert set(produced) == expected
    for path in produced.values():
        assert path.exists()
    assert "Val metrics" in produced["val-metrics.md"].read_text()
    assert "not yet available" in produced["agreement.md"].read_text()