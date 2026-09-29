from suture_scoring.data import SCORE_KEYS, load_annotations
from suture_scoring.model import OrdinalSutureModel
from suture_scoring.score import fit_aggregation, score_cohort, write_csv


def test_fit_aggregation_and_score_cohort(fixture_annotations, fixture_images_dir, tmp_path):
    annotations = load_annotations(fixture_annotations)
    model = OrdinalSutureModel(backbone="tiny", embedding=16)

    aggregation = fit_aggregation(model, fixture_images_dir, annotations, size=64)
    rows = score_cohort(model, aggregation, fixture_images_dir, size=64)

    assert len(rows) == 3
    for row in rows:
        assert set(row) == {*SCORE_KEYS, *(f"conf_{k}" for k in SCORE_KEYS), "filename"}
        for k in SCORE_KEYS:
            assert 0 <= row[k] <= 10
            assert 0.0 <= row[f"conf_{k}"] <= 1.0

    out = tmp_path / "scores.csv"
    write_csv(rows, out)
    assert out.exists()
    header = out.read_text().splitlines()[0]
    assert header.split(",") == ["filename", *SCORE_KEYS, *(f"conf_{k}" for k in SCORE_KEYS)]