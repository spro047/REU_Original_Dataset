import shutil

import openpyxl
import pytest

from suture_scoring.data import Annotation, load_annotations, join_images_to_labels

SCORE_KEYS = ("Overall", "ISD", "Slack", "Position", "Angulation", "Width")


def _write_annotations(path, rows):
    """rows: list of (index, name, scores_dict). Scores are written as TEXT to mirror the real files."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append([None, "Name", "Overall", "ISD", "Slack", "Position", "Angulation", "Width"])
    for index, name, scores in rows:
        ws.append([index, name] + [str(scores[k]) for k in SCORE_KEYS])
    wb.save(path)


def test_load_annotations_casts_text_scores_to_ints(fixture_annotations):
    annotations = load_annotations(fixture_annotations)
    assert len(annotations) == 3
    for a in annotations:
        assert isinstance(a, Annotation)
        assert set(a.scores) == set(SCORE_KEYS)
        assert all(isinstance(v, int) for v in a.scores.values()), "scores must be int-cast"
        assert all(0 <= v <= 10 for v in a.scores.values())


def test_load_annotations_parses_filename_id_matching_index_column(fixture_annotations):
    annotations = load_annotations(fixture_annotations)
    for a in annotations:
        assert a.image_id == int(a.name.split("_")[1])


def test_index_mismatch_raises(tmp_path):
    bad = tmp_path / "bad.xlsx"
    _write_annotations(bad, [(0, "Image_0001_04_0_0_4.png", {"Overall": 5, "ISD": 6, "Slack": 7, "Position": 7, "Angulation": 7, "Width": 7})])
    with pytest.raises(ValueError, match="index"):
        load_annotations(bad)


def test_missing_index_raises(tmp_path):
    bad = tmp_path / "noindex.xlsx"
    _write_annotations(bad, [(None, "Image_0000_04_0_0_4.png", {"Overall": 5, "ISD": 6, "Slack": 7, "Position": 7, "Angulation": 7, "Width": 7})])
    with pytest.raises(ValueError, match="index"):
        load_annotations(bad)


def test_join_matches_all_images_to_labels(fixture_annotations, fixture_images_dir):
    annotations = load_annotations(fixture_annotations)
    result = join_images_to_labels(fixture_images_dir, annotations)
    assert len(result.matched) == 3
    assert result.missing_images == []
    assert result.missing_labels == []


def test_join_reports_unlisted_image(fixture_annotations, tmp_path, fixture_images_dir):
    tmp_images = tmp_path / "images"
    shutil.copytree(fixture_images_dir, tmp_images)
    extra = tmp_images / "Image_0099_04_0_0_4.png"
    shutil.copy(next(fixture_images_dir.glob("*.png")), extra)

    annotations = load_annotations(fixture_annotations)
    result = join_images_to_labels(tmp_images, annotations)
    assert result.missing_labels == [extra.name]
    assert result.missing_images == []


def test_join_reports_unlabeled_annotation(fixture_annotations, tmp_path, fixture_images_dir):
    tmp_images = tmp_path / "images"
    shutil.copytree(fixture_images_dir, tmp_images)
    (tmp_images / "Image_0001_04_0_0_4.png").unlink()

    annotations = load_annotations(fixture_annotations)
    result = join_images_to_labels(tmp_images, annotations)
    assert result.missing_images == ["Image_0001_04_0_0_4.png"]
    assert result.missing_labels == []