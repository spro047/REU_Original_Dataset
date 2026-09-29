import openpyxl
import pytest

from suture_scoring.validate import (
    agreement,
    agreement_by_level,
    alignment_report,
    load_expert_labels,
)

SCORE_KEYS = ["Overall", "ISD", "Slack", "Position", "Angulation", "Width"]


def _write_expert_labels(path, rows):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append([None, "Name", *SCORE_KEYS])
    for index, name, scores in rows:
        ws.append([index, name] + [str(scores[k]) for k in SCORE_KEYS])
    wb.save(path)


def _scores_dict():
    return {"Overall": 5, "ISD": 6, "Slack": 7, "Position": 7, "Angulation": 7, "Width": 7}


def test_load_expert_labels_parses_test_names(tmp_path):
    path = tmp_path / "expert.xlsx"
    _write_expert_labels(
        path,
        [(1, "0001_Doc_01_Itr_01.png", _scores_dict()), (2, "0002_Doc_01_Itr_02.png", _scores_dict())],
    )
    parsed = load_expert_labels(path)
    assert set(parsed) == {"0001_Doc_01_Itr_01.png", "0002_Doc_01_Itr_02.png"}
    assert parsed["0001_Doc_01_Itr_01.png"]["Overall"] == 5
    assert all(isinstance(v, int) for s in parsed.values() for v in s.values())


def test_load_expert_labels_id_mismatch_raises(tmp_path):
    path = tmp_path / "bad.xlsx"
    _write_expert_labels(path, [(5, "0001_Doc_01_Itr_01.png", _scores_dict())])
    with pytest.raises(ValueError, match="index"):
        load_expert_labels(path)


def test_load_expert_labels_rejects_non_test_names(tmp_path):
    path = tmp_path / "bad.xlsx"
    _write_expert_labels(path, [(0, "Image_0000_04_0_0_4.png", _scores_dict())])
    with pytest.raises(ValueError, match="Doc"):
        load_expert_labels(path)


def test_agreement_perfect_is_one():
    scores = {
        "0001_Doc_01_Itr_01.png": {k: float(v) for k, v in _scores_dict().items()},
        "0002_Doc_01_Itr_02.png": {k: float(v) for k, v in _scores_dict().items()},
    }
    labels = {
        "0001_Doc_01_Itr_01.png": _scores_dict(),
        "0002_Doc_01_Itr_02.png": _scores_dict(),
    }
    result = agreement(scores, labels)
    assert set(result) == set(SCORE_KEYS)
    for key in SCORE_KEYS:
        assert result[key]["kappa"] == pytest.approx(1.0)
        assert result[key]["icc"] == pytest.approx(1.0, abs=1e-6)


def test_agreement_rejects_disjoint_inputs():
    scores = {"a.png": _scores_dict()}
    labels = {"b.png": _scores_dict()}
    with pytest.raises(ValueError, match="common"):
        agreement(scores, labels)


def test_agreement_clamps_out_of_range_predictions():
    scores = {"0001_Doc_01_Itr_01.png": {k: float(v) for k, v in _scores_dict().items()},
              "0002_Doc_01_Itr_02.png": {k: float(v) for k, v in _scores_dict().items()}}
    scores["0002_Doc_01_Itr_02.png"]["Overall"] = 12.0  # beyond the 0-10 scale
    labels = {
        "0001_Doc_01_Itr_01.png": _scores_dict(),
        "0002_Doc_01_Itr_02.png": _scores_dict(),
    }
    result = agreement(scores, labels)
    assert 0.0 <= result["Overall"]["kappa"] <= 1.0


def test_alignment_report_lists_unmatched():
    scores = {"a.png": _scores_dict(), "b.png": _scores_dict()}
    labels = {"b.png": _scores_dict(), "c.png": _scores_dict()}
    report = alignment_report(scores, labels)
    assert report == {"labels_without_scores": ["c.png"], "scores_without_labels": ["a.png"]}


def test_agreement_by_level_groups():
    scores = {"2week": {"0001_Doc_01_Itr_01.png": {k: float(v) for k, v in _scores_dict().items()},
                        "0002_Doc_01_Itr_02.png": {k: float(v) for k, v in _scores_dict().items()}},
              "Resident": {"0101_Doc_01_Itr_01.png": {k: float(v) for k, v in _scores_dict().items()},
                           "0102_Doc_01_Itr_02.png": {k: float(v) for k, v in _scores_dict().items()}}}
    labels = {"0001_Doc_01_Itr_01.png": _scores_dict(), "0002_Doc_01_Itr_02.png": _scores_dict(),
              "0101_Doc_01_Itr_01.png": _scores_dict(), "0102_Doc_01_Itr_02.png": _scores_dict()}
    result = agreement_by_level(scores, labels)
    assert set(result) == {"2week", "Resident"}
    assert result["2week"]["Overall"]["kappa"] == pytest.approx(1.0)