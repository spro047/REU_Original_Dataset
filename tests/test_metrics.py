import numpy as np
import pytest

from suture_scoring.metrics import (
    ece,
    exact_match_accuracy,
    icc,
    mae,
    spearman,
    weighted_cohens_kappa,
    within_tolerance_accuracy,
)


def test_mae():
    assert mae([3, 5], [1, 7]) == pytest.approx(2.0)
    assert mae([4, 4, 4], [4, 4, 4]) == 0.0


def test_exact_match_accuracy():
    assert exact_match_accuracy([1, 2, 3], [1, 2, 2]) == pytest.approx(2 / 3)
    assert exact_match_accuracy([5], [5]) == 1.0


def test_within_tolerance_accuracy():
    # diffs: 1, 1, 2 -> two within tolerance 1
    assert within_tolerance_accuracy([1, 2, 3], [2, 1, 5], tolerance=1) == pytest.approx(2 / 3)
    assert within_tolerance_accuracy([1, 2, 3], [1, 2, 3], tolerance=0) == 1.0


def test_spearman_perfect_and_inverse():
    assert spearman([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) == pytest.approx(-1.0)


def test_spearman_constant_input_is_nan():
    assert np.isnan(spearman([1, 1, 1], [1, 2, 3]))


def test_ece_perfectly_calibrated_is_zero():
    conf = [0.5, 0.5, 0.5, 0.5]
    correct = [1, 1, 0, 0]  # bin accuracy equals mean confidence -> ECE 0
    assert ece(conf, correct, n_bins=1) == pytest.approx(0.0, abs=1e-12)


def test_ece_miscalibrated_is_positive():
    conf = [0.9, 0.9, 0.1, 0.1]
    correct = [1, 0, 1, 0]  # 50% accuracy claimed with 0.9/0.1 confidence
    assert ece(conf, correct, n_bins=2) > 0.0


def test_ece_numeric_fixture():
    # hand-computed: bin (0,0.5]: acc 0.5 conf 0.1 -> 0.4; bin (0.5,1]: acc 0.5 conf 0.9 -> 0.4
    # ECE = 0.5*0.4 + 0.5*0.4 = 0.4
    conf = [0.9, 0.9, 0.1, 0.1]
    correct = [1, 0, 1, 0]
    assert ece(conf, correct, n_bins=2) == pytest.approx(0.4)


def test_ece_validates_inputs():
    with pytest.raises(ValueError, match="length"):
        ece([0.5, 0.5], [1, 0, 1])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ece([1.5, 0.5], [1, 0])


def test_metrics_raise_on_empty_input():
    with pytest.raises(ValueError, match="empty"):
        mae([], [])
    with pytest.raises(ValueError, match="empty"):
        exact_match_accuracy([], [])
    with pytest.raises(ValueError, match="empty"):
        within_tolerance_accuracy([], [])
    with pytest.raises(ValueError, match="empty"):
        spearman([], [])
    with pytest.raises(ValueError, match="empty"):
        ece([], [])
    with pytest.raises(ValueError, match="empty"):
        weighted_cohens_kappa([], [])


# Classic Shrout & Fleiss (1979) worked example (also used in the irr R package docs):
# 6 targets x 4 judges. Published ICC(2,1) = 0.29.
SHROUT_FLEISS = np.array(
    [
        [9, 2, 5, 8],
        [6, 1, 3, 2],
        [8, 4, 6, 8],
        [7, 1, 2, 6],
        [10, 5, 6, 9],
        [6, 2, 4, 7],
    ],
    dtype=float,
)


def test_icc_perfect_agreement_is_one():
    ratings = np.tile(np.array([3.0, 7.0, 9.0]), (4, 1)).T  # 3 subjects x 4 identical raters
    assert icc(ratings) == pytest.approx(1.0, abs=1e-6)


def test_icc_shrout_fleiss_example():
    assert icc(SHROUT_FLEISS) == pytest.approx(0.29, abs=0.02)


def test_icc_constant_ratings_is_one():
    ratings = np.full((3, 4), 5.0)  # zero variance: convention returns 1.0
    assert icc(ratings) == 1.0


def test_icc_raises_with_fewer_than_two_raters():
    with pytest.raises(ValueError, match="raters"):
        icc(np.array([[3.0], [7.0]]))


def test_weighted_kappa_perfect_agreement_is_one():
    y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert weighted_cohens_kappa(y, y) == pytest.approx(1.0)


def test_weighted_kappa_hand_computed_quadratic():
    # 3 categories {0,1,2}, quadratic weights w = 1 - (i-j)^2/4
    # po = 0.9167, pe = 0.6667 -> kappa = 0.75 (hand-derived)
    y1 = [0, 0, 1, 1, 2, 2]
    y2 = [0, 1, 1, 2, 2, 2]
    assert weighted_cohens_kappa(y1, y2, weights="quadratic") == pytest.approx(0.75, abs=1e-6)


def test_weighted_kappa_linear_weights_differ():
    y1 = [0, 0, 1, 1, 2, 2]
    y2 = [0, 1, 1, 2, 2, 2]
    linear = weighted_cohens_kappa(y1, y2, weights="linear")
    quadratic = weighted_cohens_kappa(y1, y2, weights="quadratic")
    assert linear != quadratic
    assert 0.0 <= linear <= 1.0


def test_weighted_kappa_full_scale_weights_differ_from_inferred():
    # Same ratings, but weights over the full 0-10 scale vs the observed range
    y1 = [0, 0, 1, 1, 2, 2]
    y2 = [0, 1, 1, 2, 2, 2]
    inferred = weighted_cohens_kappa(y1, y2, weights="quadratic")
    full_scale = weighted_cohens_kappa(y1, y2, weights="quadratic", n_categories=11)
    assert inferred != full_scale
    assert 0.0 <= full_scale <= 1.0


def test_weighted_kappa_validates_scores():
    y1 = [0, 1, 2]
    with pytest.raises(ValueError, match="non-negative"):
        weighted_cohens_kappa(y1, [-1, 1, 2])
    with pytest.raises(ValueError, match="integers"):
        weighted_cohens_kappa([0.5, 1, 2], [0, 1, 2])