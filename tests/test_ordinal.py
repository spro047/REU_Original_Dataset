import pytest
import torch

from suture_scoring.ordinal import (
    NUM_CATEGORIES,
    coral_loss,
    ordinal_confidence,
    ordinal_probs,
    ordinal_score,
    ordinal_target,
)


def test_ordinal_probs_sum_to_one():
    logits = torch.randn(2, NUM_CATEGORIES - 1)
    probs = ordinal_probs(logits)
    assert probs.shape == (2, NUM_CATEGORIES)
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(2), atol=1e-5, rtol=1e-5)


def test_ordinal_probs_are_valid_probabilities():
    logits = torch.randn(4, NUM_CATEGORIES - 1)
    probs = ordinal_probs(logits)
    assert bool((probs >= 0).all()) and bool((probs <= 1).all())


def test_ordinal_score_extremes():
    top = torch.full((1, NUM_CATEGORIES - 1), 10.0)
    bottom = torch.full((1, NUM_CATEGORIES - 1), -10.0)
    assert ordinal_score(top).item() == NUM_CATEGORIES - 1
    assert ordinal_score(bottom).item() == 0


def test_ordinal_confidence_peaked_is_high():
    logits = torch.full((1, NUM_CATEGORIES - 1), 10.0)
    assert ordinal_confidence(logits).item() == pytest.approx(1.0, abs=1e-3)


def test_ordinal_confidence_two_point_mass():
    # all-zero logits -> mass 0.5 on 0 and 0.5 on 10 -> entropy log(2)
    logits = torch.zeros(1, NUM_CATEGORIES - 1)
    expected = 1.0 - torch.log(torch.tensor(2.0)) / torch.log(torch.tensor(float(NUM_CATEGORIES)))
    assert ordinal_confidence(logits).item() == pytest.approx(expected.item(), abs=1e-3)


def test_ordinal_target_encoding():
    labels = torch.tensor([5])
    targets = ordinal_target(labels)
    assert targets.tolist() == [[1.0] * 5 + [0.0] * 5]


def test_coral_loss_zero_for_perfect_logits():
    labels = torch.tensor([5, 2, 9])
    targets = ordinal_target(labels)
    perfect = torch.where(targets > 0.5, torch.tensor(10.0), torch.tensor(-10.0))
    # BCE at sigmoid(10) is ~5e-5, not exactly 0
    assert coral_loss(perfect, labels).item() < 1e-3


def test_coral_loss_penalizes_wrong_logits():
    labels = torch.tensor([5])
    wrong = torch.full((1, NUM_CATEGORIES - 1), 10.0)  # predicts top score for label 5
    assert coral_loss(wrong, labels).item() > 0.0