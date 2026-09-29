import numpy as np
import torch

from suture_scoring.data import SCORE_KEYS
from suture_scoring.model import OrdinalSutureModel, OrdinalSutureScorer
from suture_scoring.ordinal import NUM_CATEGORIES


def test_tiny_model_forward_outputs_six_heads():
    model = OrdinalSutureModel(backbone="tiny", embedding=32)
    logits = model(torch.randn(2, 3, 64, 64))
    assert set(logits) == set(SCORE_KEYS)
    for key in SCORE_KEYS:
        assert logits[key].shape == (2, NUM_CATEGORIES - 1)


def test_scorer_outputs_scores_and_confidence_in_range():
    model = OrdinalSutureModel(backbone="tiny", embedding=32)
    scorer = OrdinalSutureScorer(model)
    image = np.random.default_rng(0).random((64, 64, 3), dtype=np.float32)
    out = scorer.score(image)
    assert set(out) == {*SCORE_KEYS, *(f"conf_{k}" for k in SCORE_KEYS)}
    for key in SCORE_KEYS:
        assert 0 <= out[key] <= NUM_CATEGORIES - 1
        assert 0.0 <= out[f"conf_{key}"] <= 1.0


def test_scorer_deterministic_in_eval():
    model = OrdinalSutureModel(backbone="tiny", embedding=32)
    scorer = OrdinalSutureScorer(model)
    image = np.random.default_rng(1).random((64, 64, 3), dtype=np.float32)
    assert scorer.score(image) == scorer.score(image)