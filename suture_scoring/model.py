"""Scoring models.

Ticket 01 ships a deterministic placeholder scorer so the pipeline works end to
end and is reproducible. It is replaced by the real ordinal-regression model in
ticket 03; the scorer interface (image array in, score dict out) is the seam
that must stay stable.
"""
from __future__ import annotations

import numpy as np

from .data import SCORE_KEYS


class ToyScorer:
    """Deterministic placeholder: scores derived from mean image brightness.

    Scores are mapped onto the 0-10 scale from the mean brightness of the
    normalized image; confidence is a fixed 0.9 placeholder. Deterministic for
    a given input, so CLI output is reproducible. Note the white padding
    (preprocess) biases brightness upward for non-square images — acceptable
    for a placeholder, replaced by the real model in ticket 03.
    """

    def score(self, image: np.ndarray) -> dict[str, float]:
        base = round(float(image.mean()) * 10.0, 1)
        out = {key: round(max(0.0, min(10.0, base + 0.1 * i)), 1) for i, key in enumerate(SCORE_KEYS)}
        out.update({f"conf_{key}": 0.9 for key in SCORE_KEYS})
        return out