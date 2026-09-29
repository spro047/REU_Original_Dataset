"""Production scoring: trained model + fitted hybrid aggregation -> per-cohort CSVs.

For each cohort directory the pipeline predicts the five parameter scores and
confidences from the ordinal heads, computes the final Overall through the
ticket-04 hybrid aggregation (fitted on model predictions vs annotated Overall
on the Train cohort), and writes one CSV per cohort. The official Val split is
scored but never used to fit anything — it is consumed exactly once, later, for
the final reported numbers.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .aggregate import LinearAggregation
from .data import SCORE_KEYS, Annotation, list_images
from .dataset import SutureDataset
from .model import OrdinalSutureModel
from .ordinal import ordinal_confidence, ordinal_expected, ordinal_score
from .preprocess import preprocess_image, to_tensor

PARAM_KEYS = [k for k in SCORE_KEYS if k != "Overall"]


def _predict(model, tensor):
    """Model forward -> (param scores, Overall-head expected score, confidences)."""
    with torch.no_grad():
        logits = model(tensor)
    params = {k: float(ordinal_score(logits[k][0])) for k in PARAM_KEYS}
    overall_expected = float(ordinal_expected(logits["Overall"][0]))
    confidences = {key: float(ordinal_confidence(logits[key][0])) for key in SCORE_KEYS}
    return params, overall_expected, confidences


def fit_aggregation(
    model: OrdinalSutureModel,
    image_dir: str | Path,
    annotations: List[Annotation],
    size: int = 224,
) -> LinearAggregation:
    """Fit the hybrid Overall aggregation on model predictions vs annotated Overall."""
    device = next(model.parameters()).device
    model.eval()
    ds = SutureDataset(image_dir, annotations, size=size, augment=False)
    features: List[List[float]] = []
    overall: List[float] = []
    for image, targets in ds:
        params, overall_expected, _ = _predict(model, image.unsqueeze(0).to(device))
        features.append([*params.values(), overall_expected])
        overall.append(float(targets["Overall"]))
    return LinearAggregation().fit(np.asarray(features), np.asarray(overall))


def score_cohort(
    model: OrdinalSutureModel,
    aggregation: LinearAggregation,
    image_dir: str | Path,
    size: int = 224,
) -> List[Dict]:
    """Score every image in a directory: 6 scores (Overall via aggregation) + confidences."""
    device = next(model.parameters()).device
    model.eval()
    rows: List[Dict] = []
    for path in list_images(image_dir):
        tensor = to_tensor(preprocess_image(path, size=size)).unsqueeze(0).to(device)
        params, overall_expected, confidences = _predict(model, tensor)
        row = {"filename": path.name, **params}
        row["Overall"] = float(aggregation.predict(np.array([[*(params.values()), overall_expected]]))[0])
        for key in SCORE_KEYS:
            row[f"conf_{key}"] = confidences[key]
        rows.append(row)
    return rows


def write_csv(rows: List[Dict], out_path: str | Path) -> None:
    header = ["filename", *SCORE_KEYS, *(f"conf_{key}" for key in SCORE_KEYS)]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)