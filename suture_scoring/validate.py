"""External validation tooling: expert-label consumer and AI-vs-expert agreement.

Application-cohort filenames are NNNN_Doc_XX_Itr_YY.png (no Image_ prefix), so
expert labels are parsed with their own loader. Agreement (ICC, quadratic
weighted Cohen's kappa) is computed per output on the 0-10 scale, grouped by
trainee level. Until expert labels exist, validation stays gracefully empty.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import openpyxl

from .data import SCORE_KEYS
from .metrics import icc, weighted_cohens_kappa

_APPLICATION_NAME_RE = re.compile(r"(\d+)_Doc_(\d+)_Itr_(\d+)\.png")


def load_expert_labels(xlsx_path: str | Path) -> Dict[str, Dict[str, int]]:
    """Read application-cohort expert labels: filename -> integer scores (0-10).

    Mirrors the Train/Val schema (sheet 'Sheet1', text-typed scores, column A
    index equal to the leading filename id) but for NNNN_Doc_XX_Itr_YY names.
    """
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    try:
        ws = wb["Sheet1"]
        rows = ws.iter_rows(values_only=True)
        header = next(rows)
        name_col = header.index("Name")
        score_cols = {key: header.index(key) for key in SCORE_KEYS}
        out: Dict[str, Dict[str, int]] = {}
        for row in rows:
            if row is None or row[name_col] is None:
                continue
            name = str(row[name_col]).strip()
            match = _APPLICATION_NAME_RE.match(name)
            if match is None:
                raise ValueError(f"filename {name!r} is not NNNN_Doc_XX_Itr_YY.png")
            file_id = int(match.group(1))
            if row[0] is None or int(row[0]) != file_id:
                raise ValueError(f"index column {row[0]!r} does not match filename id {file_id}")
            out[name] = {key: int(row[score_cols[key]]) for key in SCORE_KEYS}
        return out
    finally:
        wb.close()


def read_scores_csv(csv_path: str | Path) -> Dict[str, Dict[str, float]]:
    """Read a scored cohort CSV into {filename: {key: score, conf_key: confidence}}."""
    out: Dict[str, Dict[str, float]] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            entry = {key: float(row[key]) for key in SCORE_KEYS}
            entry.update(
                {f"conf_{key}": float(row[f"conf_{key}"]) for key in SCORE_KEYS if f"conf_{key}" in row}
            )
            out[row["filename"]] = entry
    return out


def alignment_report(scores, labels) -> Dict[str, List[str]]:
    """Names present on only one side of the scores/labels join (never silently dropped)."""
    return {
        "labels_without_scores": sorted(set(labels) - set(scores)),
        "scores_without_labels": sorted(set(scores) - set(labels)),
    }


def agreement(
    scores: Dict[str, Dict[str, float]],
    labels: Dict[str, Dict[str, int]],
    n_categories: int = 11,
) -> Dict[str, Dict[str, float]]:
    """Per-output {icc, kappa} between predicted and expert scores on the 0-10 scale."""
    common = sorted(set(scores) & set(labels))
    if not common:
        raise ValueError("no filenames in common between scores and labels")
    out: Dict[str, Dict[str, float]] = {}
    for key in SCORE_KEYS:
        y_true = [labels[n][key] for n in common]
        y_pred = [min(n_categories - 1, max(0, int(round(scores[n][key])))) for n in common]
        out[key] = {
            "icc": icc(np.column_stack([y_true, y_pred])),
            "kappa": weighted_cohens_kappa(y_true, y_pred, weights="quadratic", n_categories=n_categories),
        }
    return out


def agreement_by_level(
    scores_by_level: Dict[str, Dict[str, Dict[str, float]]],
    labels: Dict[str, Dict[str, int]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Per-trainee-level agreement: level -> {output -> {icc, kappa}}."""
    return {level: agreement(scores, labels) for level, scores in scores_by_level.items()}