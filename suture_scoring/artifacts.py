"""Paper artifacts generator: one command turns the scored cohort CSVs into
metrics tables, AI-vs-expert agreement tables, and figures.

The official Val split is consumed here, once, for the final reported numbers
(metrics table, per-output scatter and calibration grids). Application-cohort
agreement uses the per-level CSVs and the expert labels when they exist.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import metrics
from .data import SCORE_KEYS, load_annotations
from .validate import agreement_by_level, load_expert_labels, read_scores_csv

COHORT_FILES = {
    "train": "scores-Train.csv",
    "val": "scores-Val.csv",
    "2week": "scores-2week.csv",
    "4week": "scores-4week.csv",
    "Resident": "scores-Resident.csv",
}
APPLICATION_LEVELS = ["2week", "4week", "Resident"]


def _load_aligned(scores_csv: Path, annotations_xlsx: Path):
    """(common names, scores, annotations) aligned by filename; raises if none match."""
    scores = read_scores_csv(scores_csv)
    annotations = {a.name: a.scores for a in load_annotations(annotations_xlsx)}
    common = sorted(set(scores) & set(annotations))
    if not common:
        raise ValueError("no filenames in common between scores and annotations")
    return common, scores, annotations


def val_metrics_table(scores_csv: Path, val_annotations_xlsx: Path) -> Dict[str, Dict[str, float]]:
    """Per-output dev metrics comparing AI scores to the Val expert annotations.

    Predictions are rounded to the discrete 0-10 scale before exact-match and
    ECE correctness, consistent with the validation agreement path.
    """
    common, scores, annotations = _load_aligned(scores_csv, val_annotations_xlsx)
    out: Dict[str, Dict[str, float]] = {}
    for key in SCORE_KEYS:
        y_true = [annotations[n][key] for n in common]
        y_pred = [scores[n][key] for n in common]
        rounded = [round(p) for p in y_pred]
        correct = [1 if a == b else 0 for a, b in zip(rounded, y_true)]
        out[key] = {
            "mae": metrics.mae(y_true, y_pred),
            "exact": metrics.exact_match_accuracy(y_true, rounded),
            "tol1": metrics.within_tolerance_accuracy(y_true, y_pred, 1),
            "spearman": metrics.spearman(y_true, y_pred),
            "ece": metrics.ece([scores[n][f"conf_{key}"] for n in common], correct),
        }
    return out


def scatter_grid(by_key: Dict[str, tuple], out_path: Path) -> None:
    """Per-output scatter of AI vs annotated scores (2x3 grid over the outputs)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, key in zip(axes.ravel(), SCORE_KEYS):
        actual, predicted = by_key[key]
        ax.scatter(actual, predicted, alpha=0.5, s=15)
        ax.plot([0, 10], [0, 10], "k--", lw=0.8)
        ax.set_title(key)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_xlabel("annotated")
        ax.set_ylabel("AI")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def calibration_grid(by_key: Dict[str, tuple], out_path: Path, n_bins: int = 10) -> None:
    """Per-output confidence calibration curves (2x3 grid over the outputs)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, key in zip(axes.ravel(), SCORE_KEYS):
        conf, correct = by_key[key]
        bins = metrics.confidence_bins(conf, correct, n_bins)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        if bins:
            xs = [b[0] for b in bins]
            ys = [b[1] for b in bins]
            ax.plot(xs, ys, "o-")
        ax.set_title(key)
        ax.set_xlabel("confidence")
        ax.set_ylabel("accuracy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def trainee_violin(overall_by_level: Dict[str, np.ndarray], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (level, values) in enumerate(overall_by_level.items()):
        ax.violinplot(values, positions=[i], showmeans=True, widths=0.7)
    ax.set_xticks(range(len(overall_by_level)))
    ax.set_xticklabels(list(overall_by_level))
    ax.set_ylabel("AI Overall score")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _table_md(title: str, table: Dict[str, Dict[str, float]]) -> str:
    lines = [f"## {title}", "", "| output | mae | exact | +/-1 | spearman | ece |", "|---|---|---|---|---|---|"]
    for key in SCORE_KEYS:
        r = table[key]
        lines.append(f"| {key} | {r['mae']:.3f} | {r['exact']:.3f} | {r['tol1']:.3f} | {r['spearman']:.3f} | {r['ece']:.3f} |")
    return "\n".join(lines) + "\n"


def _agreement_md(table) -> str:
    lines = ["## AI-vs-expert agreement (per trainee level)", "", "| level | output | icc | kappa |", "|---|---|---|---|"]
    for level in APPLICATION_LEVELS:
        if level in table:
            for key in SCORE_KEYS:
                r = table[level][key]
                lines.append(f"| {level} | {key} | {r['icc']:.3f} | {r['kappa']:.3f} |")
    return "\n".join(lines) + "\n"


def generate_all(
    scores_dir: Path,
    val_annotations_xlsx: Path,
    output_dir: Path,
    expert_labels_xlsx: Path | None = None,
) -> Dict[str, Path]:
    """Generate all paper artifacts from the scored cohort CSVs; returns produced paths."""
    scores_dir = Path(scores_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    produced: Dict[str, Path] = {}

    val_csv = scores_dir / COHORT_FILES["val"]
    if not val_csv.exists():
        raise ValueError(f"missing scored Val CSV: {val_csv}")
    produced["val-metrics.md"] = _write(
        output_dir / "val-metrics.md",
        _table_md("Val metrics (official split, final numbers)", val_metrics_table(val_csv, val_annotations_xlsx)),
    )

    common, scores, annotations = _load_aligned(val_csv, val_annotations_xlsx)
    scatter_data = {
        key: ([annotations[n][key] for n in common], [scores[n][key] for n in common])
        for key in SCORE_KEYS
    }
    produced["scatter-grid.png"] = _figure(output_dir / "scatter-grid.png", scatter_grid, scatter_data)
    calibration_data = {
        key: (
            [scores[n][f"conf_{key}"] for n in common],
            [1 if round(scores[n][key]) == annotations[n][key] else 0 for n in common],
        )
        for key in SCORE_KEYS
    }
    produced["calibration-grid.png"] = _figure(output_dir / "calibration-grid.png", calibration_grid, calibration_data)

    level_scores: Dict[str, Dict[str, Dict[str, float]]] = {}
    for level in APPLICATION_LEVELS:
        path = scores_dir / COHORT_FILES[level]
        if path.exists():
            level_scores[level] = read_scores_csv(path)
    if level_scores:
        levels = {
            level: np.asarray([scores[n]["Overall"] for n in sorted(scores)])
            for level, scores in level_scores.items()
        }
        produced["trainee-levels.png"] = _figure(output_dir / "trainee-levels.png", trainee_violin, levels)

    if expert_labels_xlsx is not None and Path(expert_labels_xlsx).exists():
        labels = load_expert_labels(expert_labels_xlsx)
        overlapping = {level: sv for level, sv in level_scores.items() if set(sv) & set(labels)}
        if overlapping:
            produced["agreement.md"] = _write(output_dir / "agreement.md", _agreement_md(agreement_by_level(overlapping, labels)))
        else:
            produced["agreement.md"] = _write(output_dir / "agreement.md", "Expert labels do not overlap any scored cohort file.\n")
    else:
        produced["agreement.md"] = _write(output_dir / "agreement.md", "Expert labels not yet available; the agreement table will appear here once `--expert-labels` is supplied.\n")

    manifest = "Paper artifacts generated from the scored cohort CSVs.\n\n" + "\n".join(f"- {path.name}" for path in produced.values()) + "\n"
    produced["manifest.md"] = _write(output_dir / "manifest.md", manifest)
    return produced


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _figure(path: Path, fn, *args) -> Path:
    fn(*args, path)
    return path