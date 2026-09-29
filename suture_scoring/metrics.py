"""Evaluation metrics for suturing quality scores.

Dev metrics (MAE, exact-match %, ±1 accuracy, Spearman, ECE) validate the model
on held-out data; validation metrics (ICC, weighted Cohen's kappa) report the
AI-vs-expert agreement study on the application cohort. All functions take
plain arrays and work on any predicted-vs-actual data; none depend on a model.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import ConstantInputWarning, spearmanr


def mae(y_true, y_pred) -> float:
    """Mean absolute error between predictions and ground truth."""
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.size == 0:
        raise ValueError("empty input")
    return float(np.mean(np.abs(a - b)))


def exact_match_accuracy(y_true, y_pred) -> float:
    """Fraction of predictions exactly equal to the ground truth."""
    a = np.asarray(y_true)
    b = np.asarray(y_pred)
    if a.size == 0:
        raise ValueError("empty input")
    return float(np.mean(a == b))


def within_tolerance_accuracy(y_true, y_pred, tolerance: int = 1) -> float:
    """Fraction of predictions within `tolerance` points of ground truth."""
    a = np.asarray(y_true)
    b = np.asarray(y_pred)
    if a.size == 0:
        raise ValueError("empty input")
    return float(np.mean(np.abs(a - b) <= tolerance))


def spearman(y_true, y_pred) -> float:
    """Spearman rank correlation; NaN for constant input (undefined rank)."""
    a = np.asarray(y_true)
    b = np.asarray(y_pred)
    if a.size == 0:
        raise ValueError("empty input")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        rho, _ = spearmanr(a, b)
    return float(rho)


def confidence_bins(confidences, correct, n_bins: int = 10):
    """Per-bin (mean confidence, accuracy, weight) over [0, 1]; empty bins skipped.

    Shared by ECE and the calibration curve so both use the identical binning.
    """
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf <= hi) if lo == 0.0 else (conf > lo) & (conf <= hi)
        if not np.any(mask):
            continue
        bins.append((float(np.mean(conf[mask])), float(np.mean(corr[mask])), float(np.sum(mask) / conf.size)))
    return bins


def ece(confidences, correct, n_bins: int = 10) -> float:
    """Expected calibration error: mean |bin accuracy - mean confidence|.

    Bins partition [0, 1] into `n_bins` intervals (first bin includes 0); empty
    bins are skipped. Confidences should be the model's 0-1 certainty and
    `correct` the 0/1 correctness of each prediction.
    """
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correct, dtype=float)
    if conf.size == 0:
        raise ValueError("empty input")
    if conf.size != corr.size:
        raise ValueError("confidences and correct must be the same length")
    if np.any((conf < 0.0) | (conf > 1.0)):
        raise ValueError("confidences must lie in [0, 1]")
    return float(sum(weight * abs(acc - mean_conf) for mean_conf, acc, weight in confidence_bins(conf, corr, n_bins)))


def icc(ratings) -> float:
    """ICC(2,1): two-way random effects, single measures, absolute agreement.

    ``ratings`` is an (n subjects, k raters) array. ANOVA-based per
    Shrout & Fleiss (1979): (MSR - MSE) / (MSR + (k-1)*MSE + k*(MSC - MSE)/n).
    Zero-variance input (all ratings identical) returns 1.0 by convention.
    """
    x = np.asarray(ratings, dtype=float)
    n, k = x.shape
    if k < 2 or n < 2:
        raise ValueError("need at least 2 subjects and 2 raters")

    row_means = x.mean(axis=1)
    col_means = x.mean(axis=0)
    grand = x.mean()

    msr = (k * np.sum((row_means - grand) ** 2)) / (n - 1)
    msc = (n * np.sum((col_means - grand) ** 2)) / (k - 1)
    sse = np.sum((x - row_means[:, None] - col_means[None, :] + grand) ** 2)
    mse = sse / ((n - 1) * (k - 1))

    if msr == 0.0 and mse == 0.0:
        return 1.0  # no variance anywhere: all ratings identical
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def weighted_cohens_kappa(y1, y2, weights: str = "quadratic", n_categories: int | None = None) -> float:
    """Weighted Cohen's kappa for ordinal scores.

    ``weights`` is "quadratic" (Fleiss-Cohen, the default for ordinal
    agreement studies) or "linear". ``n_categories`` is the number of ordered
    categories (scores assumed 0..n_categories-1): pass 11 for the 0-10 scale
    so the weights use the full ordinal range regardless of the sample; when
    None, categories are inferred from the observed data as 0..max. kappa =
    (po - pe) / (1 - pe); perfect agreement returns 1.0.
    """
    a = np.asarray(y1)
    b = np.asarray(y2)
    if a.size == 0:
        raise ValueError("empty input")
    if not (np.array_equal(a, a.astype(int)) and np.array_equal(b, b.astype(int))):
        raise ValueError("scores must be integers")
    if np.any(a < 0) or np.any(b < 0):
        raise ValueError("scores must be non-negative")
    n_cat = n_categories if n_categories is not None else int(max(a.max(), b.max())) + 1
    if n_cat < 1:
        raise ValueError("n_categories must be positive")
    if n_cat == 1:
        return 1.0

    conf = np.zeros((n_cat, n_cat))
    np.add.at(conf, (a, b), 1)

    idx = np.arange(n_cat)
    if weights == "quadratic":
        w = 1.0 - ((idx[:, None] - idx[None, :]) ** 2) / (n_cat - 1) ** 2
    elif weights == "linear":
        w = 1.0 - np.abs(idx[:, None] - idx[None, :]) / (n_cat - 1)
    else:
        raise ValueError(f"unknown weights: {weights!r}")

    n = a.size
    po = float(np.sum(w * conf) / n)
    expected = np.outer(conf.sum(axis=1), conf.sum(axis=0)) / n
    pe = float(np.sum(w * expected) / n)

    denom = 1.0 - pe
    if denom == 0.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / denom