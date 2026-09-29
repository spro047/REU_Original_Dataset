"""Overall hybrid aggregation (ADR-0001).

The final Overall score is a learned function of the five predicted parameters
plus a residual channel that reads the image directly (the ordinal model's
Overall-head expected score). Both variants are fitted on held-out data and the
residual is kept only if it actually lowers MAE — the ADR's documented fallback.
"""
from __future__ import annotations

import numpy as np


def clamp_score(value: float) -> float:
    """Clamp a raw aggregation output onto the native 0-10 score scale."""
    return max(0.0, min(10.0, float(value)))


class LinearAggregation:
    """Least-squares linear model from parameter scores (and optional residual) to Overall."""

    def __init__(self):
        self.coefs_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(self, X, y) -> "LinearAggregation":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), y, rcond=None)
        self.intercept_ = float(beta[0])
        self.coefs_ = beta[1:]
        return self

    def predict(self, X) -> np.ndarray:
        if self.coefs_ is None:
            raise RuntimeError("fit before predict")
        X = np.asarray(X, dtype=float)
        return np.clip(X @ self.coefs_ + self.intercept_, 0.0, 10.0)


def evaluate_residual(params, overall, residual, folds: int = 5, seed: int = 0) -> dict:
    """Fit the aggregation with and without the residual channel; report the winner.

    ``params`` is an (n, 5) array of predicted parameter scores, ``overall`` the
    (n,) annotated Overall, ``residual`` the (n,) image-derived signal. Both
    variants are compared by out-of-fold MAE (K-fold CV) — comparing on the
    fitting data would always favor the extra channel. Returns both MAEs and
    which variant wins (residual kept only if it genuinely lowers MAE).
    """
    params = np.asarray(params, dtype=float)
    overall = np.asarray(overall, dtype=float)
    residual = np.asarray(residual, dtype=float)
    n = len(overall)
    if n == 0:
        raise ValueError("empty input")
    if not (np.all(np.isfinite(params)) and np.all(np.isfinite(overall)) and np.all(np.isfinite(residual))):
        raise ValueError("inputs must be finite")
    if n <= folds:
        raise ValueError(f"need more than {folds} samples for {folds}-fold evaluation")

    order = np.random.default_rng(seed).permutation(n)
    fold_size = int(np.ceil(n / folds))
    total_without = 0.0
    total_with = 0.0
    for start in range(0, n, fold_size):
        val_idx = order[start : start + fold_size]
        train_mask = np.ones(n, dtype=bool)
        train_mask[val_idx] = False

        X_val, y_val = params[val_idx], overall[val_idx]
        Xw_val = np.column_stack([X_val, residual[val_idx]])

        pure = LinearAggregation().fit(params[train_mask], overall[train_mask])
        with_res = LinearAggregation().fit(
            np.column_stack([params[train_mask], residual[train_mask]]),
            overall[train_mask],
        )
        total_without += len(val_idx) * float(np.mean(np.abs(pure.predict(X_val) - y_val)))
        total_with += len(val_idx) * float(np.mean(np.abs(with_res.predict(Xw_val) - y_val)))

    mae_without_val = total_without / n
    mae_with_val = total_with / n
    return {
        "mae_without": mae_without_val,
        "mae_with": mae_with_val,
        "winner": "with_residual" if mae_with_val < mae_without_val else "without_residual",
    }