import numpy as np
import pytest

from suture_scoring.aggregate import LinearAggregation, evaluate_residual, clamp_score


def test_clamp_score_bounds():
    assert clamp_score(12.0) == 10.0
    assert clamp_score(-3.0) == 0.0
    assert clamp_score(5.0) == pytest.approx(5.0)


def test_linear_aggregation_recovers_known_coefficients():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 5))
    coefs = np.array([0.5, -0.3, 0.2, 0.0, 0.4])
    y = X @ coefs + 1.0
    agg = LinearAggregation()
    agg.fit(X, y)
    np.testing.assert_allclose(agg.coefs_, coefs, atol=1e-6)
    assert agg.intercept_ == pytest.approx(1.0, abs=1e-6)


def test_linear_aggregation_predicts_known_value():
    # 3 points, 3 unknowns: fully determined -> exact recovery
    agg = LinearAggregation()
    agg.fit(
        np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        np.array([2.0, 3.0, 5.0]),
    )
    pred = agg.predict(np.array([[1.0, 1.0]]))
    assert pred[0] == pytest.approx(5.0)  # 0 + 2 + 3


def test_linear_aggregation_predict_clamps_to_score_scale():
    agg = LinearAggregation()
    agg.fit(np.array([[0.0], [1.0]]), np.array([0.0, 10.0]))
    pred = agg.predict(np.array([[3.0], [-5.0]]))
    assert pred.tolist() == [10.0, 0.0]


def test_evaluate_residual_rejects_too_few_samples():
    rng = np.random.default_rng(0)
    params = rng.normal(size=(4, 5))
    y = rng.normal(size=4)
    residual = rng.normal(size=4)
    with pytest.raises(ValueError, match="samples"):
        evaluate_residual(params, y, residual, folds=5)


def test_evaluate_residual_rejects_non_finite():
    rng = np.random.default_rng(0)
    params = rng.normal(size=(50, 5))
    y = rng.normal(size=50)
    residual = rng.normal(size=50)
    residual[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        evaluate_residual(params, y, residual)


def test_evaluate_residual_rewards_helpful_signal():
    rng = np.random.default_rng(0)
    n = 300
    params = rng.normal(size=(n, 5))
    y = params @ np.array([0.6, 0.4, 0.3, 0.2, 0.1]) + rng.normal(scale=0.5, size=n)

    # residual perfectly encodes the leftover: pure aggregation cannot reach it
    residual = y - params @ np.array([0.6, 0.4, 0.3, 0.2, 0.1])
    decision = evaluate_residual(params, y, residual)
    assert decision["mae_with"] < decision["mae_without"]
    assert decision["winner"] == "with_residual"


def test_evaluate_residual_ignores_noise_signal():
    rng = np.random.default_rng(1)
    n = 300
    params = rng.normal(size=(n, 5))
    y = params @ np.array([0.6, 0.4, 0.3, 0.2, 0.1]) + rng.normal(scale=0.5, size=n)

    residual = rng.normal(size=n)  # pure noise: must not meaningfully help
    decision = evaluate_residual(params, y, residual)
    assert decision["winner"] in {"with_residual", "without_residual"}
    assert decision["mae_without"] - decision["mae_with"] < 0.01