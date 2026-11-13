"""Metric definitions must be mathematically correct."""
from __future__ import annotations

import numpy as np
import pytest

from src.models.evaluation import (
    brier_multiclass,
    calibration_curve,
    evaluate,
    expected_calibration_error,
    wilson_interval,
)


def test_perfect_predictions():
    y = np.array([0, 1, 2, 0, 1, 2])
    proba = np.eye(3)[y]
    metrics = evaluate(y, proba)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["log_loss"] < 1e-9
    assert metrics["brier"] == pytest.approx(0.0)
    assert metrics["ece"] == pytest.approx(0.0)


def test_uniform_predictions_match_priors():
    y = np.array([0] * 20 + [1] * 10 + [2] * 30)
    proba = np.tile([0.5, 0.25, 0.25], (len(y), 1))
    metrics = evaluate(y, proba)
    # The argmax is always class 0, which is correct for 20 of 60 matches.
    assert metrics["accuracy"] == pytest.approx(20 / 60)
    assert metrics["n"] == 60
    assert metrics["correct"] + metrics["incorrect"] == 60
    assert sum(metrics["prediction_distribution"].values()) == 60


def test_brier_definition_is_sum_of_squared_errors():
    y = np.array([0, 2])
    proba = np.array([[0.8, 0.1, 0.1], [0.2, 0.3, 0.5]])
    expected = np.mean([
        (0.8 - 1) ** 2 + (0.1 - 0) ** 2 + (0.1 - 0) ** 2,
        (0.2 - 0) ** 2 + (0.3 - 0) ** 2 + (0.5 - 1) ** 2,
    ])
    assert brier_multiclass(y, proba) == pytest.approx(expected)


def test_ece_zero_for_calibrated_binary_style_output():
    y = np.array([2] * 80 + [0] * 20)
    proba = np.tile([0.2, 0.0, 0.8], (100, 1))
    assert expected_calibration_error(y, proba) == pytest.approx(0.0, abs=1e-9)


def test_ece_penalises_overconfidence():
    y = np.array([0, 1, 2] * 10)
    proba = np.tile([0.9, 0.05, 0.05], (30, 1))  # confident and mostly wrong
    assert expected_calibration_error(y, proba) > 0.5


def test_confusion_matrix_structure():
    y = np.array([0, 1, 2])
    proba = np.eye(3)[[0, 1, 2]]
    metrics = evaluate(y, proba)
    assert metrics["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert set(metrics["per_class"]) == {"Away win", "Draw", "Home win"}


def test_per_class_supports_are_counts():
    y = np.array([0, 0, 1, 2, 2, 2])
    metrics = evaluate(y, np.eye(3)[y])
    assert metrics["per_class"]["Away win"]["support"] == 2
    assert metrics["per_class"]["Draw"]["support"] == 1
    assert metrics["per_class"]["Home win"]["support"] == 3


def test_wilson_interval_bounds():
    low, high = wilson_interval(500, 1000)
    assert low < 0.5 < high
    assert 0 <= low <= high <= 1
    low, high = wilson_interval(0, 100)
    assert low == 0.0 and high > 0.0
    low, high = wilson_interval(100, 100)
    # The upper bound of a perfect record is 1.0 up to floating-point error.
    assert high == pytest.approx(1.0) and low < 1.0


def test_wilson_interval_shrinks_with_sample_size():
    small = wilson_interval(50, 100)
    large = wilson_interval(500, 1000)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_calibration_curve_bins_sum_to_sample():
    rng = np.random.default_rng(7)
    y = rng.integers(0, 3, 400)
    proba = rng.dirichlet([2, 2, 2], 400)
    curve = calibration_curve(y, proba, bins=10)
    assert sum(row["count"] for row in curve) == 400
    for row in curve:
        assert 0 <= row["mean_confidence"] <= 1
        assert 0 <= row["observed_accuracy"] <= 1


def test_accuracy_matches_hand_count():
    y = np.array([2, 2, 0, 1, 2, 0, 1, 2])
    proba = np.tile([0.2, 0.2, 0.6], (len(y), 1))
    metrics = evaluate(y, proba)
    # The argmax is always class 2, which is correct on the four rows where
    # the observed label is also 2.
    assert metrics["accuracy"] == pytest.approx(4 / 8)
    assert metrics["correct"] == 4 and metrics["incorrect"] == 4
