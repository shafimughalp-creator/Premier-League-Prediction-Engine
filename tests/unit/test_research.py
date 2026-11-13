"""Tests for the research diagnostics (draw rules, calibration, paired stats)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research import diagnostics as diag


def _frame(proba: np.ndarray, results: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame({
        "_result": results,
        "_season": ["2023/24"] * len(results),
    })
    frame["pred_away"] = proba[:, 0]
    frame["pred_draw"] = proba[:, 1]
    frame["pred_home"] = proba[:, 2]
    return frame


def test_choose_matches_argmax_without_policy():
    proba = np.array([[0.5, 0.2, 0.3], [0.2, 0.25, 0.55]])
    frame = _frame(proba, ["A", "H"])
    assert list(diag.choose(proba[:, 1], proba[:, 2], proba[:, 0])) == [0, 2]
    assert list(diag.choose(proba[:, 1], proba[:, 2], proba[:, 0],
                            margin=0.0)) == [0, 2]


def test_threshold_rule_forces_draw_when_probability_high():
    proba = np.array([[0.40, 0.31, 0.29], [0.20, 0.20, 0.60]])
    prediction = diag.choose(proba[:, 1], proba[:, 2], proba[:, 0], threshold=0.30)
    assert list(prediction) == [1, 2]


def test_decision_metrics_perfect_predictions():
    proba = np.eye(3)
    frame = _frame(proba, ["A", "D", "H"])
    metrics = diag.decision_metrics(diag.labels(frame), proba.argmax(axis=1))
    assert metrics["accuracy"] == 1.0
    assert metrics["draw_recall"] == 1.0
    assert metrics["draws_predicted"] == 1


def test_reliability_counts_sum_to_sample():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 3, 300)
    proba = rng.dirichlet([2, 2, 2], 300)
    rows = diag.reliability_table(y, proba, bins=10)
    assert sum(row["count"] for row in rows) == 300
    for row in rows:
        assert 0 <= row["observed_accuracy"] <= 1
        assert 0 <= row["mean_confidence"] <= 1


def test_brier_decomposition_components_non_negative():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 3, 400)
    proba = rng.dirichlet([2, 2, 2], 400)
    decomposition = diag.brier_decomposition(y, proba, bins=10)
    assert decomposition["reliability"] >= 0
    assert decomposition["resolution"] >= 0
    assert decomposition["uncertainty"] >= 0
    assert set(decomposition["per_class"]) == {"Away win", "Draw", "Home win"}


def test_paired_comparison_identical_models_has_zero_difference():
    rng = np.random.default_rng(5)
    y = rng.integers(0, 3, 120)
    proba = rng.dirichlet([2, 2, 2], 120)
    frame = _frame(proba, [("A", "D", "H")[i] for i in y])
    frame["market_away"] = proba[:, 0]
    frame["market_draw"] = proba[:, 1]
    frame["market_home"] = proba[:, 2]
    result = diag.paired_comparison(frame, "pred", "market", n_boot=200)
    assert result["agreement_table"]["only_a_correct"] == 0
    assert result["agreement_table"]["only_b_correct"] == 0
    assert result["mcnemar_exact_p"] == 1.0
    assert result["accuracy_difference"]["match_bootstrap"]["mean"] == 0.0


def test_vector_scaling_is_a_distribution():
    rng = np.random.default_rng(6)
    y = rng.integers(0, 3, 200)
    proba = rng.dirichlet([2, 2, 2], 200)
    theta = diag.fit_vector_scaling(y, proba)
    calibrated = diag.apply_vector_scaling(proba, theta)
    assert calibrated.shape == proba.shape
    assert np.allclose(calibrated.sum(axis=1), 1.0)
    assert np.isfinite(calibrated).all()


def test_select_draw_policy_respects_accuracy_gate():
    # A policy that predicts every match as a draw would lose accuracy; the
    # selector must fall back to argmax rather than force draws.
    rng = np.random.default_rng(7)
    y = rng.integers(0, 3, 400)
    proba = rng.dirichlet([1, 1, 1], 400)
    frame = _frame(proba, [("A", "D", "H")[i] for i in y])
    result = diag.select_draw_policy(frame, max_accuracy_drop=0.005,
                                     min_macro_gain=0.01)
    if result["selected"] is None:
        assert "No policy" in result["reason"]
    else:
        assert result["selected"]["accuracy"] >= result["baseline"]["accuracy"] - 0.005


# --------------------------------------------------------------------------
# Promotion gates
# --------------------------------------------------------------------------
def _draw_payload(selected: dict | None, baseline_macro: float = 0.436) -> dict:
    return {
        "policy_selection": {
            "selected": selected,
            "baseline": {"accuracy": 0.5921, "macro_f1": baseline_macro},
        },
        "policy_applied": {
            "test": {"accuracy": 0.5039, "macro_f1": 0.4099},
            "test_argmax": {"accuracy": 0.5184, "macro_f1": 0.3887},
        },
    }


def _forward_payload(changes: list[tuple[float, float]]) -> dict:
    seasons = ["2019/20", "2020/21", "2021/22", "2022/23", "2023/24"]
    origins = [
        {
            "season": season,
            "draw_policy": {"accuracy_change": accuracy, "macro_f1_change": macro,
                            "selected": {"policy": "threshold", "threshold": 0.28}},
        }
        for season, (accuracy, macro) in zip(seasons, changes)
    ]
    gains = sum(1 for _, macro in changes if macro > 0)
    return {
        "protocol": {"selection_origins": seasons},
        "origins": origins,
        "draw_policy_stability": {
            "mean_accuracy_change": float(np.mean([a for a, _ in changes])),
            "mean_macro_f1_change": float(np.mean([m for _, m in changes])),
            "origins_with_macro_gain": gains,
            "origins": len(changes),
        },
    }


def test_finalize_promotes_draw_policy_only_when_it_replicates():
    from src.research import finalize

    selected = {"policy": "threshold", "threshold": 0.28, "accuracy": 0.5921,
                "macro_f1": 0.4633}
    consistent = _forward_payload([(0.001, 0.02), (0.0, 0.03), (0.002, 0.02),
                                   (0.0, 0.025), (0.001, 0.02)])
    decision = finalize._draw_decision(_draw_payload(selected), consistent)
    assert decision["promoted"] is True
    assert decision["validation_gate"] and decision["replication_gate"]

    inconsistent = _forward_payload([(0.003, 0.008), (-0.026, 0.088), (0.0, 0.0),
                                     (0.0, 0.026), (-0.003, -0.001)])
    decision = finalize._draw_decision(_draw_payload(selected), inconsistent)
    assert decision["promoted"] is False
    assert decision["validation_gate"] is True
    assert decision["replication_gate"] is False


def test_finalize_rejects_draw_policy_without_validation_gate():
    from src.research import finalize

    decision = finalize._draw_decision(_draw_payload(None), _forward_payload(
        [(0.0, 0.0)] * 5))
    assert decision["promoted"] is False
    assert "Argmax is retained" in decision["summary"]


def test_finalize_rejects_core_features_that_lose_accuracy():
    from src.research import finalize

    challengers = {
        "core_feature_set": {
            "n_features": 143,
            "validation": {"log_loss": 0.9331, "accuracy": 0.5789},
        },
        "reference": {"no_odds_validation": {"log_loss": 0.9376, "accuracy": 0.6053}},
    }
    decision = finalize._core_decision(challengers)
    assert decision["promoted"] is False
    assert decision["validation_accuracy_change"] < -0.005


def test_finalize_rejects_dixon_coles_blend_below_margin():
    from src.research import finalize

    challengers = {
        "dixon_coles": {
            "configuration": {},
            "validation": {"accuracy": 0.5684},
            "blend_with_production": {
                "validation_selection": {"weight": 0.08, "log_loss": 0.90548},
            },
        },
        "reference": {"production_validation": {"log_loss": 0.90576}},
    }
    decision = finalize._dixon_decision(challengers)
    assert decision["promoted"] is False
    assert 0 < decision["validation_log_loss_change"] < 0.002
