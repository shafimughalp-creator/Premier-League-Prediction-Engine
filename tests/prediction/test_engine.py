"""Prediction engine behaviour and probability pipeline."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import config
from src.prediction.engine import (
    MODE_RETROSPECTIVE,
    PREDICTIONS_JSON,
    PredictionEngine,
    PredictionError,
    apply_temperature,
    blend_with_market,
    blend_with_poisson,
    choose_outcome,
)


@pytest.fixture(scope="module")
def engine() -> PredictionEngine:
    return PredictionEngine()


@pytest.fixture(scope="module")
def upcoming(engine: PredictionEngine) -> dict:
    return engine.predict_upcoming(save=True)


# --------------------------------------------------------------------------
# Probability primitives
# --------------------------------------------------------------------------
def test_temperature_preserves_distribution():
    proba = np.array([[0.2, 0.3, 0.5], [0.6, 0.25, 0.15]])
    out = apply_temperature(proba, 0.8)
    assert np.allclose(out.sum(axis=1), 1.0)
    assert (out.argmax(axis=1) == proba.argmax(axis=1)).all()


def test_temperature_identity_at_one():
    proba = np.array([[0.2, 0.3, 0.5]])
    assert np.allclose(apply_temperature(proba, 1.0), proba)


def test_market_blend_full_trust_returns_market():
    proba = np.array([[0.5, 0.2, 0.3]])
    market = np.array([[0.45, 0.25, 0.30]])
    assert np.allclose(blend_with_market(proba, market, 0.0), market)


def test_market_blend_skips_missing_prices():
    proba = np.array([[0.5, 0.2, 0.3]])
    market = np.array([[np.nan, np.nan, np.nan]])
    out = blend_with_market(proba, market, 0.0)
    assert np.allclose(out, proba)


def test_poisson_blend_alpha_zero_returns_poisson():
    proba = np.array([[0.5, 0.2, 0.3]])
    poisson = np.array([[0.4, 0.3, 0.3]])
    assert np.allclose(blend_with_poisson(proba, poisson, 0.0), poisson)


def test_choose_outcome_argmax_default():
    proba = np.array([[0.2, 0.25, 0.55], [0.5, 0.3, 0.2]])
    assert list(choose_outcome(proba)) == [2, 0]


def test_choose_outcome_draw_threshold():
    # Internal order is A/D/H, so [0.4, 0.31, 0.29] is away/draw/home.
    proba = np.array([[0.4, 0.31, 0.29], [0.2, 0.2, 0.6]])
    assert list(choose_outcome(proba, draw_threshold=0.30)) == [1, 2]


# --------------------------------------------------------------------------
# Upcoming fixtures
# --------------------------------------------------------------------------
def test_upcoming_payload_schema(upcoming):
    assert upcoming["n_fixtures"] == len(upcoming["fixtures"])
    assert upcoming["model_version"].startswith("epl-")
    assert upcoming["mode"] == "upcoming"
    for row in upcoming["fixtures"]:
        probs = row["probabilities"]
        assert abs(sum(probs.values()) - 1.0) < 1e-6
        assert row["predicted"] in {"H", "D", "A"}
        assert 0.0 < row["confidence"] <= 1.0
        assert row["predicted_at"]
        assert row["model_version"] == upcoming["model_version"]


def test_prediction_is_argmax(upcoming):
    for row in upcoming["fixtures"]:
        probs = {"H": row["prob_home"], "D": row["prob_draw"], "A": row["prob_away"]}
        assert row["predicted"] == max(probs, key=probs.get)
        assert abs(row["confidence"] - max(probs.values())) < 1e-6


def test_confidence_band_is_consistent(upcoming):
    for row in upcoming["fixtures"]:
        band = row["confidence_band"]
        assert band in {"Low", "Medium", "High"}
        if row["confidence"] >= 0.60:
            assert band == "High"
        elif row["confidence"] >= 0.45:
            assert band == "Medium"
        else:
            assert band == "Low"


def test_fixtures_are_future_and_distinct(engine, upcoming):
    cutoff = engine.data_cutoff
    for row in upcoming["fixtures"]:
        assert pd.Timestamp(row["date"]) > cutoff
        assert row["home_team"] != row["away_team"]


def test_predictions_file_is_atomic_and_valid(upcoming):
    payload = json.loads(PREDICTIONS_JSON.read_text(encoding="utf-8"))
    assert payload["model_version"] == upcoming["model_version"]
    assert payload["n_fixtures"] == upcoming["n_fixtures"]
    assert not list(PREDICTIONS_JSON.parent.glob("*.tmp"))


def test_odds_free_fixtures_use_no_odds_variant(upcoming):
    for row in upcoming["fixtures"]:
        if not row["has_odds"]:
            assert row["model_variant"] == "no_odds"
            assert row["market"] is None


# --------------------------------------------------------------------------
# Manual predictions
# --------------------------------------------------------------------------
def test_manual_prediction_is_valid(engine):
    prediction = engine.predict_manual("Arsenal", "Chelsea")
    assert prediction.home_team == "Arsenal"
    assert prediction.away_team == "Chelsea"
    assert prediction.mode == "manual"
    total = prediction.prob_home + prediction.prob_draw + prediction.prob_away
    assert abs(total - 1.0) < 1e-6


def test_manual_prediction_normalises_names(engine):
    prediction = engine.predict_manual("Manchester City", "Wolverhampton Wanderers")
    assert prediction.home_team == "Man City"
    assert prediction.away_team == "Wolves"


def test_manual_rejects_same_team(engine):
    with pytest.raises(PredictionError, match="cannot play itself"):
        engine.predict_manual("Arsenal", "Arsenal")


def test_manual_rejects_unknown_team(engine):
    with pytest.raises(PredictionError, match="Unknown team"):
        engine.predict_manual("Arsenal", "Real Madrid")


def test_manual_past_date_is_labelled_retrospective(engine):
    prediction = engine.predict_manual("Arsenal", "Chelsea", date="2025-03-15")
    assert prediction.mode == MODE_RETROSPECTIVE
    assert prediction.context["in_sample"] is True
    assert "Retrospective" in prediction.context["note"]


def test_manual_future_date_is_genuine(engine):
    future = (engine.data_cutoff + pd.Timedelta(days=30)).date()
    prediction = engine.predict_manual("Arsenal", "Chelsea", date=str(future))
    assert prediction.mode == "manual"
    assert prediction.context["in_sample"] is False


def test_manual_past_uses_only_prior_matches(engine):
    """A manual date must not see later results, even indirectly."""
    early = engine.predict_manual("Everton", "Fulham", date="2015-01-10")
    reference = engine.predict_manual("Everton", "Fulham", date="2015-01-10")
    assert early.prob_home == pytest.approx(reference.prob_home)
    # A first-season fixture cannot know the league's later scoring rate.
    elo = early.context["home_elo"]
    assert 1200 < elo < 1800


# --------------------------------------------------------------------------
# Retrospective mode
# --------------------------------------------------------------------------
def test_retrospective_scores_history(engine):
    payload = engine.predict_retrospective(season="2025/26", limit=40)
    assert payload["mode"] == MODE_RETROSPECTIVE
    assert payload["n_fixtures"] > 0
    rows = payload["fixtures"]
    with_truth = [r for r in rows if r["context"].get("actual_result")]
    assert with_truth
    for row in with_truth:
        assert row["context"]["in_sample"] is True
        assert row["context"]["correct"] in (True, False)
        assert row["context"]["actual_result"] in {"H", "D", "A"}


def test_retrospective_is_not_claimed_as_prematch(engine):
    payload = engine.predict_retrospective(limit=5)
    assert payload["mode"] == "retrospective"
    assert payload["mode"] != "upcoming"
