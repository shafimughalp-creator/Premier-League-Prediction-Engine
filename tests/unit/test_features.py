"""Feature engineering invariants."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.features.builder import (
    FeatureBuilder,
    feature_columns,
    odds_free_columns,
)


def test_feature_count_is_stable(feature_names):
    assert len(feature_names) >= 230


def test_expected_feature_families_exist(feature_names):
    expected = {
        "h_elo", "a_elo", "elo_diff", "elo_expected_home", "elo_closeness",
        "h_form3_ppg", "h_form5_ppg", "h_form10_ppg", "a_form5_ppg",
        "h_venue5_ppg", "a_venue5_ppg", "h_rest_days", "a_rest_days",
        "h_matches_14d", "a_matches_14d", "h_attack_home", "a_defence_away",
        "pois_p_home", "pois_p_draw", "pois_p_away", "pois_exp_home",
        "h2h_ppg", "h2h_draw_rate", "mkt_home", "mkt_draw", "mkt_away",
        "mkt_overround", "mkt_entropy", "mkt_books",
        "h_prev_ppg", "a_prev_ppg",
    }
    missing = expected - set(feature_names)
    assert not missing, f"missing feature families: {sorted(missing)}"


def test_no_odds_subset_excludes_market(feature_names):
    no_odds = odds_free_columns(feature_names)
    assert len(no_odds) < len(feature_names)
    assert not any(c.startswith(("mkt_", "odds_")) for c in no_odds)
    assert "h_elo" in no_odds


def test_tree_missingness_is_bounded(features, feature_names):
    ratio = features[feature_names].isna().mean().mean()
    assert ratio < 0.02, f"too many missing values: {ratio:.3%}"


def test_labels_align_with_results(features):
    mapping = {"A": 0, "D": 1, "H": 2}
    expected = features["_result"].map(mapping).to_numpy()
    assert (features["_label"].to_numpy() == expected).all()


def test_poisson_probabilities_are_distributions(features):
    cols = ["pois_p_home", "pois_p_draw", "pois_p_away"]
    subset = features.dropna(subset=cols)
    assert len(subset) > 4000
    assert np.allclose(subset[cols].sum(axis=1), 1.0, atol=1e-6)
    assert (subset[cols].to_numpy() >= 0).all()


def test_elo_ratings_stay_bounded(features):
    for column in ("h_elo", "a_elo"):
        values = features[column].dropna()
        assert values.min() > 1000
        assert values.max() < 2100


def test_no_feature_is_constant(features, feature_names):
    std = features[feature_names].std(numeric_only=True)
    constant = [c for c in feature_names if c in std.index and std[c] == 0]
    assert not constant, f"constant features: {constant[:10]}"


def test_rebuilding_is_deterministic(matches):
    first = FeatureBuilder().build(matches.tail(300))
    second = FeatureBuilder().build(matches.tail(300))
    columns = feature_columns(first)
    a = first[columns].to_numpy(dtype=float)
    b = second[columns].to_numpy(dtype=float)
    assert np.allclose(a, b, equal_nan=True)


def test_same_date_matches_are_scored_before_any_update(matches):
    """Two matches on the same date must not see each other's results."""
    date = pd.Timestamp("2019-12-26")
    same_day = matches[matches["date"] == date]
    assert len(same_day) > 1
    built = FeatureBuilder().build(matches[matches["date"] <= date])
    rows = built[built["_date"] == date]
    assert len(rows) == len(same_day)
    # Identical team state for a team that appears once that day means the
    # state was not advanced by the earlier kick-off.
    teams = set(rows["_home_team"]) | set(rows["_away_team"])
    for team in teams:
        home_rows = rows[rows["_home_team"] == team]
        away_rows = rows[rows["_away_team"] == team]
        if len(home_rows) + len(away_rows) > 1:
            values = pd.concat([
                home_rows["h_form5_ppg"], away_rows["a_form5_ppg"]])
            assert values.nunique(dropna=True) == 1


def test_fixture_features_are_complete(matches):
    fixtures = pd.DataFrame([{
        "date": pd.Timestamp("2026-12-05"), "time": "15:00",
        "home_team": "Arsenal", "away_team": "Chelsea",
        "source": "test", "matchweek": 15,
    }])
    builder = FeatureBuilder()
    builder.build(matches)
    frame = builder.build_fixtures(fixtures)
    columns = feature_columns(frame)
    assert len(frame) == 1
    assert len(columns) >= 230
    assert frame["h_elo"].iloc[0] > 0
    assert not pd.isna(frame["pois_p_home"].iloc[0])


def test_season_transition_regresses_elo(features):
    """The first matchday of a season should pull ratings toward the mean."""
    seasons = features["_season_code"].unique().tolist()
    assert len(seasons) == 13
    first_2026 = features[features["_season_code"] == "2627"].iloc[0]
    assert config.ELO_START - 400 < first_2026["h_elo"] < config.ELO_START + 400
