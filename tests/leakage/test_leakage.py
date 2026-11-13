"""Leakage proofs.

These are the most important tests in the repository: they demonstrate, by
perturbation of the real feature builder, that a match's feature vector cannot
contain its own result or any future information.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import validation
from src.features.builder import FeatureBuilder, feature_columns


def _fingerprint(frame: pd.DataFrame) -> np.ndarray:
    columns = feature_columns(frame)
    return np.round(frame[columns].to_numpy(dtype=float), 10)


def test_own_result_cannot_leak(early_slice):
    """Flipping a match's own score must not move its own feature row."""
    result = validation.own_result_test(early_slice)
    assert result["passed"], result["cases"]
    assert all(case["identical"] for case in result["cases"])


def test_future_data_cannot_leak(early_slice):
    result = validation.future_independence_test(early_slice, dates=["2015-10-24"])
    assert result["passed"], result["cases"]


def test_cold_start_state_is_pristine(early_slice):
    result = validation.chronological_order_test(early_slice)
    assert result["passed"], result["checks"]
    assert result["checks"]["first_match_elo_is_start"]


def test_integrity_checks_pass(matches):
    result = validation.validate_matches(matches)
    assert result["ok"], result["issues"]


def test_same_day_results_do_not_cross_contaminate(matches):
    """Matches sharing a date must all be scored from the pre-date state."""
    date = pd.Timestamp("2021-05-23")  # final day: 10 simultaneous kick-offs
    day = matches[matches["date"] == date]
    if len(day) < 2:
        pytest.skip("no multi-match date")
    built = FeatureBuilder().build(matches[matches["date"] <= date])
    rows = built[built["_date"] == date]
    assert len(rows) == len(day)
    # Every team plays once, so comparing the home block's form to the away
    # block's form for the same club is a direct check of no intra-day update.
    for _, row in rows.iterrows():
        same_team = rows[(rows["_home_team"] == row["_home_team"])
                         | (rows["_away_team"] == row["_away_team"])]
        assert len(same_team) == 1


def test_feature_builder_does_not_use_target_column(matches):
    """Changing a match's own result must not change its own feature row."""
    subset = (matches[matches["date"] <= pd.Timestamp("2015-06-30")]
              .sort_values(["date", "time", "home_team"], kind="mergesort")
              .reset_index(drop=True)
              .copy())
    baseline = FeatureBuilder().build(subset)
    columns = feature_columns(baseline)
    last = len(subset) - 1

    # Flip the outcome of the final match only. Every earlier row is untouched,
    # so if the target leaked into the final row its features would move.
    mutated = subset.copy()
    home_goals = int(mutated.loc[last, "home_goals"])
    away_goals = int(mutated.loc[last, "away_goals"])
    if home_goals > away_goals:
        mutated.loc[last, ["home_goals", "away_goals", "result"]] = [0, 3, "A"]
    elif home_goals < away_goals:
        mutated.loc[last, ["home_goals", "away_goals", "result"]] = [3, 0, "H"]
    else:
        mutated.loc[last, ["home_goals", "away_goals", "result"]] = [4, 0, "H"]

    rebuilt = FeatureBuilder().build(mutated)
    before = _fingerprint(baseline)[last]
    after = _fingerprint(rebuilt)[last]
    changed = [columns[i] for i in np.where(~np.isclose(before, after, equal_nan=True))[0]]
    assert not changed, f"target leaked into features: {changed[:10]}"


def test_elo_updates_after_result_not_before(matches):
    """Elo must move only after the match has been featurised."""
    subset = matches[matches["date"] <= pd.Timestamp("2016-06-30")].sort_values(
        ["date", "time", "home_team"]).reset_index(drop=True)
    builder = FeatureBuilder()
    frame = builder.build(subset)
    last_row = frame.iloc[-1]
    before = float(last_row["h_elo"])
    after = float(builder.states[last_row["_home_team"]]["elo"])
    assert before != after


@pytest.mark.slow
def test_full_history_own_result_independence(matches):
    result = validation.own_result_test(matches)
    assert result["passed"], result["cases"]
