"""Processed-dataset integrity and odds derivation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.processing import data_quality_report, load_matches, normalize_team


def test_match_table_shape(matches):
    assert len(matches) > 4000
    assert matches["date"].notna().all()
    assert matches["result"].isin(["H", "D", "A"]).all()


def test_no_duplicate_fixtures(matches):
    assert not matches.duplicated(["date", "home_team", "away_team"]).any()


def test_result_matches_scoreline(matches):
    expected = np.where(matches["home_goals"] > matches["away_goals"], "H",
                        np.where(matches["home_goals"] < matches["away_goals"], "A", "D"))
    assert (matches["result"].to_numpy() == expected).all()


def test_teams_never_play_themselves(matches):
    assert (matches["home_team"] != matches["away_team"]).all()


def test_market_probabilities_sum_to_one(matches):
    subset = matches.dropna(subset=["mkt_home", "mkt_draw", "mkt_away"])
    totals = subset[["mkt_home", "mkt_draw", "mkt_away"]].sum(axis=1)
    assert len(subset) > 4000
    assert np.allclose(totals, 1.0, atol=1e-6)


def test_market_probabilities_are_probabilities(matches):
    subset = matches.dropna(subset=["mkt_home", "mkt_draw", "mkt_away"])
    values = subset[["mkt_home", "mkt_draw", "mkt_away"]].to_numpy()
    assert (values >= 0).all() and (values <= 1).all()


def test_odds_coverage_is_high(matches):
    assert matches["mkt_home"].notna().mean() > 0.97


def test_stat_coverage_is_high(matches):
    for column in ("home_shots", "away_shots", "home_sot", "home_corners"):
        assert matches[column].notna().mean() > 0.95, column


def test_seasons_are_complete(matches):
    counts = matches.groupby("season").size()
    for season, count in counts.items():
        if season == "2026/27":
            assert 60 <= count <= 380
        else:
            assert 370 <= count <= 382, f"{season}: {count}"


def test_date_range_is_chronological(matches):
    assert matches["date"].is_monotonic_increasing


def test_every_team_name_is_canonical(matches):
    teams = set(matches["home_team"]) | set(matches["away_team"])
    for team in teams:
        assert normalize_team(team) == team


def test_data_quality_report_is_consistent(matches, tmp_path):
    report = data_quality_report(matches)
    assert report["n_matches"] == len(matches)
    assert report["duplicate_fixtures"] == 0
    assert report["invalid_results"] == 0
    assert sum(report["seasons"].values()) == len(matches)
    assert 0.4 < report["odds_coverage"] <= 1.0


def test_processed_csv_matches_loader(matches):
    reloaded = load_matches(refresh=False)
    assert len(reloaded) == len(matches)
    assert list(reloaded.columns) == list(matches.columns)
