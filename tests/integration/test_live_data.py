"""Live DATA loader and results-page tests with mocked stores.

Every test builds its own store files inside ``tmp_path`` and points the
loader at them, so the suite is deterministic and never touches the real
``data/`` artifacts. Covers the integration requirements: latest fixtures
displayed, probabilities matching production outputs, matchweek selection,
aggregates, freshness, missing-data resilience and cache invalidation.
"""
from __future__ import annotations

import json

import pytest

import data as data_module
import render


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
@pytest.fixture
def stores(tmp_path, monkeypatch):
    """Point every loader constant at an isolated directory."""
    paths = {
        "RESULTS_JSON": tmp_path / "results.json",
        "PREDICTIONS_JSON": tmp_path / "predictions.json",
        "MATCHWEEKS_JSON": tmp_path / "matchweeks.json",
        "FRESHNESS_JSON": tmp_path / "freshness.json",
        "MATCHES_CSV": tmp_path / "matches.csv",
    }
    for name, path in paths.items():
        monkeypatch.setattr(data_module, name, path)
    data_module._last_fingerprint = None
    data_module._last_built_at = 0.0
    data_module.DATA.clear()
    yield tmp_path
    data_module._last_fingerprint = None
    data_module._last_built_at = 0.0
    data_module.DATA.clear()


def _build():
    """Rebuild DATA from the mocked stores and return it."""
    data_module.refresh(ttl_seconds=0.0)
    return data_module.DATA


def _prediction(fixture_id: str, home: str, away: str, date: str = "2026-11-21",
                ph: float = 0.5, pd_: float = 0.3, pa: float = 0.2,
                predicted: str = "H", matchweek: int = 11) -> dict:
    return {
        "fixture_id": fixture_id,
        "match_id": f"{date}_{home}_{away}",
        "date": date,
        "time": "15:00",
        "home_team": home,
        "away_team": away,
        "prob_home": ph,
        "prob_draw": pd_,
        "prob_away": pa,
        "predicted": predicted,
        "model_version": "test-model-001",
        "model_variant": "no_odds",
        "matchweek": matchweek,
        "season": "2026/27",
        "status": "scheduled",
    }


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _result(fixture_id: str, home: str, away: str, gh: int, ga: int,
            matchweek: int = 11, date: str = "2026-11-21") -> dict:
    return {
        "fixture_id": fixture_id,
        "season_code": "2627",
        "season": "2026/27",
        "matchweek": matchweek,
        "date": date,
        "time": "15:00",
        "home_team": home,
        "away_team": away,
        "goals_home": gh,
        "goals_away": ga,
        "actual_outcome": "H" if gh > ga else "D" if gh == ga else "A",
        "status": "completed",
        "source": "football-data",
        "first_synced_at": "2026-11-22T00:00:00+00:00",
        "last_synced_at": "2026-11-22T00:00:00+00:00",
        "corrections": [],
    }


# --------------------------------------------------------------------------
# Fixtures display and probabilities
# --------------------------------------------------------------------------
def test_latest_fixtures_come_from_predictions(stores):
    _write(data_module.PREDICTIONS_JSON, {"fixtures": [
        _prediction("2627:Arsenal:Chelsea", "Arsenal", "Chelsea"),
        _prediction("2627:Hull:Everton", "Hull", "Everton", date="2026-11-22"),
    ]})
    fresh = _build()
    assert [f["home"] for f in fresh["fixtures"]] == ["Arsenal", "Hull"]
    assert fresh["fixtures"][0]["id"] == "2026-11-21_Arsenal_Chelsea"
    assert fresh["meta"]["matchweek"] == 11


def test_chances_are_production_outputs_summing_to_100(stores):
    _write(data_module.PREDICTIONS_JSON, {"fixtures": [
        _prediction("2627:Arsenal:Chelsea", "Arsenal", "Chelsea",
                    ph=0.4895351540432115, pd_=0.2742383809557038,
                    pa=0.2362264650010846),
    ]})
    chances = _build()["fixtures"][0]["chances"]
    # The approved prototype's rounding for these exact production values.
    assert (chances["home"], chances["draw"], chances["away"]) == (49, 27, 24)
    assert sum(chances.values()) == 100


def test_pick_follows_the_argmax_rule(stores):
    _write(data_module.PREDICTIONS_JSON, {"fixtures": [
        _prediction("2627:Hull:Everton", "Hull", "Everton",
                    ph=0.2, pd_=0.25, pa=0.55, predicted="A"),
        _prediction("2627:Arsenal:Chelsea", "Arsenal", "Chelsea",
                    ph=0.25, pd_=0.5, pa=0.25, predicted="D"),
    ]})
    picks = {f["home"]: f["pick"] for f in _build()["fixtures"]}
    assert picks["Hull"] == "away" and picks["Arsenal"] == "draw"


# --------------------------------------------------------------------------
# Missing-data resilience
# --------------------------------------------------------------------------
def test_empty_stores_never_crash_and_show_honest_empty_states(stores):
    fresh = _build()
    assert fresh["fixtures"] == [] and fresh["table"] == []
    assert fresh["seasonNumbers"]["matchesPlayed"] == 0
    assert fresh["meta"]["dataCutoff"] is None
    # Every page still renders.
    for page in ("overview", "week", "table", "results"):
        html, _ = render.shell(page)
        assert html


def test_results_page_empty_state(stores):
    _build()
    html, _ = render.shell("results")
    assert "No scored predictions yet" in html


# --------------------------------------------------------------------------
# Results, aggregates and freshness
# --------------------------------------------------------------------------
def test_results_page_shows_aggregates_and_verdicts(stores):
    _write(data_module.RESULTS_JSON, {"updated_at": "2026-11-22T00:00:00+00:00",
                                      "results": {
        "2627:Arsenal:Chelsea": _result("2627:Arsenal:Chelsea", "Arsenal", "Chelsea", 2, 0),
        "2627:Hull:Everton": _result("2627:Hull:Everton", "Hull", "Everton", 1, 1),
    }})
    _write(data_module.PREDICTIONS_JSON, {"fixtures": []})
    _write(data_module.MATCHWEEKS_JSON, {"seasons": {"2627": {
        "matchweeks": {"11": {
            "matchweek": 11, "scheduled": 2, "completed": 2, "evaluated": 2,
            "correct": 1, "incorrect": 1, "pending": 0, "missing_prediction": 0,
            "accuracy": 50.0, "coverage": 100.0}},
        "season": {"scheduled": 2, "completed": 2, "evaluated": 2, "correct": 1,
                   "incorrect": 1, "pending": 0, "missing_prediction": 0,
                   "accuracy": 50.0},
    }}})
    _write(data_module.FRESHNESS_JSON, {
        "fixtures": {"last_sync_at": "2026-11-14T05:45:00+00:00"},
        "predictions": {"last_run": "2026-11-14T06:15:00+00:00"},
        "results": {"last_sync_at": "2026-11-22T00:00:00+00:00"},
        "evaluation": {"last_run": "2026-11-22T00:01:00+00:00"},
    })
    fresh = _build()
    assert fresh["performance"]["season"]["accuracy"] == 50.0

    html, _ = render.shell("results")
    assert "Season so far" in html
    assert "Matchweek 11 performance" in html
    assert "1 / 2" in html and "50.0% accuracy" in html
    assert "Data freshness" in html
    assert "2026-11-14 05:45 UTC" in html   # fixture sync freshness
    assert "2026-11-14 06:15 UTC" in html   # prediction freshness, distinct
    # Pending and missing-prediction are reported, never folded into accuracy.
    assert "Pending: 0" in html and "No eligible prediction: 0" in html


def test_matchweek_selection_shows_requested_week(stores):
    _write(data_module.MATCHWEEKS_JSON, {"seasons": {"2627": {
        "matchweeks": {
            "10": {"matchweek": 10, "scheduled": 10, "completed": 10,
                   "evaluated": 0, "correct": 0, "incorrect": 0, "pending": 0,
                   "missing_prediction": 10, "accuracy": None, "coverage": 0.0},
            "11": {"matchweek": 11, "scheduled": 10, "completed": 0,
                   "evaluated": 0, "correct": 0, "incorrect": 0, "pending": 10,
                   "missing_prediction": 0, "accuracy": None, "coverage": None},
        },
        "season": {},
    }}})
    fresh = _build()
    assert len(fresh["performance"]["matchweeks"]) == 2

    html, _ = render.shell("results", mw=10)
    assert "Matchweek 10 performance" in html
    assert "No eligible prediction: 10" in html
    # Zero evaluated predictions -> honest em dash, never a 0%.
    assert "0.0% accuracy" not in html


# --------------------------------------------------------------------------
# Cache behaviour
# --------------------------------------------------------------------------
def test_refresh_rebuilds_when_a_store_changes(stores):
    _write(data_module.PREDICTIONS_JSON, {"fixtures": [
        _prediction("2627:Arsenal:Chelsea", "Arsenal", "Chelsea")]})
    _build()
    assert len(data_module.DATA["fixtures"]) == 1

    # A newly synchronized fixture must appear without a restart.
    _write(data_module.PREDICTIONS_JSON, {"fixtures": [
        _prediction("2627:Arsenal:Chelsea", "Arsenal", "Chelsea"),
        _prediction("2627:Hull:Everton", "Hull", "Everton")]})
    data_module.refresh(ttl_seconds=0.0)
    assert len(data_module.DATA["fixtures"]) == 2
    assert render.DATA["fixtures"][1]["home"] == "Hull", \
        "render's imported binding must see fresh values"


def test_refresh_skips_rebuild_when_nothing_changed(stores):
    _write(data_module.PREDICTIONS_JSON, {"fixtures": []})
    _build()
    first_built = data_module._last_built_at
    data_module.refresh(ttl_seconds=3600)
    assert data_module._last_built_at == first_built


# --------------------------------------------------------------------------
# Structural parity with the approved prototype
# --------------------------------------------------------------------------
def test_meta_and_full_match_prototype_conventions(stores):
    _write(data_module.PREDICTIONS_JSON, {"fixtures": [
        _prediction("2627:Arsenal:Chelsea", "Arsenal", "Chelsea")]})
    fresh = _build()
    assert fresh["meta"]["season"] == "2026/27"
    assert fresh["meta"]["dateLong"] == "Saturday, 21 November 2026"
    # The prototype expands exactly these two display names.
    assert fresh["full"] == {"Man United": "Manchester United",
                             "Man City": "Manchester City"}


def test_table_and_season_numbers_use_real_results(stores):
    import pandas as pd

    frame = pd.DataFrame([
        {"date": "2026-08-21", "time": "20:00", "season_code": "2627",
         "season": "2026/27", "home_team": "Arsenal", "away_team": "Chelsea",
         "home_goals": 3, "away_goals": 0, "result": "H"},
        {"date": "2026-08-22", "time": "15:00", "season_code": "2627",
         "season": "2026/27", "home_team": "Hull", "away_team": "Everton",
         "home_goals": 1, "away_goals": 1, "result": "D"},
    ])
    frame.to_csv(data_module.MATCHES_CSV, index=False)
    fresh = _build()
    table = {t["team"]: t for t in fresh["table"]}
    assert table["Arsenal"]["played"] == 1 and table["Arsenal"]["points"] == 3
    assert table["Arsenal"]["form"] == ["W"]          # newest first
    assert table["Everton"]["form"] == ["D"]
    sn = fresh["seasonNumbers"]
    assert sn["matchesPlayed"] == 2 and sn["goals"] == 5
    assert sn["goalsPerGame"] == 2.5
    assert sn["homeWins"] == 1 and sn["draws"] == 1 and sn["awayWins"] == 0
    assert sn["bothScored"] == 1                      # the 1-1 draw
    assert sn["cleanSheets"] == 1                     # Arsenal's clean sheet
    assert fresh["meta"]["dataCutoff"] == "2026-08-22"


# --------------------------------------------------------------------------
# Matchweek progression: active / last completed / next are separate keys,
# and confirmed cancellations are reported, never folded into pending.
# --------------------------------------------------------------------------
def test_meta_reports_last_completed_and_next_matchweek_separately(stores):
    _write(data_module.PREDICTIONS_JSON, {"fixtures": [
        _prediction("2627:Arsenal:Chelsea", "Arsenal", "Chelsea", matchweek=11),
        _prediction("2627:Hull:Everton", "Hull", "Everton", date="2026-11-22",
                    matchweek=11),
    ]})
    _write(data_module.MATCHWEEKS_JSON, {"seasons": {"2627": {
        "matchweeks": {
            # MW10 fully completed.
            "10": {"matchweek": 10, "scheduled": 10, "completed": 10,
                   "evaluated": 10, "correct": 6, "incorrect": 4, "pending": 0,
                   "missing_prediction": 0, "accuracy": 60.0, "coverage": 100.0},
            # MW11 in progress: still outstanding fixtures.
            "11": {"matchweek": 11, "scheduled": 10, "completed": 2,
                   "evaluated": 2, "correct": 1, "incorrect": 1, "pending": 8,
                   "missing_prediction": 0, "accuracy": 50.0, "coverage": 100.0},
            # MW12 published, not started.
            "12": {"matchweek": 12, "scheduled": 10, "completed": 0,
                   "evaluated": 0, "correct": 0, "incorrect": 0, "pending": 10,
                   "missing_prediction": 0, "accuracy": None, "coverage": None},
        },
        "season": {},
    }}})
    fresh = _build()

    assert fresh["meta"]["matchweek"] == 11               # active: still to play
    assert fresh["meta"]["lastCompletedMatchweek"] == 10  # fully completed
    assert fresh["meta"]["nextMatchweek"] == 12           # next with upcoming


def test_results_page_reports_confirmed_cancellations(stores):
    _write(data_module.PREDICTIONS_JSON, {"fixtures": []})
    _write(data_module.MATCHWEEKS_JSON, {"seasons": {"2627": {
        "matchweeks": {
            "11": {"matchweek": 11, "scheduled": 9, "completed": 9,
                   "evaluated": 9, "correct": 5, "incorrect": 4, "pending": 0,
                   "cancelled": 1, "missing_prediction": 0, "accuracy": 55.56,
                   "coverage": 100.0},
            "12": {"matchweek": 12, "scheduled": 10, "completed": 0,
                   "evaluated": 0, "correct": 0, "incorrect": 0, "pending": 10,
                   "cancelled": 0, "missing_prediction": 0, "accuracy": None,
                   "coverage": None},
        },
        "season": {"cancelled": 1},
    }}})
    _build()
    html, _ = render.shell("results", mw=11)
    assert "Cancelled: 1" in html
    assert "Pending: 0" in html

    # A week without cancellations shows no cancelled entry at all.
    html, _ = render.shell("results", mw=12)
    assert "Cancelled" not in html
