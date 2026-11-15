"""Evaluation and matchweek-aggregation tests.

Snapshots and results are produced by the real ``snapshots`` and ``results``
modules from mocked inputs; everything lives inside ``tmp_path``. Includes
the concrete example from the specification: 10 scheduled fixtures, 9
completed with valid predictions (7 correct, 2 incorrect), 1 pending ->
77.78% accuracy.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from src.evaluation import evaluate
from src.prediction import snapshots
from src.results import sync as result_sync

NOW = "2026-11-22T12:00:00+00:00"           # after the MW11 fixtures
KICKOFF = "2026-11-21T15:00:00+00:00"
PREDICTED_AT = "2026-11-14T10:00:00+00:00"
SEASON = "2026/27"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _store_rows(specs: list[dict]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        rows.append({
            "fixture_id": spec["fixture_id"],
            "season_code": "2627",
            "matchweek": spec.get("matchweek", 11),
            "date": spec.get("date", "2026-11-21"),
            "time": spec.get("time", "15:00"),
            "status": spec.get("status", "scheduled"),
        })
    return pd.DataFrame(rows)


def _prediction(fixture_id: str, predicted: str = "H",
                created: str = PREDICTED_AT, matchweek: int = 11) -> dict:
    probabilities = {"H": (0.55, 0.25, 0.20), "D": (0.25, 0.50, 0.25),
                     "A": (0.20, 0.25, 0.55)}[predicted]
    return {
        "fixture_id": fixture_id,
        "home_team": fixture_id.split(":")[1],
        "away_team": fixture_id.split(":")[2],
        "season": SEASON,
        "matchweek": matchweek,
        "prob_home": probabilities[0],
        "prob_draw": probabilities[1],
        "prob_away": probabilities[2],
        "predicted": predicted,
        "model_version": "test-model-001",
        "model_variant": "no_odds",
        "has_odds": False,
        "predicted_at": created,
    }


def _snapshot_store(tmp_path, fixtures: list[str], predicted: dict[str, str],
                    store_rows: pd.DataFrame) -> None:
    weeks = dict(zip(store_rows["fixture_id"], store_rows["matchweek"]))
    dates = dict(zip(store_rows["fixture_id"], store_rows["date"]))
    predictions = []
    for fid in fixtures:
        # Keep the synthetic data internally consistent: a prediction for a
        # fixture that kicked off before PREDICTED_AT must carry an earlier
        # (pre-kickoff) creation timestamp.
        created = PREDICTED_AT
        row_date = dates.get(fid)
        if row_date is not None and pd.Timestamp(row_date) < pd.Timestamp(PREDICTED_AT[:10]):
            created = (f"{pd.Timestamp(row_date) - pd.Timedelta(days=1):%Y-%m-%d}"
                       "T10:00:00+00:00")
        predictions.append(_prediction(fid, predicted.get(fid, "H"),
                                       created=created, matchweek=weeks.get(fid, 11)))
    snapshots.reconcile(store_rows, predictions, now=PREDICTED_AT,
                        path=tmp_path / "snapshots.json")


def _result(fixture_id: str, gh: int, ga: int, matchweek: int = 11) -> dict:
    home, away = fixture_id.split(":")[1:]
    return {
        "fixture_id": fixture_id,
        "season_code": "2627",
        "season": SEASON,
        "matchweek": matchweek,
        "date": "2026-11-21",
        "time": "15:00",
        "home_team": home,
        "away_team": away,
        "goals_home": gh,
        "goals_away": ga,
        "actual_outcome": "H" if gh > ga else "D" if gh == ga else "A",
        "status": "completed",
        "source": "football-data",
        "first_synced_at": NOW,
        "last_synced_at": NOW,
        "corrections": [],
    }


def _write_results(tmp_path, results: list[dict]) -> None:
    payload = {"updated_at": NOW,
               "results": {r["fixture_id"]: r for r in results}}
    (tmp_path / "results.json").write_text(json.dumps(payload), encoding="utf-8")


def _evaluate(tmp_path, now: str = NOW) -> dict:
    return evaluate.evaluate(now=now,
                             snapshots_path=tmp_path / "snapshots.json",
                             results_path=tmp_path / "results.json",
                             evaluations_path=tmp_path / "evaluations.json")


def _aggregate(tmp_path, store_rows: pd.DataFrame, now: str = NOW) -> dict:
    return evaluate.aggregate(now=now,
                              results_path=tmp_path / "results.json",
                              evaluations_path=tmp_path / "evaluations.json",
                              store_path=tmp_path / "matchweeks.json",
                              snapshots_path=tmp_path / "snapshots.json",
                              fixture_store=store_rows)


def _weeks(aggregate_report: dict) -> dict:
    return aggregate_report["seasons"]["2627"]["matchweeks"]


def _evaluations(tmp_path) -> dict:
    with open(tmp_path / "evaluations.json", encoding="utf-8") as handle:
        return json.load(handle)["evaluations"]


# --------------------------------------------------------------------------
# Evaluation correctness
# --------------------------------------------------------------------------
def test_correct_incorrect_and_draw_evaluation(tmp_path):
    fixtures = ["2627:Arsenal:Chelsea", "2627:Hull:Everton", "2627:Leeds:Burnley"]
    _snapshot_store(tmp_path, fixtures,
                    {"2627:Arsenal:Chelsea": "H", "2627:Hull:Everton": "D",
                     "2627:Leeds:Burnley": "A"},
                    _store_rows([{"fixture_id": f} for f in fixtures]))
    _write_results(tmp_path, [
        _result("2627:Arsenal:Chelsea", 2, 0),   # predicted H -> correct
        _result("2627:Hull:Everton", 1, 1),      # predicted D -> correct
        _result("2627:Leeds:Burnley", 2, 1),     # predicted A -> incorrect
    ])
    report = _evaluate(tmp_path)
    assert report["new"] == 3 and report["evaluated"] == 3
    evaluations = _evaluations(tmp_path)
    assert evaluations["2627:Arsenal:Chelsea"]["correct"] is True
    assert evaluations["2627:Hull:Everton"]["correct"] is True
    assert evaluations["2627:Leeds:Burnley"]["correct"] is False
    record = evaluations["2627:Arsenal:Chelsea"]
    # Spec section 7: every required field is present.
    for field in ("fixture_id", "season", "matchweek", "home_team", "away_team",
                  "predicted_outcome", "prob_home", "prob_draw", "prob_away",
                  "goals_home", "goals_away", "actual_outcome", "correct",
                  "prediction_created_at", "result_synced_at", "model_id"):
        assert field in record, field


def test_evaluation_idempotent(tmp_path):
    fixtures = ["2627:Arsenal:Chelsea"]
    _snapshot_store(tmp_path, fixtures, {}, _store_rows([{"fixture_id": fixtures[0]}]))
    _write_results(tmp_path, [_result(fixtures[0], 2, 1)])
    _evaluate(tmp_path)
    first = (tmp_path / "evaluations.json").read_bytes()
    report = _evaluate(tmp_path, now="2026-11-22T13:00:00+00:00")
    assert report["new"] == 0 and report["unchanged"] == 1 and report["recomputed"] == 0
    assert (tmp_path / "evaluations.json").read_bytes() == first


def test_completed_without_prediction_never_incorrect(tmp_path):
    fixtures = ["2627:Arsenal:Chelsea", "2627:Hull:Everton"]
    # Only Arsenal:Chelsea gets a pre-match snapshot.
    _snapshot_store(tmp_path, [fixtures[0]], {}, _store_rows(
        [{"fixture_id": f} for f in fixtures]))
    # Hull:Everton has an official result but no snapshot: reported, not guessed.
    _write_results(tmp_path, [
        _result("2627:Arsenal:Chelsea", 2, 0),
        _result("2627:Hull:Everton", 3, 3),
    ])
    report = _evaluate(tmp_path)
    assert report["missing_predictions"] == 1
    assert "2627:Hull:Everton" not in _evaluations(tmp_path)
    aggregate = _aggregate(tmp_path, _store_rows([{"fixture_id": f} for f in fixtures]))
    week = _weeks(aggregate)["11"]
    assert week["evaluated"] == 1 and week["missing_prediction"] == 1
    assert week["incorrect"] == 0 and week["correct"] == 1
    assert week["accuracy"] == 100.0


def test_corrected_result_recomputes_with_audit_trail(tmp_path):
    fixture = "2627:Arsenal:Chelsea"
    _snapshot_store(tmp_path, [fixture], {"2627:Arsenal:Chelsea": "H"},
                    _store_rows([{"fixture_id": fixture}]))
    _write_results(tmp_path, [_result(fixture, 2, 1)])
    _evaluate(tmp_path)
    assert _evaluations(tmp_path)[fixture]["correct"] is True

    # Authoritative correction: the draw was confirmed after review.
    _write_results(tmp_path, [_result(fixture, 2, 2)])
    report = _evaluate(tmp_path, now="2026-11-23T09:00:00+00:00")
    assert report["recomputed"] == 1
    record = _evaluations(tmp_path)[fixture]
    assert record["correct"] is False
    assert record["goals_home"] == 2 and record["goals_away"] == 2
    assert record["actual_outcome"] == "D"
    assert len(record["history"]) == 1
    assert record["history"][0]["previous_correct"] is True
    assert record["history"][0]["previous_goals_home"] == 2
    assert record["history"][0]["previous_goals_away"] == 1
    assert record["history"][0]["reason"] == "result correction"


# --------------------------------------------------------------------------
# The specification's concrete example
# --------------------------------------------------------------------------
def test_matchweek_example_ten_scheduled_seven_correct(tmp_path):
    """10 scheduled, 9 completed (7 correct, 2 incorrect), 1 pending."""
    fixtures = [f"2627:Team{i}:Rival{i}" for i in range(1, 11)]
    predicted = {}
    outcomes = ["H", "H", "H", "H", "H", "H", "H",  # 7 correct home wins
                "A", "A",                            # 2 incorrect
                "P"]                                 # 1 pending (no result)
    for fixture, outcome in zip(fixtures, outcomes):
        # The two away wins were predicted as draws -> the 2 incorrect ones.
        predicted[fixture] = "D" if outcome == "A" else "H"

    rows = _store_rows([{"fixture_id": f} for f in fixtures])
    _snapshot_store(tmp_path, fixtures, predicted, rows)

    results = []
    for fixture, outcome in zip(fixtures, outcomes):
        if outcome == "H":
            results.append(_result(fixture, 2, 0))
        elif outcome == "A":
            results.append(_result(fixture, 0, 1))
    _write_results(tmp_path, results)

    report = _evaluate(tmp_path)
    assert report["evaluated"] == 9 and report["new"] == 9

    aggregate = _aggregate(tmp_path, rows)
    week = _weeks(aggregate)["11"]
    assert week["scheduled"] == 10
    assert week["completed"] == 9
    assert week["evaluated"] == 9
    assert week["correct"] == 7
    assert week["incorrect"] == 2
    assert week["pending"] == 1
    assert week["missing_prediction"] == 0
    assert week["accuracy"] == pytest.approx(77.78)
    assert week["coverage"] == 100.0

    season = aggregate["seasons"]["2627"]["season"]
    assert season["scheduled"] == 10 and season["evaluated"] == 9
    assert season["correct"] == 7 and season["incorrect"] == 2
    assert season["accuracy"] == pytest.approx(77.78)


# --------------------------------------------------------------------------
# Aggregation edge cases
# --------------------------------------------------------------------------
def test_zero_evaluated_shows_null_accuracy_not_zero(tmp_path):

    fixture = "2627:Arsenal:Chelsea"
    rows = _store_rows([{"fixture_id": fixture}])
    _snapshot_store(tmp_path, [fixture], {}, rows)
    _write_results(tmp_path, [])  # nothing completed yet
    aggregate = _aggregate(tmp_path, rows)
    week = _weeks(aggregate)["11"]
    assert week["scheduled"] == 1 and week["pending"] == 1
    assert week["evaluated"] == 0 and week["accuracy"] is None


def test_postponed_fixture_is_pending_never_incorrect(tmp_path):
    fixtures = ["2627:Arsenal:Chelsea", "2627:Hull:Everton"]
    rows = _store_rows([
        {"fixture_id": fixtures[0]},
        {"fixture_id": fixtures[1], "status": "postponed", "date": "2026-12-02"},
    ])
    _snapshot_store(tmp_path, fixtures, {}, rows)
    _write_results(tmp_path, [_result(fixtures[0], 2, 1)])
    _evaluate(tmp_path)
    aggregate = _aggregate(tmp_path, rows)
    week = _weeks(aggregate)["11"]
    assert week["scheduled"] == 2 and week["completed"] == 1
    assert week["evaluated"] == 1 and week["pending"] == 1
    assert week["correct"] + week["incorrect"] == 1


def test_multiple_matchweeks_and_season_sum(tmp_path):
    fixtures = [f"2627:Team{i}:Rival{i}" for i in range(1, 5)]
    rows = _store_rows([
        {"fixture_id": fixtures[0], "matchweek": 10, "date": "2026-11-08"},
        {"fixture_id": fixtures[1], "matchweek": 10, "date": "2026-11-08"},
        {"fixture_id": fixtures[2], "matchweek": 11},
        {"fixture_id": fixtures[3], "matchweek": 11},
    ])
    _snapshot_store(tmp_path, fixtures, {}, rows)
    _write_results(tmp_path, [
        _result(fixtures[0], 2, 0, matchweek=10),   # H correct
        _result(fixtures[1], 0, 2, matchweek=10),   # H incorrect
        _result(fixtures[2], 1, 1, matchweek=11),   # H incorrect
    ])
    _evaluate(tmp_path)
    aggregate = _aggregate(tmp_path, rows)
    weeks = _weeks(aggregate)
    assert weeks["10"]["correct"] == 1 and weeks["10"]["incorrect"] == 1
    assert weeks["10"]["accuracy"] == 50.0
    assert weeks["11"]["evaluated"] == 1 and weeks["11"]["incorrect"] == 1
    assert weeks["11"]["accuracy"] == 0.0
    season = aggregate["seasons"]["2627"]["season"]
    assert season["scheduled"] == 4 and season["completed"] == 3
    assert season["evaluated"] == 3 and season["correct"] == 1
    assert season["incorrect"] == 2
    assert season["accuracy"] == pytest.approx(33.33)


def test_duplicate_evaluation_never_double_counts(tmp_path):
    fixture = "2627:Arsenal:Chelsea"
    rows = _store_rows([{"fixture_id": fixture}])
    _snapshot_store(tmp_path, [fixture], {}, rows)
    _write_results(tmp_path, [_result(fixture, 2, 1)])
    _evaluate(tmp_path)
    _evaluate(tmp_path, now="2026-11-22T13:00:00+00:00")
    evaluations = _evaluations(tmp_path)
    assert len(evaluations) == 1
    aggregate = _aggregate(tmp_path, rows)
    week = _weeks(aggregate)["11"]
    assert week["evaluated"] == 1 and week["correct"] == 1


# --------------------------------------------------------------------------
# Freshness
# --------------------------------------------------------------------------
def test_freshness_reports_each_source(tmp_path, monkeypatch):
    fixture = "2627:Arsenal:Chelsea"
    rows = _store_rows([{"fixture_id": fixture}])
    _snapshot_store(tmp_path, [fixture], {}, rows)
    _write_results(tmp_path, [_result(fixture, 2, 1)])
    _evaluate(tmp_path)
    evaluate._save_freshness.__wrapped__ if False else None

    monkeypatch.setattr(evaluate, "FRESHNESS_JSON", tmp_path / "freshness.json")
    monkeypatch.setattr(evaluate, "EVALUATIONS_JSON", tmp_path / "evaluations.json")
    monkeypatch.setattr(snapshots, "SNAPSHOTS_JSON", tmp_path / "snapshots.json")
    monkeypatch.setattr(result_sync, "RESULTS_JSON", tmp_path / "results.json")
    (tmp_path / "sync_state.json").write_text(json.dumps(
        {"last_run": {"finished_at": "2026-11-14T05:45:00+00:00", "status": "ok"}}),
        encoding="utf-8")
    monkeypatch.setattr(evaluate.config, "FIXTURE_SYNC_STATE", tmp_path / "sync_state.json")
    (tmp_path / "pipeline_state.json").write_text(json.dumps(
        {"last_success": "2026-11-14T06:15:00+00:00"}), encoding="utf-8")
    monkeypatch.setattr(evaluate.config, "DATA_DIR", tmp_path)

    payload = evaluate.freshness(now=NOW)
    assert payload["fixtures"]["last_sync_at"] == "2026-11-14T05:45:00+00:00"
    assert payload["predictions"]["last_run"] == "2026-11-14T06:15:00+00:00"
    assert payload["predictions"]["snapshots"] == 1
    assert payload["results"]["completed"] == 1
    assert payload["evaluation"]["evaluated"] == 1


# --------------------------------------------------------------------------
# Cancelled fixtures and born-ineligible (post-kickoff) snapshots
# --------------------------------------------------------------------------
def test_cancelled_fixtures_are_separate_never_pending_or_incorrect(tmp_path):
    fixtures = ["2627:Arsenal:Chelsea", "2627:Hull:Everton", "2627:Leeds:Tottenham"]
    rows = _store_rows([
        {"fixture_id": fixtures[0]},
        {"fixture_id": fixtures[1]},
        {"fixture_id": fixtures[2], "status": "cancelled"},
    ])
    _snapshot_store(tmp_path, fixtures, {fixtures[0]: "H", fixtures[1]: "D"}, rows)
    _write_results(tmp_path, [_result(fixtures[0], 2, 0), _result(fixtures[1], 1, 1)])
    _evaluate(tmp_path)
    week = _weeks(_aggregate(tmp_path, rows))["11"]

    # The cancelled fixture stays in its own bucket: never scheduled-pending,
    # never evaluated, never incorrect.
    assert week["scheduled"] == 2
    assert week["completed"] == 2
    assert week["evaluated"] == 2
    assert week["pending"] == 0
    assert week["cancelled"] == 1
    assert week["missing_prediction"] == 0
    assert week["correct"] == 2 and week["incorrect"] == 0
    assert week["accuracy"] == 100.0


def test_cancelled_only_matchweek_still_reports_its_cancellations(tmp_path):
    fixture = "2627:Arsenal:Chelsea"
    rows = _store_rows([{"fixture_id": fixture, "status": "cancelled"}])
    _snapshot_store(tmp_path, [fixture], {fixture: "H"}, rows)
    week = _weeks(_aggregate(tmp_path, rows))["11"]
    assert week["scheduled"] == 0
    assert week["pending"] == 0
    assert week["cancelled"] == 1
    assert week["accuracy"] is None


def test_born_ineligible_snapshot_is_missing_prediction_never_scored(tmp_path):
    """A snapshot created after kickoff can never be evaluated (acceptance 1)."""
    fixture = "2627:Arsenal:Chelsea"
    rows = _store_rows([{"fixture_id": fixture}])
    # The workflow ran after kickoff: the record is born locked and ineligible.
    after_kickoff = "2026-11-21T17:00:00+00:00"
    snapshots.reconcile(rows, [_prediction(fixture, created=after_kickoff)],
                        now=after_kickoff, path=tmp_path / "snapshots.json")
    _write_results(tmp_path, [_result(fixture, 2, 0)])

    report = _evaluate(tmp_path)
    assert report["evaluated"] == 0
    assert report["missing_predictions"] == 1
    assert report["needs_review"] and "eligible pre-match prediction" in report["needs_review"][0]["reason"]

    week = _weeks(_aggregate(tmp_path, rows))["11"]
    assert week["completed"] == 1
    assert week["evaluated"] == 0
    assert week["missing_prediction"] == 1
    assert week["correct"] == 0 and week["incorrect"] == 0
    assert week["accuracy"] is None           # honest null, never 0%
