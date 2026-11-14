"""Immutable pre-match prediction snapshot tests.

Everything is mocked: the fixture store rows and prediction payload rows are
built in-memory, the clock is injected through ``now`` and every test writes
into ``tmp_path``. The real ``data/predictions`` store is never touched and
the production engine is never loaded.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from src.prediction import snapshots

NOW = "2026-11-13T12:00:00+00:00"
KICKOFF = "2026-11-21T15:00:00+00:00"          # Europe/London resolved to UTC
AFTER_KICKOFF = "2026-11-21T17:00:00+00:00"
FIXTURE_ID = "2627:Arsenal:Chelsea"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _store_row(fixture_id: str = FIXTURE_ID, date: str = "2026-11-21",
               time: str = "15:00", status: str = "scheduled",
               season_code: str = "2627", matchweek: int = 11) -> pd.DataFrame:
    return pd.DataFrame([{
        "fixture_id": fixture_id,
        "season_code": season_code,
        "matchweek": matchweek,
        "date": date,
        "time": time,
        "status": status,
    }])


def _prediction(fixture_id: str = FIXTURE_ID,
                created: str = "2026-11-13T10:00:00+00:00",
                **over) -> dict:
    row = {
        "fixture_id": fixture_id,
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "season": "2026/27",
        "matchweek": 11,
        "prob_home": 0.50,
        "prob_draw": 0.30,
        "prob_away": 0.20,
        "predicted": "H",
        "confidence": 0.50,
        "model_version": "test-model-001",
        "model_variant": "no_odds",
        "has_odds": False,
        "predicted_at": created,
    }
    row.update(over)
    return row


def _reconcile(store: pd.DataFrame, predictions: list[dict],
               now: str = NOW, tmp_path=None, first: bool = False):
    path = tmp_path / "snapshots.json"
    return snapshots.reconcile(store, predictions, now=now, path=path)


def _snap(tmp_path, fixture_id: str = FIXTURE_ID) -> dict:
    with open(tmp_path / "snapshots.json", encoding="utf-8") as handle:
        return json.load(handle)["snapshots"][fixture_id]


# --------------------------------------------------------------------------
# Creation
# --------------------------------------------------------------------------
def test_creates_snapshot_with_utc_kickoff(tmp_path):
    report = _reconcile(_store_row(), [_prediction()], tmp_path=tmp_path)
    assert report["created"] == 1
    snapshot = _snap(tmp_path)
    assert snapshot["status"] == "pre_match"
    assert snapshot["kickoff_at"] == KICKOFF
    eligible = snapshot["eligible"]
    assert eligible["prob_home"] == pytest.approx(0.50)
    assert eligible["prob_draw"] == pytest.approx(0.30)
    assert eligible["prob_away"] == pytest.approx(0.20)
    assert eligible["predicted"] == "H"
    assert eligible["model_id"] == "test-model-001"
    assert eligible["prediction_status"] == "pre_match"


def test_probability_validation_rejects_bad_payloads(tmp_path):
    report = _reconcile(
        _store_row(),
        [_prediction(prob_home=0.6, prob_draw=0.3, prob_away=0.2)],  # sum 1.1
        tmp_path=tmp_path,
    )
    assert report["invalid"] == 1
    assert report["needs_review"] and "sum" in report["needs_review"][0]["reason"]
    with open(tmp_path / "snapshots.json", encoding="utf-8") as handle:
        assert json.load(handle)["snapshots"] == {}

    report = _reconcile(
        _store_row(),
        [_prediction(prob_home=1.5, prob_draw=-0.3, prob_away=-0.2)],
        tmp_path=tmp_path,
    )
    assert report["invalid"] == 1


def test_unknown_kickoff_is_locked_at_creation_and_never_revised(tmp_path):
    row = _store_row(date=None, time=None)
    row["date"] = pd.NaT
    report = _reconcile(row, [_prediction()], tmp_path=tmp_path)
    snapshot = _snap(tmp_path)
    assert snapshot["kickoff_at"] is None
    assert snapshot["locked_at_creation"] is True
    assert snapshot["status"] == "locked"

    # A later, different payload must never replace an unverifiable snapshot.
    report = _reconcile(
        row, [_prediction(prob_home=0.4, prob_draw=0.35, prob_away=0.25)],
        tmp_path=tmp_path,
    )
    assert report["revised"] == 0
    assert _snap(tmp_path)["eligible"]["prob_home"] == pytest.approx(0.50)


# --------------------------------------------------------------------------
# Idempotency / preservation
# --------------------------------------------------------------------------
def test_idempotent_rerun_changes_nothing(tmp_path):
    _reconcile(_store_row(), [_prediction()], tmp_path=tmp_path)
    first = _snap(tmp_path)["eligible"]
    report = _reconcile(_store_row(), [_prediction()], tmp_path=tmp_path)
    assert report["created"] == 0 and report["revised"] == 0
    assert report["unchanged"] == 1
    assert _snap(tmp_path)["eligible"] == first


def test_missing_prediction_is_reported_not_deleted(tmp_path):
    _reconcile(_store_row(), [_prediction()], tmp_path=tmp_path)
    report = _reconcile(_store_row(), [], tmp_path=tmp_path)
    assert report["missing"] == 1
    assert _snap(tmp_path)["eligible"]["prob_home"] == pytest.approx(0.50)


# --------------------------------------------------------------------------
# Locking after kickoff
# --------------------------------------------------------------------------
def test_locks_after_kickoff_and_freezes_eligible(tmp_path):
    _reconcile(_store_row(), [_prediction()], tmp_path=tmp_path)
    report = _reconcile(
        _store_row(),
        [_prediction(created=AFTER_KICKOFF,
                     prob_home=0.40, prob_draw=0.35, prob_away=0.25)],
        now=AFTER_KICKOFF, tmp_path=tmp_path,
    )
    assert report["locked"] == 1
    snapshot = _snap(tmp_path)
    assert snapshot["status"] == "locked"
    assert snapshot["eligible"]["prob_home"] == pytest.approx(0.50)


def test_post_kickoff_candidate_never_replaces_pre_match_version(tmp_path):
    _reconcile(_store_row(), [_prediction()], tmp_path=tmp_path)
    # Payload differs but was generated after kickoff.
    report = _reconcile(
        _store_row(),
        [_prediction(created="2026-11-21T16:00:00+00:00",
                     prob_home=0.10, prob_draw=0.20, prob_away=0.70,
                     predicted="A")],
        now="2026-11-21T18:00:00+00:00", tmp_path=tmp_path,
    )
    snapshot = _snap(tmp_path)
    assert snapshot["status"] == "locked"
    assert snapshot["eligible"]["prob_away"] == pytest.approx(0.20)
    assert snapshot["eligible"]["predicted"] == "H"
    assert snapshot["revisions"] == []


# --------------------------------------------------------------------------
# Authorized pre-kickoff revisions
# --------------------------------------------------------------------------
def test_pre_kickoff_revision_appends_history(tmp_path):
    _reconcile(_store_row(), [_prediction()], tmp_path=tmp_path)
    report = _reconcile(
        _store_row(),
        [_prediction(created="2026-11-20T09:00:00+00:00",
                     prob_home=0.45, prob_draw=0.30, prob_away=0.25)],
        now="2026-11-20T10:00:00+00:00", tmp_path=tmp_path,
    )
    assert report["revised"] == 1
    snapshot = _snap(tmp_path)
    assert snapshot["eligible"]["prob_home"] == pytest.approx(0.45)
    assert snapshot["eligible"]["prediction_created_at"] == "2026-11-20T09:00:00+00:00"
    assert len(snapshot["revisions"]) == 1
    superseded = snapshot["revisions"][0]
    assert superseded["prob_home"] == pytest.approx(0.50)
    assert superseded["prediction_created_at"] == "2026-11-13T10:00:00+00:00"
    assert superseded["superseded_at"] == "2026-11-20T10:00:00+00:00"


def test_revision_created_after_kickoff_is_rejected_even_pre_lock(tmp_path):
    _reconcile(_store_row(), [_prediction()], tmp_path=tmp_path)
    # Still pre-kickoff "now", but the candidate claims a post-kickoff creation.
    report = _reconcile(
        _store_row(),
        [_prediction(created="2026-11-21T16:30:00+00:00",
                     prob_home=0.40, prob_draw=0.35, prob_away=0.25)],
        now="2026-11-21T14:00:00+00:00", tmp_path=tmp_path,
    )
    snapshot = _snap(tmp_path)
    assert snapshot["eligible"]["prob_home"] == pytest.approx(0.50)
    assert snapshot["revisions"] == []


# --------------------------------------------------------------------------
# Postponements, reschedules and cancellations
# --------------------------------------------------------------------------
def test_reschedule_moves_kickoff_and_keeps_prediction(tmp_path):
    _reconcile(_store_row(), [_prediction()], tmp_path=tmp_path)
    rescheduled = _store_row(date="2026-11-22", time="20:00")
    report = _reconcile(rescheduled, [_prediction()],
                        now="2026-11-19T12:00:00+00:00", tmp_path=tmp_path)
    snapshot = _snap(tmp_path)
    assert snapshot["kickoff_at"] == "2026-11-22T20:00:00+00:00"
    assert snapshot["status"] == "pre_match"
    assert report["status_updates"] == 1
    assert snapshot["eligible"]["prob_home"] == pytest.approx(0.50)


def test_postponed_then_rescheduled_allows_revision_before_new_kickoff(tmp_path):
    _reconcile(_store_row(), [_prediction()], tmp_path=tmp_path)
    postponed = _store_row(status="postponed", date=None)
    postponed["date"] = pd.NaT
    _reconcile(postponed, [_prediction()],
               now="2026-11-18T12:00:00+00:00", tmp_path=tmp_path)
    assert _snap(tmp_path)["status"] == "postponed"

    # Provider publishes the new date before kickoff: snapshot returns to
    # pre-match and an authorized revision (e.g. odds now published) applies.
    rescheduled = _store_row(date="2026-12-02", time="19:45")
    report = _reconcile(
        rescheduled,
        [_prediction(created="2026-12-01T09:00:00+00:00",
                     prob_home=0.48, prob_draw=0.28, prob_away=0.24)],
        now="2026-12-01T10:00:00+00:00", tmp_path=tmp_path,
    )
    snapshot = _snap(tmp_path)
    assert snapshot["kickoff_at"] == "2026-12-02T19:45:00+00:00"
    assert snapshot["status"] == "pre_match"
    assert report["revised"] == 1
    assert snapshot["eligible"]["prob_home"] == pytest.approx(0.48)
    assert len(snapshot["revisions"]) == 1


def test_cancelled_fixture_is_never_revised(tmp_path):
    _reconcile(_store_row(), [_prediction()], tmp_path=tmp_path)
    report = _reconcile(_store_row(status="cancelled"), [_prediction()],
                        tmp_path=tmp_path)
    snapshot = _snap(tmp_path)
    assert snapshot["status"] == "cancelled"
    assert snapshot["eligible"]["prob_home"] == pytest.approx(0.50)
    assert report["revised"] == 0


# --------------------------------------------------------------------------
# Bookkeeping helpers
# --------------------------------------------------------------------------
def test_mark_completed_is_idempotent(tmp_path):
    _reconcile(_store_row(), [_prediction()], tmp_path=tmp_path)
    assert snapshots.mark_completed([FIXTURE_ID], path=tmp_path / "snapshots.json") == 1
    assert _snap(tmp_path)["status"] == "completed"
    assert snapshots.mark_completed([FIXTURE_ID], path=tmp_path / "snapshots.json") == 0
    # Eligible prediction survives the status change.
    assert _snap(tmp_path)["eligible"]["prob_home"] == pytest.approx(0.50)


def test_eligible_predictions_map(tmp_path):
    _reconcile(_store_row(), [_prediction()], tmp_path=tmp_path)
    eligible = snapshots.eligible_predictions(path=tmp_path / "snapshots.json")
    assert list(eligible) == [FIXTURE_ID]
    assert eligible[FIXTURE_ID]["predicted"] == "H"
