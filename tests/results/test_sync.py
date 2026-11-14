"""Result synchronisation tests.

Provider payloads (a football-data season CSV and the openfootball schedule)
are mocked as small local files inside ``tmp_path``. The real ``data/raw``
and ``data/results`` locations are never touched.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from src.results import sync

NOW = "2026-11-13T12:00:00+00:00"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _row(home: str, away: str, date: str, gh: int, ga: int,
         ftr: str | None = None, time: str = "15:00") -> dict:
    d = pd.Timestamp(date)
    return {
        "Div": "E0",
        "Date": d.strftime("%d/%m/%Y"),
        "Time": time,
        "HomeTeam": home,
        "AwayTeam": away,
        "FTHG": gh,
        "FTAG": ga,
        "FTR": ftr if ftr is not None else ("H" if gh > ga else "D" if gh == ga else "A"),
    }


def _raw_dir(tmp_path, rows: list[dict], openfootball: list[dict] | None = None,
             season: str = "2627") -> None:
    (tmp_path / "raw").mkdir(exist_ok=True)
    frame = pd.DataFrame(rows, columns=[
        "Div", "Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])
    frame.to_csv(tmp_path / "raw" / f"E0_{season}.csv", index=False)
    if openfootball is not None:
        payload = {"name": "English Premier League 2026/27",
                   "matches": openfootball}
        (tmp_path / "raw" / "openfootball_2026-27.json").write_text(
            json.dumps(payload), encoding="utf-8")


def _openfootball_match(home: str, away: str, date: str, round_: str,
                        ft: list[int] | None = None, time: str = "15:00") -> dict:
    match = {"round": round_, "date": date, "time": time,
             "team1": f"{home} FC", "team2": f"{away} FC"}
    if ft is not None:
        match["score"] = {"ht": [0, 0], "ft": ft}
    return match


def _run(tmp_path, now: str = NOW, **over) -> dict:
    return sync.sync(raw_dir=tmp_path / "raw", path=tmp_path / "results.json",
                     now=now, **over)


def _store(tmp_path) -> dict:
    with open(tmp_path / "results.json", encoding="utf-8") as handle:
        return json.load(handle)


def _result(tmp_path, fixture_id: str) -> dict:
    return _store(tmp_path)["results"][fixture_id]


# --------------------------------------------------------------------------
# Successful synchronisation
# --------------------------------------------------------------------------
def test_syncs_completed_results_with_matchweek(tmp_path):
    _raw_dir(tmp_path, [
        _row("Arsenal", "Coventry", "2026-08-21", 3, 0, time="20:00"),
        _row("Hull", "Man United", "2026-08-22", 2, 0),
    ], [
        _openfootball_match("Arsenal", "Coventry", "2026-08-21", "Matchday 1", [3, 0], "20:00"),
        _openfootball_match("Hull", "Man United", "2026-08-22", "Matchday 1", [2, 0]),
    ])
    report = _run(tmp_path)
    assert report["new"] == 2 and report["total"] == 2
    record = _result(tmp_path, "2627:Arsenal:Coventry")
    assert record["goals_home"] == 3 and record["goals_away"] == 0
    assert record["actual_outcome"] == "H"
    assert record["matchweek"] == 1
    assert record["status"] == "completed"
    assert record["source"] == "football-data"
    assert record["time"] == "20:00"


def test_incomplete_and_unplayed_matches_are_not_results(tmp_path):
    # FTHG/FTAG absent -> pandas writes empty cells; no scores published yet.
    _raw_dir(tmp_path, [
        {"Div": "E0", "Date": "21/11/2026", "Time": "15:00",
         "HomeTeam": "Arsenal", "AwayTeam": "Chelsea",
         "FTHG": None, "FTAG": None, "FTR": None},
    ], [
        # openfootball knows the fixture but has no score: scheduled, not played
        _openfootball_match("Arsenal", "Chelsea", "2026-11-21", "Matchday 12"),
    ])
    report = _run(tmp_path)
    assert report["total"] == 0 and report["new"] == 0


def test_provider_outcome_must_agree_with_scores(tmp_path):
    _raw_dir(tmp_path, [_row("Arsenal", "Chelsea", "2026-11-08", 2, 1, ftr="A")])
    report = _run(tmp_path)
    assert report["total"] == 0
    assert any("FTR" in entry["reason"] for entry in report["rejected"])


def test_provider_failure_preserves_store(tmp_path):
    _raw_dir(tmp_path, [_row("Arsenal", "Chelsea", "2026-11-08", 2, 1)])
    _run(tmp_path)
    before = _store(tmp_path)
    # Next run: the season CSV disappeared (download failure leaves nothing).
    (tmp_path / "raw" / "E0_2627.csv").unlink()
    report = sync.sync(raw_dir=tmp_path / "raw", path=tmp_path / "results.json",
                       now="2026-11-13T13:00:00+00:00")
    assert report["total"] == 1
    assert any("season CSV missing" in r["reason"] for r in report["rejected"])
    assert _store(tmp_path)["results"] == before["results"]


# --------------------------------------------------------------------------
# Idempotency and duplicates
# --------------------------------------------------------------------------
def test_idempotent_rerun_is_unchanged_and_byte_identical(tmp_path):
    _raw_dir(tmp_path, [_row("Arsenal", "Chelsea", "2026-11-08", 2, 1)],
             [_openfootball_match("Arsenal", "Chelsea", "2026-11-08", "Matchday 10", [2, 1])])
    _run(tmp_path)
    first = (tmp_path / "results.json").read_bytes()
    report = _run(tmp_path, now="2026-11-13T13:00:00+00:00")
    assert report["new"] == 0 and report["unchanged"] == 1
    assert (tmp_path / "results.json").read_bytes() == first


def test_duplicate_rows_collapse_into_one_result(tmp_path):
    _raw_dir(tmp_path, [
        _row("Arsenal", "Chelsea", "2026-11-08", 2, 1),
        _row("Arsenal", "Chelsea", "2026-11-08", 2, 1),
    ])
    report = _run(tmp_path)
    assert report["new"] == 1 and report["total"] == 1


def test_conflicting_scores_are_flagged_not_guessed(tmp_path):
    _raw_dir(tmp_path, [
        _row("Arsenal", "Chelsea", "2026-11-08", 2, 1),
        _row("Arsenal", "Chelsea", "2026-11-08", 1, 1),
    ])
    report = _run(tmp_path)
    assert report["total"] == 0
    assert any("conflicting scores" in entry["reason"]
               for entry in report["needs_review"])


# --------------------------------------------------------------------------
# Corrections
# --------------------------------------------------------------------------
def test_corrected_final_score_keeps_audit_trail(tmp_path):
    _raw_dir(tmp_path, [_row("Arsenal", "Chelsea", "2026-11-08", 2, 1)])
    _run(tmp_path)
    _raw_dir(tmp_path, [_row("Arsenal", "Chelsea", "2026-11-08", 3, 1)])
    report = _run(tmp_path, now="2026-11-14T09:00:00+00:00")
    assert report["corrections"] == 1
    record = _result(tmp_path, "2627:Arsenal:Chelsea")
    assert record["goals_home"] == 3 and record["actual_outcome"] == "H"
    assert len(record["corrections"]) == 1
    correction = record["corrections"][0]
    assert correction["previous_goals_home"] == 2
    assert correction["previous_outcome"] == "H"
    assert correction["corrected_at"] == "2026-11-14T09:00:00+00:00"


def test_score_correction_to_draw_updates_outcome(tmp_path):
    _raw_dir(tmp_path, [_row("Arsenal", "Chelsea", "2026-11-08", 2, 1)])
    _run(tmp_path)
    _raw_dir(tmp_path, [_row("Arsenal", "Chelsea", "2026-11-08", 2, 2)])
    report = _run(tmp_path, now="2026-11-14T09:00:00+00:00")
    record = _result(tmp_path, "2627:Arsenal:Chelsea")
    assert record["actual_outcome"] == "D"
    assert record["corrections"][0]["previous_outcome"] == "H"


# --------------------------------------------------------------------------
# Fallback source
# --------------------------------------------------------------------------
def test_openfootball_fallback_used_when_football_data_lacks_result(tmp_path):
    _raw_dir(tmp_path, [], [
        _openfootball_match("Arsenal", "Chelsea", "2026-11-08", "Matchday 10", [2, 1]),
    ])
    report = _run(tmp_path)
    assert report["new"] == 1
    record = _result(tmp_path, "2627:Arsenal:Chelsea")
    assert record["source"] == "openfootball"
    assert record["goals_home"] == 2 and record["actual_outcome"] == "H"


def test_fallback_upgraded_when_authoritative_provider_catches_up(tmp_path):
    _raw_dir(tmp_path, [], [
        _openfootball_match("Arsenal", "Chelsea", "2026-11-08", "Matchday 10", [2, 1]),
    ])
    _run(tmp_path)
    _raw_dir(tmp_path, [_row("Arsenal", "Chelsea", "2026-11-08", 2, 1)],
             [_openfootball_match("Arsenal", "Chelsea", "2026-11-08", "Matchday 10", [2, 1])])
    report = _run(tmp_path, now="2026-11-14T09:00:00+00:00")
    assert report["source_upgrades"] == 1
    assert _result(tmp_path, "2627:Arsenal:Chelsea")["source"] == "football-data"


def test_conflicting_fallback_result_never_overwrites_authoritative(tmp_path):
    _raw_dir(tmp_path, [_row("Arsenal", "Chelsea", "2026-11-08", 2, 1)])
    _run(tmp_path)
    # openfootball disagrees with the authoritative provider.
    _raw_dir(tmp_path, [_row("Arsenal", "Chelsea", "2026-11-08", 2, 1)], [
        _openfootball_match("Arsenal", "Chelsea", "2026-11-08", "Matchday 10", [9, 0]),
    ])
    report = _run(tmp_path, now="2026-11-14T09:00:00+00:00")
    record = _result(tmp_path, "2627:Arsenal:Chelsea")
    assert record["goals_home"] == 2 and record["source"] == "football-data"
    assert record["corrections"] == []
    # The openfootball candidate was not silently adopted.
    assert report["corrections"] == 0


# --------------------------------------------------------------------------
# Matchweek resolution
# --------------------------------------------------------------------------
def test_matchweek_from_schedule_pair_when_date_was_rescheduled(tmp_path):
    # The match was played on 2026-11-08 but the schedule still lists it on
    # the provisional 2026-11-01 date: the team pair identifies the round.
    _raw_dir(tmp_path, [_row("Arsenal", "Chelsea", "2026-11-08", 2, 1)], [
        _openfootball_match("Arsenal", "Chelsea", "2026-11-01", "Matchday 10", None),
    ])
    report = _run(tmp_path)
    assert _result(tmp_path, "2627:Arsenal:Chelsea")["matchweek"] == 10


def test_matchweek_filled_on_later_run(tmp_path):
    _raw_dir(tmp_path, [_row("Arsenal", "Chelsea", "2026-11-08", 2, 1)])
    _run(tmp_path)
    assert _result(tmp_path, "2627:Arsenal:Chelsea")["matchweek"] is None
    _raw_dir(tmp_path, [_row("Arsenal", "Chelsea", "2026-11-08", 2, 1)], [
        _openfootball_match("Arsenal", "Chelsea", "2026-11-08", "Matchday 10", None),
    ])
    report = _run(tmp_path, now="2026-11-13T13:00:00+00:00")
    assert report["matchweek_filled"] == 1
    assert _result(tmp_path, "2627:Arsenal:Chelsea")["matchweek"] == 10


def test_ambiguous_schedule_pair_excluded_from_pair_lookup(tmp_path):
    # Two scheduled meetings for the same pair (replay/cup collision): the
    # pair map must exclude it, so the matchweek stays unset (not guessed).
    _raw_dir(tmp_path, [_row("Arsenal", "Chelsea", "2026-11-08", 2, 1)], [
        _openfootball_match("Arsenal", "Chelsea", "2026-11-01", "Matchday 10", None),
        _openfootball_match("Arsenal", "Chelsea", "2027-02-20", "Matchday 24", None),
    ])
    report = _run(tmp_path)
    assert _result(tmp_path, "2627:Arsenal:Chelsea")["matchweek"] is None
    assert report["total"] == 1
