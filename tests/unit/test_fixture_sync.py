"""Fixture synchronisation tests.

All provider responses are mocked local payloads, so the suite is deterministic
and never touches the network. Each test writes into ``tmp_path``; the real
``data/fixtures`` store is never modified.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from src import config
from src.data import fixture_sync

NOW = "2026-11-13T00:00:00+00:00"
CUTOFF = pd.Timestamp("2026-11-08")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _matches() -> pd.DataFrame:
    """A minimal completed-match frame: sets the cutoff and the known teams."""
    rows = [
        ("2026-10-03", "Arsenal", "Chelsea"),
        ("2026-10-03", "Liverpool", "Man City"),
        ("2026-10-10", "Hull", "Everton"),
        ("2026-10-17", "Tottenham", "Newcastle"),
        ("2026-10-24", "Crystal Palace", "Nott'm Forest"),
        ("2026-10-31", "Aston Villa", "Sunderland"),
        ("2026-11-01", "Fulham", "Leeds"),
        ("2026-11-07", "Ipswich", "Coventry"),
        ("2026-11-08", "Man United", "Bournemouth"),
        ("2026-11-08", "Brighton", "Brentford"),
    ]
    frame = pd.DataFrame(rows, columns=["date", "home_team", "away_team"])
    frame["date"] = pd.to_datetime(frame["date"])
    frame["season_code"] = "2627"
    return frame


def _fixture(team1: str = "Arsenal FC", team2: str = "Liverpool FC",
             date: str = "2026-11-21", time: str = "15:00",
             round_: str = "Matchday 11", **extra) -> dict:
    row = {"round": round_, "date": date, "time": time,
           "team1": team1, "team2": team2}
    row.update(extra)
    return row


def _raw(tmp_path, matches: list[dict] | None = None):
    raw = tmp_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    if matches is not None:
        (raw / "openfootball_2026-27.json").write_text(
            json.dumps({"name": "English Premier League 2026/27",
                        "matches": matches}),
            encoding="utf-8",
        )
    return raw


def _sync(tmp_path, raw, matches: pd.DataFrame | None = None) -> dict:
    return fixture_sync.sync_fixtures(
        fetch=False,
        matches=_matches() if matches is None else matches,
        store_path=tmp_path / "upcoming.csv",
        state_path=tmp_path / "sync_state.json",
        raw_dir=raw,
        now=NOW,
    )


def _store(tmp_path) -> pd.DataFrame:
    return pd.read_csv(tmp_path / "upcoming.csv", parse_dates=["date"])


# --------------------------------------------------------------------------
# Date parsing
# --------------------------------------------------------------------------
@pytest.mark.parametrize("value,expected", [
    ("2026-10-11", "2026-10-11"),   # ISO must not be day/month swapped
    ("2026-10-12", "2026-10-12"),
    ("2026-11-21", "2026-11-21"),
    ("11/10/2026", "2026-10-11"),   # football-data day-first
    ("06/11/2026", "2026-11-06"),
])
def test_parse_date_handles_both_provider_formats(value, expected):
    assert fixture_sync._parse_date(value) == pd.Timestamp(expected)


@pytest.mark.parametrize("value", [None, "", "not-a-date", float("nan")])
def test_parse_date_rejects_unusable_values(value):
    assert pd.isna(fixture_sync._parse_date(value))


# --------------------------------------------------------------------------
# Parsing and validation
# --------------------------------------------------------------------------
def test_parse_openfootball_requires_a_matches_list():
    with pytest.raises(fixture_sync.ProviderError):
        fixture_sync.parse_openfootball({"name": "broken"}, CUTOFF)


def test_parse_football_data_requires_core_columns():
    with pytest.raises(fixture_sync.ProviderError):
        fixture_sync.parse_football_data(pd.DataFrame({"Div": ["E0"]}), CUTOFF)


def test_unknown_team_is_rejected(tmp_path):
    raw = _raw(tmp_path, [_fixture("Arsenal FC", "Barcelona FC")])
    result = _sync(tmp_path, raw)

    assert result["counts"]["rejected"] == 1
    assert result["store_rows"] == 0
    assert "unknown team" in result["rejected"][0]["reason"]


def test_identical_home_and_away_is_rejected(tmp_path):
    raw = _raw(tmp_path, [_fixture("Arsenal FC", "Arsenal FC")])
    result = _sync(tmp_path, raw)

    assert result["counts"]["rejected"] == 1
    assert result["store_rows"] == 0


def test_past_fixture_is_skipped_as_history(tmp_path):
    raw = _raw(tmp_path, [_fixture(date="2026-10-11")])
    result = _sync(tmp_path, raw)

    assert result["counts"]["new"] == 0
    assert result["historical_skipped"] == 1


# --------------------------------------------------------------------------
# Insertion, dedupe and idempotency
# --------------------------------------------------------------------------
def test_sync_inserts_new_fixtures(tmp_path):
    raw = _raw(tmp_path, [_fixture(), _fixture("Chelsea FC", "Man City FC")])
    result = _sync(tmp_path, raw)

    assert result["ok"] is True
    assert result["status"] == "ok"
    assert result["counts"]["new"] == 2
    assert result["store_rows"] == 2
    assert result["upcoming_rows"] == 2

    store = _store(tmp_path)
    assert set(store["fixture_id"]) == {
        "2627:Arsenal:Liverpool", "2627:Chelsea:Man City",
    }
    assert store["status"].eq("scheduled").all()
    assert store["timezone"].eq("Europe/London").all()
    assert store["first_seen_at"].eq(NOW).all()


def test_duplicate_fixtures_are_collapsed(tmp_path):
    raw = _raw(tmp_path, [
        _fixture(),
        _fixture(date="2026-11-22", round_="Matchday 12"),
    ])
    result = _sync(tmp_path, raw)

    assert result["counts"]["duplicates"] == 1
    assert result["store_rows"] == 1


def test_sync_is_idempotent(tmp_path):
    raw = _raw(tmp_path, [_fixture()])
    first = _sync(tmp_path, raw)
    before = (tmp_path / "upcoming.csv").read_bytes()

    second = _sync(tmp_path, raw)
    after = (tmp_path / "upcoming.csv").read_bytes()

    assert first["counts"]["new"] == 1
    assert second["counts"]["new"] == 0
    assert second["counts"]["updated"] == 0
    assert second["counts"]["unchanged"] == 1
    assert second["store_changed"] is False
    assert before == after


# --------------------------------------------------------------------------
# Reschedules, postponements, cancellations
# --------------------------------------------------------------------------
def test_kickoff_change_marks_rescheduled(tmp_path):
    raw = _raw(tmp_path, [_fixture()])
    _sync(tmp_path, raw)

    raw = _raw(tmp_path, [_fixture(date="2026-11-22", time="17:30")])
    result = _sync(tmp_path, raw)
    row = _store(tmp_path).iloc[0]

    assert result["counts"]["updated"] == 1
    assert row["status"] == "rescheduled"
    assert row["date"] == pd.Timestamp("2026-11-22")
    assert row["time"] == "17:30"
    assert row["fixture_id"] == "2627:Arsenal:Liverpool"  # stable identity
    assert row["match_id"] == "2026-11-22_Arsenal_Liverpool"  # engine key


def test_postponed_fixture_is_kept_but_not_upcoming(tmp_path):
    raw = _raw(tmp_path, [_fixture()])
    _sync(tmp_path, raw)

    raw = _raw(tmp_path, [{"round": "Matchday 11",
                           "team1": "Arsenal FC", "team2": "Liverpool FC"}])
    result = _sync(tmp_path, raw)
    row = _store(tmp_path).iloc[0]

    assert result["counts"]["postponed"] == 1
    assert row["status"] == "postponed"
    assert result["upcoming_rows"] == 0


def test_reinstated_fixture_returns_to_upcoming(tmp_path):
    raw = _raw(tmp_path, [_fixture()])
    _sync(tmp_path, raw)
    _sync(tmp_path, _raw(tmp_path, [{"round": "Matchday 11",
                                     "team1": "Arsenal FC",
                                     "team2": "Liverpool FC"}]))

    result = _sync(tmp_path, _raw(tmp_path, [_fixture()]))
    row = _store(tmp_path).iloc[0]

    assert row["status"] == "scheduled"
    assert result["upcoming_rows"] == 1


def test_cancelled_fixture_is_recorded_but_not_upcoming(tmp_path):
    raw = _raw(tmp_path, [_fixture(status="cancelled")])
    result = _sync(tmp_path, raw)
    row = _store(tmp_path).iloc[0]

    assert result["counts"]["cancelled"] == 1
    assert row["status"] == "cancelled"
    assert result["upcoming_rows"] == 0


# --------------------------------------------------------------------------
# History protection
# --------------------------------------------------------------------------
def test_completed_match_is_not_resurrected(tmp_path):
    # Hull v Everton is already in matches.csv (2026-10-10).
    raw = _raw(tmp_path, [_fixture("Hull City AFC", "Everton FC")])
    result = _sync(tmp_path, raw)

    assert result["counts"]["completed"] == 1
    assert result["counts"]["new"] == 0
    assert result["store_rows"] == 0


def test_fixture_missing_from_feed_is_retained(tmp_path):
    raw = _raw(tmp_path, [_fixture(), _fixture("Chelsea FC", "Man City FC")])
    _sync(tmp_path, raw)

    result = _sync(tmp_path, _raw(tmp_path, [_fixture()]))

    assert result["counts"]["retained"] == 1
    assert result["store_rows"] == 2
    assert set(_store(tmp_path)["fixture_id"]) == {
        "2627:Arsenal:Liverpool", "2627:Chelsea:Man City",
    }


# --------------------------------------------------------------------------
# Provider failures
# --------------------------------------------------------------------------
def test_all_providers_unavailable_preserves_store(tmp_path):
    raw = _raw(tmp_path, [_fixture()])
    _sync(tmp_path, raw)
    before = (tmp_path / "upcoming.csv").read_bytes()

    empty = tmp_path / "empty"
    empty.mkdir()
    result = _sync(tmp_path, empty)

    assert result["ok"] is False
    assert result["status"] == "failed"
    assert (tmp_path / "upcoming.csv").read_bytes() == before
    assert result["store_rows"] == 1
    assert result["upcoming_rows"] == 1
    assert "preserved" in result["error"]


def test_malformed_provider_is_isolated(tmp_path):
    raw = _raw(tmp_path, [_fixture()])
    (raw / "openfootball_2026-27.json").write_text("{not json", encoding="utf-8")

    result = _sync(tmp_path, raw)

    assert result["ok"] is False
    assert result["status"] == "failed"
    assert result["sources"]["openfootball"]["status"] == "failed"


def test_partial_provider_failure_is_reported(tmp_path):
    raw = _raw(tmp_path, [_fixture()])
    # football-data feed exists but lacks the required columns.
    (raw / "fixtures.csv").write_text("Div\nE0\n", encoding="utf-8")

    result = _sync(tmp_path, raw)

    assert result["ok"] is True
    assert result["status"] == "partial"
    assert result["sources"]["football-data"]["status"] == "failed"
    assert result["counts"]["new"] == 1


# --------------------------------------------------------------------------
# Store views and pipeline integration
# --------------------------------------------------------------------------
def test_upcoming_returns_only_the_next_matchweek(tmp_path):
    matchweek_11 = [
        ("Arsenal FC", "Liverpool FC"), ("Chelsea FC", "Manchester City FC"),
        ("Hull City AFC", "Tottenham Hotspur FC"), ("Everton FC", "Newcastle United FC"),
        ("Crystal Palace FC", "Aston Villa FC"),
        ("Nottingham Forest FC", "Sunderland AFC"),
        ("Fulham FC", "Ipswich Town FC"), ("Leeds United FC", "Coventry City FC"),
        ("Manchester United FC", "Brighton & Hove Albion FC"),
        ("AFC Bournemouth", "Brentford FC"),
    ]
    matchweek_12 = [
        ("Arsenal FC", "Manchester City FC"), ("Chelsea FC", "Liverpool FC"),
        ("Hull City AFC", "Newcastle United FC"),
    ]
    payload = [_fixture(home, away, round_="Matchday 11") for home, away in matchweek_11]
    payload += [_fixture(home, away, date="2026-11-28", round_="Matchday 12")
                for home, away in matchweek_12]

    result = _sync(tmp_path, _raw(tmp_path, payload))
    view = fixture_sync.upcoming(matches=_matches(),
                                 store_path=tmp_path / "upcoming.csv")

    assert result["store_rows"] == 13
    assert result["upcoming_rows"] == 10
    assert len(view) == 10
    assert view["matchweek"].eq(11).all()
    assert view["match_id"].str.match(r"2026-11-21_.+_.+").all()


# --------------------------------------------------------------------------
# Matchweek progression: the provider-assigned matchweek decides the active
# week, however few of its fixtures are still outstanding.
# --------------------------------------------------------------------------
def test_single_outstanding_fixture_keeps_its_matchweek_active(tmp_path):
    """One rescheduled MW11 fixture must not be skipped for MW12.

    Nine MW11 fixtures were played (they no longer exist as upcoming rows);
    the last one was rescheduled far into the future while MW12 is the next
    full round. The active matchweek must stay 11.
    """
    payload = [_fixture("Arsenal FC", "Manchester City FC", date="2026-11-28",
                        round_="Matchday 12"),
               _fixture("Chelsea FC", "Liverpool FC", date="2026-11-28",
                        round_="Matchday 12"),
               # The outstanding rescheduled MW11 fixture.
               _fixture("Hull City AFC", "Tottenham Hotspur FC", date="2027-01-06",
                        round_="Matchday 11")]
    _sync(tmp_path, _raw(tmp_path, payload))
    view = fixture_sync.upcoming(matches=_matches(),
                                 store_path=tmp_path / "upcoming.csv")

    assert len(view) == 1
    assert view["matchweek"].eq(11).all()
    assert view.iloc[0]["fixture_id"] == "2627:Hull:Tottenham"


def test_small_outstanding_remainder_of_a_matchweek_is_selected(tmp_path):
    """A matchweek finishing midweek (2 fixtures left) stays active."""
    payload = [_fixture("Arsenal FC", "Liverpool FC", date="2026-11-23",
                        round_="Matchday 11"),
               _fixture("Chelsea FC", "Manchester City FC", date="2026-11-24",
                        round_="Matchday 11")]
    payload += [_fixture(home, away, date="2026-11-28", round_="Matchday 12")
                for home, away in [("Hull City AFC", "Newcastle United FC"),
                                   ("Tottenham Hotspur FC", "Everton FC"),
                                   ("Crystal Palace FC", "Aston Villa FC"),
                                   ("Nottingham Forest FC", "Sunderland AFC"),
                                   ("Fulham FC", "Ipswich Town FC"),
                                   ("Leeds United FC", "Coventry City FC")]]
    _sync(tmp_path, _raw(tmp_path, payload))
    view = fixture_sync.upcoming(matches=_matches(),
                                 store_path=tmp_path / "upcoming.csv")

    assert view["matchweek"].eq(11).all()
    assert len(view) == 2


def test_matchweek_advances_once_its_fixtures_are_resolved(tmp_path):
    """When every MW11 fixture is resolved the view moves to MW12."""
    payload = [_fixture(home, away, date="2026-11-07", round_="Matchday 11")
               for home, away in [("Arsenal FC", "Liverpool FC"),
                                  ("Chelsea FC", "Manchester City FC")]]
    payload += [_fixture(home, away, date="2026-11-28", round_="Matchday 12")
                for home, away in [("Everton FC", "Newcastle United FC"),
                                   ("Tottenham Hotspur FC", "Aston Villa FC"),
                                   ("Hull City AFC", "Fulham FC")]]
    _sync(tmp_path, _raw(tmp_path, payload))
    # The 2026-11-19 rows are now before the last completed match: resolved.
    view = fixture_sync.upcoming(matches=_matches(),
                                 store_path=tmp_path / "upcoming.csv")

    assert view["matchweek"].eq(12).all()
    assert len(view) == 3


def test_cancelled_fixture_does_not_hold_the_matchweek_open(tmp_path):
    """A confirmed cancellation is terminal: it neither plays nor blocks MW12."""
    payload = [_fixture("Arsenal FC", "Liverpool FC", date="2026-11-21",
                        round_="Matchday 11", status="CANCELLED")]
    payload += [_fixture(home, away, date="2026-11-28", round_="Matchday 12")
                for home, away in [("Hull City AFC", "Tottenham Hotspur FC"),
                                   ("Everton FC", "Newcastle United FC"),
                                   ("Crystal Palace FC", "Aston Villa FC"),
                                   ("Nottingham Forest FC", "Sunderland AFC"),
                                   ("Fulham FC", "Ipswich Town FC")]]
    result = _sync(tmp_path, _raw(tmp_path, payload))
    view = fixture_sync.upcoming(matches=_matches(),
                                 store_path=tmp_path / "upcoming.csv")

    assert result["counts"]["cancelled"] == 1
    assert len(view) == 5
    assert view["matchweek"].eq(12).all()
    assert not (view["status"] == "cancelled").any()


def test_without_matchweek_numbers_the_horizon_fallback_remains(tmp_path):
    """Fixtures without provider rounds fall back to the date horizon."""
    payload = [_fixture("Arsenal FC", "Liverpool FC", date="2026-11-21", round_=""),
               _fixture("Chelsea FC", "Manchester City FC", date="2026-11-21", round_=""),
               _fixture("Hull City AFC", "Newcastle United FC", date="2026-12-05", round_="")]
    _sync(tmp_path, _raw(tmp_path, payload))
    view = fixture_sync.upcoming(matches=_matches(),
                                 store_path=tmp_path / "upcoming.csv")

    # Within 6 days of the earliest valid fixture: the two, not the December one.
    assert len(view) == 2
    assert view["date"].max() <= pd.Timestamp("2026-11-27")


def test_load_fixtures_reads_the_synced_store(tmp_path, monkeypatch):
    from src.data import processing

    raw = _raw(tmp_path, [_fixture()])
    _sync(tmp_path, raw)
    monkeypatch.setattr(config, "FIXTURES_CSV", tmp_path / "upcoming.csv")
    monkeypatch.setattr(config, "FIXTURE_SYNC_STATE", tmp_path / "sync_state.json")

    fixtures = processing.load_fixtures(matches=_matches())

    assert len(fixtures) == 1
    assert fixtures.iloc[0]["match_id"] == "2026-11-21_Arsenal_Liverpool"
    assert "fixture_id" in fixtures.columns
    assert "status" in fixtures.columns


def test_load_fixtures_bootstraps_from_cached_payloads(tmp_path, monkeypatch):
    from src.data import processing

    raw = _raw(tmp_path, [_fixture()])
    monkeypatch.setattr(config, "FIXTURES_CSV", tmp_path / "upcoming.csv")
    monkeypatch.setattr(config, "FIXTURE_SYNC_STATE", tmp_path / "sync_state.json")
    monkeypatch.setattr(config, "RAW_DIR", raw)

    fixtures = processing.load_fixtures(matches=_matches())

    assert len(fixtures) == 1
    assert (tmp_path / "upcoming.csv").exists()


def test_state_records_sources_counts_and_timestamp(tmp_path):
    _sync(tmp_path, _raw(tmp_path, [_fixture()]))
    state = fixture_sync.load_state(tmp_path / "sync_state.json")
    run = state["last_run"]

    assert run["started_at"] == NOW
    assert run["status"] == "ok"
    assert run["sources"]["openfootball"]["status"] == "cached"
    assert run["counts"]["new"] == 1
    assert len(state["history"]) == 1


# --------------------------------------------------------------------------
# Workflow configuration (text assertions: PyYAML is not a runtime dependency)
# --------------------------------------------------------------------------
def test_sync_workflow_is_configured(project_root):
    text = (project_root / ".github" / "workflows" / "sync-fixtures.yml").read_text(
        encoding="utf-8")

    assert "schedule:" in text and "cron:" in text
    assert "workflow_dispatch:" in text
    assert 'python-version: "3.12"' in text
    assert "contents: write" in text
    assert "concurrency:" in text
    assert "python -m src.data.fixture_sync --fetch" in text
    assert "data/fixtures/upcoming.csv" in text
    assert "data/fixtures/sync_state.json" in text
    assert "git diff --cached --quiet" in text
    assert "git push" in text
    # No retraining and no secrets are part of this workflow.
    assert "--retrain" not in text
    assert "${{ secrets." not in text
