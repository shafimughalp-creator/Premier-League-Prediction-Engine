"""Official final-result synchronisation for the current EPL season.

The pipeline already downloads fresh copies of the football-data season CSV
(``E0_<code>.csv``) and the openfootball schedule on every run. This module
turns those feeds into a persistent, idempotent store of *official final
results* keyed by the same stable fixture id the fixture store and the
prediction snapshots use: ``"{season_code}:{home}:{away}"``.

Store contract
--------------
``data/results/results.json``::

    {
      "updated_at": "...",
      "results": {
        "2627:Arsenal:Coventry": {
          "fixture_id":  "2627:Arsenal:Coventry",
          "season_code": "2627",
          "season":      "2026/27",
          "matchweek":   1,
          "date":        "2026-08-21",
          "time":        "20:00",
          "home_team":   "Arsenal",
          "away_team":   "Coventry",
          "goals_home":  3,
          "goals_away":  0,
          "actual_outcome": "H",          # H / D / A (project convention)
          "status":      "completed",      # only official finals are stored
          "source":      "football-data",
          "first_synced_at": "...",
          "last_synced_at":  "...",
          "corrections": []
        }
      }
    }

Rules
-----
* A match enters the store only when a provider publishes final scores.
  A kickoff time passing is **not** evidence of completion; pending matches
  are simply absent and the evaluation layer reports them as pending.
* football-data.co.uk is the authoritative source; openfootball ``score.ft``
  is a fallback used only for fixtures football-data does not cover. A
  fallback record is upgraded to the authoritative source automatically once
  football-data publishes the same result.
* An authoritative score change (rare provider correction) updates the
  record and appends the previous values to ``corrections`` with both
  timestamps, so evaluation can recompute affected aggregates with an audit
  trail.
* Result rows are validated: scores must be non-negative integers and a
  provider-declared outcome (``FTR``) must agree with the score-derived
  outcome. Anything ambiguous is reported in ``needs_review`` and never
  guessed.
* The sync is idempotent: with unchanged feeds the store file is left
  byte-identical and the report says ``new=0 updated=0 corrections=0``.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.data import fixture_sync, processing
from src.utilities.io import write_json_atomic

log = logging.getLogger(__name__)

RESULTS_JSON = config.DATA_DIR / "results" / "results.json"

SOURCE_FOOTBALL_DATA = "football-data"
SOURCE_OPENFOOTBALL = "openfootball"
STATUS_COMPLETED = "completed"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _store_path(path: str | Path | None) -> Path:
    return Path(path) if path is not None else RESULTS_JSON


def load(path: str | Path | None = None) -> dict:
    target = _store_path(path)
    if not target.exists():
        return {"updated_at": None, "results": {}}
    with open(target, encoding="utf-8") as handle:
        return json.load(handle)


def _save(store: dict, path: str | Path | None = None) -> None:
    store["updated_at"] = _now()
    write_json_atomic(_store_path(path), store)


def _season_slug(code: str) -> str:
    """"2627" -> "2026-27" (openfootball payload naming)."""
    label = config.SEASONS.get(code)
    if label:
        return label.replace("/", "-")
    return f"20{code[:2]}-{code[2:]}"


def _score(value: object) -> int | None:
    """Validate one provider score: a non-negative integer."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number) or number < 0 or number != int(number):
        return None
    return int(number)


def _outcome(goals_home: int, goals_away: int) -> str:
    if goals_home > goals_away:
        return "H"
    if goals_home < goals_away:
        return "A"
    return "D"


def matchweek_lookup(
    raw_dir: str | Path | None = None,
    season_code: str = config.CURRENT_SEASON,
) -> tuple[dict[tuple[str, str, str], int], dict[tuple[str, str], int]]:
    """Matchweek lookups from the openfootball payload.

    Returns ``(exact, by_pair)``: ``exact`` maps ``(ISO date, home, away)``
    to the round number and ``by_pair`` maps ``(home, away)`` to it. The
    provider publishes a ``round`` for every fixture of the season, so both
    maps cover the whole schedule even for unplayed rounds. The pair map is
    the fallback for fixtures whose kickoff was rescheduled after the
    schedule was published (a league season has every team pair exactly
    once, so the pair alone identifies the round); pairs that appear more
    than once are excluded from it rather than guessed.
    """
    directory = Path(raw_dir) if raw_dir is not None else config.RAW_DIR
    payload_path = directory / f"openfootball_{_season_slug(season_code)}.json"
    if not payload_path.exists():
        return {}, {}
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("openfootball schedule unreadable: %s", exc)
        return {}, {}
    exact: dict[tuple[str, str, str], int] = {}
    by_pair: dict[tuple[str, str], list[int]] = {}
    for match in payload.get("matches", []):
        round_label = str(match.get("round") or "")
        digits = "".join(ch for ch in round_label if ch.isdigit())
        if not digits or not match.get("date"):
            continue
        home = processing.normalize_team(match.get("team1"))
        away = processing.normalize_team(match.get("team2"))
        week = int(digits)
        exact[(str(match["date"]), home, away)] = week
        by_pair.setdefault((home, away), []).append(week)
    unambiguous = {pair: weeks[0] for pair, weeks in by_pair.items() if len(weeks) == 1}
    return exact, unambiguous


def parse_football_data_results(
    frame: pd.DataFrame, season_code: str
) -> tuple[list[dict], list[dict]]:
    """Extract validated final results from a football-data season CSV."""
    required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"football-data feed missing columns: {sorted(missing)}")
    rows: list[dict] = []
    rejected: list[dict] = []
    feed = frame[frame["Div"].astype(str).str.strip().str.upper() == "E0"]
    for _, source_row in feed.iterrows():
        home = processing.normalize_team(source_row.get("HomeTeam"))
        away = processing.normalize_team(source_row.get("AwayTeam"))
        goals_home = _score(source_row.get("FTHG"))
        goals_away = _score(source_row.get("FTAG"))
        if goals_home is None or goals_away is None:
            continue  # no final score published yet -> not a result
        date = fixture_sync._parse_date(source_row.get("Date"))
        if pd.isna(date):
            rejected.append({"fixture": f"{home} vs {away}",
                             "reason": "unparseable date"})
            continue
        outcome = _outcome(goals_home, goals_away)
        provider_outcome = str(source_row.get("FTR") or "").strip().upper()
        if provider_outcome and provider_outcome not in {"H", "D", "A"}:
            rejected.append({"fixture": f"{home} vs {away}",
                             "reason": f"invalid FTR {provider_outcome!r}"})
            continue
        if provider_outcome and provider_outcome != outcome:
            # Scores and the declared outcome disagree: do not guess.
            rejected.append({"fixture": f"{home} vs {away}",
                             "reason": f"FTR {provider_outcome} != scores {goals_home}-{goals_away}"})
            continue
        time_text = fixture_sync._normalise_time(source_row.get("Time"))
        rows.append({
            "fixture_id": f"{season_code}:{home}:{away}",
            "season_code": season_code,
            "season": config.SEASONS.get(season_code, season_code),
            "date": str(date.date()),
            "time": time_text or "15:00",
            "home_team": home,
            "away_team": away,
            "goals_home": goals_home,
            "goals_away": goals_away,
            "actual_outcome": outcome,
            "source": SOURCE_FOOTBALL_DATA,
        })
    return rows, rejected


def parse_openfootball_results(payload: object) -> tuple[list[dict], list[dict]]:
    """Extract fallback final results (``score.ft``) from the openfootball feed."""
    rows: list[dict] = []
    rejected: list[dict] = []
    if not isinstance(payload, dict) or not isinstance(payload.get("matches"), list):
        return rows, [{"reason": "openfootball payload has no 'matches' list"}]
    for match in payload["matches"]:
        if not isinstance(match, dict):
            continue
        score = match.get("score") or {}
        final = score.get("ft") if isinstance(score, dict) else None
        if not isinstance(final, list) or len(final) != 2:
            continue
        goals_home = _score(final[0])
        goals_away = _score(final[1])
        if goals_home is None or goals_away is None:
            continue
        home = processing.normalize_team(match.get("team1"))
        away = processing.normalize_team(match.get("team2"))
        if not home or not away or not match.get("date"):
            rejected.append({"fixture": f"{home} vs {away}",
                             "reason": "missing date or team"})
            continue
        rows.append({
            "fixture_id": f"{config.CURRENT_SEASON}:{home}:{away}",
            "season_code": config.CURRENT_SEASON,
            "season": config.SEASONS.get(config.CURRENT_SEASON, config.CURRENT_SEASON),
            "date": str(match["date"]),
            "time": str(match.get("time") or "15:00"),
            "home_team": home,
            "away_team": away,
            "goals_home": goals_home,
            "goals_away": goals_away,
            "actual_outcome": _outcome(goals_home, goals_away),
            "source": SOURCE_OPENFOOTBALL,
        })
    return rows, rejected


def _candidate_map(rows: list[dict], source: str) -> tuple[dict[str, dict], list[dict]]:
    """Deduplicate one source's rows; conflicts are flagged, not guessed."""
    candidates: dict[str, dict] = {}
    conflicts: list[dict] = []
    for row in rows:
        fixture_id = row["fixture_id"]
        existing = candidates.get(fixture_id)
        if existing is None:
            candidates[fixture_id] = row
            continue
        same = (existing["goals_home"] == row["goals_home"]
                and existing["goals_away"] == row["goals_away"])
        if same:
            continue
        conflicts.append({"fixture_id": fixture_id,
                          "source": source,
                          "reason": "conflicting scores within one source"})
        candidates.pop(fixture_id, None)
    return candidates, conflicts


def sync(
    fetch: bool = False,
    seasons: list[str] | None = None,
    now: str | None = None,
    raw_dir: str | Path | None = None,
    path: str | Path | None = None,
) -> dict:
    """Synchronise official final results into the results store.

    Returns a report with ``new / updated / unchanged / corrections /
    source_upgrades / matchweek_filled / rejected / needs_review / total``.
    The store file is only rewritten when something actually changed, so an
    unchanged run leaves it byte-identical.
    """
    moment = now or _now()
    season_codes = seasons or [config.CURRENT_SEASON]
    directory = Path(raw_dir) if raw_dir is not None else config.RAW_DIR

    authoritative: dict[str, dict] = {}
    fallback: dict[str, dict] = {}
    rejected: list[dict] = []
    conflicts: list[dict] = []

    for code in season_codes:
        season_path = directory / f"E0_{code}.csv"
        if not season_path.exists():
            rejected.append({"season": code, "reason": "season CSV missing"})
            continue
        frame = pd.read_csv(season_path, encoding="utf-8-sig", low_memory=False)
        rows, rejected_rows = parse_football_data_results(frame, code)
        rejected.extend(rejected_rows)
        candidates, source_conflicts = _candidate_map(rows, SOURCE_FOOTBALL_DATA)
        conflicts.extend(source_conflicts)
        authoritative.update(candidates)

    for code in season_codes:
        payload_path = directory / f"openfootball_{_season_slug(code)}.json"
        if not payload_path.exists():
            continue
        try:
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("openfootball payload unreadable: %s", exc)
            continue
        rows, rejected_rows = parse_openfootball_results(payload)
        rejected.extend(rejected_rows)
        candidates, source_conflicts = _candidate_map(rows, SOURCE_OPENFOOTBALL)
        conflicts.extend(source_conflicts)
        for fixture_id, row in candidates.items():
            if fixture_id not in authoritative:
                fallback[fixture_id] = row

    exact_weeks, pair_weeks = matchweek_lookup(raw_dir=directory)

    def resolve_week(candidate: dict) -> int | None:
        return (
            exact_weeks.get((candidate["date"], candidate["home_team"], candidate["away_team"]))
            or pair_weeks.get((candidate["home_team"], candidate["away_team"]))
        )

    store = load(path)
    results = store.setdefault("results", {})
    report = {
        "at": moment,
        "seasons": season_codes,
        "new": 0,
        "updated": 0,
        "unchanged": 0,
        "corrections": 0,
        "source_upgrades": 0,
        "matchweek_filled": 0,
        "rejected": [],
        "needs_review": [],
        "total": 0,
    }

    def needs_review(fixture_id: str, reason: str) -> None:
        report["needs_review"].append({"fixture_id": fixture_id, "reason": reason})

    report["needs_review"].extend(conflicts)

    for fixture_id in sorted(set(authoritative) | set(fallback) | set(results)):
        candidate = authoritative.get(fixture_id) or fallback.get(fixture_id)
        if candidate is None:
            continue  # stored row whose source feeds dropped it: retained
        if fixture_id not in results:
            record = {
                **candidate,
                "matchweek": resolve_week(candidate),
                "status": STATUS_COMPLETED,
                "first_synced_at": moment,
                "last_synced_at": moment,
                "corrections": [],
            }
            results[fixture_id] = record
            report["new"] += 1
            continue

        record = results[fixture_id]
        same_scores = (record.get("goals_home") == candidate["goals_home"]
                       and record.get("goals_away") == candidate["goals_away"])
        if not same_scores:
            report["corrections"] += 1
            record["corrections"].append({
                "previous_goals_home": record.get("goals_home"),
                "previous_goals_away": record.get("goals_away"),
                "previous_outcome": record.get("actual_outcome"),
                "previous_source": record.get("source"),
                "corrected_at": moment,
                "source": candidate["source"],
            })
            record["goals_home"] = candidate["goals_home"]
            record["goals_away"] = candidate["goals_away"]
            record["actual_outcome"] = candidate["actual_outcome"]
            record["source"] = candidate["source"]
            record["last_synced_at"] = moment
            record["status"] = STATUS_COMPLETED
            continue
        if record.get("source") != SOURCE_FOOTBALL_DATA and candidate["source"] == SOURCE_FOOTBALL_DATA:
            # The authoritative provider caught up with the fallback.
            record["source"] = SOURCE_FOOTBALL_DATA
            record["last_synced_at"] = moment
            report["source_upgrades"] += 1
            continue
        if record.get("matchweek") is None:
            week = resolve_week(candidate)
            if week is not None:
                record["matchweek"] = week
                record["last_synced_at"] = moment
                report["matchweek_filled"] += 1
                continue
        report["unchanged"] += 1

    report["total"] = len(results)
    report["rejected"] = rejected[:20]
    report["needs_review"] = report["needs_review"][:20]
    changed = (report["new"] or report["updated"] or report["corrections"]
               or report["source_upgrades"] or report["matchweek_filled"])
    if changed:
        _save(store, path)
    return report


def completed_results(path: str | Path | None = None) -> dict[str, dict]:
    """Map fixture_id -> official final result."""
    return load(path)["results"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Official result synchronisation")
    parser.add_argument("--fetch", action="store_true",
                        help="re-download the season CSVs before parsing")
    parser.add_argument("--season", action="append", dest="seasons",
                        help="season code to sync (repeatable; default: current)")
    parser.add_argument("--json", action="store_true", help="print a JSON report")
    args = parser.parse_args(argv)
    if args.fetch:
        for code in (args.seasons or [config.CURRENT_SEASON]):
            try:
                from src.data import acquisition
                acquisition.download_season(code, force=True)
                acquisition.download_openfootball(f"20{code[:2]}-{code[2:]}", force=True)
            except Exception as exc:  # keep cached copies usable
                log.warning("download failed for %s: %s", code, exc)
    report = sync(seasons=args.seasons)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        log.info("%s", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
