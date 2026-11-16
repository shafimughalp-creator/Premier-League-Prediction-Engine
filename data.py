"""DATA for the EPL Intelligence match centre — live loader.

Builds the exact structure the approved prototype shipped (same shape, same
formulas) from the automation stores, so the approved design consumes live
production data without a single visual change:

* ``data/results/results.json``      official finals (workflow C, freshest)
* ``data/processed/matches.csv``     all completed matches incl. past seasons
* ``data/predictions.json``          production predictions, next matchweek
* ``data/evaluation/matchweeks.json`` matchweek/season accuracy aggregates
* ``data/evaluation/freshness.json`` last sync timestamps

Chances are the production model's probabilities rounded to whole
percentages with the largest-remainder method so they always sum to 100,
exactly like the prototype. Form, table, head-to-head and the season
numbers use real Premier League results with the prototype's formulas.

Missing stores degrade gracefully: empty fixtures, an honest empty table
and null performance numbers — the app never crashes and never invents
data. ``refresh()`` rebuilds ``DATA`` in place (same object identity) when
any underlying store changes, so reruns show fresh data without a restart.
"""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

import pandas as pd

from src import config

ROOT = Path(__file__).resolve().parent
RESULTS_JSON = ROOT / "data" / "results" / "results.json"
PREDICTIONS_JSON = ROOT / "data" / "predictions.json"
MATCHWEEKS_JSON = ROOT / "data" / "evaluation" / "matchweeks.json"
FRESHNESS_JSON = ROOT / "data" / "evaluation" / "freshness.json"
MATCHES_CSV = ROOT / "data" / "processed" / "matches.csv"
SYNC_STATE_JSON = ROOT / "data" / "fixtures" / "sync_state.json"
PIPELINE_STATE_JSON = ROOT / "data" / "pipeline_state.json"
SNAPSHOTS_JSON = ROOT / "data" / "predictions" / "snapshots.json"

SEASON_CODE = config.CURRENT_SEASON
SEASON_LABEL = config.SEASONS[SEASON_CODE]

PICK = {"H": "home", "D": "draw", "A": "away"}


# --------------------------------------------------------------------------
# Club badge identity: primary colour, secondary colour, pattern key.
# --------------------------------------------------------------------------
CLUB = {
    "ARS": ["#ef0107", "#fff", "chev"],
    "BHA": ["#0057b8", "#fff", "v"],
    "MCI": ["#6cabdd", "#1c2c5b", "band"],
    "LIV": ["#c8102e", "#f6eb61", "top"],
    "CHE": ["#034694", "#fff", "cross"],
    "EVE": ["#003399", "#fff", "diag"],
    "BOU": ["#da291c", "#111", "v"],
    "BRE": ["#e30613", "#fff", "v"],
    "LEE": ["#1d428a", "#ffcd00", "h"],
    "AVL": ["#670e36", "#95bfe5", "half"],
    "TOT": ["#132257", "#fff", "chev"],
    "NEW": ["#111", "#fff", "v"],
    "HUL": ["#f5a12d", "#111", "v"],
    "IPS": ["#0044aa", "#fff", "top"],
    "FUL": ["#f2f2f2", "#111", "half"],
    "NFO": ["#dd0000", "#fff", "diag"],
    "COV": ["#2aa3d6", "#1c5b84", "cross"],
    "MUN": ["#da291c", "#fbe122", "chev"],
    "SUN": ["#eb172b", "#fff", "half"],
    "CRY": ["#1b458f", "#c4122e", "v"],
}

# Official badges from the Premier League image host. Coventry has no id and
# always uses the built-in SVG badge.
_BADGE_IDS = {
    "ARS": 3,
    "AVL": 7,
    "BOU": 91,
    "BRE": 94,
    "BHA": 36,
    "CHE": 8,
    "CRY": 31,
    "EVE": 11,
    "FUL": 54,
    "HUL": 88,
    "IPS": 40,
    "LEE": 2,
    "LIV": 14,
    "MCI": 43,
    "MUN": 1,
    "NEW": 4,
    "NFO": 17,
    "SUN": 56,
    "TOT": 6,
}
LOGOS = {
    short: f"https://resources.premierleague.com/premierleague/badges/70/t{badge_id}.png"
    for short, badge_id in _BADGE_IDS.items()
}


# --------------------------------------------------------------------------
# Store readers
# --------------------------------------------------------------------------
def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def _effective_freshness() -> dict:
    """Combine freshness.json with authoritative timestamps from event stores.

    Each field uses the most direct, truthful timestamp available for that
    operation and never fabricates a synchronization time.
    """
    stored = _read_json(FRESHNESS_JSON, {})
    if not isinstance(stored, dict):
        stored = {}

    sync_state = _read_json(SYNC_STATE_JSON, {})
    pipeline = _read_json(PIPELINE_STATE_JSON, {})
    predictions = _read_json(PREDICTIONS_JSON, {})
    snapshots = _read_json(SNAPSHOTS_JSON, {})
    results = _read_json(RESULTS_JSON, {})
    matchweeks = _read_json(MATCHWEEKS_JSON, {})

    def _merge(section: str) -> dict:
        value = stored.get(section) or {}
        return dict(value) if isinstance(value, dict) else {}

    fixtures = _merge("fixtures")
    last_run = (sync_state.get("last_run") or {}) if isinstance(sync_state, dict) else {}
    sync_ok = last_run.get("status") == "ok" or last_run.get("ok") is True
    sync_at = last_run.get("finished_at") or last_run.get("started_at")
    if sync_ok and sync_at:
        fixtures["last_sync_at"] = sync_at

    preds = _merge("predictions")
    pred_at = predictions.get("generated_at") if isinstance(predictions, dict) else None
    if not pred_at and isinstance(pipeline, dict):
        for step in reversed(pipeline.get("steps") or []):
            if step.get("step") == "predict" and step.get("status") == "ok" and step.get("at"):
                pred_at = step.get("at")
                break
        if not pred_at and pipeline.get("last_success") and pipeline.get("n_fixtures"):
            pred_at = pipeline.get("last_success")
    if pred_at:
        preds["last_run"] = pred_at
    snap_at = snapshots.get("updated_at") if isinstance(snapshots, dict) else None
    if not snap_at and isinstance(pipeline, dict):
        for step in reversed(pipeline.get("steps") or []):
            if step.get("step") == "snapshots" and step.get("status") == "ok" and step.get("at"):
                snap_at = step.get("at")
                break
    if snap_at:
        preds["snapshots_updated_at"] = snap_at
    if isinstance(snapshots, dict) and isinstance(snapshots.get("snapshots"), dict):
        preds["snapshots"] = len(snapshots["snapshots"])

    results_block = _merge("results")
    res_at = results.get("updated_at") if isinstance(results, dict) else None
    if res_at:
        results_block["last_sync_at"] = res_at
    if isinstance(results, dict) and isinstance(results.get("results"), dict):
        results_block["completed"] = len(results["results"])

    evaluation = _merge("evaluation")
    eval_at = None
    if isinstance(matchweeks, dict):
        eval_at = matchweeks.get("at") or matchweeks.get("updated_at")
    if eval_at:
        evaluation["last_run"] = eval_at

    fresh = dict(stored)
    fresh["fixtures"] = fixtures
    fresh["predictions"] = preds
    fresh["results"] = results_block
    fresh["evaluation"] = evaluation
    return fresh


def _completed() -> list[dict]:
    """All completed matches, freshest sources winning, sorted by kickoff."""
    rows: dict[tuple[str, str, str], dict] = {}
    if MATCHES_CSV.exists():
        try:
            frame = pd.read_csv(MATCHES_CSV, low_memory=False)
        except (OSError, pd.errors.EmptyDataError):
            frame = pd.DataFrame()
        for _, r in frame.iterrows():
            if pd.isna(r.get("home_goals")) or pd.isna(r.get("away_goals")):
                continue
            key = (str(r["date"]), str(r["home_team"]), str(r["away_team"]))
            rows[key] = {
                "date": str(r["date"]),
                "time": str(r.get("time") or "15:00"),
                "season_code": str(r.get("season_code") or ""),
                "season": str(r.get("season") or config.SEASONS.get(str(r.get("season_code")), "")),
                "home": str(r["home_team"]),
                "away": str(r["away_team"]),
                "home_goals": int(r["home_goals"]),
                "away_goals": int(r["away_goals"]),
                "outcome": str(r.get("result") or ""),
                "matchweek": None,
            }
    # Workflow C may have synced results the pipeline has not processed yet.
    store = _read_json(RESULTS_JSON, {})
    for record in (store.get("results") or {}).values():
        key = (record["date"], record["home_team"], record["away_team"])
        rows[key] = {
            "date": record["date"],
            "time": str(record.get("time") or "15:00"),
            "season_code": str(record.get("season_code") or ""),
            "season": str(record.get("season") or ""),
            "home": record["home_team"],
            "away": record["away_team"],
            "home_goals": int(record["goals_home"]),
            "away_goals": int(record["goals_away"]),
            "outcome": str(record.get("actual_outcome") or ""),
            "matchweek": record.get("matchweek"),
        }
    return sorted(rows.values(), key=lambda m: (m["date"], m["time"], m["home"]))


def _current_season(completed: list[dict]) -> list[dict]:
    return [m for m in completed if m["season_code"] == SEASON_CODE]


# --------------------------------------------------------------------------
# Prototype formulas
# --------------------------------------------------------------------------
def _largest_remainder(probabilities: tuple[float, float, float]) -> dict:
    """Round shares of 100 keeping the sum exactly 100."""
    scaled = [p * 100.0 for p in probabilities]
    base = [int(x // 1) for x in scaled]
    remainder = 100 - sum(base)
    order = sorted(range(3), key=lambda i: scaled[i] - base[i], reverse=True)
    for i in range(max(0, remainder)):
        base[order[i % 3]] += 1
    return {"home": base[0], "draw": base[1], "away": base[2]}


def _points_for(team: str, match: dict) -> int:
    if match["outcome"]:
        return {"H": 3, "D": 1, "A": 0}[match["outcome"]] \
            if match["home"] == team else \
            {"A": 3, "D": 1, "H": 0}[match["outcome"]]
    won = (match["home_goals"] > match["away_goals"]) == (match["home"] == team)
    drawn = match["home_goals"] == match["away_goals"]
    return 3 if won else 1 if drawn else 0


def _last5(matches: list[dict]) -> list[dict]:
    return matches[-5:]


def _team_matches(completed: list[dict], team: str,
                  venue: str | None = None) -> list[dict]:
    picked = [
        m for m in completed
        if (m["home"] == team or m["away"] == team)
        and (venue is None or (m["home"] == team if venue == "home" else m["away"] == team))
    ]
    return _last5(picked)


def _team_row(team: str, season_matches: list[dict], all_completed: list[dict]) -> dict:
    played = won = drawn = lost = gf = ga = 0
    for m in season_matches:
        mine, theirs = (m["home_goals"], m["away_goals"]) if m["home"] == team \
            else (m["away_goals"], m["home_goals"])
        gf += mine
        ga += theirs
        played += 1
        if mine > theirs:
            won += 1
        elif mine == theirs:
            drawn += 1
        else:
            lost += 1
    form = []
    for m in _team_matches(all_completed, team):
        mine, theirs = (m["home_goals"], m["away_goals"]) if m["home"] == team \
            else (m["away_goals"], m["home_goals"])
        form.append("W" if mine > theirs else "D" if mine == theirs else "L")
    form.reverse()  # the prototype lists the most recent result first
    short = config.SHORT_NAMES.get(team, "".join(ch for ch in team if ch.isalnum())[:3].upper())
    return {
        "team": team,
        "short": short,
        "played": played,
        "won": won,
        "drawn": drawn,
        "lost": lost,
        "gf": gf,
        "ga": ga,
        "gd": gf - ga,
        "points": won * 3 + drawn,
        "form": form,
    }


def _meetings(completed: list[dict], a: str, b: str, count: int = 5) -> list[dict]:
    """Meetings between two clubs, newest first (the prototype's order)."""
    pair = {a, b}
    found = [m for m in completed if {m["home"], m["away"]} == pair]
    return list(reversed(found[-count:]))


def _number(value: float) -> float | int:
    rounded = round(value, 1)
    return int(rounded) if rounded == int(rounded) else rounded


def _rates(completed: list[dict], home: str, away: str) -> dict:
    home5 = _team_matches(completed, home)
    away5 = _team_matches(completed, away)
    home_h = _team_matches(completed, home, "home")
    away_a = _team_matches(completed, away, "away")
    meetings = _meetings(completed, home, away)

    def ppg(matches: list[dict], side: str) -> float | None:
        if not matches:
            return None
        return sum(_points_for(side, m) for m in matches) / len(matches)

    def goals(matches: list[dict], team: str) -> tuple[float | None, float | None]:
        if not matches:
            return None, None
        gf = sum(m["home_goals"] if m["home"] == team else m["away_goals"] for m in matches)
        ga = sum(m["away_goals"] if m["home"] == team else m["home_goals"] for m in matches)
        return gf / len(matches), ga / len(matches)

    h_gf, h_ga = goals(home5, home)
    a_gf, a_ga = goals(away5, away)
    level = sum(1 for m in meetings if m["home_goals"] == m["away_goals"])
    return {
        "homeForm5Ppg": None if ppg(home5, home) is None else _number(ppg(home5, home)),
        "awayForm5Ppg": None if ppg(away5, away) is None else _number(ppg(away5, away)),
        "homeHomePpg": None if ppg(home_h, home) is None else _number(ppg(home_h, home)),
        "awayAwayPpg": None if ppg(away_a, away) is None else _number(ppg(away_a, away)),
        "homeGf5": None if h_gf is None else _number(h_gf),
        "homeGa5": None if h_ga is None else _number(h_ga),
        "awayGf5": None if a_gf is None else _number(a_gf),
        "awayGa5": None if a_ga is None else _number(a_ga),
        "h2hPpg": None if ppg(meetings, home) is None else _number(ppg(meetings, home)),
        "h2hLevelPct": None if not meetings else round(100.0 * level / len(meetings)),
        "xg": None,  # no reliable xG source for upcoming fixtures: honest dash
    }


# --------------------------------------------------------------------------
# DATA build
# --------------------------------------------------------------------------
def _predictions() -> list[dict]:
    payload = _read_json(PREDICTIONS_JSON, {})
    rows = [r for r in (payload.get("fixtures") or [])
            if r.get("home_team") and r.get("prob_home") is not None]
    return sorted(rows, key=lambda r: (str(r.get("date")), str(r.get("time"))))


def build() -> dict:
    completed = _completed()
    current = _current_season(completed)
    predictions = _predictions()

    teams: set[str] = set()
    for m in current:
        teams.add(m["home"])
        teams.add(m["away"])
    if not teams:
        for r in predictions:
            teams.add(r["home_team"])
            teams.add(r["away_team"])
    table = [_team_row(team, [m for m in current if team in (m["home"], m["away"])], completed)
             for team in sorted(teams)]
    table.sort(key=lambda t: (-t["points"], -t["gd"], -t["gf"], t["team"]))
    for position, entry in enumerate(table, start=1):
        entry["position"] = position

    fixtures = []
    for r in predictions:
        chances = _largest_remainder((r["prob_home"], r["prob_draw"], r["prob_away"]))
        fixtures.append({
            "id": str(r.get("match_id") or f"{r['date']}_{r['home_team']}_{r['away_team']}"),
            "fixtureId": str(r.get("fixture_id") or ""),
            "status": str(r.get("status") or "scheduled"),
            "matchweek": r.get("matchweek"),
            "home": r["home_team"],
            "away": r["away_team"],
            "date": str(r.get("date")),
            "time": str(r.get("time") or "15:00"),
            "chances": chances,
            "pick": PICK.get(r.get("predicted"), "home"),
            "modelVersion": r.get("model_version"),
            "rates": _rates(completed, r["home_team"], r["away_team"]),
            "headToHead": [
                {
                    "date": m["date"],
                    "home": m["home"],
                    "away": m["away"],
                    "homeGoals": m["home_goals"],
                    "awayGoals": m["away_goals"],
                    "season": m["season"],
                }
                for m in _meetings(completed, r["home_team"], r["away_team"])
            ],
        })

    matches_played = len(current)
    goals = sum(m["home_goals"] + m["away_goals"] for m in current)
    season_numbers = {
        "matchesPlayed": matches_played,
        "goals": goals,
        "goalsPerGame": round(goals / matches_played, 2) if matches_played else 0,
        "homeWins": sum(1 for m in current if m["home_goals"] > m["away_goals"]),
        "draws": sum(1 for m in current if m["home_goals"] == m["away_goals"]),
        "awayWins": sum(1 for m in current if m["home_goals"] < m["away_goals"]),
        "bothScored": sum(1 for m in current if m["home_goals"] > 0 and m["away_goals"] > 0),
        # Clean sheets kept, counted per team like the approved prototype:
        # a scoreless draw keeps one for each side.
        "cleanSheets": sum(1 for m in current if m["home_goals"] == 0)
                       + sum(1 for m in current if m["away_goals"] == 0),
    }

    fixture_weeks = [r.get("matchweek") for r in predictions if r.get("matchweek") is not None]
    cutoff = max((m["date"] for m in current), default=None)

    aggregates = _read_json(MATCHWEEKS_JSON, {})
    season_aggregates = ((aggregates.get("seasons") or {}).get(SEASON_CODE) or {})

    evaluations = _read_json(ROOT / "data" / "evaluation" / "evaluations.json", {})
    evaluation_map = evaluations.get("evaluations") or {}
    results_list = []
    for m in current:
        evaluation = evaluation_map.get(
            f"{SEASON_CODE}:{m['home']}:{m['away']}")
        results_list.append({
            "date": m["date"],
            "matchweek": (evaluation or {}).get("matchweek")
                         if (evaluation or {}).get("matchweek") is not None
                         else m.get("matchweek"),
            "home": m["home"],
            "away": m["away"],
            "homeGoals": m["home_goals"],
            "awayGoals": m["away_goals"],
            "outcome": m["outcome"],
            "predicted": (evaluation or {}).get("predicted_outcome"),
            "correct": (evaluation or {}).get("correct"),
            "evaluatedAt": (evaluation or {}).get("evaluated_at"),
        })

    first_day = predictions[0]["date"] if predictions else cutoff

    # Matchweek progression context. The active matchweek is the provider-
    # assigned matchweek of the fixtures still to be played (the prediction
    # payload's view of the synced fixture store) — never a calendar guess.
    # The last fully completed and the next upcoming weeks come from the
    # aggregates, which bucket fixtures by the same authoritative matchweek.
    # Both keys are additive: the approved pages never read them.
    def _week_number(key):
        try:
            return int(key)
        except (TypeError, ValueError):
            return None

    progress: dict[int, dict] = {}
    for key, week in (season_aggregates.get("matchweeks") or {}).items():
        number = _week_number(key)
        if number is None:
            continue
        progress[number] = {
            "scheduled": int(week.get("scheduled") or 0),
            "completed": int(week.get("completed") or 0),
            "pending": int(week.get("pending") or 0),
        }
    finished = sorted(
        number for number, counts in progress.items()
        if counts["scheduled"] and counts["completed"] >= counts["scheduled"])
    active = min(fixture_weeks) if fixture_weeks else None
    later_outstanding = sorted(
        number for number, counts in progress.items()
        if counts["pending"] > 0 and (active is None or number > active))

    meta = {
        "season": SEASON_LABEL,
        "matchweek": active,
        "lastCompletedMatchweek": finished[-1] if finished else None,
        "nextMatchweek": later_outstanding[0] if later_outstanding else None,
        "dataCutoff": cutoff,
        "dateLong": (
            lambda d: f"{d.strftime('%A')}, {d.day} {d.strftime('%B')} {d.year}"
        )(date.fromisoformat(first_day)) if first_day else "",
    }

    return {
        "meta": meta,
        "seasonNumbers": season_numbers,
        "full": {"Man United": "Manchester United", "Man City": "Manchester City"},
        "club": CLUB,
        "logos": LOGOS,
        "table": table,
        "fixtures": fixtures,
        # Additive keys — the approved pages never read them.
        "performance": {
            "matchweeks": season_aggregates.get("matchweeks") or {},
            "season": season_aggregates.get("season") or {},
        },
        "results": results_list,
        "freshness": _effective_freshness(),
    }


# --------------------------------------------------------------------------
# Live refresh (same object identity, so render.py's binding stays valid)
# --------------------------------------------------------------------------
_STORES = (RESULTS_JSON, PREDICTIONS_JSON, MATCHWEEKS_JSON, FRESHNESS_JSON,
           MATCHES_CSV, SYNC_STATE_JSON, PIPELINE_STATE_JSON, SNAPSHOTS_JSON)
_last_fingerprint: tuple | None = None
_last_built_at: float = 0.0
TTL_SECONDS = 300.0

DATA: dict = {}


def _fingerprint() -> tuple:
    return tuple(
        (str(path), path.stat().st_mtime_ns) if path.exists() else (str(path), None)
        for path in _STORES
    )


def refresh(ttl_seconds: float = TTL_SECONDS) -> dict:
    """Rebuild DATA in place when a store changed or the TTL elapsed.

    Same dict object every time, so ``from data import DATA`` bindings keep
    seeing fresh values. Cheap when nothing changed (a few stat calls).
    """
    global _last_fingerprint, _last_built_at
    now = time.monotonic()
    fingerprint = _fingerprint()
    if DATA and fingerprint == _last_fingerprint and now - _last_built_at < ttl_seconds:
        return DATA
    fresh = build()
    DATA.clear()
    DATA.update(fresh)
    _last_fingerprint = fingerprint
    _last_built_at = now
    return DATA


refresh(ttl_seconds=0.0)
