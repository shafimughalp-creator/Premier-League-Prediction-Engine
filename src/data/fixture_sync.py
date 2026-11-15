"""Synchronise upcoming Premier League fixtures into a persistent store.

The module reuses the two sources the project already ships with:

* football-data.co.uk ``fixtures.csv`` - next-round fixtures, usually with prices;
* openfootball ``football.json`` - the published full-season schedule.

Design rules
------------
* The store is ``data/fixtures/upcoming.csv`` (same path every existing consumer
  already reads), enriched with a stable ``fixture_id`` and an explicit
  ``status``. ``data/fixtures/sync_state.json`` records the run metadata.
* A fixture is keyed by ``"{season_code}:{home}:{away}"``. League fixtures are
  unique on that key, so a rescheduled kickoff keeps its identity while the
  date-derived ``match_id`` (the prediction-engine contract) is recomputed.
* The synchronisation is idempotent: re-running it with unchanged provider
  payloads leaves the store byte-identical and reports ``new=0, updated=0``.
* A fixture that disappears from a provider response is never deleted. Only an
  explicit provider cancellation, a provider postponement or a completed result
  in ``matches.csv`` changes its status or removes it.
* A failed or malformed provider response never overwrites the store. The run is
  reported as ``failed`` (all sources down) or ``partial`` (one source down).
* No credentials, no scraping and no new data provider are introduced.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

log = logging.getLogger(__name__)

TIMEZONE = "Europe/London"
STATUS_SCHEDULED = "scheduled"
STATUS_RESCHEDULED = "rescheduled"
STATUS_POSTPONED = "postponed"
STATUS_CANCELLED = "cancelled"
STATUS_COMPLETED = "completed"
UPCOMING_STATUSES = (STATUS_SCHEDULED, STATUS_RESCHEDULED)

ODDS_COLUMNS = (
    "mkt_home", "mkt_draw", "mkt_away", "mkt_overround", "mkt_disp_home",
    "mkt_disp_draw", "mkt_disp_away", "mkt_books", "mkt_balance",
    "mkt_favourite", "mkt_entropy", "odds_home", "odds_draw", "odds_away",
)

STORE_COLUMNS = (
    "fixture_id", "season_code", "season", "matchweek", "round",
    "date", "time", "timezone", "home_team", "away_team", "status",
    "source", "match_id", *ODDS_COLUMNS, "first_seen_at", "updated_at",
)


class ProviderError(RuntimeError):
    """A provider response is missing or structurally unusable."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _season_code(date: pd.Timestamp) -> str:
    if pd.isna(date):
        return config.CURRENT_SEASON
    start = date.year if date.month >= 7 else date.year - 1
    label = f"{start}/{str(start + 1)[-2:]}"
    for code, season_label in config.SEASONS.items():
        if season_label == label:
            return code
    return config.CURRENT_SEASON


def _fixture_id(season_code: str, home: str, away: str) -> str:
    return f"{season_code}:{home}:{away}"


def _store_path(path: str | Path | None = None) -> Path:
    return Path(path) if path is not None else config.FIXTURES_CSV


def _state_path(path: str | Path | None = None) -> Path:
    return Path(path) if path is not None else config.FIXTURE_SYNC_STATE


def _empty_store() -> pd.DataFrame:
    frame = pd.DataFrame(columns=list(STORE_COLUMNS))
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _normalise_time(value: object) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    parts = text.replace(".", ":").split(":")
    if len(parts) != 2:
        return None
    try:
        hour, minute = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return f"{hour:02d}:{minute:02d}"


def _parse_date(value: object) -> pd.Timestamp:
    """Parse a provider date without swapping ISO month/day.

    openfootball publishes ISO ``YYYY-MM-DD`` dates while football-data uses
    ``DD/MM/YYYY``. Parsing both with ``dayfirst=True`` silently turns
    ``2026-10-11`` into 10 November (pandas warns about this). ISO is therefore
    tried first and day-first parsing is only the fallback.
    """
    if value is None:
        return pd.NaT
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(value).normalize()
    if isinstance(value, float) and np.isnan(value):
        return pd.NaT
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat"}:
        return pd.NaT
    try:
        parsed = pd.to_datetime(text, format="ISO8601", errors="coerce")
    except (TypeError, ValueError):
        parsed = pd.NaT
    if pd.isna(parsed):
        try:
            parsed = pd.to_datetime(text, dayfirst=True, errors="coerce")
        except (TypeError, ValueError):
            return pd.NaT
    if isinstance(parsed, pd.Series):
        parsed = parsed.iloc[0]
    if pd.isna(parsed):
        return pd.NaT
    return pd.Timestamp(parsed).normalize()


def _status_from_openfootball(match: dict) -> str:
    """Map the provider's optional flags to a canonical status.

    The current file only publishes ``score`` for played matches, so absence of
    a status is treated as scheduled. A match without a date is postponed
    (kickoff to be confirmed) rather than invented.
    """
    raw = str(match.get("status") or "").strip().lower()
    if match.get("canceled") or match.get("cancelled") or raw in {
            "cancelled", "canceled", "abandoned"}:
        return STATUS_CANCELLED
    if raw in {"postponed", "pp", "pst"}:
        return STATUS_POSTPONED
    if match.get("score"):
        return STATUS_COMPLETED
    if not match.get("date"):
        return STATUS_POSTPONED
    return STATUS_SCHEDULED


# --------------------------------------------------------------------------
# Provider parsing
# --------------------------------------------------------------------------
def parse_football_data(
    frame: pd.DataFrame, cutoff: pd.Timestamp
) -> tuple[list[dict], list[dict], int]:
    """Parse the football-data fixture feed. Returns (rows, rejected, skipped)."""
    required = {"Div", "Date", "HomeTeam", "AwayTeam"}
    missing = required - set(frame.columns)
    if missing:
        raise ProviderError(f"football-data feed missing columns: {sorted(missing)}")
    from src.data import processing  # local import avoids a cycle

    rows: list[dict] = []
    rejected: list[dict] = []
    skipped = 0
    feed = frame[frame["Div"].astype(str).str.strip().str.upper() == "E0"]
    for _, source_row in feed.iterrows():
        date = _parse_date(source_row.get("Date"))
        home = processing.normalize_team(source_row.get("HomeTeam"))
        away = processing.normalize_team(source_row.get("AwayTeam"))
        if pd.isna(date):
            rejected.append({"source": "football-data", "home": home, "away": away,
                             "reason": "unparseable date"})
            continue
        if date <= cutoff:
            skipped += 1  # already completed (or stale feed); matches.csv is truth
            continue
        try:
            odds = processing._fixture_odds(source_row)
        except Exception as exc:  # prices are optional
            log.debug("odds extraction failed for %s vs %s: %s", home, away, exc)
            odds = {column: np.nan for column in ODDS_COLUMNS}
        rows.append({
            "date": date,
            "time": _normalise_time(source_row.get("Time")) or "15:00",
            "home_team": home,
            "away_team": away,
            "source": "football-data fixtures",
            "round": None,
            "status": STATUS_SCHEDULED,
            **{column: odds.get(column, np.nan) for column in ODDS_COLUMNS},
        })
    return rows, rejected, skipped


def parse_openfootball(
    payload: object, cutoff: pd.Timestamp
) -> tuple[list[dict], list[dict], int]:
    """Parse the openfootball published schedule. Returns (rows, rejected, skipped)."""
    if not isinstance(payload, dict) or not isinstance(payload.get("matches"), list):
        raise ProviderError("openfootball payload has no 'matches' list")
    from src.data import processing

    rows: list[dict] = []
    rejected: list[dict] = []
    skipped = 0
    for match in payload["matches"]:
        if not isinstance(match, dict):
            rejected.append({"source": "openfootball", "reason": "match is not an object"})
            continue
        status = _status_from_openfootball(match)
        if status == STATUS_COMPLETED:
            skipped += 1  # historical; matches.csv is the authority
            continue
        date = _parse_date(match.get("date"))
        home = processing.normalize_team(match.get("team1"))
        away = processing.normalize_team(match.get("team2"))
        if pd.notna(date) and date <= cutoff:
            skipped += 1
            continue
        rows.append({
            "date": date,
            "time": _normalise_time(match.get("time")) or "15:00",
            "home_team": home,
            "away_team": away,
            "source": "openfootball schedule",
            "round": match.get("round"),
            "status": status,
            **{column: np.nan for column in ODDS_COLUMNS},
        })
    return rows, rejected, skipped


def collect_sources(
    fetch: bool, cutoff: pd.Timestamp, raw_dir: str | Path | None = None
) -> tuple[list[dict], list[dict], dict, int]:
    """Read both providers from cache (or download them when *fetch* is true)."""
    from src.data import acquisition

    raw = Path(raw_dir) if raw_dir is not None else config.RAW_DIR
    rows: list[dict] = []
    rejected: list[dict] = []
    status: dict[str, dict] = {}
    skipped = 0

    fixtures_path = raw / "fixtures.csv"
    try:
        if fetch:
            acquisition.download_fixtures(force=True)
            source_status = "downloaded"
        else:
            source_status = "cached" if fixtures_path.exists() else "missing"
        if fixtures_path.exists():
            frame = pd.read_csv(fixtures_path, encoding="utf-8-sig", low_memory=False)
            parsed, bad, old = parse_football_data(frame, cutoff)
            rows.extend(parsed)
            rejected.extend(bad)
            skipped += old
            status["football-data"] = {"status": source_status, "rows": len(parsed),
                                       "url": config.FIXTURES_URL}
        else:
            status["football-data"] = {"status": "missing", "rows": 0,
                                       "url": config.FIXTURES_URL}
    except Exception as exc:
        log.warning("football-data fixtures unavailable: %s", exc)
        status["football-data"] = {"status": "failed", "rows": 0,
                                   "detail": f"{type(exc).__name__}: {exc}",
                                   "url": config.FIXTURES_URL}

    season_label = config.SEASONS.get(config.CURRENT_SEASON, "2026/27")
    openfootball_season = season_label.replace("/", "-")
    openfootball_path = raw / f"openfootball_{openfootball_season}.json"
    try:
        if fetch:
            acquisition.download_openfootball(season=openfootball_season, force=True)
            source_status = "downloaded"
        else:
            source_status = "cached" if openfootball_path.exists() else "missing"
        if openfootball_path.exists():
            payload = json.loads(openfootball_path.read_text(encoding="utf-8"))
            parsed, bad, old = parse_openfootball(payload, cutoff)
            rows.extend(parsed)
            rejected.extend(bad)
            skipped += old
            status["openfootball"] = {"status": source_status, "rows": len(parsed),
                                      "url": f"{config.OPENFOOTBALL_BASE}/"
                                             f"{openfootball_season}/en.1.json"}
        else:
            status["openfootball"] = {"status": "missing", "rows": 0,
                                      "url": f"{config.OPENFOOTBALL_BASE}/"
                                             f"{openfootball_season}/en.1.json"}
    except Exception as exc:
        log.warning("openfootball fixtures unavailable: %s", exc)
        status["openfootball"] = {"status": "failed", "rows": 0,
                                  "detail": f"{type(exc).__name__}: {exc}",
                                  "url": f"{config.OPENFOOTBALL_BASE}/"
                                         f"{openfootball_season}/en.1.json"}
    return rows, rejected, status, skipped


# --------------------------------------------------------------------------
# Store handling
# --------------------------------------------------------------------------
def load_store(path: str | Path | None = None) -> pd.DataFrame | None:
    """Return the persisted store, or ``None`` for a missing/legacy file."""
    store = _store_path(path)
    if not store.exists() or store.stat().st_size == 0:
        return None
    try:
        frame = pd.read_csv(store, dtype={"season_code": str})
    except Exception as exc:
        log.warning("Fixture store unreadable: %s", exc)
        return None
    if "fixture_id" not in frame.columns or "status" not in frame.columns:
        return None  # legacy schema; the next sync migrates it
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in STORE_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[list(STORE_COLUMNS)]


def _records(frame: pd.DataFrame | None, now: str) -> list[dict]:
    """Canonical dict records from a store frame (migrates the legacy schema)."""
    if frame is None or len(frame) == 0:
        return []
    records = []
    for _, row in frame.iterrows():
        date = _parse_date(row.get("date"))
        raw_code = row.get("season_code")
        season_code = ("" if pd.isna(raw_code) else str(raw_code)) or _season_code(date)
        home = str(row.get("home_team") or "").strip()
        away = str(row.get("away_team") or "").strip()
        record = {column: row.get(column) for column in STORE_COLUMNS}
        record.update({
            "date": date,
            "season_code": season_code,
            "season": config.SEASONS.get(season_code, ""),
            "home_team": home,
            "away_team": away,
            "fixture_id": str(row.get("fixture_id") or _fixture_id(season_code, home, away)),
            "status": str(row.get("status") or STATUS_SCHEDULED).lower(),
            "time": _normalise_time(row.get("time")) or "15:00",
            "timezone": TIMEZONE,
            "match_id": _match_id(date, home, away),
            "first_seen_at": row.get("first_seen_at") or now,
            "updated_at": row.get("updated_at") or now,
        })
        records.append(record)
    return records


def _match_id(date: pd.Timestamp, home: str, away: str) -> str:
    stamp = "" if pd.isna(date) else pd.Timestamp(date).strftime("%Y-%m-%d")
    return f"{stamp}_{home}_{away}"


def _completed_ids(matches: pd.DataFrame) -> set[str]:
    if matches is None or len(matches) == 0:
        return set()
    frame = matches.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    codes = (frame["season_code"].astype(str) if "season_code" in frame.columns
             else frame["date"].map(_season_code))
    return {_fixture_id(code, home, away)
            for code, home, away in zip(codes, frame["home_team"], frame["away_team"])}


def _validate_row(row: dict, known_teams: set[str]) -> str | None:
    home, away = row["home_team"], row["away_team"]
    if not home or not away or home.lower() == "nan" or away.lower() == "nan":
        return "missing team"
    if home == away:
        return "identical home and away team"
    unknown = {home, away} - known_teams
    if unknown:
        return f"unknown team mapping: {sorted(unknown)}"
    if row["status"] in (STATUS_SCHEDULED, STATUS_RESCHEDULED) and pd.isna(row["date"]):
        return "scheduled fixture without a date"
    return None


def _same(a: object, b: object) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    if isinstance(a, pd.Timestamp) or isinstance(b, pd.Timestamp):
        if pd.isna(a) or pd.isna(b):
            return False
        return pd.Timestamp(a).normalize() == pd.Timestamp(b).normalize()
    try:
        return bool(a == b)
    except (TypeError, ValueError):
        return False


def _merge_odds(old: dict, new: dict) -> dict:
    merged = {}
    for column in ODDS_COLUMNS:
        value = new.get(column)
        merged[column] = old.get(column) if pd.isna(value) else value
    return merged


def _merge(
    existing: list[dict], incoming: list[dict], matches: pd.DataFrame, now: str
) -> tuple[pd.DataFrame, dict, list[dict]]:
    known = set(matches["home_team"]) | set(matches["away_team"])
    completed = _completed_ids(matches)
    counts = {"new": 0, "updated": 0, "unchanged": 0, "postponed": 0,
              "cancelled": 0, "completed": 0, "retained": 0, "duplicates": 0,
              "rejected": 0}
    rejected: list[dict] = []

    # Fill schedule metadata (round/matchweek) across sources before merging:
    # football-data rows are authoritative for kickoff/prices, openfootball for
    # the matchday number.
    schedule = {}
    for row in incoming:
        if row.get("round"):
            schedule[_fixture_id(_season_code(row["date"]), row["home_team"],
                                 row["away_team"])] = row["round"]

    existing_by_id = {record["fixture_id"]: record for record in existing}
    seen: set[str] = set()
    merged: list[dict] = []
    ordered = sorted(incoming, key=lambda row: 0 if row["source"].startswith("football-data") else 1)

    for row in ordered:
        row = dict(row)
        date = row.get("date", pd.NaT)
        if pd.notna(date):
            date = pd.Timestamp(date).normalize()
        season_code = _season_code(date)
        fixture_id = _fixture_id(season_code, row["home_team"], row["away_team"])
        row.update({"date": date, "season_code": season_code,
                    "season": config.SEASONS.get(season_code, ""),
                    "fixture_id": fixture_id,
                    "timezone": TIMEZONE,
                    "match_id": _match_id(date, row["home_team"], row["away_team"])})
        row["round"] = row.get("round") or schedule.get(fixture_id)
        if row["round"]:
            matchweek = pd.to_numeric(
                pd.Series([row["round"]]).astype(str).str.extract(r"(\d+)")[0],
                errors="coerce").iloc[0]
            row["matchweek"] = matchweek

        reason = _validate_row(row, known)
        if reason:
            rejected.append({"source": row["source"], "home": row["home_team"],
                             "away": row["away_team"], "reason": reason})
            continue
        if fixture_id in completed:
            # A provider entry that matches a result already in matches.csv is
            # stale (openfootball sometimes lags); history is authoritative.
            counts["completed"] += 1
            continue
        if fixture_id in seen:
            counts["duplicates"] += 1
            continue
        seen.add(fixture_id)

        old = existing_by_id.get(fixture_id)
        if old is None:
            if row["status"] == STATUS_CANCELLED:
                counts["cancelled"] += 1
            elif row["status"] == STATUS_POSTPONED:
                counts["postponed"] += 1
            else:
                counts["new"] += 1
            row.update({"first_seen_at": now, "updated_at": now})
            merged.append({column: row.get(column) for column in STORE_COLUMNS})
            continue

        candidate = dict(old)
        candidate.update({
            "season_code": season_code,
            "season": row["season"],
            "round": row.get("round") or old.get("round"),
            "matchweek": row.get("matchweek") if pd.notna(row.get("matchweek")) else old.get("matchweek"),
            "source": row["source"] if not pd.isna(row.get("mkt_home")) else old.get("source"),
            **_merge_odds(old, row),
        })
        date_changed = not _same(old.get("date"), row["date"]) or not _same(
            old.get("time"), row.get("time"))

        if row["status"] == STATUS_CANCELLED:
            candidate["status"] = STATUS_CANCELLED
            candidate["date"] = old.get("date") if pd.isna(row["date"]) else row["date"]
            counts["cancelled"] += 1
        elif row["status"] == STATUS_POSTPONED:
            candidate["status"] = STATUS_POSTPONED
            candidate["date"] = old.get("date") if pd.isna(row["date"]) else row["date"]
            counts["postponed"] += 1
        else:
            if date_changed:
                candidate["status"] = STATUS_RESCHEDULED
            elif old.get("status") in (STATUS_POSTPONED, STATUS_CANCELLED):
                candidate["status"] = STATUS_SCHEDULED  # reinstated by the provider
            else:
                candidate["status"] = old.get("status") or STATUS_SCHEDULED
            candidate["date"] = row["date"]
            candidate["time"] = row.get("time") or old.get("time")
            candidate["match_id"] = _match_id(candidate["date"], candidate["home_team"],
                                              candidate["away_team"])

        if any(not _same(candidate.get(column), old.get(column)) for column in STORE_COLUMNS
               if column not in ("first_seen_at", "updated_at")):
            candidate["updated_at"] = now
            counts["updated"] += 1
        else:
            counts["unchanged"] += 1
        merged.append(candidate)

    # Fixtures absent from the response are retained unless a result exists.
    for fixture_id, old in existing_by_id.items():
        if fixture_id in seen:
            continue
        if fixture_id in completed or old.get("status") == STATUS_COMPLETED:
            counts["completed"] += 1
            continue
        counts["retained"] += 1
        merged.append(old)

    counts["rejected"] = len(rejected)
    frame = pd.DataFrame(merged, columns=list(STORE_COLUMNS))
    if len(frame):
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.sort_values(["date", "time", "home_team"],
                                  kind="mergesort", na_position="last")
    frame = frame.reset_index(drop=True)
    return frame, counts, rejected


# --------------------------------------------------------------------------
# Synchronisation
# --------------------------------------------------------------------------
def sync_fixtures(
    fetch: bool = False,
    matches: pd.DataFrame | None = None,
    store_path: str | Path | None = None,
    state_path: str | Path | None = None,
    raw_dir: str | Path | None = None,
    now: str | None = None,
) -> dict:
    """Synchronise the fixture store. Never raises for provider failures."""
    started = now or _now()
    store = _store_path(store_path)
    state_file = _state_path(state_path)
    from src.data import processing

    matches = matches if matches is not None else processing.load_processed()
    cutoff = pd.Timestamp(matches["date"].max()).normalize()

    existing_frame = load_store(store)
    if existing_frame is None and store.exists():
        # Legacy schema: read it for migration, but do not expose it as the store.
        try:
            legacy = pd.read_csv(store)
            existing_frame = None
            existing = _records(legacy, started)
        except Exception:
            existing = []
    else:
        existing = _records(existing_frame, started)

    try:
        incoming, rejected, sources, skipped = collect_sources(fetch, cutoff, raw_dir)
    except Exception as exc:  # defensive: collect_sources already isolates sources
        incoming, rejected = [], []
        sources = {"all": {"status": "failed", "detail": f"{type(exc).__name__}: {exc}"}}
        skipped = 0

    available = [name for name, block in sources.items()
                 if block.get("status") in ("downloaded", "cached")]
    if not available:
        view = upcoming(matches=matches, store_path=store)
        payload = {
            "started_at": started,
            "finished_at": _now(),
            "ok": False,
            "status": "failed",
            "sources": sources,
            "counts": {"new": 0, "updated": 0, "unchanged": 0, "postponed": 0,
                       "cancelled": 0, "completed": 0, "retained": 0,
                       "duplicates": 0, "rejected": 0},
            "historical_skipped": skipped,
            "store_rows": len(existing),
            "upcoming_rows": int(len(view)),
            "store_changed": False,
            "rejected": rejected[:20],
            "error": "no fixture provider available; existing store preserved",
        }
        _write_state(state_file, payload)
        log.error("Fixture sync failed: no provider available")
        return payload

    frame, counts, merge_rejected = _merge(existing, incoming, matches, started)
    rejected = rejected + merge_rejected
    counts["rejected"] = len(rejected)

    before = store.read_bytes() if store.exists() else b""
    _write_store(store, frame)
    after = store.read_bytes() if store.exists() else b""
    changed = before != after

    view = upcoming(matches=matches, store_path=store)
    partial = any(block.get("status") == "failed" for block in sources.values())
    payload = {
        "started_at": started,
        "finished_at": _now(),
        "ok": True,
        "status": "partial" if partial else "ok",
        "sources": sources,
        "counts": counts,
        "historical_skipped": skipped,
        "store_rows": int(len(frame)),
        "upcoming_rows": int(len(view)),
        "store_changed": bool(changed),
        "rejected": rejected[:20],
    }
    _write_state(state_file, payload)
    log.info("Fixture sync %s: new=%d updated=%d unchanged=%d retained=%d "
             "postponed=%d cancelled=%d rejected=%d (store %d, upcoming %d)",
             payload["status"], counts["new"], counts["updated"], counts["unchanged"],
             counts["retained"], counts["postponed"], counts["cancelled"],
             counts["rejected"], payload["store_rows"], payload["upcoming_rows"])
    return payload


def _write_store(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    # Explicit LF keeps the store byte-identical on Windows and Linux, so an
    # unchanged sync never produces a line-ending-only commit.
    frame.to_csv(tmp, index=False, lineterminator="\n")
    os.replace(tmp, path)


def _write_state(path: Path, run: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            history = payload.get("history", [])
        except (json.JSONDecodeError, AttributeError):
            history = []
    payload = {"last_run": run, "history": (history + [run])[-60:]}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def load_state(path: str | Path | None = None) -> dict:
    state = _state_path(path)
    if not state.exists():
        return {}
    try:
        return json.loads(state.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


# --------------------------------------------------------------------------
# Upcoming view (consumed by the prediction engine)
# --------------------------------------------------------------------------
def upcoming(matches: pd.DataFrame | None = None,
             store_path: str | Path | None = None) -> pd.DataFrame:
    """Return the next matchweek of valid upcoming fixtures from the store."""
    frame = load_store(store_path)
    if frame is None or len(frame) == 0:
        return _empty_store()
    if matches is None:
        from src.data import processing
        matches = processing.load_processed()
    cutoff = pd.Timestamp(matches["date"].max()).normalize()

    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    valid = frame[frame["status"].isin(UPCOMING_STATUSES)
                  & frame["date"].notna() & (frame["date"] > cutoff)].copy()
    if len(valid) == 0:
        return _empty_store()

    weeks = pd.to_numeric(valid["matchweek"], errors="coerce")
    known = weeks.dropna()
    if len(known):
        # The authoritative provider matchweek decides the active week — even
        # when only a few of its fixtures are still outstanding (e.g. one
        # rescheduled fixture after the rest of the round finished). A
        # matchweek with outstanding fixtures must not be skipped for being
        # small, and progression must not be inferred from calendar dates.
        target = known.min()
        selected = valid[weeks == target]
    else:
        # No provider matchweek information: fall back to a one-week horizon.
        horizon = valid["date"].min() + pd.Timedelta(days=6)
        selected = valid[valid["date"] <= horizon]
    selected = selected.sort_values(["date", "time", "home_team"],
                                    kind="mergesort").reset_index(drop=True)
    selected["match_id"] = (selected["date"].dt.strftime("%Y-%m-%d") + "_"
                            + selected["home_team"] + "_" + selected["away_team"])
    return selected[list(STORE_COLUMNS)]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Synchronise upcoming EPL fixtures into data/fixtures/upcoming.csv")
    parser.add_argument("--fetch", action="store_true",
                        help="download the provider feeds before syncing")
    parser.add_argument("--json", action="store_true", help="print the full run payload")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = sync_fixtures(fetch=args.fetch)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        counts = result["counts"]
        print(json.dumps({
            "status": result["status"],
            "new": counts["new"], "updated": counts["updated"],
            "unchanged": counts["unchanged"], "postponed": counts["postponed"],
            "cancelled": counts["cancelled"], "completed": counts["completed"],
            "retained": counts["retained"], "rejected": counts["rejected"],
            "store_rows": result["store_rows"],
            "upcoming_rows": result["upcoming_rows"],
            "store_changed": result["store_changed"],
        }, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
