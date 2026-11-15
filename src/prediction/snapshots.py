"""Immutable pre-match prediction snapshots.

The prediction engine regenerates ``data/predictions.json`` on every run;
that payload is a *display* artefact, not an evaluation record. This module
persists the eligible pre-match prediction for every fixture in an
append-only store so a later run — or a model variant upgrade, or an
authorized pre-kickoff revision — can never silently replace the
probabilities a result is scored against.

Store contract
--------------
``data/predictions/snapshots.json`` maps the stable fixture id
(``"{season_code}:{home}:{away}"`` from the fixture store) to a record:

    {
      "fixture_id":  "2627:Aston Villa:Sunderland",
      "season_code": "2627",
      "season":      "2026/27",
      "matchweek":   11,
      "home_team":   "Aston Villa",
      "away_team":   "Sunderland",
      "kickoff_at":  "2026-11-21T15:00:00+00:00",   # Europe/London resolved
      "created_at":  "...",
      "updated_at":  "...",
      "status":      "pre_match" | "locked" | "completed" | "cancelled" | "postponed",
      "eligible": {
        "prob_home": 0.4895, "prob_draw": 0.2742, "prob_away": 0.2362,
        "predicted": "H",
        "model_id":  "epl-20261113-1478e7d5",
        "model_variant": "no_odds",
        "has_odds": false,
        "prediction_created_at": "...",
        "prediction_status": "pre_match"
      },
      "revisions": [ {…earlier versions with superseded_at…} ]
    }

Eligibility policy
------------------
* The **eligible** prediction is the latest version whose
  ``prediction_created_at`` is at or before the fixture kickoff.
* A version may be replaced **only before kickoff** (e.g. bookmaker prices
  are published and the full model variant replaces the no-odds variant).
  The superseded version is retained in ``revisions`` with both timestamps.
* Once ``now >= kickoff_at`` the record is **locked**: ``eligible`` is frozen
  for evaluation, whatever the current engine would output.
* Postponements and reschedules keep the snapshot (and its history); the
  kickoff instant simply moves with the fixture store.
* Fixtures without a usable kickoff time are snapshotted on creation but are
  never eligible for revision (``locked_at_creation`` flag) because
  pre-matchness cannot be verified.

Probabilities are validated (each in [0, 1], sum within 1e-6 of 1) before a
version is accepted. All writes are atomic and the store is only ever
touched by the pipeline, so a provider outage never affects it.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from src import config
from src.utilities.io import write_json_atomic

log = logging.getLogger(__name__)

SNAPSHOTS_JSON = config.DATA_DIR / "predictions" / "snapshots.json"
KICKOFF_TZ = ZoneInfo("Europe/London")

STATUS_PRE_MATCH = "pre_match"
STATUS_LOCKED = "locked"
STATUS_COMPLETED = "completed"
STATUS_CANCELLED = "cancelled"
STATUS_POSTPONED = "postponed"

_PROBABILITY_FIELDS = ("prob_home", "prob_draw", "prob_away")


class SnapshotError(ValueError):
    """A candidate prediction cannot be persisted."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def kickoff_instant(date: object, time: object) -> str | None:
    """Resolve a fixture date + Europe/London kickoff time to a UTC instant."""
    if date is None or (isinstance(date, float) and pd.isna(date)):
        return None
    try:
        day = pd.Timestamp(date)
    except (TypeError, ValueError):
        return None
    if pd.isna(day):
        return None
    if not time or str(time).strip().lower() in {"nan", "none"}:
        return None
    parts = str(time).replace(".", ":").split(":")
    if len(parts) != 2:
        return None
    try:
        hour, minute = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    local = datetime(day.year, day.month, day.day, hour, minute, tzinfo=KICKOFF_TZ)
    return local.astimezone(timezone.utc).isoformat(timespec="seconds")


def validate_probabilities(prob_home: float, prob_draw: float, prob_away: float) -> None:
    values = (prob_home, prob_draw, prob_away)
    for value in values:
        if not isinstance(value, (int, float)) or pd.isna(value) or not 0.0 <= float(value) <= 1.0:
            raise SnapshotError(f"probability outside [0, 1]: {values}")
    if abs(sum(map(float, values)) - 1.0) > 1e-6:
        raise SnapshotError(f"probabilities do not sum to 1: {values}")


def _store_path(path: str | Path | None) -> Path:
    return Path(path) if path is not None else SNAPSHOTS_JSON


def load(path: str | Path | None = None) -> dict:
    target = _store_path(path)
    if not target.exists():
        return {"updated_at": None, "snapshots": {}}
    with open(target, encoding="utf-8") as handle:
        return json.load(handle)


def _save(store: dict, path: str | Path | None = None) -> None:
    store["updated_at"] = _now()
    write_json_atomic(_store_path(path), store)


def _candidate_version(row: dict) -> dict:
    """Extract an immutable version record from a prediction payload row."""
    probabilities = {key: float(row[key]) for key in _PROBABILITY_FIELDS}
    validate_probabilities(*probabilities.values())
    model_id = str(row.get("model_version") or "")
    if not model_id:
        raise SnapshotError("prediction payload without a model identifier")
    created = str(row.get("predicted_at") or _now())
    return {
        **probabilities,
        "predicted": str(row.get("predicted") or ""),
        "model_id": model_id,
        "model_variant": str(row.get("model_variant") or ""),
        "has_odds": bool(row.get("has_odds")),
        "prediction_created_at": created,
        "prediction_status": STATUS_PRE_MATCH,
    }


def reconcile(
    store_rows: pd.DataFrame,
    predictions: list[dict],
    now: str | None = None,
    path: str | Path | None = None,
) -> dict:
    """Create, revise or lock snapshots from the current prediction payload.

    ``store_rows`` is the full fixture store (any status); ``predictions`` is
    the list of prediction payload rows from the engine. Returns a report
    with ``created / revised / unchanged / locked / missing / invalid`` counts
    and is idempotent: an unchanged payload leaves the store byte-identical.
    """
    moment = now or _now()
    moment_ts = pd.Timestamp(moment)
    store = load(path)
    snapshots = store.setdefault("snapshots", {})

    by_fixture: dict[str, dict] = {}
    for row in predictions:
        fixture_id = row.get("fixture_id")
        if fixture_id:
            by_fixture[str(fixture_id)] = row

    meta_rows = {
        str(row["fixture_id"]): row
        for _, row in store_rows.iterrows()
        if row.get("fixture_id")
    } if store_rows is not None and len(store_rows) else {}

    report = {
        "at": moment,
        "total": len(snapshots),
        "created": 0,
        "revised": 0,
        "unchanged": 0,
        "locked": 0,
        "missing": 0,
        "invalid": 0,
        "post_kickoff": 0,
        "status_updates": 0,
        "needs_review": [],
    }

    touched = set(meta_rows) | set(snapshots) | set(by_fixture)
    for fixture_id in sorted(touched):
        meta = meta_rows.get(fixture_id)
        snapshot = snapshots.get(fixture_id)

        if snapshot is None:
            if fixture_id not in by_fixture:
                continue  # fixture not predicted yet; nothing to persist
            row = by_fixture[fixture_id]
            try:
                candidate = _candidate_version(row)
            except SnapshotError as exc:
                report["invalid"] += 1
                report["needs_review"].append({"fixture_id": fixture_id, "reason": str(exc)})
                continue
            kickoff = kickoff_instant(meta.get("date"), meta.get("time")) if meta is not None else None
            # A prediction whose ACTUAL creation time is at or after kickoff is
            # never a valid pre-match prediction (e.g. a delayed workflow or a
            # fixture row that surfaced late): the record is born locked and
            # ineligible, the actual candidate is kept for audit with its real
            # timestamp, and nothing is backdated. Evaluation then reports the
            # fixture under ``missing_prediction`` instead of scoring it.
            born_after_kickoff = (
                kickoff is not None
                and pd.Timestamp(candidate["prediction_created_at"]) >= pd.Timestamp(kickoff))
            if born_after_kickoff:
                snapshots[fixture_id] = {
                    "fixture_id": fixture_id,
                    "season_code": str(meta.get("season_code") or "") if meta is not None else "",
                    "season": str(row.get("season") or ""),
                    "matchweek": None if row.get("matchweek") is None else int(row["matchweek"]),
                    "home_team": str(row.get("home_team") or ""),
                    "away_team": str(row.get("away_team") or ""),
                    "kickoff_at": kickoff,
                    "created_at": moment,
                    "updated_at": moment,
                    "status": STATUS_LOCKED,
                    "locked_at_creation": True,
                    "eligible": None,
                    "revisions": [],
                    "ineligible_candidates": [{
                        **candidate,
                        "ineligible_reason": "prediction created at or after kickoff",
                    }],
                }
                report["created"] += 1
                report["post_kickoff"] += 1
                continue
            snapshots[fixture_id] = {
                "fixture_id": fixture_id,
                "season_code": str(meta.get("season_code") or "") if meta is not None else "",
                "season": str(row.get("season") or ""),
                "matchweek": None if row.get("matchweek") is None else int(row["matchweek"]),
                "home_team": str(row.get("home_team") or ""),
                "away_team": str(row.get("away_team") or ""),
                "kickoff_at": kickoff,
                "created_at": moment,
                "updated_at": moment,
                "status": STATUS_PRE_MATCH if kickoff is not None else STATUS_LOCKED,
                "locked_at_creation": kickoff is None,
                "eligible": candidate,
                "revisions": [],
            }
            report["created"] += 1
            continue

        # Existing snapshot ---------------------------------------------------
        kickoff_now = kickoff_instant(meta.get("date"), meta.get("time")) if meta is not None else None
        if kickoff_now is not None and kickoff_now != snapshot.get("kickoff_at"):
            # Reschedule or postponement: the instant moves, history stays.
            snapshot["kickoff_at"] = kickoff_now
            snapshot["updated_at"] = moment
            report["status_updates"] += 1

        fixture_status = str(meta.get("status") or "") if meta is not None else ""
        if fixture_status == STATUS_CANCELLED and snapshot["status"] != STATUS_CANCELLED:
            snapshot["status"] = STATUS_CANCELLED
            snapshot["updated_at"] = moment
            report["status_updates"] += 1
            continue
        if fixture_status == STATUS_POSTPONED and snapshot["status"] == STATUS_PRE_MATCH:
            snapshot["status"] = STATUS_POSTPONED
            snapshot["updated_at"] = moment
            report["status_updates"] += 1
        elif (fixture_status in {"scheduled", "rescheduled"}
                and snapshot["status"] == STATUS_POSTPONED
                and kickoff_now is not None and moment_ts < pd.Timestamp(kickoff_now)):
            # The provider published a new date for a previously postponed
            # fixture: back to pre-match, history and prediction preserved.
            snapshot["status"] = STATUS_PRE_MATCH
            snapshot["updated_at"] = moment
            report["status_updates"] += 1

        kickoff_ts = pd.Timestamp(snapshot["kickoff_at"]) if snapshot.get("kickoff_at") else None
        after_kickoff = kickoff_ts is not None and moment_ts >= kickoff_ts

        if snapshot["status"] in {STATUS_LOCKED, STATUS_COMPLETED, STATUS_CANCELLED} or (
                after_kickoff and snapshot["status"] not in {STATUS_POSTPONED}):
            if snapshot["status"] == STATUS_PRE_MATCH and after_kickoff:
                snapshot["status"] = STATUS_LOCKED
                snapshot["updated_at"] = moment
                report["locked"] += 1
            continue  # frozen: never silently replaced

        if fixture_id not in by_fixture:
            report["missing"] += 1
            continue

        row = by_fixture[fixture_id]
        try:
            candidate = _candidate_version(row)
        except SnapshotError as exc:
            report["invalid"] += 1
            report["needs_review"].append({"fixture_id": fixture_id, "reason": str(exc)})
            continue

        if snapshot.get("locked_at_creation") or kickoff_ts is None:
            continue  # created without a verifiable kickoff: never revised

        current = snapshot["eligible"]
        differs = any(
            abs(float(candidate[key]) - float(current[key])) > 1e-9
            for key in _PROBABILITY_FIELDS
        ) or candidate["model_id"] != current["model_id"] or candidate["predicted"] != current["predicted"]

        if not differs:
            report["unchanged"] += 1
            continue

        created_ts = pd.Timestamp(candidate["prediction_created_at"])
        if created_ts >= kickoff_ts:
            # The engine generated this at or after kickoff; it must not
            # replace the pre-match version. Keep the frozen record untouched.
            snapshot["status"] = STATUS_LOCKED
            snapshot["updated_at"] = moment
            report["locked"] += 1
            continue

        superseded = dict(current)
        superseded["superseded_at"] = moment
        snapshot["revisions"].append(superseded)
        snapshot["eligible"] = candidate
        snapshot["updated_at"] = moment
        report["revised"] += 1

    report["needs_review"] = report["needs_review"][:20]
    report["total"] = len(snapshots)
    _save(store, path)
    return report


def eligible_predictions(path: str | Path | None = None) -> dict[str, dict]:
    """Map fixture_id -> the frozen eligible pre-match prediction."""
    return {
        fixture_id: record["eligible"]
        for fixture_id, record in load(path)["snapshots"].items()
        if record.get("eligible")
    }


def mark_completed(fixture_ids: list[str], path: str | Path | None = None) -> int:
    """Flag snapshots whose matches have a final result (status bookkeeping)."""
    store = load(path)
    changed = 0
    for fixture_id in fixture_ids:
        snapshot = store.get("snapshots", {}).get(fixture_id)
        if snapshot and snapshot["status"] != STATUS_COMPLETED:
            snapshot["status"] = STATUS_COMPLETED
            snapshot["updated_at"] = _now()
            changed += 1
    if changed:
        _save(store, path)
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prediction snapshot store utilities")
    parser.add_argument("--json", action="store_true", help="print a JSON summary")
    args = parser.parse_args(argv)
    store = load()
    summary = {
        "updated_at": store.get("updated_at"),
        "snapshots": len(store.get("snapshots", {})),
        "locked": sum(
            1 for record in store.get("snapshots", {}).values()
            if record.get("status") != STATUS_PRE_MATCH
        ),
    }
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        log.info("%s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
