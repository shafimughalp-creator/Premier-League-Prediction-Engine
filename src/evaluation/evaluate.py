"""Prediction evaluation and matchweek/season aggregation.

Compares the *frozen* pre-match prediction snapshots against official final
results and maintains three stores:

* ``data/evaluation/evaluations.json``  — one record per evaluated fixture
  (idempotent, correction-aware, with an audit history when a final result
  or an eligible prediction was recomputed);
* ``data/evaluation/matchweeks.json``   — per-matchweek and season-to-date
  aggregates, separating *pending* and *missing prediction* fixtures so
  neither is ever counted as incorrect;
* ``data/evaluation/freshness.json``    — last sync/generation timestamps
  for fixture sync, predictions, results and evaluation.

Matchweek accuracy formula
--------------------------
``accuracy = correct / evaluated`` where *evaluated* counts only completed
fixtures that had a valid eligible pre-match prediction. Pending matches
(scheduled/postponed, no official result yet) and completed matches without
an eligible prediction are reported separately — never as incorrect. With
zero evaluated predictions the accuracy is ``null`` (an honest empty state),
never 0%.

A match belongs to the matchweek recorded by the authoritative provider
schedule (carried on the snapshot / result / fixture-store row), not the
calendar week it happened to be played.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src import config
from src.data import fixture_sync
from src.prediction import snapshots
from src.results import sync as result_sync
from src.utilities.io import write_json_atomic

log = logging.getLogger(__name__)

EVALUATIONS_JSON = config.DATA_DIR / "evaluation" / "evaluations.json"
MATCHWEEKS_JSON = config.DATA_DIR / "evaluation" / "matchweeks.json"
FRESHNESS_JSON = config.DATA_DIR / "evaluation" / "freshness.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _store_path(path: str | Path | None, default: Path) -> Path:
    return Path(path) if path is not None else default


def load_evaluations(path: str | Path | None = None) -> dict:
    target = _store_path(path, EVALUATIONS_JSON)
    if not target.exists():
        return {"updated_at": None, "evaluations": {}}
    with open(target, encoding="utf-8") as handle:
        return json.load(handle)


def _save(store: dict, path: str | Path | None, default: Path) -> None:
    store["updated_at"] = _now()
    write_json_atomic(_store_path(path, default), store)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def evaluate(
    now: str | None = None,
    snapshots_path: str | Path | None = None,
    results_path: str | Path | None = None,
    evaluations_path: str | Path | None = None,
) -> dict:
    """Evaluate final results against frozen pre-match predictions.

    Idempotent: an unchanged (result, prediction) pair keeps its record and
    is reported as ``unchanged``. If an authoritative result correction
    changes the scores, or an eligible prediction was revised before a
    postponed fixture's new kickoff, the evaluation is recomputed and the
    previous verdict is retained in the record's ``history``.
    """
    moment = now or _now()
    snap_store = snapshots.load(snapshots_path)["snapshots"]
    results = result_sync.load(results_path)["results"]
    store = load_evaluations(evaluations_path)
    evaluations = store.setdefault("evaluations", {})

    report = {
        "at": moment,
        "evaluated": 0,
        "new": 0,
        "unchanged": 0,
        "recomputed": 0,
        "missing_predictions": 0,
        "total": 0,
        "needs_review": [],
    }

    evaluated_ids: list[str] = []
    for fixture_id in sorted(results):
        result = results[fixture_id]
        record = snap_store.get(fixture_id)
        eligible = (record or {}).get("eligible")
        if not eligible:
            report["missing_predictions"] += 1
            report["needs_review"].append({
                "fixture_id": fixture_id,
                "reason": "completed match without an eligible pre-match prediction",
            })
            continue

        correct = eligible.get("predicted") == result["actual_outcome"]
        evaluated_ids.append(fixture_id)
        existing = evaluations.get(fixture_id)

        if existing is not None:
            same_result = (existing.get("goals_home") == result["goals_home"]
                           and existing.get("goals_away") == result["goals_away"])
            same_prediction = (
                existing.get("prediction_created_at") == eligible.get("prediction_created_at")
                and existing.get("predicted_outcome") == eligible.get("predicted")
            )
            if same_result and same_prediction:
                report["unchanged"] += 1
                continue
            # Recompute with an audit trail of the previous verdict.
            existing.setdefault("history", []).append({
                "at": moment,
                "previous_correct": existing.get("correct"),
                "previous_goals_home": existing.get("goals_home"),
                "previous_goals_away": existing.get("goals_away"),
                "previous_prediction_created_at": existing.get("prediction_created_at"),
                "reason": ("result correction" if not same_result
                           else "eligible prediction revised"),
            })
            existing.update({
                "goals_home": result["goals_home"],
                "goals_away": result["goals_away"],
                "actual_outcome": result["actual_outcome"],
                "result_source": result.get("source"),
                "result_synced_at": result.get("last_synced_at"),
                "predicted_outcome": eligible.get("predicted"),
                "prob_home": eligible.get("prob_home"),
                "prob_draw": eligible.get("prob_draw"),
                "prob_away": eligible.get("prob_away"),
                "prediction_created_at": eligible.get("prediction_created_at"),
                "model_id": eligible.get("model_id"),
                "model_variant": eligible.get("model_variant"),
                "correct": correct,
                "evaluated_at": moment,
            })
            report["recomputed"] += 1
            continue

        evaluations[fixture_id] = {
            "fixture_id": fixture_id,
            "season_code": result.get("season_code") or (record or {}).get("season_code") or "",
            "season": result.get("season") or (record or {}).get("season") or "",
            "matchweek": result.get("matchweek")
                         if result.get("matchweek") is not None else (record or {}).get("matchweek"),
            "home_team": result["home_team"],
            "away_team": result["away_team"],
            "predicted_outcome": eligible.get("predicted"),
            "prob_home": eligible.get("prob_home"),
            "prob_draw": eligible.get("prob_draw"),
            "prob_away": eligible.get("prob_away"),
            "model_id": eligible.get("model_id"),
            "model_variant": eligible.get("model_variant"),
            "prediction_created_at": eligible.get("prediction_created_at"),
            "prediction_status": "pre_match",
            "goals_home": result["goals_home"],
            "goals_away": result["goals_away"],
            "actual_outcome": result["actual_outcome"],
            "correct": correct,
            "result_source": result.get("source"),
            "result_synced_at": result.get("last_synced_at"),
            "evaluated_at": moment,
            "history": [],
        }
        report["new"] += 1

    report["evaluated"] = len(evaluated_ids)
    report["total"] = len(evaluations)
    report["needs_review"] = report["needs_review"][:20]

    if report["new"] or report["recomputed"]:
        _save(store, evaluations_path, EVALUATIONS_JSON)
        # Bookkeeping only: flag evaluated snapshots as completed. Eligible
        # predictions are never touched here.
        changed = snapshots.mark_completed(evaluated_ids, path=snapshots_path)
        report["snapshots_marked_completed"] = changed
    else:
        report["snapshots_marked_completed"] = 0
    return report


# --------------------------------------------------------------------------
# Matchweek / season aggregation
# --------------------------------------------------------------------------
def aggregate(
    now: str | None = None,
    results_path: str | Path | None = None,
    evaluations_path: str | Path | None = None,
    store_path: str | Path | None = None,
    snapshots_path: str | Path | None = None,
    fixture_store: "pd.DataFrame | None" = None,
) -> dict:
    """Compute per-matchweek and season-to-date aggregates.

    Scheduled fixtures are the union of completed results, fixture-store
    rows and prediction snapshots for a matchweek, so postponed matches and
    unplayed future weeks are counted honestly. ``pending`` is
    ``scheduled - completed``; ``missing_prediction`` is
    ``completed - evaluated`` and is never treated as incorrect.
    """
    import pandas as pd  # local import keeps the module importable without pandas users

    moment = now or _now()
    evaluations = load_evaluations(evaluations_path)["evaluations"]
    results = result_sync.load(results_path)["results"]

    scheduled: dict[tuple[str, int | None], set[str]] = {}
    completed: dict[tuple[str, int | None], set[str]] = {}
    cancelled: dict[tuple[str, int | None], set[str]] = {}

    def bucket(season_code: str, matchweek) -> tuple[str, int | None]:
        return (season_code, None if matchweek is None else int(matchweek))

    def is_cancelled(status) -> bool:
        return str(status or "").strip().lower() == "cancelled"

    # Fixture store: every stored row is scheduled unless the provider
    # confirmed a cancellation — cancelled fixtures are terminal and are
    # reported separately, never as pending and never as incorrect.
    if fixture_store is None:
        fixture_store = fixture_sync.load_store()
    if fixture_store is not None and len(fixture_store):
        for _, row in fixture_store.iterrows():
            code = str(row.get("season_code") or "")
            key = bucket(code, row.get("matchweek"))
            if is_cancelled(row.get("status")):
                cancelled.setdefault(key, set()).add(str(row["fixture_id"]))
            else:
                scheduled.setdefault(key, set()).add(str(row["fixture_id"]))

    # Snapshots: fixtures predicted but possibly not in the store.
    snap_store = snapshots.load(snapshots_path)["snapshots"]
    for fixture_id, record in snap_store.items():
        code = str(record.get("season_code") or "")
        key = bucket(code, record.get("matchweek"))
        state = cancelled if is_cancelled(record.get("status")) else scheduled
        state.setdefault(key, set()).add(fixture_id)

    # Results: officially completed.
    for fixture_id, result in results.items():
        code = str(result.get("season_code") or "")
        key = bucket(code, result.get("matchweek"))
        scheduled.setdefault(key, set()).add(fixture_id)
        completed.setdefault(key, set()).add(fixture_id)

    evaluated_by_key: dict[tuple[str, int | None], set[str]] = {}
    for fixture_id, evaluation in evaluations.items():
        key = bucket(str(evaluation.get("season_code") or ""),
                     evaluation.get("matchweek"))
        evaluated_by_key.setdefault(key, set()).add(fixture_id)

    seasons: dict[str, dict] = {}
    for (season_code, matchweek) in sorted(set(scheduled) | set(cancelled),
                                           key=lambda k: (k[0], k[1] is None, k[1] or 0)):
        cancelled_ids = cancelled.get((season_code, matchweek), set())
        # Cancelled fixtures are terminal: exclude them from the scheduled /
        # pending maths entirely and count them in their own bucket.
        fixture_ids = scheduled.get((season_code, matchweek), set()) - cancelled_ids
        done = completed.get((season_code, matchweek), set())
        evaluated = evaluated_by_key.get((season_code, matchweek), set()) & done
        correct = sum(1 for fid in evaluated if evaluations[fid]["correct"])
        incorrect = len(evaluated) - correct
        missing_prediction = len(done - evaluated)
        entry = {
            "matchweek": matchweek,
            "scheduled": len(fixture_ids),
            "completed": len(done),
            "evaluated": len(evaluated),
            "correct": correct,
            "incorrect": incorrect,
            "pending": len(fixture_ids - done),
            "cancelled": len(cancelled_ids - done),
            "missing_prediction": missing_prediction,
            "accuracy": round(100.0 * correct / len(evaluated), 2) if evaluated else None,
            "coverage": round(100.0 * len(evaluated) / len(done), 2) if done else None,
        }
        season = seasons.setdefault(season_code, {"matchweeks": {}, "season": None})
        season["matchweeks"][str(matchweek) if matchweek is not None else "unknown"] = entry

    for season_code, season in seasons.items():
        weeks = [entry for key, entry in season["matchweeks"].items() if key != "unknown"]
        total_evaluated = sum(entry["evaluated"] for entry in weeks)
        season["season"] = {
            "matchweeks": len(weeks),
            "scheduled": sum(entry["scheduled"] for entry in weeks),
            "completed": sum(entry["completed"] for entry in weeks),
            "evaluated": total_evaluated,
            "correct": sum(entry["correct"] for entry in weeks),
            "incorrect": sum(entry["incorrect"] for entry in weeks),
            "pending": sum(entry["pending"] for entry in weeks),
            "cancelled": sum(entry["cancelled"] for entry in weeks),
            "missing_prediction": sum(entry["missing_prediction"] for entry in weeks),
            "accuracy": round(100.0 * sum(entry["correct"] for entry in weeks) / total_evaluated, 2)
                        if total_evaluated else None,
        }

    payload = {"at": moment, "seasons": seasons}
    target = _store_path(store_path, MATCHWEEKS_JSON)
    if target.exists():
        with open(target, encoding="utf-8") as handle:
            previous = json.load(handle)
        if previous.get("seasons") == seasons:
            return {"at": moment, "changed": False, "seasons": seasons}
    write_json_atomic(target, payload)
    return {"at": moment, "changed": True, "seasons": seasons}


# --------------------------------------------------------------------------
# Freshness
# --------------------------------------------------------------------------
def freshness(now: str | None = None) -> dict:
    """Last synchronization timestamps for every automated data source."""
    moment = now or _now()

    def _read(path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    fixture_state = _read(config.FIXTURE_SYNC_STATE)
    pipeline_state = _read(config.DATA_DIR / "pipeline_state.json")
    snap_store = _read(snapshots.SNAPSHOTS_JSON)
    result_store = _read(result_sync.RESULTS_JSON)
    evaluations = _read(EVALUATIONS_JSON)
    payload = {
        "updated_at": moment,
        "fixtures": {
            "last_sync_at": (fixture_state.get("last_run") or {}).get("finished_at"),
            "status": (fixture_state.get("last_run") or {}).get("status"),
        },
        "predictions": {
            "last_run": pipeline_state.get("last_success") or pipeline_state.get("last_run"),
            "snapshots_updated_at": snap_store.get("updated_at"),
            "snapshots": len(snap_store.get("snapshots", {})),
        },
        "results": {
            "last_sync_at": result_store.get("updated_at"),
            "completed": len(result_store.get("results", {})),
        },
        "evaluation": {
            "last_run": evaluations.get("updated_at"),
            "evaluated": len(evaluations.get("evaluations", {})),
        },
    }
    return payload


def _save_freshness(payload: dict) -> None:
    write_json_atomic(FRESHNESS_JSON, payload)


def run(now: str | None = None, **paths) -> dict:
    """Evaluate + aggregate + freshness in one idempotent step."""
    moment = now or _now()
    report = {
        "at": moment,
        "evaluation": evaluate(now=moment, **{k: v for k, v in paths.items()
                                              if k in ("snapshots_path", "results_path",
                                                       "evaluations_path")}),
        "aggregates": aggregate(now=moment, **{k: v for k, v in paths.items()
                                               if k in ("results_path", "evaluations_path",
                                                        "store_path", "snapshots_path")}),
        "freshness": freshness(now=moment),
    }
    _save_freshness(report["freshness"])
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prediction evaluation and aggregation")
    parser.add_argument("--json", action="store_true", help="print a JSON report")
    args = parser.parse_args(argv)
    report = run()
    if args.json:
        printable = {key: report[key] for key in ("at", "evaluation", "aggregates")}
        printable["aggregates"] = {k: v for k, v in report["aggregates"].items() if k != "at"}
        print(json.dumps(printable, indent=2))
    else:
        log.info("%s", report["evaluation"])
        log.info("aggregates changed: %s", report["aggregates"]["changed"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
