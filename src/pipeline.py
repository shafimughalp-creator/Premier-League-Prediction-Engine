"""End-to-end orchestration: refresh -> validate -> retrain (guarded) -> predict.

Design rules
------------
* A refresh failure never destroys the existing dataset; the cached copy is
  used and the failure is logged.
* Retraining is skipped unless there is genuinely new completed data or the
  model is missing/stale - rebuilding every day on identical data wastes time
  and churns the artifact.
* A newly trained model is only promoted when it is not materially worse than
  the model already in production. The previous artifact is always preserved.
* Every run appends a record to ``data/pipeline_state.json`` so the dashboard
  and CI can show exactly what happened and when.

Usage
-----
    python -m src.pipeline                 # auto refresh + auto retrain
    python -m src.pipeline --offline       # use cached data only
    python -m src.pipeline --retrain always
    python -m src.pipeline --predict-only
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from src import config
from src.data import acquisition, fixture_sync, processing, validation
from src.prediction import snapshots
from src.prediction.engine import PredictionEngine
from src.utilities.io import read_json, write_json_atomic

log = logging.getLogger(__name__)

STATE_JSON = config.DATA_DIR / "pipeline_state.json"
BACKUP_PKL = config.MODELS_DIR / "model.previous.pkl"

# Retrain only after this many new completed matches have arrived. One match
# day is 10 matches, so a round of fixtures triggers a retrain at most once.
MIN_NEW_MATCHES = 10
# A new model may not be worse than the incumbent by more than this margin.
MAX_ACCURACY_REGRESSION = 0.01


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_state() -> dict:
    return read_json(STATE_JSON, default={}) or {}


def _save_state(state: dict) -> None:
    write_json_atomic(STATE_JSON, state)


def _push_step(state: dict, name: str, status: str, detail: str = "") -> None:
    state.setdefault("steps", []).append({
        "at": _now(), "step": name, "status": status, "detail": detail,
    })
    state["steps"] = state["steps"][-25:]
    log.info("[%s] %s %s", status.upper(), name, detail)


# --------------------------------------------------------------------------
# Individual stages
# --------------------------------------------------------------------------
def stage_refresh(state: dict, offline: bool = False) -> pd.DataFrame:
    if not offline:
        try:
            record = acquisition.refresh(force=False)
            _push_step(state, "refresh", "ok",
                       f"seasons={record.get('seasons')} fixtures={record.get('fixtures')}")
            state["last_data_refresh"] = record.get("finished_at")
        except Exception as exc:  # keep going with the cache
            _push_step(state, "refresh", "degraded", f"{type(exc).__name__}: {exc}")
    else:
        _push_step(state, "refresh", "skipped", "offline mode")
    matches = processing.load_matches(refresh=False)
    _push_step(state, "process", "ok", f"{len(matches)} matches")
    return matches


def stage_fixtures(state: dict, matches: pd.DataFrame) -> dict:
    """Synchronise upcoming fixtures from the cached provider payloads.

    The refresh stage has already downloaded the feeds when the pipeline is
    online, so this stage never performs its own network call. Predictions are
    read from the resulting store by the existing engine.
    """
    from src.data import fixture_sync

    result = fixture_sync.sync_fixtures(fetch=False, matches=matches)
    counts = result.get("counts", {})
    _push_step(
        state, "fixtures", "ok" if result.get("ok") else "degraded",
        f"new={counts.get('new', 0)} updated={counts.get('updated', 0)} "
        f"postponed={counts.get('postponed', 0)} cancelled={counts.get('cancelled', 0)} "
        f"rejected={counts.get('rejected', 0)} upcoming={result.get('upcoming_rows', 0)}",
    )
    state["fixture_sync"] = {
        key: result.get(key)
        for key in ("status", "sources", "counts", "store_rows", "upcoming_rows",
                    "store_changed")
    }
    return result


def stage_validate(state: dict, matches: pd.DataFrame) -> dict:
    integrity = validation.validate_matches(matches)
    quality = processing.data_quality_report(matches)
    state["data_quality"] = {
        "ok": integrity["ok"],
        "checks": integrity["checks"],
        "issues": integrity["issues"],
        "n_matches": integrity["n_matches"],
        "last_date": integrity["last_date"],
    }
    if not integrity["ok"]:
        _push_step(state, "validate", "failed", "; ".join(integrity["issues"]))
        raise RuntimeError(f"Data integrity failed: {integrity['issues']}")
    _push_step(state, "validate", "ok",
               f"odds={quality['odds_coverage']:.0%} shots={quality['shots_coverage']:.0%}")
    return integrity


def _model_state() -> dict:
    artifact_path = config.MODELS_DIR / "model.pkl"
    if not artifact_path.exists():
        return {"exists": False}
    try:
        artifact = joblib.load(artifact_path)
    except Exception as exc:
        return {"exists": True, "loadable": False, "error": str(exc)}
    return {
        "exists": True,
        "loadable": True,
        "data_cutoff": artifact.get("data_cutoff"),
        "trained_at": artifact.get("trained_at"),
        "test_accuracy": (artifact.get("metrics", {}).get("test") or {}).get("accuracy"),
        "accepted": artifact.get("accepted"),
    }


def should_retrain(matches: pd.DataFrame, mode: str = "auto") -> tuple[bool, str]:
    if mode == "never":
        return False, "retraining disabled"
    if mode == "always":
        return True, "forced by caller"
    state = _model_state()
    if not state.get("exists"):
        return True, "no model artifact"
    if not state.get("loadable"):
        return True, "existing artifact is unreadable"
    latest = str(pd.Timestamp(matches["date"].max()).date())
    if state.get("data_cutoff") != latest:
        new_rows = int((pd.to_datetime(matches["date"]) > pd.Timestamp(state["data_cutoff"])).sum()) \
            if state.get("data_cutoff") else len(matches)
        if new_rows <= 0:
            return True, "model cutoff differs from data cutoff"
        if new_rows >= MIN_NEW_MATCHES:
            return True, f"{new_rows} new completed matches since {state['data_cutoff']}"
        return False, f"only {new_rows} new matches (threshold {MIN_NEW_MATCHES})"
    return False, "model already trained on the latest data"


def stage_train(state: dict, force_reason: str) -> dict:
    from src.models import training

    previous = _model_state()
    if (config.MODELS_DIR / "model.pkl").exists():
        shutil.copy2(config.MODELS_DIR / "model.pkl", BACKUP_PKL)
        _push_step(state, "backup", "ok", f"saved {BACKUP_PKL.name}")

    try:
        report = training.train(seed_data=True)
    except Exception as exc:
        _push_step(state, "train", "failed", f"{type(exc).__name__}: {exc}")
        if BACKUP_PKL.exists():
            shutil.copy2(BACKUP_PKL, config.MODELS_DIR / "model.pkl")
            _push_step(state, "restore", "ok", "previous model restored after failure")
        raise

    current = _model_state()
    new_acc = current.get("test_accuracy") or 0.0
    old_acc = previous.get("test_accuracy")
    regression = (old_acc is not None and previous.get("accepted")
                  and new_acc < old_acc - MAX_ACCURACY_REGRESSION)

    if regression:
        # Keep the incumbent: promote nothing, but keep the experiment record.
        shutil.copy2(BACKUP_PKL, config.MODELS_DIR / "model.pkl")
        _push_step(state, "promote", "rejected",
                   f"new {new_acc:.4f} vs incumbent {old_acc:.4f} "
                   f"(margin {MAX_ACCURACY_REGRESSION})")
        report["promotion"] = {"promoted": False, "reason": "accuracy regression"}
        state["promotion"] = report["promotion"]
        return report

    _push_step(state, "train", "ok",
               f"{report['production_model']} acc={report['test_metrics']['accuracy']:.4f} "
               f"ll={report['test_metrics']['log_loss']:.4f}")
    _push_step(state, "promote", "ok", f"model {report['selected_model']}")
    report["promotion"] = {"promoted": True, "reason": force_reason}
    state["promotion"] = report["promotion"]
    return report


def stage_predict(state: dict) -> dict:
    engine = PredictionEngine()
    payload = engine.predict_upcoming(save=True)
    _push_step(state, "predict", "ok",
               f"{payload['n_fixtures']} fixtures, model {payload['model_version']}")
    # Immutable pre-match snapshots: persist/revise/lock the eligible
    # prediction for every fixture the payload covers (see
    # src/prediction/snapshots.py for the eligibility policy).
    store_rows = fixture_sync.load_store()
    snapshot_report = snapshots.reconcile(store_rows, payload["fixtures"])
    state["snapshots"] = {
        key: snapshot_report[key]
        for key in ("created", "revised", "unchanged", "locked", "missing",
                    "invalid", "post_kickoff", "status_updates", "total")
    }
    _push_step(state, "snapshots", "ok",
               "created={created} revised={revised} unchanged={unchanged} "
               "locked={locked} missing={missing} invalid={invalid}".format(**snapshot_report))
    return payload


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def run(offline: bool = False, retrain: str = "auto", predict_only: bool = False) -> dict:
    clock = time.monotonic()
    started = _now()
    state = _load_state()
    state.update({"last_run": started, "status": "running", "steps": []})
    _save_state(state)
    log.info("Pipeline start %s (offline=%s retrain=%s predict_only=%s)",
             started, offline, retrain, predict_only)

    try:
        matches = stage_refresh(state, offline=offline)
        stage_validate(state, matches)
        stage_fixtures(state, matches)

        if predict_only:
            _push_step(state, "train", "skipped", "predict-only mode")
            report = None
        else:
            needs, reason = should_retrain(matches, retrain)
            if needs:
                report = stage_train(state, reason)
            else:
                _push_step(state, "train", "skipped", reason)
                report = None

        payload = stage_predict(state)
        meta = _model_state()

        state.update({
            "status": "ok",
            # A successful run must not leave the previous failure visible.
            "error": None,
            "traceback": None,
            "last_success": _now(),
            "n_matches": int(len(matches)),
            "data_cutoff": str(pd.Timestamp(matches["date"].max()).date()),
            "n_fixtures": payload["n_fixtures"],
            "model_version": payload["model_version"],
            "model_type": payload.get("model_type"),
            "test_accuracy": meta.get("test_accuracy"),
            "retrained": bool(report is not None),
            "duration_seconds": round(time.monotonic() - clock, 1),
        })
        _save_state(state)
        log.info("Pipeline finished OK in %.1fs", state["duration_seconds"])
        return state
    except Exception as exc:
        state.update({
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-4000:],
            "last_success": state.get("last_success"),
        })
        _save_state(state)
        log.error("Pipeline failed: %s", exc)
        raise


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EPL Intelligence pipeline")
    parser.add_argument("--offline", action="store_true", help="use cached raw data only")
    parser.add_argument("--retrain", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--predict-only", action="store_true", help="skip training entirely")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        state = run(offline=args.offline, retrain=args.retrain, predict_only=args.predict_only)
    except Exception:
        return 1
    print(json.dumps({k: state.get(k) for k in (
        "status", "last_success", "n_matches", "data_cutoff", "n_fixtures",
        "model_version", "test_accuracy", "retrained", "duration_seconds",
    )}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_main())
