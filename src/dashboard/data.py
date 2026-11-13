"""Cached data access for the dashboard.

Every loader degrades gracefully: a missing artifact yields an empty frame and
a flag rather than an exception, so the app never shows a raw traceback.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src import config
from src.utilities.io import read_json

TTL = 120  # seconds; matches the automated update cadence tolerance


def _read_csv(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, parse_dates=parse_dates or [])
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=TTL, show_spinner=False)
def matches() -> pd.DataFrame:
    df = _read_csv(config.MATCHES_CSV, ["date"])
    if df.empty:
        return df
    return df.sort_values(["date", "time"], kind="mergesort").reset_index(drop=True)


@st.cache_data(ttl=TTL, show_spinner=False)
def fixtures() -> pd.DataFrame:
    return _read_csv(config.FIXTURES_CSV, ["date"])


@st.cache_data(ttl=TTL, show_spinner=False)
def predictions() -> dict:
    return read_json(config.DATA_DIR / "predictions.json", default={}) or {}


@st.cache_data(ttl=TTL, show_spinner=False)
def report() -> dict:
    return read_json(config.EVAL_DIR / "training_report.json", default={}) or {}


@st.cache_data(ttl=TTL, show_spinner=False)
def meta() -> dict:
    return read_json(config.METADATA_DIR / "model_meta.json", default={}) or {}


@st.cache_data(ttl=TTL, show_spinner=False)
def pipeline_state() -> dict:
    return read_json(config.DATA_DIR / "pipeline_state.json", default={}) or {}


@st.cache_data(ttl=TTL, show_spinner=False)
def standings() -> pd.DataFrame:
    return _read_csv(config.STANDINGS_CSV)


@st.cache_data(ttl=TTL, show_spinner=False)
def quality() -> dict:
    return read_json(config.DATA_QUALITY_JSON, default={}) or {}


@st.cache_data(ttl=TTL, show_spinner=False)
def leakage() -> dict:
    return read_json(config.QUALITY_DIR / "leakage_report.json", default={}) or {}


@st.cache_data(ttl=TTL, show_spinner=False)
def update_log() -> list[dict]:
    data = read_json(config.UPDATE_LOG, default=[]) or []
    return data if isinstance(data, list) else []


@st.cache_data(ttl=TTL, show_spinner=False)
def test_predictions() -> pd.DataFrame:
    df = _read_csv(config.EVAL_DIR / "test_predictions.csv", ["_date"])
    if not df.empty:
        df = df.rename(columns={
            "_date": "date", "_season": "season", "_home_team": "home_team",
            "_away_team": "away_team", "_home_goals": "home_goals",
            "_away_goals": "away_goals", "_result": "result",
            "pred_home": "prob_home", "pred_draw": "prob_draw", "pred_away": "prob_away",
        })
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


@st.cache_data(ttl=TTL, show_spinner=False)
def experiments() -> list[dict]:
    data = read_json(config.EXPERIMENT_DIR / "experiment_log.json", default=[]) or []
    return data if isinstance(data, list) else []


@st.cache_resource(show_spinner=False)
def engine():
    from src.prediction.engine import PredictionEngine
    return PredictionEngine()


def clear_cache() -> None:
    st.cache_data.clear()


# --------------------------------------------------------------------------
# Lightweight derivations
# --------------------------------------------------------------------------
def season_table(matches_df: pd.DataFrame) -> pd.DataFrame:
    if matches_df.empty:
        return pd.DataFrame()
    rows = []
    for season, group in matches_df.groupby("season"):
        rows.append({
            "season": season,
            "matches": int(len(group)),
            "home_wins": int((group["result"] == "H").sum()),
            "draws": int((group["result"] == "D").sum()),
            "away_wins": int((group["result"] == "A").sum()),
            "goals": int(group["total_goals"].sum()),
            "goals_per_match": round(float(group["total_goals"].mean()), 3),
        })
    return pd.DataFrame(rows).sort_values("season")


def latest_results(matches_df: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    if matches_df.empty:
        return pd.DataFrame()
    return matches_df.sort_values(["date", "time"]).tail(n).iloc[::-1].reset_index(drop=True)


def upcoming_payload() -> list[dict]:
    payload = predictions()
    rows = payload.get("fixtures") or []
    return rows if isinstance(rows, list) else []


def data_health() -> dict:
    """A single, honest status object used across pages."""
    state = pipeline_state()
    payload = predictions()
    report_data = report()
    matches_df = matches()
    last_result = matches_df["date"].max() if not matches_df.empty else None

    generated = payload.get("generated_at")
    age_hours = None
    if generated:
        try:
            age_hours = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(generated)).total_seconds() / 3600
        except Exception:
            age_hours = None

    if state.get("status") == "failed":
        status = "degraded"
    elif not report_data:
        status = "degraded"
    elif age_hours is not None and age_hours > 72:
        status = "stale"
    else:
        status = "ok"

    return {
        "status": status,
        "last_update": state.get("last_success") or payload.get("generated_at"),
        "last_result_date": str(pd.Timestamp(last_result).date()) if last_result is not None else None,
        "predictions_generated": generated,
        "prediction_age_hours": round(age_hours, 1) if age_hours is not None else None,
        "n_matches": int(len(matches_df)),
        "n_fixtures": int(payload.get("n_fixtures", 0)),
        "model_version": payload.get("model_version") or meta().get("model_version"),
        "model_trained_at": report_data.get("trained_at") or meta().get("trained_at"),
        "test_accuracy": (report_data.get("test_metrics") or {}).get("accuracy"),
    }
