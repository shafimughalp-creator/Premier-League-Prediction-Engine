"""End-to-end pipeline behaviour."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from src import config
from src.data import processing, validation
from src.pipeline import MIN_NEW_MATCHES, run, should_retrain
from src.utilities.io import read_json


def test_fixtures_are_valid():
    matches = processing.load_processed()
    fixtures = processing.load_fixtures(matches=matches)
    report = validation.validate_fixtures(fixtures, matches)
    assert report["ok"], report["issues"]
    assert report["n"] >= 1
    assert not fixtures.duplicated(["date", "home_team", "away_team"]).any()
    assert (fixtures["date"] > matches["date"].max()).all()


def test_fixture_teams_are_known():
    matches = processing.load_processed()
    fixtures = processing.load_fixtures(matches=matches)
    known = set(matches["home_team"]) | set(matches["away_team"])
    assert set(fixtures["home_team"]) | set(fixtures["away_team"]) <= known


def test_retrain_policy_is_conservative():
    matches = processing.load_processed()
    needed, reason = should_retrain(matches, "never")
    assert needed is False
    assert "disabled" in reason
    assert MIN_NEW_MATCHES >= 1


def test_pipeline_predict_only_run():
    state = run(offline=True, retrain="never", predict_only=True)
    assert state["status"] == "ok"
    assert state["n_matches"] > 4000
    assert state["n_fixtures"] >= 1
    assert state["model_version"].startswith("epl-")
    assert state["duration_seconds"] > 0
    steps = {step["step"]: step["status"] for step in state["steps"]}
    assert steps["predict"] == "ok"
    assert steps["train"] == "skipped"


def test_pipeline_state_is_persisted():
    saved = read_json(config.DATA_DIR / "pipeline_state.json", default={})
    assert saved["status"] == "ok"
    assert saved["last_success"]
    assert saved["data_cutoff"]


def test_predictions_snapshot_matches_state():
    predictions = read_json(config.DATA_DIR / "predictions.json", default={})
    state = read_json(config.DATA_DIR / "pipeline_state.json", default={})
    assert predictions["model_version"] == state["model_version"]
    assert predictions["n_fixtures"] == state["n_fixtures"]
    assert predictions["data_cutoff"] == state["data_cutoff"]


def test_leakage_report_is_recorded():
    report = read_json(config.QUALITY_DIR / "leakage_report.json", default={})
    if not report:
        pytest.skip("leakage report not generated yet")
    assert report["passed"] is True
    assert report["own_result"]["passed"] is True
    assert report["future_independence"]["passed"] is True


def test_model_metadata_is_consistent():
    meta = read_json(config.METADATA_DIR / "model_meta.json", default={})
    predictions = read_json(config.DATA_DIR / "predictions.json", default={})
    assert meta
    assert meta["data_cutoff"] == predictions["data_cutoff"]
    assert 0.45 <= meta["test_accuracy"] <= 0.65
    assert meta["test_log_loss"] > 0.9


def test_predictions_are_stable_across_repeated_runs():
    from src.prediction.engine import PredictionEngine
    engine = PredictionEngine()
    first = engine.predict_upcoming(save=False)
    second = engine.predict_upcoming(save=False)
    assert first["n_fixtures"] == second["n_fixtures"]
    for left, right in zip(first["fixtures"], second["fixtures"]):
        assert left["match_id"] == right["match_id"]
        assert left["prob_home"] == pytest.approx(right["prob_home"], abs=1e-9)
        assert left["predicted"] == right["predicted"]


def test_refresh_failure_falls_back_to_cached_data(monkeypatch):
    from src import pipeline

    def _boom(**kwargs):
        raise RuntimeError("simulated network failure")

    monkeypatch.setattr(pipeline.acquisition, "refresh", _boom)
    state = pipeline.run(offline=False, retrain="never", predict_only=True)
    assert state["status"] == "ok"
    steps = {step["step"]: step["status"] for step in state["steps"]}
    assert steps["refresh"] == "degraded"
    assert steps["process"] == "ok"
    assert state["n_matches"] > 4000


def test_training_failure_keeps_the_incumbent_artifact(monkeypatch):
    from src import pipeline

    model_path = config.MODELS_DIR / "model.pkl"
    before = model_path.read_bytes()

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated training failure")

    monkeypatch.setattr("src.models.training.train", _boom)
    monkeypatch.setattr(pipeline, "should_retrain", lambda *a, **k: (True, "test"))
    with pytest.raises(RuntimeError):
        pipeline.run(offline=True, retrain="always", predict_only=False)

    assert model_path.read_bytes() == before
    failed = read_json(config.DATA_DIR / "pipeline_state.json", default={})
    assert failed["status"] == "failed"
    assert "simulated training failure" in (failed.get("error") or "")

    # Leave the pipeline state healthy for the remaining tests.
    pipeline.run(offline=True, retrain="never", predict_only=True)

