"""Focused Stage L regression tests for dashboard repairs."""
from __future__ import annotations

import json

import pytest


def test_match_lookup_accepts_id_and_fixture_id(project_root):
    import data
    import render

    data.refresh()
    fixtures = data.DATA.get("fixtures") or []
    assert fixtures, "expected live fixtures for lookup test"
    f = fixtures[0]
    h_id, _ = render.shell("match", f["id"])
    assert "Fixture not found" not in h_id
    assert f["home"] in h_id and f["away"] in h_id

    fixture_id = f.get("fixtureId")
    assert fixture_id, "fixture must expose fixtureId"
    h_fid, _ = render.shell("match", fixture_id)
    assert "Fixture not found" not in h_fid
    assert f["home"] in h_fid and f["away"] in h_fid

    h_bad, _ = render.shell("match", "NO_SUCH_FIXTURE:ABC:DEF")
    assert "Fixture not found" in h_bad


def test_match_route_with_fixture_id_apptest(project_root):
    from streamlit.testing.v1 import AppTest
    import data

    data.refresh()
    f = (data.DATA.get("fixtures") or [])[0]
    at = AppTest.from_file(str(project_root / "app.py"), default_timeout=300)
    at.query_params["page"] = "match"
    at.query_params["id"] = f["fixtureId"]
    at.run()
    assert not at.exception
    payload = at.main.html[0].value
    assert "Fixture not found" not in payload
    assert f["home"] in payload


def test_data_payload_includes_club_and_logos(project_root):
    import data

    data.refresh()
    assert "club" in data.DATA and isinstance(data.DATA["club"], dict)
    assert "logos" in data.DATA and isinstance(data.DATA["logos"], dict)
    assert "COV" in data.DATA["club"], "COV must be present for badge hydration"


def test_effective_freshness_prefers_authoritative_event_timestamps(tmp_path, monkeypatch):
    import data as data_module

    paths = {
        "RESULTS_JSON": tmp_path / "results.json",
        "PREDICTIONS_JSON": tmp_path / "predictions.json",
        "MATCHWEEKS_JSON": tmp_path / "matchweeks.json",
        "FRESHNESS_JSON": tmp_path / "freshness.json",
        "MATCHES_CSV": tmp_path / "matches.csv",
        "SYNC_STATE_JSON": tmp_path / "sync_state.json",
        "PIPELINE_STATE_JSON": tmp_path / "pipeline_state.json",
        "SNAPSHOTS_JSON": tmp_path / "snapshots.json",
    }
    for name, path in paths.items():
        monkeypatch.setattr(data_module, name, path)
    monkeypatch.setattr(data_module, "_STORES", tuple(paths.values()))
    monkeypatch.setattr(data_module, "_last_fingerprint", None)
    monkeypatch.setattr(data_module, "_last_built_at", 0.0)

    paths["RESULTS_JSON"].write_text(json.dumps({"updated_at": "2026-11-14T16:37:18+00:00", "results": {}}), encoding="utf-8")
    paths["PREDICTIONS_JSON"].write_text(json.dumps({"generated_at": "2026-11-16T17:11:32+00:00", "fixtures": []}), encoding="utf-8")
    paths["MATCHWEEKS_JSON"].write_text(json.dumps({"at": "2026-11-15T18:58:24+00:00", "seasons": {}}), encoding="utf-8")
    paths["FRESHNESS_JSON"].write_text(json.dumps({
        "fixtures": {"last_sync_at": "2026-11-14T20:09:46+00:00", "status": "ok"},
        "predictions": {"last_run": "2026-11-14T20:10:07+00:00", "snapshots_updated_at": "2026-11-14T20:10:06+00:00", "snapshots": 10},
        "results": {"last_sync_at": "2026-11-14T16:00:00+00:00", "completed": 1},
        "evaluation": {"last_run": None, "evaluated": 0},
    }), encoding="utf-8")
    paths["MATCHES_CSV"].write_text("season,date,home_team,away_team,FTHG,FTAG,FTR,HS,AS,HST,AST,HF,AF,HC,AC,HY,AY,HR,AR,B365H,B365D,B365A\n", encoding="utf-8")
    paths["SYNC_STATE_JSON"].write_text(json.dumps({"last_run": {"finished_at": "2026-11-16T17:09:39+00:00", "status": "ok", "ok": True}}), encoding="utf-8")
    paths["PIPELINE_STATE_JSON"].write_text(json.dumps({"steps": [{"step": "predict", "status": "ok", "at": "2026-11-16T17:09:57+00:00"}]}), encoding="utf-8")
    paths["SNAPSHOTS_JSON"].write_text(json.dumps({"updated_at": "2026-11-16T17:09:57+00:00", "snapshots": {"a": {}, "b": {}}}), encoding="utf-8")

    data_module.refresh(ttl_seconds=0.0)
    f = data_module.DATA["freshness"]
    assert f["fixtures"]["last_sync_at"] == "2026-11-16T17:09:39+00:00"
    assert f["predictions"]["last_run"] == "2026-11-16T17:11:32+00:00"
    assert f["predictions"]["snapshots_updated_at"] == "2026-11-16T17:09:57+00:00"
    assert f["results"]["last_sync_at"] == "2026-11-14T16:37:18+00:00"
    assert f["evaluation"]["last_run"] == "2026-11-15T18:58:24+00:00"
