"""Headless tests for the approved Streamlit match centre (root ``app.py``).

Routing is query-parameter based (?page=...), so each test sets a query
param and re-runs. The suite is deliberately small: every run boots the
real Streamlit runtime, so module-scoped reuse keeps it fast.
"""
from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from data import DATA

PAGES = ["overview", "week", "table", "results"]


def test_app_boots_without_exception(project_root):
    at = AppTest.from_file(str(project_root / "app.py"), default_timeout=300)
    at.run()
    assert not at.exception, f"initial render failed: {at.exception}"
    assert not [e for e in at.error], f"app reported: {[str(e.value) for e in at.error]}"


def test_payload_embeds_live_data(project_root):
    at = AppTest.from_file(str(project_root / "app.py"), default_timeout=300)
    at.run()
    assert not at.exception
    payload = at.main.html[0].value
    assert "EPL_DATA" in payload
    # The approved pages consume the latest synchronized fixtures: the
    # served payload must carry exactly the production prediction rows.
    assert payload.count('"pick"') >= len(DATA["fixtures"])
    assert 'href="?page=results"' in payload, "Results nav entry missing"


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exception(project_root, page):
    at = AppTest.from_file(str(project_root / "app.py"), default_timeout=300)
    at.query_params["page"] = page
    at.run()
    assert not at.exception, f"{page} raised: {at.exception}"
    assert not [e for e in at.error], f"{page} reported: {[str(e.value) for e in at.error]}"


def test_results_page_shows_matchweek_content(project_root):
    at = AppTest.from_file(str(project_root / "app.py"), default_timeout=300)
    at.query_params["page"] = "results"
    at.run()
    assert not at.exception
    payload = at.main.html[0].value
    assert "Prediction tracking" in payload
    assert "Data freshness" in payload


def test_matchweek_selection_on_results_page(project_root):
    at = AppTest.from_file(str(project_root / "app.py"), default_timeout=300)
    at.query_params["page"] = "results"
    at.query_params["mw"] = "10"
    at.run()
    assert not at.exception
    payload = at.main.html[0].value
    assert "Matchweek 10" in payload


def test_match_page_renders_first_fixture(project_root):
    at = AppTest.from_file(str(project_root / "app.py"), default_timeout=300)
    if not DATA["fixtures"]:
        pytest.skip("no fixtures synced yet")
    at.query_params["page"] = "match"
    at.query_params["id"] = DATA["fixtures"][0]["id"]
    at.run()
    assert not at.exception
    payload = at.main.html[0].value
    assert "The chances" in payload
