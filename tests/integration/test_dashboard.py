"""Headless dashboard tests using Streamlit's AppTest harness.

Every page is rendered for real, including Plotly figures and dataframes. The
manual-prediction workflow is exercised end to end (select teams, click the
button, read the probabilities).
"""
from __future__ import annotations

import pytest

from streamlit.testing.v1 import AppTest

PAGES = [
    "Overview",
    "Predictions",
    "Match Centre",
    "Manual Prediction",
    "Historical Analysis",
    "Model Performance",
    "Data Centre",
]


@pytest.fixture(scope="module")
def app_path(project_root):
    return str(project_root / "app.py")


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exception(app_path, page):
    at = AppTest.from_file(app_path, default_timeout=240)
    at.run()
    assert not at.exception, f"initial render failed: {at.exception}"
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, f"{page} raised: {at.exception}"
    assert len(at.markdown) > 5, f"{page} produced almost no output"
    assert not [e for e in at.error], f"{page} reported: {[str(e.value) for e in at.error]}"


def test_navigation_has_all_seven_pages(app_path):
    at = AppTest.from_file(app_path, default_timeout=240)
    at.run()
    options = list(at.sidebar.radio[0].options)
    assert len(options) == 7
    for page in PAGES:
        assert any(page in option for option in options)


def test_sidebar_shows_data_freshness(app_path):
    at = AppTest.from_file(app_path, default_timeout=240)
    at.run()
    joined = " ".join(str(block.value) for block in at.sidebar.markdown)
    assert "Matches" in joined
    assert "Model" in joined


def test_manual_prediction_workflow(app_path):
    at = AppTest.from_file(app_path, default_timeout=300)
    at.run()
    at.sidebar.radio[0].set_value("Manual Prediction").run()
    assert not at.exception

    selects = at.selectbox
    assert len(selects) >= 2
    selects[0].set_value("Arsenal")
    selects[1].set_value("Chelsea")
    at.run()
    assert not at.exception

    buttons = at.button
    assert buttons, "Generate prediction button missing"
    buttons[0].click().run()

    assert not at.exception, at.exception
    assert not [e for e in at.error], [str(e.value) for e in at.error]
    rendered = " ".join(str(block.value) for block in at.markdown)
    assert "Arsenal" in rendered and "Chelsea" in rendered
    assert "confidence" in rendered.lower()
    # Probabilities are rendered as percentages inside the big-probability block.
    assert "%" in rendered


def test_manual_rejects_same_team(app_path):
    at = AppTest.from_file(app_path, default_timeout=300)
    at.run()
    at.sidebar.radio[0].set_value("Manual Prediction").run()
    at.selectbox[0].set_value("Arsenal")
    at.selectbox[1].set_value("Arsenal")
    at.run()
    assert not at.exception
    assert [e for e in at.error], "identical teams should be rejected"


def test_predictions_page_filters(app_path):
    at = AppTest.from_file(app_path, default_timeout=240)
    at.run()
    at.sidebar.radio[0].set_value("Predictions").run()
    assert not at.exception
    assert len(at.selectbox) == 3
    # Filter to a single team; the page must still render.
    teams = list(at.selectbox[1].options)
    assert len(teams) > 2
    at.selectbox[1].set_value(teams[1])
    at.run()
    assert not at.exception
    assert not [e for e in at.error]


def test_data_centre_has_refresh_control(app_path):
    at = AppTest.from_file(app_path, default_timeout=240)
    at.run()
    at.sidebar.radio[0].set_value("Data Centre").run()
    assert not at.exception
    labels = [button.label for button in at.button]
    assert any("latest data" in label.lower() for label in labels)


def test_model_performance_shows_research_evidence(app_path):
    at = AppTest.from_file(app_path, default_timeout=300)
    at.run()
    at.sidebar.radio[0].set_value("Model Performance").run()
    assert not at.exception
    assert not [e for e in at.error], [str(e.value) for e in at.error]
    text = " ".join(str(block.value) for block in at.markdown)
    for heading in (
        "Draw prediction analysis",
        "Statistical comparison with the bookmaker consensus",
        "Calibration, sharpness and reliability",
        "Season-by-season historical performance",
        "Walk-forward validation",
        "Challenger experiments",
        "Model status and limitations",
    ):
        assert heading in text, f"missing section: {heading}"
    assert len(at.dataframe) >= 6, "research tables did not render"

