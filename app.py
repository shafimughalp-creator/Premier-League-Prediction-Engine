"""EPL Intelligence - premium football analytics dashboard.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="EPL Intelligence",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "EPL Intelligence - leakage-audited Premier League match outcome modelling.",
    },
)

from src.dashboard import data, theme  # noqa: E402
from src.dashboard.pages import (  # noqa: E402
    data_centre,
    historical,
    manual,
    match_centre,
    model_performance,
    overview,
    predictions,
)

PAGES = {
    "Overview": ("◈", overview.render),
    "Predictions": ("◎", predictions.render),
    "Match Centre": ("⬢", match_centre.render),
    "Manual Prediction": ("✧", manual.render),
    "Historical Analysis": ("◷", historical.render),
    "Model Performance": ("▲", model_performance.render),
    "Data Centre": ("⛁", data_centre.render),
}


def sidebar() -> str:
    with st.sidebar:
        st.markdown(
            '<div class="side-brand">'
            '<div class="side-mark">EPL</div>'
            '<div><div class="side-title">EPL Intelligence</div>'
            '<div class="side-sub">Match Analytics</div></div>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)

        choice = st.radio(
            "Navigation",
            list(PAGES),
            format_func=lambda name: f"{PAGES[name][0]}   {name}",
            label_visibility="collapsed",
        )

        health = data.data_health()
        status = health.get("status")
        dot = {"ok": "", "stale": "stale"}.get(status, "bad")
        label = {"ok": "Live data current", "stale": "Snapshot stale"}.get(status, "Needs attention")
        accuracy = health.get("test_accuracy")
        st.markdown(
            '<div class="side-meta">'
            f'<div><span class="dot {dot}"></span>{label}</div>'
            f'<div style="margin-top:.4rem">Matches <b>{health.get("n_matches", 0):,}</b></div>'
            f'<div>Fixtures <b>{health.get("n_fixtures", 0)}</b></div>'
            f'<div>Test accuracy <b>{f"{accuracy:.1%}" if accuracy else "—"}</b></div>'
            f'<div style="margin-top:.4rem;font-size:.66rem;opacity:.8">'
            f'Cutoff {health.get("last_result_date") or "—"}</div>'
            f'<div style="font-size:.66rem;opacity:.8">'
            f'Model {health.get("model_version") or "—"}</div>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
        st.caption(
            "Probabilities are model estimates, not guarantees. Historical figures are "
            "out-of-sample unless labelled otherwise."
        )
    return choice


def main() -> None:
    theme.inject(st)
    page = sidebar()
    renderer = PAGES[page][1]
    try:
        renderer()
    except Exception as exc:  # never expose a raw traceback to the user
        st.error(f"Something went wrong while rendering **{page}**: {type(exc).__name__}: {exc}")
        st.caption("The rest of the dashboard remains available. Check the Data Centre for "
                   "pipeline status, then rerun the affected stage.")
        with st.expander("Technical detail"):
            import traceback
            st.code(traceback.format_exc()[-3000:])

    st.markdown(
        '<div style="margin-top:3rem;padding-top:1rem;border-top:1px solid #1F2C4A;'
        'display:flex;justify-content:space-between;color:#8C99B4;font-size:.7rem;'
        'flex-wrap:wrap;gap:.5rem">'
        "<span>EPL Intelligence · three-outcome match modelling</span>"
        "<span>Data: football-data.co.uk · openfootball</span>"
        "<span>Chronological validation · no random splits</span>"
        "</div>",
        unsafe_allow_html=True,
    )


main()
