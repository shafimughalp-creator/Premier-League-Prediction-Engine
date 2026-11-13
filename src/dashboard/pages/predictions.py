"""Page 2 - Match Predictions."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config
from src.dashboard import components as ui
from src.dashboard import data


def render() -> None:
    payload = data.predictions()
    rows = data.upcoming_payload()

    ui.page_header(
        "Predictions · Upcoming fixtures",
        "Match Predictions",
        "Calibrated home / draw / away probabilities for the next Premier League round. "
        "Odds-free fixtures are scored with the dedicated no-odds model rather than "
        "imputing a betting market that does not exist yet.",
    )

    if not rows:
        ui.empty_state("No predictions available",
                       "Run the pipeline from the Data Centre to refresh fixtures and score them.")
        return

    pipeline = payload.get("pipeline") or {}
    st.markdown(
        '<div class="plabel" style="gap:1rem;flex-wrap:wrap">'
        f'{ui.tag("Generated " + str(payload.get("generated_at", ""))[:16].replace("T", " ") + " UTC")}'
        f'{ui.tag("Model " + str(payload.get("model_version", "—")))}'
        f'{ui.tag("Poisson α " + str(pipeline.get("poisson_alpha", "—")), "muted")}'
        f'{ui.tag("Market β " + str(pipeline.get("market_beta", "—")), "muted")}'
        f'{ui.tag("Temperature " + str(round(float(pipeline.get("temperature", 1.0)), 2)), "muted")}'
        "</div>",
        unsafe_allow_html=True,
    )

    # ---------------- filters ----------------
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    weeks = sorted(frame["matchweek"].dropna().unique().tolist())
    teams = sorted(set(frame["home_display"]) | set(frame["away_display"]))

    with st.container():
        c1, c2, c3 = st.columns([1, 1, 1.2], gap="small")
        with c1:
            week_options = ["All matchweeks"] + [f"Matchweek {int(w)}" for w in weeks]
            week = st.selectbox("Matchweek", week_options, key="pred_week")
        with c2:
            team = st.selectbox("Team", ["All teams"] + teams, key="pred_team")
        with c3:
            sort_by = st.selectbox(
                "Sort by", ["Kick-off (earliest)", "Highest confidence", "Biggest mismatch",
                            "Closest match"],
                key="pred_sort")

    view = frame.copy()
    if week != "All matchweeks" and weeks:
        view = view[view["matchweek"] == float(week.split()[-1])]
    if team != "All teams":
        view = view[(view["home_display"] == team) | (view["away_display"] == team)]

    if sort_by == "Highest confidence":
        view = view.sort_values("confidence", ascending=False)
    elif sort_by == "Biggest mismatch":
        view = view.assign(edge=(view["prob_home"] - view["prob_away"]).abs()) \
                   .sort_values("edge", ascending=False)
    elif sort_by == "Closest match":
        view = view.assign(edge=(view["prob_home"] - view["prob_away"]).abs()) \
                   .sort_values("edge")
    else:
        view = view.sort_values(["date", "time"])

    if view.empty:
        ui.empty_state("No fixtures match those filters",
                       "Clear the filters to see the full round.")
        return

    st.caption(f"{len(view)} fixture{'s' if len(view) != 1 else ''} shown")
    ui.render_match_cards(view.to_dict("records"), columns=2, show_market=True)

    # ---------------- full probability table ----------------
    ui.section("Full probability table")
    table = view[[
        "date", "time", "home_display", "away_display", "prob_home", "prob_draw",
        "prob_away", "predicted_label", "confidence", "confidence_band", "model_variant",
    ]].copy()
    table["date"] = table["date"].dt.strftime("%Y-%m-%d")
    st.dataframe(
        table, hide_index=True,
        column_config={
            "date": st.column_config.TextColumn("Date", width="small"),
            "time": st.column_config.TextColumn("KO", width="small"),
            "home_display": st.column_config.TextColumn("Home", width="medium"),
            "away_display": st.column_config.TextColumn("Away", width="medium"),
            "prob_home": st.column_config.ProgressColumn(
                "Home", min_value=0.0, max_value=1.0, format="%.1f%%"),
            "prob_draw": st.column_config.ProgressColumn(
                "Draw", min_value=0.0, max_value=1.0, format="%.1f%%"),
            "prob_away": st.column_config.ProgressColumn(
                "Away", min_value=0.0, max_value=1.0, format="%.1f%%"),
            "predicted_label": st.column_config.TextColumn("Call", width="small"),
            "confidence": st.column_config.NumberColumn("Conf.", format="%.1f%%", width="small"),
            "confidence_band": st.column_config.TextColumn("Band", width="small"),
            "model_variant": st.column_config.TextColumn("Variant", width="small"),
        },
    )

    st.caption(
        "Progress bars show each probability on a 0–100% scale. The three probabilities for "
        "every fixture sum to 100%."
    )

    ui.note(
        "Confidence is the maximum calibrated probability, not a promise. A 55% home "
        "probability means the model expects the home side to win roughly 55 times in 100 "
        "comparable matches."
    )
