"""Page 4 - Manual Prediction."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config
from src.dashboard import components as ui
from src.dashboard import data
from src.dashboard.theme import PALETTE
from src.prediction.engine import PredictionError


def _memo() -> dict:
    if "manual_cache" not in st.session_state:
        st.session_state["manual_cache"] = {}
    return st.session_state["manual_cache"]


def render() -> None:
    ui.page_header(
        "Manual Prediction · What-if",
        "Manual Prediction",
        "Choose any two Premier League teams and a date. The engine rebuilds the feature "
        "state using only matches played before that date, so the inputs are always "
        "point-in-time correct.",
    )

    try:
        engine = data.engine()
        teams = engine.known_teams
    except Exception as exc:
        ui.empty_state("Model unavailable",
                       f"Could not load the production artifact: {exc}")
        return

    matches = data.matches()
    last_date = pd.Timestamp(matches["date"].max()) if not matches.empty else pd.Timestamp.today()
    default_date = (last_date + pd.Timedelta(days=7)).date()

    with st.container():
        c1, c2, c3 = st.columns([1, 1, 0.85], gap="small")
        with c1:
            home = st.selectbox("Home team", teams, index=teams.index("Arsenal")
                                if "Arsenal" in teams else 0,
                                format_func=config.display_name, key="manual_home")
        with c2:
            away = st.selectbox("Away team", teams, index=teams.index("Chelsea")
                                if "Chelsea" in teams else min(1, len(teams) - 1),
                                format_func=config.display_name, key="manual_away")
        with c3:
            date_value = st.date_input(
                "Match date", value=default_date,
                min_value=pd.Timestamp("2014-08-16").date(),
                max_value=(pd.Timestamp.today() + pd.Timedelta(days=400)).date(),
                key="manual_date",
            )

    same_team = home == away
    if same_team:
        st.error("A team cannot play itself. Choose two different clubs.")
        return

    target = pd.Timestamp(date_value)
    in_sample = target <= pd.Timestamp(engine.data_cutoff)
    if in_sample:
        ui.note(
            "This date is inside the training window. Features are still built only from "
            "earlier matches, but the <b>model weights</b> have seen this period, so treat "
            "the output as a retrospective simulation rather than a genuine pre-match call.",
            "warn",
        )

    button = st.button("Generate prediction", type="primary", key="manual_run")

    key = (home, away, str(target.date()), "15:00")
    cache = _memo()
    if button or key in cache:
        if key not in cache:
            with st.spinner("Rebuilding chronological state and scoring…"):
                try:
                    prediction = engine.predict_manual(home, away, date=target)
                    cache[key] = prediction.as_dict()
                except PredictionError as exc:
                    st.error(str(exc))
                    return
                except Exception as exc:  # keep the page alive
                    st.error(f"Prediction failed: {type(exc).__name__}: {exc}")
                    return
        result = cache[key]
    else:
        ui.note("Select two teams and press <b>Generate prediction</b>.")
        return

    st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)

    # ---------------- headline ----------------
    left, right = st.columns([1.35, 1], gap="medium")
    with left:
        st.markdown(
            f'<div class="panel" style="padding:1.1rem 1.2rem">'
            f'<div style="display:flex;justify-content:space-between;align-items:center">'
            f'<div><div class="kpi-label">Fixture</div>'
            f'<div style="font-size:1.15rem;font-weight:750;margin-top:.2rem">'
            f'{result["home_display"]} <span style="color:{PALETTE["muted"]};font-weight:400">vs</span> '
            f'{result["away_display"]}</div>'
            f'<div style="font-size:.74rem;color:{PALETTE["muted"]};margin-top:.2rem">'
            f'{pd.Timestamp(result["date"]).strftime("%A %d %B %Y")}</div></div>'
            f'<div style="text-align:right"><span class="pill '
            f'{ "pill-H" if result["predicted"] == "H" else "pill-D" if result["predicted"] == "D" else "pill-A"}">'
            f'{result["predicted_label"].upper()}</span>'
            f'<div style="font-size:.7rem;color:{PALETTE["muted"]};margin-top:.35rem">'
            f'{result["confidence"]:.1%} · {result["confidence_band"]} confidence</div></div>'
            f'</div></div>', unsafe_allow_html=True)

        st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)
        ui.big_probabilities(result["prob_home"], result["prob_draw"], result["prob_away"],
                             result["predicted"])
        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
        st.markdown(ui.prob_bar_html(result["prob_home"], result["prob_draw"],
                                     result["prob_away"], "26px"), unsafe_allow_html=True)

        st.markdown(
            '<div class="plabel" style="margin-top:.6rem;gap:.5rem;flex-wrap:wrap">'
            f'{ui.tag("Variant: " + result["model_variant"])}'
            f'{ui.tag("Mode: " + result["mode"], "warn" if result["mode"] == "retrospective" else "ok")}'
            f'{ui.tag("Model " + result["model_version"], "muted")}'
            f'{ui.tag("Scored " + result["predicted_at"][:16].replace("T", " ") + " UTC", "muted")}'
            "</div>", unsafe_allow_html=True)

    with right:
        st.plotly_chart(ui.probability_donut(result["prob_home"], result["prob_draw"],
                                             result["prob_away"]), key="manual_donut")

    # ---------------- context ----------------
    context = result.get("context") or {}
    ui.section("Supporting pre-match indicators")
    c1, c2, c3, c4 = st.columns(4, gap="small")
    tiles = [
        ("Home Elo", context.get("home_elo"), "{:.0f}"),
        ("Away Elo", context.get("away_elo"), "{:.0f}"),
        ("Home form (5)", context.get("home_form5_ppg"), "{:.2f} ppg"),
        ("Away form (5)", context.get("away_form5_ppg"), "{:.2f} ppg"),
    ]
    for column, (label, value, fmt) in zip((c1, c2, c3, c4), tiles):
        with column:
            text = fmt.format(value) if value is not None else "—"
            st.markdown(ui.kpi(label, text), unsafe_allow_html=True)

    with st.expander("Expected goals and head-to-head"):
        frame = pd.DataFrame([
            {"indicator": "Poisson home win", "value": context.get("poisson_home")},
            {"indicator": "Poisson draw", "value": context.get("poisson_draw")},
            {"indicator": "Poisson away win", "value": context.get("poisson_away")},
            {"indicator": "Expected goals — home", "value": context.get("expected_goals_home")},
            {"indicator": "Expected goals — away", "value": context.get("expected_goals_away")},
            {"indicator": "Head-to-head PPG (home view)", "value": context.get("h2h_ppg")},
            {"indicator": "Head-to-head draw rate", "value": context.get("h2h_draw_rate")},
            {"indicator": "Home rest days", "value": context.get("home_rest_days")},
            {"indicator": "Away rest days", "value": context.get("away_rest_days")},
        ])
        st.dataframe(frame, hide_index=True, height=310,
                     column_config={"value": st.column_config.NumberColumn(format="%.4f")})

    ui.note(
        f"<b>{result.get('context', {}).get('note', '')}</b>" if result.get("context", {}).get("note")
        else "Probabilities are model estimates, not guarantees. Football outcomes carry "
             "irreducible variance: even a perfectly calibrated 60% favourite loses four "
             "times in ten.")
