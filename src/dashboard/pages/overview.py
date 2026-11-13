"""Page 1 - Overview."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config
from src.dashboard import components as ui
from src.dashboard import data
from src.dashboard.theme import PALETTE, plotly_layout


def _freshness(state: dict) -> str:
    status = state.get("status")
    if status == "ok":
        return '<span class="dot"></span>Systems nominal'
    if status == "stale":
        return '<span class="dot stale"></span>Prediction snapshot is stale'
    return '<span class="dot bad"></span>Needs attention'


def render() -> None:
    health = data.data_health()
    report = data.report()
    matches = data.matches()
    payload = data.predictions()
    upcoming = data.upcoming_payload()
    test_metrics = report.get("test_metrics") or {}
    market = report.get("market_baseline") or {}
    no_odds = report.get("no_odds_metrics") or {}

    ui.page_header(
        "EPL Intelligence · Overview",
        "Premier League Match Intelligence",
        "A leakage-audited three-outcome model (home / draw / away) with chronological "
        "validation, calibration and a continuously refreshed fixture pipeline.",
    )

    accuracy = test_metrics.get("accuracy")
    edge = (accuracy - market.get("accuracy", 0)) * 100 if accuracy and market else 0.0
    ui.kpi_row([
        {"label": "Matches analysed", "value": f"{health['n_matches']:,}",
         "delta": f"{matches['season'].nunique() if not matches.empty else 0} seasons · "
                  f"{matches['date'].min().date() if not matches.empty else '—'} → "
                  f"{matches['date'].max().date() if not matches.empty else '—'}"},
        {"label": "Out-of-sample accuracy", "value": f"{accuracy:.1%}" if accuracy else "—",
         "delta": f"{test_metrics.get('correct', 0)} correct / "
                  f"{test_metrics.get('incorrect', 0)} incorrect",
         "direction": "up" if accuracy and accuracy >= 0.5 else ""},
        {"label": "Upcoming fixtures", "value": f"{health['n_fixtures']}",
         "delta": "Next matchweek predictions ready" if health["n_fixtures"] else "Awaiting fixtures"},
        {"label": "Last data update", "value": (health.get("last_update") or "—")[:10],
         "delta": _freshness(health)},
    ])

    st.markdown(
        '<div class="plabel" style="margin-top:.9rem;gap:1rem;flex-wrap:wrap">'
        f'{ui.tag("Model " + str(health.get("model_version") or "n/a"))}'
        f'{ui.tag("Data cutoff " + str(health.get("last_result_date") or "n/a"))}'
        f'{ui.tag("Accuracy edge vs market " + (f"{edge:+.2f} pp" if accuracy and market else "n/a"), "ok" if edge > 0 else "muted")}'
        f'{ui.tag("Draw recall " + f"{test_metrics.get('per_class', {}).get('Draw', {}).get('recall', 0):.1%}", "warn")}'
        "</div>",
        unsafe_allow_html=True,
    )

    # ---------------- upcoming ----------------
    ui.section("Next fixtures")
    if upcoming:
        ui.render_match_cards(upcoming[:4], columns=2, show_market=True)
        if len(upcoming) > 4:
            st.caption(f"+ {len(upcoming) - 4} more fixtures on the Predictions page")
    else:
        ui.empty_state("No upcoming fixtures resolved",
                       "The fixture feed returned nothing new. Run the pipeline from the Data Centre.")

    # ---------------- latest results ----------------
    left, right = st.columns([1.35, 1], gap="medium")
    with left:
        ui.section("Latest results")
        results = data.latest_results(matches, 7)
        if results.empty:
            ui.empty_state("No results", "The processed match table is empty.")
        else:
            for _, row in results.iterrows():
                outcome = row["result"]
                cls = {"H": "pill-H", "D": "pill-D", "A": "pill-A"}[outcome]
                st.markdown(
                    f'<div class="panel" style="padding:.55rem .8rem;margin-bottom:.4rem">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;gap:.6rem">'
                    f'<span style="font-size:.72rem;color:{PALETTE["muted"]};min-width:64px">'
                    f'{pd.Timestamp(row["date"]).strftime("%d %b")}</span>'
                    f'<span style="flex:1;text-align:right;font-size:.83rem;font-weight:600">'
                    f'{config.display_name(row["home_team"])}</span>'
                    f'<span style="padding:.05rem .45rem;border-radius:6px;background:#0A111F;'
                    f'font-weight:800;font-size:.82rem;margin:0 .35rem">'
                    f'{int(row["home_goals"])} – {int(row["away_goals"])}</span>'
                    f'<span style="flex:1;font-size:.83rem;font-weight:600">'
                    f'{config.display_name(row["away_team"])}</span>'
                    f'<span class="pill {cls}" style="min-width:38px;text-align:center">'
                    f'{outcome}</span>'
                    f"</div></div>",
                    unsafe_allow_html=True,
                )

    with right:
        ui.section("Model vs baselines")
        rows = [
            ("This model", accuracy, PALETTE["accent"]),
            ("Market consensus", market.get("accuracy"), PALETTE["accent_2"]),
            ("No-odds model", no_odds.get("accuracy"), PALETTE["good"]),
        ]
        rows = [(label, value, colour) for label, value, colour in rows if value]
        if rows:
            fig = ui.metric_bar([r[0] for r in rows], [r[1] for r in rows],
                                [r[2] for r in rows], height=190)
            st.plotly_chart(fig, key="overview_baselines")
        if rows:
            st.caption(
                f"Test window: {report.get('split', {}).get('test', {}).get('start', '—')} → "
                f"{report.get('split', {}).get('test', {}).get('end', '—')} · "
                f"{test_metrics.get('n', 0)} unseen matches"
            )

    # ---------------- season trend + standings ----------------
    left, right = st.columns([1.2, 1], gap="medium")
    with left:
        ui.section("Season-by-season accuracy")
        season_rows = report.get("season_breakdown") or []
        if season_rows:
            st.plotly_chart(ui.season_trend_figure(season_rows), key="overview_trend")
        else:
            ui.empty_state("No season breakdown", "Train a model to populate this panel.")
    with right:
        ui.section("Current standings")
        table = data.standings()
        if table.empty:
            ui.empty_state("Standings unavailable", "The standings table has not been built yet.")
        else:
            view = table.head(8)[["position", "team", "played", "won", "drawn", "lost",
                                  "goal_diff", "points"]].copy()
            view["team"] = view["team"].map(config.display_name)
            st.dataframe(
                view, hide_index=True, height=290,
                column_config={
                    "position": st.column_config.NumberColumn("#", width="small"),
                    "team": st.column_config.TextColumn("Team", width="medium"),
                    "played": st.column_config.NumberColumn("P", width="small"),
                    "won": st.column_config.NumberColumn("W", width="small"),
                    "drawn": st.column_config.NumberColumn("D", width="small"),
                    "lost": st.column_config.NumberColumn("L", width="small"),
                    "goal_diff": st.column_config.NumberColumn("GD", width="small"),
                    "points": st.column_config.NumberColumn("Pts", width="small"),
                },
            )
            st.caption(f"{table.iloc[0]['season']} · computed from the processed match table")

    # ---------------- prediction distribution ----------------
    counts = payload.get("prediction_counts") or {}
    if counts:
        ui.section("How the model is calling the next round")
        cols = st.columns(4, gap="small")
        total = max(sum(counts.values()), 1)
        for column, (label, code, colour) in zip(
            cols, [("Home wins", "H", PALETTE["home"]), ("Draws", "D", PALETTE["draw"]),
                   ("Away wins", "A", PALETTE["away"]), ("Fixtures", "", PALETTE["accent_2"])]
        ):
            with column:
                value = total if code == "" else counts.get(code, 0)
                share = "" if code == "" else f"{counts.get(code, 0) / total:.0%} of fixtures"
                st.markdown(
                    f'<div class="kpi"><div class="kpi-label">{label}</div>'
                    f'<div class="kpi-value" style="color:{colour}">{value}</div>'
                    f'<div class="kpi-delta">{share}</div></div>',
                    unsafe_allow_html=True,
                )
        st.caption(
            "Outcome selection is a plain argmax over the calibrated probabilities - no "
            "artificial draw forcing. Draw accuracy is reported separately on Model Performance."
        )
