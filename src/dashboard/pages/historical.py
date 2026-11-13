"""Page 5 - Historical Analysis (genuine out-of-sample predictions)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config
from src.dashboard import components as ui
from src.dashboard import data
from src.dashboard.theme import PALETTE


def _pct(frame: pd.DataFrame, columns: list[str], decimals: int = 1) -> pd.DataFrame:
    """Scale fraction columns to 0-100 so printf formats render as percentages."""
    frame = frame.copy()
    for column in columns:
        if column in frame.columns:
            frame[column] = (pd.to_numeric(frame[column], errors="coerce") * 100).round(decimals)
    return frame


def render() -> None:
    frame = data.test_predictions()
    report = data.report()

    ui.page_header(
        "Historical Analysis · Held-out seasons",
        "Historical Analysis",
        "Every prediction below was produced by a model that never saw these seasons during "
        "training, model selection, calibration or ensembling. This is the honest "
        "out-of-sample record.",
    )

    if frame.empty:
        ui.empty_state("No historical predictions",
                       "Run the training pipeline to generate the held-out prediction file.")
        return

    split = report.get("split", {}).get("test", {})
    st.markdown(
        '<div class="plabel" style="gap:.5rem;flex-wrap:wrap">'
        f'{ui.tag("Out-of-sample window", "ok")}'
        f'{ui.tag("Test seasons 2024/25 + 2025/26")}'
        f'{ui.tag(f"{split.get('start', '—')} → {split.get('end', '—')}")}'
        f'{ui.tag("Model " + str(report.get("production_model", "—")), "muted")}'
        "</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)

    # ---------------- filters ----------------
    seasons = ["All seasons"] + sorted(frame["season"].dropna().unique().tolist())
    c1, c2, c3 = st.columns([1, 1, 1], gap="small")
    with c1:
        season = st.selectbox("Season", seasons, key="hist_season")
    with c2:
        outcome = st.selectbox("Actual outcome", ["All", "Home win", "Draw", "Away win"],
                               key="hist_outcome")
    with c3:
        correctness = st.selectbox("Show", ["All predictions", "Correct only", "Incorrect only"],
                                   key="hist_correct")

    view = frame.copy()
    if season != "All seasons":
        view = view[view["season"] == season]
    mapping = {"Home win": "H", "Draw": "D", "Away win": "A"}
    if outcome != "All":
        view = view[view["result"] == mapping[outcome]]
    if correctness == "Correct only":
        view = view[view["correct"] == 1]
    elif correctness == "Incorrect only":
        view = view[view["correct"] == 0]

    accuracy = float(frame["correct"].mean()) if len(frame) else 0.0

    def _sub_accuracy(code: str) -> float:
        subset = frame.loc[frame["result"] == code, "correct"]
        return float(subset.mean()) if len(subset) else 0.0

    ui.kpi_row([
        {"label": "Matches evaluated", "value": f"{len(frame)}",
         "delta": "Two complete untouched seasons"},
        {"label": "Accuracy", "value": f"{accuracy:.1%}",
         "delta": f"{int(frame['correct'].sum())} correct / "
                  f"{int(len(frame) - frame['correct'].sum())} incorrect",
         "direction": "up" if accuracy >= 0.5 else "down"},
        {"label": "Home win accuracy", "value": f"{_sub_accuracy('H'):.1%}",
         "delta": f"{int((frame['result'] == 'H').sum())} home wins in window"},
        {"label": "Draw accuracy", "value": f"{_sub_accuracy('D'):.1%}",
         "delta": f"{int(frame['predicted'].eq('D').sum())} draws predicted"},
    ])

    # ---------------- trend + distribution ----------------
    left, right = st.columns([1.4, 1], gap="medium")
    with left:
        ui.section("Rolling accuracy")
        st.plotly_chart(ui.rolling_accuracy_figure(view if not view.empty else frame),
                        key="hist_rolling")
        st.caption("Rolling 40-match accuracy across the order the matches were played.")
    with right:
        ui.section("Actual vs predicted outcomes")
        actual = frame["result"].value_counts()
        predicted = frame["predicted"].value_counts()
        comparison = pd.DataFrame({
            "outcome": ["Home win", "Draw", "Away win"],
            "actual": [int(actual.get("H", 0)), int(actual.get("D", 0)), int(actual.get("A", 0))],
            "predicted": [int(predicted.get("H", 0)), int(predicted.get("D", 0)),
                          int(predicted.get("A", 0))],
        })
        st.dataframe(comparison, hide_index=True, height=160,
                     column_config={
                         "actual": st.column_config.NumberColumn("Actual", width="small"),
                         "predicted": st.column_config.NumberColumn("Predicted", width="small"),
                     })
        st.caption(
            "The model is deliberately conservative about draws: draws are the hardest "
            "class and forcing them would trade accuracy for recall without improving the "
            "probability estimates."
        )

    # ---------------- confusion + per season ----------------
    left, right = st.columns([1, 1], gap="medium")
    with left:
        ui.section("Confusion matrix")
        matrix = report.get("test_metrics", {}).get("confusion_matrix")
        if matrix:
            st.plotly_chart(ui.confusion_figure(matrix), key="hist_confusion")
    with right:
        ui.section("Season breakdown")
        rows = report.get("season_breakdown") or []
        if rows:
            table = pd.DataFrame(rows).rename(columns={
                "season": "Season", "n": "Matches", "accuracy": "Accuracy",
                "log_loss": "Log loss", "macro_f1": "Macro F1",
                "draw_recall": "Draw recall", "draw_predicted": "Draws called",
            })
            table = _pct(table, ["Accuracy", "Draw recall"])
            st.dataframe(
                table, hide_index=True, height=200,
                column_config={
                    "Accuracy": st.column_config.NumberColumn(format="%.1f%%"),
                    "Log loss": st.column_config.NumberColumn(format="%.4f"),
                    "Macro F1": st.column_config.NumberColumn(format="%.3f"),
                    "Draw recall": st.column_config.NumberColumn(format="%.1f%%"),
                })
        else:
            ui.empty_state("No breakdown", "Train the model to populate this table.")

    # ---------------- biggest surprises ----------------
    ui.section("Highest-confidence misses")
    wrong = frame[frame["correct"] == 0].sort_values("confidence", ascending=False).head(8)
    if wrong.empty:
        st.caption("The model was never wrong in this window (statistically implausible).")
    else:
        for _, row in wrong.iterrows():
            st.markdown(
                f'<div class="panel" style="padding:.55rem .85rem;margin-bottom:.4rem">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;gap:.7rem;flex-wrap:wrap">'
                f'<span style="font-size:.73rem;color:{PALETTE["muted"]};min-width:120px">'
                f'{pd.Timestamp(row["date"]).strftime("%d %b %Y")}</span>'
                f'<span style="flex:1;font-size:.84rem;font-weight:600">'
                f'{config.display_name(row["home_team"])} {int(row["home_goals"])}–'
                f'{int(row["away_goals"])} {config.display_name(row["away_team"])}</span>'
                f'<span class="tag muted">called {row["predicted"]} '
                f'({row["confidence"]:.0%})</span>'
                f'<span class="tag bad">actual {row["result"]}</span>'
                f'</div></div>', unsafe_allow_html=True)
        st.caption("High confidence does not mean certainty - this list exists to make that explicit.")

    # ---------------- detailed table ----------------
    ui.section(f"Prediction ledger · {len(view)} matches")
    if view.empty:
        ui.empty_state("No matches match those filters", "Loosen the filters above.")
        return
    table = view.sort_values("date", ascending=False).copy()
    table["match"] = (table["home_team"].map(config.display_name) + " vs "
                      + table["away_team"].map(config.display_name))
    table["score"] = (table["home_goals"].astype(int).astype(str) + "–"
                      + table["away_goals"].astype(int).astype(str))
    table["result_tag"] = table["correct"].map({1: "✓", 0: "✗"})
    display = table[["date", "season", "match", "score", "predicted", "result",
                     "prob_home", "prob_draw", "prob_away", "confidence", "result_tag"]]
    display = _pct(display, ["prob_home", "prob_draw", "prob_away", "confidence"])
    st.dataframe(
        display, hide_index=True, height=420,
        column_config={
            "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD", width="small"),
            "season": st.column_config.TextColumn("Season", width="small"),
            "match": st.column_config.TextColumn("Fixture", width="large"),
            "score": st.column_config.TextColumn("Score", width="small"),
            "predicted": st.column_config.TextColumn("Call", width="small"),
            "result": st.column_config.TextColumn("Actual", width="small"),
            "prob_home": st.column_config.NumberColumn("P(H)", format="%.1f%%", width="small"),
            "prob_draw": st.column_config.NumberColumn("P(D)", format="%.1f%%", width="small"),
            "prob_away": st.column_config.NumberColumn("P(A)", format="%.1f%%", width="small"),
            "confidence": st.column_config.NumberColumn("Conf.", format="%.1f%%", width="small"),
            "result_tag": st.column_config.TextColumn("", width="small"),
        })
