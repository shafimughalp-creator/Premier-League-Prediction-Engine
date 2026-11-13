"""Page 7 - Data Centre."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config
from src.dashboard import components as ui
from src.dashboard import data
from src.dashboard.theme import PALETTE


def _status_tag(ok: bool, label: str) -> str:
    return ui.tag(label, "ok" if ok else "bad")


def render() -> None:
    health = data.data_health()
    quality = data.quality()
    leakage = data.leakage()
    state = data.pipeline_state()
    updates = data.update_log()
    meta = data.meta()
    report = data.report()

    ui.page_header(
        "Data Centre · Operations",
        "Data Centre",
        "Everything the platform knows, where it came from, when it was last refreshed and "
        "whether the integrity and leakage audits still pass.",
    )

    status = health.get("status")
    if status == "ok":
        ui.note("All systems nominal. Cached data is valid and the prediction snapshot is "
                "current.", "")
    elif status == "stale":
        ui.note("The prediction snapshot is older than 72 hours. Run a refresh before "
                "treating the fixtures as current.", "warn")
    else:
        ui.note("The pipeline needs attention - the last run did not complete cleanly. "
                f"Recorded error: <code>{state.get('error', 'unknown')}</code>", "bad")

    ui.kpi_row([
        {"label": "Processed matches", "value": f"{health['n_matches']:,}",
         "delta": f"{quality.get('n_teams', '—')} clubs · "
                  f"{len(quality.get('seasons', {}))} seasons"},
        {"label": "Odds coverage", "value": f"{quality.get('odds_coverage', 0):.1%}",
         "delta": "Matches with a usable bookmaker price"},
        {"label": "Match-stat coverage", "value": f"{quality.get('shots_coverage', 0):.1%}",
         "delta": "Shots, corners, cards, fouls"},
        {"label": "Upcoming fixtures", "value": f"{health['n_fixtures']}",
         "delta": f"Scored {str(health.get('predictions_generated', '—'))[:16].replace('T', ' ')} UTC"},
    ])

    # ---------------- refresh ----------------
    ui.section("Refresh pipeline")
    c1, c2 = st.columns([1, 2.2], gap="medium")
    with c1:
        refresh = st.button("Fetch latest data", type="primary", key="dc_refresh")
    with c2:
        st.caption("Downloads the current season, next-round fixtures and published schedule, "
                   "re-validates the dataset, then re-scores upcoming fixtures. Retraining is "
                   "only triggered when enough new results exist.")
    if refresh:
        with st.spinner("Refreshing sources and re-scoring fixtures…"):
            try:
                from src.pipeline import run
                result = run(offline=False, retrain="auto", predict_only=True)
                data.clear_cache()
                st.success(
                    f"Refresh complete - {result.get('n_fixtures', 0)} fixtures scored, "
                    f"data cutoff {result.get('data_cutoff', '—')}."
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Refresh failed: {type(exc).__name__}: {exc}")

    # ---------------- sources ----------------
    ui.section("Data sources")
    sources = pd.DataFrame([
        {"Source": "football-data.co.uk — season results (E0)",
         "Provides": "Dates, teams, scores, shots, corners, cards, fouls, bookmaker odds",
         "Coverage": "2014/15 → 2026/27 · 4,660 matches", "Status": "primary"},
        {"Source": "football-data.co.uk — fixtures feed",
         "Provides": "Upcoming fixtures with live prices when published",
         "Coverage": "Rolling next round", "Status": "primary"},
        {"Source": "openfootball / football.json",
         "Provides": "Full 2026/27 schedule and round numbers",
         "Coverage": "2026/27 season", "Status": "fallback"},
    ])
    st.dataframe(sources, hide_index=True, height=180)

    last_update = (updates[-1] if updates else {}) or {}
    st.caption(
        f"Last refresh started {last_update.get('started_at', '—')} and finished "
        f"{last_update.get('finished_at', '—')}. Raw payloads are cached unchanged in "
        f"data/raw/ and never edited."
    )

    # ---------------- quality + leakage ----------------
    left, right = st.columns([1, 1], gap="medium")
    with left:
        ui.section("Integrity audit")
        checks = quality
        if checks:
            rows = [
                ("Matches parsed", f"{checks.get('n_matches', 0):,}"),
                ("Duplicate fixtures", str(checks.get("duplicate_fixtures", 0))),
                ("Invalid result codes", str(checks.get("invalid_results", 0))),
                ("Date range", f"{checks.get('first_date', '—')} → {checks.get('last_date', '—')}"),
                ("Teams", str(checks.get("n_teams", 0))),
            ]
            st.dataframe(pd.DataFrame(rows, columns=["Check", "Value"]),
                         hide_index=True, height=215)
            distribution = checks.get("outcome_distribution", {})
            if distribution:
                st.caption(" · ".join(f"{k}: {v:.1%}" for k, v in distribution.items()))
        else:
            ui.empty_state("No quality report", "Run the pipeline to generate one.")
    with right:
        ui.section("Leakage audit")
        if leakage:
            checks = leakage.get("integrity", {}).get("checks", {})
            rows = [("Dataset integrity", leakage.get("integrity", {}).get("ok", False))]
            for key, block in (("own_result", leakage.get("own_result", {})),
                               ("future_independence", leakage.get("future_independence", {})),
                               ("cold_start", leakage.get("cold_start", {}))):
                rows.append((key.replace("_", " ").title(), block.get("passed", False)))
            frame = pd.DataFrame(
                [{"Test": label, "Result": "PASS" if ok else "FAIL"} for label, ok in rows])
            st.dataframe(frame, hide_index=True, height=215,
                         column_config={"Result": st.column_config.TextColumn(width="small")})
            cases = leakage.get("own_result", {}).get("cases", [])
            if cases:
                st.caption(
                    f"Own-result independence verified on {len(cases)} matches across the "
                    f"full history and {leakage.get('own_result', {}).get('cases', [{}])[0].get('n_features', 0)} "
                    "features. Flipping a match's score never changes its own feature row."
                )
        else:
            ui.empty_state("No leakage report",
                           "Run <code>python -m src.data.validation</code> to generate one.")

    # ---------------- pipeline log ----------------
    left, right = st.columns([1.2, 1], gap="medium")
    with left:
        ui.section("Recent pipeline runs")
        if updates:
            frame = pd.DataFrame(updates).tail(8).iloc[::-1].copy()
            keep = [c for c in ("finished_at", "fixtures", "openfootball", "seasons")
                    if c in frame.columns]
            frame = frame[keep]
            frame["seasons"] = frame["seasons"].astype(str).str.slice(0, 60)
            st.dataframe(frame, hide_index=True, height=230)
        else:
            ui.empty_state("No update log", "No refresh has been recorded yet.")
    with right:
        ui.section("Last pipeline run")
        steps = (state.get("steps") or [])[-8:]
        if steps:
            for step in steps:
                kind = {"ok": "ok", "failed": "bad", "degraded": "warn",
                        "skipped": "muted", "rejected": "warn"}.get(step.get("status"), "muted")
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;gap:.5rem;'
                    f'padding:.28rem 0;border-bottom:1px solid rgba(255,255,255,.05);'
                    f'font-size:.76rem">'
                    f'<span style="color:{PALETTE["muted"]};min-width:82px">'
                    f'{str(step.get("at", ""))[11:19]}</span>'
                    f'<span style="flex:1;font-weight:600">{step.get("step")}</span>'
                    f'{ui.tag(str(step.get("status", "")).upper(), kind)}</div>',
                    unsafe_allow_html=True)
            st.caption(f"Last success: {state.get('last_success', '—')} · "
                       f"duration {state.get('duration_seconds', '—')}s")
        else:
            ui.empty_state("No runs recorded", "Run the pipeline to populate the activity log.")

    # ---------------- model metadata ----------------
    ui.section("Model metadata")
    rows = [
        ("Model version", health.get("model_version") or "—"),
        ("Production model", report.get("production_model", meta.get("model_type", "—"))),
        ("Trained at", meta.get("trained_at", report.get("trained_at", "—"))),
        ("Data cutoff", meta.get("data_cutoff", report.get("data_cutoff", "—"))),
        ("Features (with odds)", str(report.get("n_features", "—"))),
        ("Features (no odds)", str(report.get("n_features_no_odds", "—"))),
        ("Poisson alpha", str(report.get("poisson_alpha", "—"))),
        ("Market beta", str(report.get("market_beta", "—"))),
        ("Temperature", str(report.get("temperature", "—"))),
        ("Test accuracy", f"{report.get('test_metrics', {}).get('accuracy', 0):.4f}"),
        ("Accepted", "yes" if report.get("accepted") else "no"),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Property", "Value"]),
                 hide_index=True, height=390)

    st.caption(
        "Seasons: " + ", ".join(f"{v}" for v in config.SEASONS.values())
        + f". Raw cache: {len(list(config.RAW_DIR.glob('*.csv')))} CSV files, "
        f"{len(list(config.RAW_DIR.glob('*.json')))} JSON files."
    )
