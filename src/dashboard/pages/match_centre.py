"""Page 3 - Match Centre."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src import config
from src.dashboard import components as ui
from src.dashboard import data
from src.dashboard.theme import PALETTE, plotly_layout


def _recent_form(matches: pd.DataFrame, team: str, before: pd.Timestamp, n: int = 5) -> list[dict]:
    if matches.empty:
        return []
    history = matches[(matches["date"] < before)
                      & ((matches["home_team"] == team) | (matches["away_team"] == team))]
    history = history.sort_values(["date", "time"]).tail(n).iloc[::-1]
    rows = []
    for _, row in history.iterrows():
        home = row["home_team"] == team
        goals_for = row["home_goals"] if home else row["away_goals"]
        goals_against = row["away_goals"] if home else row["home_goals"]
        if goals_for > goals_against:
            code = "W"
        elif goals_for < goals_against:
            code = "L"
        else:
            code = "D"
        rows.append({
            "code": code,
            "score": f"{int(goals_for)}–{int(goals_against)}",
            "opponent": row["away_team"] if home else row["home_team"],
            "venue": "H" if home else "A",
            "date": pd.Timestamp(row["date"]).strftime("%d %b"),
        })
    return rows


def _form_html(rows: list[dict], align: str = "left") -> str:
    if not rows:
        return f'<div style="color:{PALETTE["muted"]};font-size:.76rem">No recent matches</div>'
    colors = {"W": "#3BE58C", "D": "#FFB547", "L": "#FF5C7A"}
    chips = "".join(
        f'<span style="display:inline-flex;width:20px;height:20px;border-radius:6px;'
        f'align-items:center;justify-content:center;font-size:.62rem;font-weight:800;'
        f'background:{colors[r["code"]]}22;color:{colors[r["code"]]};'
        f'border:1px solid {colors[r["code"]]}55;margin-right:4px">{r["code"]}</span>'
        for r in rows
    )
    detail = " · ".join(f'{r["venue"]} {r["score"]} v {config.short_name(r["opponent"])}'
                        for r in rows[:3])
    return (f'<div style="text-align:{align}">{chips}'
            f'<div style="font-size:.68rem;color:{PALETTE["muted"]};margin-top:.25rem">{detail}</div></div>')


def _scaled_pair(home_value, away_value) -> tuple[float, float]:
    try:
        h = float(home_value)
        a = float(away_value)
    except (TypeError, ValueError):
        return 50.0, 50.0
    if not np.isfinite(h) or not np.isfinite(a) or (h + a) <= 0:
        return 50.0, 50.0
    return 100.0 * h / (h + a), 100.0 * a / (h + a)


def render() -> None:
    rows = data.upcoming_payload()
    matches = data.matches()

    ui.page_header(
        "Match Centre · Deep dive",
        "Match Centre",
        "Side-by-side team profile for any upcoming fixture: Elo, rolling form, venue "
        "splits, attacking and defensive strength, head-to-head, rest and the model's "
        "probability decomposition.",
    )

    if not rows:
        ui.empty_state("No upcoming fixtures", "Refresh the pipeline to load the next round.")
        return

    labels = [
        f"{r['home_display']} vs {r['away_display']} — "
        f"{pd.Timestamp(r['date']).strftime('%d %b')} {r.get('time', '')}"
        for r in rows
    ]
    choice = st.selectbox("Select fixture", labels, key="mc_fixture")
    row = rows[labels.index(choice)]
    context = row.get("context") or {}
    home = row["home_display"]
    away = row["away_display"]
    kickoff = pd.Timestamp(row["date"])

    # ---------------- header ----------------
    left, middle, right = st.columns([1.15, 1.25, 1.15], gap="medium")
    with left:
        st.markdown(
            f'<div class="panel" style="text-align:center;padding:1.1rem">'
            f'<div class="crest" style="--c1:{PALETTE["home"]};--c2:#4C2FD6;margin:0 auto 8px;'
            f'width:46px;height:46px;font-size:.8rem">{row["home_short"]}</div>'
            f'<div style="font-size:1.05rem;font-weight:750">{home}</div>'
            f'<div style="font-size:.72rem;color:{PALETTE["muted"]};margin-top:.25rem">Home</div>'
            f'</div>', unsafe_allow_html=True)
        st.markdown(_form_html(_recent_form(matches, row["home_team"], kickoff)),
                    unsafe_allow_html=True)
    with middle:
        ui.big_probabilities(row["prob_home"], row["prob_draw"], row["prob_away"],
                             row["predicted"])
        st.markdown(
            f'<div style="text-align:center;margin-top:.6rem;font-size:.74rem;'
            f'color:{PALETTE["muted"]}">Model call: '
            f'<b style="color:#fff">{row["predicted_label"]}</b> · '
            f'{row["confidence"]:.1%} confidence · {row["confidence_band"]} band</div>',
            unsafe_allow_html=True)
    with right:
        st.markdown(
            f'<div class="panel" style="text-align:center;padding:1.1rem">'
            f'<div class="crest" style="--c1:{PALETTE["away"]};--c2:#0E8FA8;margin:0 auto 8px;'
            f'width:46px;height:46px;font-size:.8rem">{row["away_short"]}</div>'
            f'<div style="font-size:1.05rem;font-weight:750">{away}</div>'
            f'<div style="font-size:.72rem;color:{PALETTE["muted"]};margin-top:.25rem">Away</div>'
            f'</div>', unsafe_allow_html=True)
        st.markdown(_form_html(_recent_form(matches, row["away_team"], kickoff), "right"),
                    unsafe_allow_html=True)

    st.markdown(f'<div style="text-align:center;font-size:.74rem;color:{PALETTE["muted"]};'
                f'margin:.4rem 0 0">Kick-off {kickoff.strftime("%A %d %B %Y")} · '
                f'{row.get("time", "15:00")} · {row.get("season", "")} '
                f'{"· Matchweek " + str(int(row["matchweek"])) if row.get("matchweek") else ""}'
                f'</div>', unsafe_allow_html=True)

    # ---------------- team comparison ----------------
    left, right = st.columns([1.15, 1], gap="medium")
    with left:
        ui.section("Team profile")
        rows_html = "".join([
            ui.comparison_row("Elo rating", context.get("home_elo"), context.get("away_elo"), "{:.0f}"),
            ui.comparison_row("Form · points per game", context.get("home_form5_ppg"),
                              context.get("away_form5_ppg")),
            ui.comparison_row("Venue form (5)", context.get("home_venue5_ppg"),
                              context.get("away_venue5_ppg")),
            ui.comparison_row("Goals scored (5)", context.get("home_form5_gf"),
                              context.get("away_form5_gf")),
            ui.comparison_row("Goals conceded (5)", context.get("home_form5_ga"),
                              context.get("away_form5_ga"), invert=True),
            ui.comparison_row("Season points per game", context.get("home_season_ppg"),
                              context.get("away_season_ppg")),
            ui.comparison_row("Rest days", context.get("home_rest_days"),
                              context.get("away_rest_days"), "{:.0f}"),
            ui.comparison_row("Matches in last 14 days", context.get("home_matches_14d"),
                              context.get("away_matches_14d"), "{:.0f}", invert=True),
            ui.comparison_row("Expected goals (Poisson)", context.get("expected_goals_home"),
                              context.get("expected_goals_away")),
            ui.comparison_row("Head-to-head PPG", context.get("h2h_ppg"),
                              (3.0 - float(context.get("h2h_ppg") or 0)) if context.get("h2h_ppg") is not None else None),
        ])
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;font-size:.7rem;'
            f'color:{PALETTE["muted"]};margin-bottom:.5rem">'
            f'<b style="color:#A98CFF">{home}</b><b style="color:#6DE3F7">{away}</b></div>'
            + rows_html, unsafe_allow_html=True)
        st.caption("Goals conceded and fixture congestion are inverted: the longer bar is better.")

    with right:
        ui.section("Strength profile")
        categories = ["Attack", "Defence", "Form", "Venue", "Elo", "Freshness"]
        home_vals, away_vals = [], []
        for home_key, away_key, invert in [
            ("home_form5_gf", "away_form5_gf", False),
            ("home_form5_ga", "away_form5_ga", True),
            ("home_form5_ppg", "away_form5_ppg", False),
            ("home_venue5_ppg", "away_venue5_ppg", False),
            ("home_elo", "away_elo", False),
            ("home_rest_days", "away_rest_days", False),
        ]:
            hv, av = context.get(home_key), context.get(away_key)
            if invert:
                hv = -float(hv) if hv is not None else None
                av = -float(av) if av is not None else None
            h, a = _scaled_pair(hv, av)
            home_vals.append(round(h, 1))
            away_vals.append(round(a, 1))
        st.plotly_chart(
            ui.radar_figure(home, away, home_vals, away_vals, categories),
            key="mc_radar")

    # ---------------- probability decomposition ----------------
    ui.section("Probability decomposition")
    cols = st.columns(3, gap="small")
    with cols[0]:
        st.markdown('<div class="panel"><div class="kpi-label">Calibrated model</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(ui.probability_donut(row["prob_home"], row["prob_draw"],
                                             row["prob_away"]), key="mc_donut")
        st.markdown("</div>", unsafe_allow_html=True)
    with cols[1]:
        st.markdown('<div class="panel"><div class="kpi-label">Poisson goal model</div>',
                    unsafe_allow_html=True)
        pois = [context.get("poisson_home"), context.get("poisson_draw"), context.get("poisson_away")]
        if all(v is not None for v in pois):
            st.plotly_chart(ui.probability_donut(*pois), key="mc_poisson")
        else:
            st.caption("Poisson prior unavailable for this fixture.")
        st.markdown("</div>", unsafe_allow_html=True)
    with cols[2]:
        st.markdown('<div class="panel"><div class="kpi-label">Bookmaker consensus</div>',
                    unsafe_allow_html=True)
        market = row.get("market")
        if market:
            st.plotly_chart(ui.probability_donut(market["home"], market["draw"], market["away"]),
                            key="mc_market")
        else:
            st.markdown(
                f'<div class="empty" style="padding:1.4rem .8rem">'
                f'<b>No market data</b>Prices are not published this far ahead. '
                f'This fixture is scored with the no-odds model.</div>',
                unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if row.get("market"):
        market = row["market"]
        st.caption(
            f"Model minus market: home {(row['prob_home'] - market['home']) * 100:+.1f} pp · "
            f"draw {(row['prob_draw'] - market['draw']) * 100:+.1f} pp · "
            f"away {(row['prob_away'] - market['away']) * 100:+.1f} pp"
        )
    else:
        ui.note("This fixture has no published prices yet, so the no-odds variant "
                "(223 features, no market information) produced the probabilities above. "
                "It is retrained from scratch on the same chronological split.", "warn")

    with st.expander("Raw feature context (pre-match only)"):
        context_frame = pd.DataFrame(
            [{"feature": k, "value": v} for k, v in context.items()
             if not isinstance(v, (dict, bool)) and v is not None])
        st.dataframe(context_frame, hide_index=True, height=300)
        st.caption("Every value above was computed from matches played strictly before this "
                   "fixture's kick-off.")
