"""Reusable HTML and Plotly building blocks for the dashboard."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.theme import (
    AWAY_COLOR,
    DRAW_COLOR,
    HOME_COLOR,
    PALETTE,
    plotly_layout,
    team_gradient,
)

OUTCOME_TEXT = {"H": "HOME WIN", "D": "DRAW", "A": "AWAY WIN"}
OUTCOME_CLASS = {"H": "pill-H", "D": "pill-D", "A": "pill-A"}


# --------------------------------------------------------------------------
# Layout primitives
# --------------------------------------------------------------------------
def page_header(eyebrow: str, title: str, subtitle: str = "") -> None:
    st.markdown(
        f'<div class="page-eyebrow">{eyebrow}</div>'
        f'<div class="page-title">{title}</div>'
        f'<div class="page-sub">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def section(title: str) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def kpi(label: str, value: str, delta: str = "", direction: str = "") -> str:
    cls = {"up": "up", "down": "down"}.get(direction, "")
    return (
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-delta {cls}">{delta}</div></div>'
    )


def kpi_row(items: list[dict]) -> None:
    columns = st.columns(len(items), gap="small")
    for column, item in zip(columns, items):
        with column:
            st.markdown(
                kpi(item.get("label", ""), item.get("value", "—"),
                    item.get("delta", ""), item.get("direction", "")),
                unsafe_allow_html=True,
            )


def note(text: str, kind: str = "") -> None:
    st.markdown(f'<div class="note {kind}">{text}</div>', unsafe_allow_html=True)


def tag(text: str, kind: str = "") -> str:
    return f'<span class="tag {kind}">{text}</span>'


def empty_state(title: str, message: str) -> None:
    st.markdown(
        f'<div class="empty"><b>{title}</b>{message}</div>', unsafe_allow_html=True
    )


# --------------------------------------------------------------------------
# Match presentation
# --------------------------------------------------------------------------
def _date_label(date_value, time_value) -> str:
    try:
        stamp = pd.Timestamp(date_value)
        label = stamp.strftime("%a %d %b").replace(" 0", " ")
        return f"{label} · {time_value or '15:00'}"
    except Exception:
        return f"{date_value} · {time_value or ''}"


def prob_bar_html(home: float, draw: float, away: float, height: str = "22px") -> str:
    total = max(float(home) + float(draw) + float(away), 1e-9)
    h, d, a = (100 * float(home) / total, 100 * float(draw) / total, 100 * float(away) / total)
    parts = []
    if h > 0.5:
        parts.append(f'<div class="h" style="width:{h:.2f}%">{h:.0f}%</div>')
    if d > 0.5:
        parts.append(f'<div class="d" style="width:{d:.2f}%">{d:.0f}%</div>')
    if a > 0.5:
        parts.append(f'<div class="a" style="width:{a:.2f}%">{a:.0f}%</div>')
    return f'<div class="pbar" style="height:{height}">{"".join(parts)}</div>'


def match_card(row: dict, show_market: bool = False) -> str:
    home = row.get("home_display") or row.get("home_team", "")
    away = row.get("away_display") or row.get("away_team", "")
    hs = row.get("home_short") or home[:3].upper()
    aw = row.get("away_short") or away[:3].upper()
    c1a, c2a = team_gradient(home)
    c1b, c2b = team_gradient(away)
    code = row.get("predicted", "H")
    confidence = float(row.get("confidence") or 0.0) * 100
    home_p = float(row.get("prob_home") or 0.0)
    draw_p = float(row.get("prob_draw") or 0.0)
    away_p = float(row.get("prob_away") or 0.0)
    week = row.get("matchweek")
    badge = f"MW {int(week)}" if week else (row.get("season") or "")

    market_html = ""
    market = row.get("market")
    if show_market and market:
        market_html = (
            '<div class="plabel" style="margin-top:.45rem">'
            f'<span>Market: H {market["home"]:.0%}</span>'
            f'<span>D {market["draw"]:.0%}</span>'
            f'<span>A {market["away"]:.0%}</span></div>'
        )

    variant = row.get("model_variant", "full")
    variant_tag = ("odds-enhanced" if variant == "full"
                   else "no-odds model")

    return f"""
<div class="mc">
  <div class="mc-top">
    <span>{_date_label(row.get('date'), row.get('time'))}</span>
    <span class="mc-badge">{badge}</span>
  </div>
  <div class="mc-teams">
    <div class="mc-team">
      <div class="crest" style="--c1:{c1a};--c2:{c2a}">{hs}</div>
      <span>{home}</span>
    </div>
    <div class="mc-pred">
      <span class="pill {OUTCOME_CLASS.get(code, 'pill-H')}">{OUTCOME_TEXT.get(code, code)}</span>
      <span class="mc-conf">{confidence:.0f}% confidence</span>
    </div>
    <div class="mc-team away">
      <div class="crest" style="--c1:{c1b};--c2:{c2b}">{aw}</div>
      <span>{away}</span>
    </div>
  </div>
  {prob_bar_html(home_p, draw_p, away_p)}
  <div class="plabel">
    <span>Home {home_p:.1%}</span><span>Draw {draw_p:.1%}</span><span>Away {away_p:.1%}</span>
  </div>
  {market_html}
  <div class="plabel" style="margin-top:.4rem;opacity:.75">
    <span>{variant_tag}</span>
    <span>model {row.get('model_version', '')}</span>
  </div>
</div>
"""


def render_match_cards(rows: list[dict], columns: int = 2, show_market: bool = False) -> None:
    if not rows:
        empty_state("No fixtures available",
                    "Run the pipeline to refresh fixtures and generate predictions.")
        return
    for start in range(0, len(rows), columns):
        cols = st.columns(columns, gap="small")
        for column, row in zip(cols, rows[start:start + columns]):
            with column:
                st.markdown(match_card(row, show_market=show_market), unsafe_allow_html=True)


def big_probabilities(home: float, draw: float, away: float, predicted: str) -> None:
    items = [
        ("bp-h", "Home win", home, "H"),
        ("bp-d", "Draw", draw, "D"),
        ("bp-a", "Away win", away, "A"),
    ]
    html = '<div class="big-prob">'
    for cls, label, value, code in items:
        win = " win" if code == predicted else ""
        html += (f'<div class="bp-item {cls}{win}"><div class="v">{value:.1%}</div>'
                 f'<div class="l">{label}</div></div>')
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def comparison_row(label: str, home_value, away_value, fmt: str = "{:.2f}",
                   invert: bool = False) -> str:
    try:
        hv = float(home_value)
        av = float(away_value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(hv) or not np.isfinite(av):
        return ""
    total = abs(hv) + abs(av)
    left_pct = 50.0 if total == 0 else 100.0 * abs(hv) / total
    if invert:
        left_pct = 100.0 - left_pct
    right_pct = 100.0 - left_pct
    return (
        f'<div class="cmp-row">'
        f'<div class="cmp-val" style="text-align:right">{fmt.format(hv)}</div>'
        f'<div><div class="cmp-label">{label}</div>'
        f'<div class="cmp-track"><div class="l" style="width:{left_pct:.1f}%"></div>'
        f'<div class="r" style="width:{right_pct:.1f}%"></div></div></div>'
        f'<div class="cmp-val" style="text-align:left">{fmt.format(av)}</div>'
        f'</div>'
    )


# --------------------------------------------------------------------------
# Plotly charts
# --------------------------------------------------------------------------
def probability_donut(home: float, draw: float, away: float) -> go.Figure:
    fig = go.Figure(go.Pie(
        values=[home, draw, away],
        labels=["Home", "Draw", "Away"],
        hole=0.68,
        sort=False,
        direction="clockwise",
        marker=dict(colors=[HOME_COLOR, DRAW_COLOR, AWAY_COLOR],
                    line=dict(color=PALETTE["bg"], width=2)),
        textinfo="label+percent",
        textfont=dict(size=12, color="#FFFFFF"),
        hovertemplate="%{label}: %{percent}<extra></extra>",
    ))
    fig.update_layout(
        height=260, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def metric_bar(labels: list[str], values: list[float], colors: list[str] | None = None,
               title: str = "", fmt: str = "{:.1%}", height: int = 300) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colors or [PALETTE["accent"]] * len(values),
                    line=dict(width=0)),
        text=[fmt.format(v) for v in values],
        textposition="outside",
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(title=title or None)
    ordered = list(reversed(labels))
    fig.update_yaxes(categoryorder="array", categoryarray=ordered, automargin=True)
    fig.update_xaxes(showticklabels=False, range=[0, max(values) * 1.22 if values else 1])
    return plotly_layout(fig, height=height, legend=False)


def calibration_figure(curve: list[dict]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
        line=dict(color="rgba(255,255,255,0.25)", dash="dash", width=1),
    ))
    if curve:
        fig.add_trace(go.Scatter(
            x=[c["mean_confidence"] for c in curve],
            y=[c["observed_accuracy"] for c in curve],
            mode="lines+markers", name="Model",
            line=dict(color=PALETTE["accent"], width=2.5),
            marker=dict(size=4 + np.sqrt([c["count"] for c in curve]) * 0.9,
                        color=PALETTE["accent_2"], line=dict(color="#0B1120", width=1)),
            customdata=[[c["count"]] for c in curve],
            hovertemplate="confidence %{x:.2f}<br>accuracy %{y:.2f}"
                          "<br>n=%{customdata[0]}<extra></extra>",
        ))
    fig.update_xaxes(title="Predicted confidence", range=[0, 1], tickformat=".0%")
    fig.update_yaxes(title="Observed accuracy", range=[0, 1], tickformat=".0%")
    return plotly_layout(fig, height=340)


def confusion_figure(matrix: list[list[int]], labels: list[str] | None = None) -> go.Figure:
    labels = labels or ["Away", "Draw", "Home"]
    matrix = np.asarray(matrix, dtype=float)
    annotations = []
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalised = matrix / np.where(row_sums == 0, 1, row_sums)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            annotations.append(dict(
                x=labels[j], y=labels[i],
                text=f"{int(matrix[i, j])}<br><span style='font-size:10px'>{normalised[i, j]:.0%}</span>",
                showarrow=False, font=dict(color="#FFFFFF" if normalised[i, j] > 0.3 else "#9AA6BF"),
            ))
    fig = go.Figure(go.Heatmap(
        z=normalised, x=labels, y=labels, colorscale="Purples", showscale=False,
        hovertemplate="Actual %{y} · Predicted %{x}<br>%{z:.1%}<extra></extra>",
    ))
    fig.update_layout(annotations=annotations)
    fig.update_xaxes(title="Predicted", side="bottom")
    fig.update_yaxes(title="Actual", autorange="reversed")
    return plotly_layout(fig, height=340, legend=False)


def season_trend_figure(rows: list[dict]) -> go.Figure:
    fig = go.Figure()
    seasons = [r["season"] for r in rows]
    fig.add_trace(go.Bar(
        x=seasons, y=[r["accuracy"] for r in rows], name="Accuracy",
        marker_color=PALETTE["accent"], opacity=0.85,
        text=[f"{r['accuracy']:.1%}" for r in rows], textposition="outside",
    ))
    fig.add_trace(go.Scatter(
        x=seasons, y=[r["log_loss"] for r in rows], name="Log loss",
        mode="lines+markers", yaxis="y2",
        line=dict(color=PALETTE["accent_2"], width=2),
    ))
    fig.update_layout(yaxis=dict(title="Accuracy", tickformat=".0%",
                                 range=[0, max([r["accuracy"] for r in rows] or [0.6]) * 1.25]),
                      yaxis2=dict(title="Log loss", overlaying="y", side="right",
                                  showgrid=False))
    return plotly_layout(fig, height=320)


def rolling_accuracy_figure(frame: pd.DataFrame, window: int = 40) -> go.Figure:
    fig = go.Figure()
    if frame.empty:
        return plotly_layout(fig, height=300)
    ordered = frame.sort_values("date").reset_index(drop=True)
    rolling = ordered["correct"].rolling(window, min_periods=max(10, window // 4)).mean()
    fig.add_trace(go.Scatter(
        x=ordered["date"], y=rolling, mode="lines",
        line=dict(color=PALETTE["accent"], width=2.2),
        fill="tozeroy", fillcolor="rgba(124,92,255,0.12)", name=f"{window}-match accuracy",
    ))
    fig.add_hline(y=float(ordered["correct"].mean()), line_dash="dash",
                  line_color="rgba(255,255,255,0.35)",
                  annotation_text=f"overall {ordered['correct'].mean():.1%}",
                  annotation_position="top left")
    fig.update_yaxes(tickformat=".0%", range=[0.2, 0.8])
    return plotly_layout(fig, height=320)


def radar_figure(home_label: str, away_label: str,
                 home_values: list[float], away_values: list[float],
                 categories: list[str]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=home_values + home_values[:1], theta=categories + categories[:1],
        fill="toself", name=home_label,
        line=dict(color=HOME_COLOR, width=2), fillcolor="rgba(124,92,255,0.22)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=away_values + away_values[:1], theta=categories + categories[:1],
        fill="toself", name=away_label,
        line=dict(color=AWAY_COLOR, width=2), fillcolor="rgba(34,211,238,0.18)",
    ))
    fig.update_layout(
        height=360, polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, gridcolor="rgba(255,255,255,0.08)",
                            linecolor="rgba(255,255,255,0.1)", tickfont=dict(size=9)),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.08)",
                             linecolor="rgba(255,255,255,0.12)"),
        ),
        margin=dict(l=40, r=40, t=40, b=20), showlegend=True,
        legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", color=PALETTE["text"], size=11),
    )
    return fig


def importance_figure(rows: list[dict], top: int = 15) -> go.Figure:
    rows = rows[:top][::-1]
    labels = [r["feature"].replace("_", " ") for r in rows]
    values = [r["importance"] for r in rows]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=values, colorscale="Purples", line=dict(width=0)),
        hovertemplate="%{y}: %{x:.2%}<extra></extra>",
    ))
    fig.update_xaxes(showticklabels=False)
    return plotly_layout(fig, height=380, legend=False)


def outcome_distribution_figure(home: int, draw: int, away: int) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=["Home win", "Draw", "Away win"], y=[home, draw, away],
        marker_color=[HOME_COLOR, DRAW_COLOR, AWAY_COLOR],
        text=[home, draw, away], textposition="outside",
    ))
    fig.update_yaxes(title="Matches")
    return plotly_layout(fig, height=280, legend=False)


# --------------------------------------------------------------------------
# Research evidence figures
# --------------------------------------------------------------------------
def draw_decile_figure(rows: list[dict]) -> go.Figure:
    """Mean predicted draw probability against the observed draw rate."""
    labels = [f"{r['mean_p_draw']:.0%}" for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=[r["mean_p_draw"] for r in rows], name="Mean predicted",
        marker_color=DRAW_COLOR, opacity=0.55,
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=[r["actual_draw_rate"] for r in rows], name="Observed draw rate",
        mode="lines+markers", line=dict(color=PALETTE["accent_2"], width=2.5),
        marker=dict(size=7),
    ))
    fig.update_xaxes(title="Draw-probability decile (mean p)")
    fig.update_yaxes(title="Probability", tickformat=".0%", range=[0, 1])
    return plotly_layout(fig, height=320)


def policy_curve_figure(rows: list[dict], x_key: str = "threshold") -> go.Figure:
    """Accuracy and macro F1 as a function of a draw decision policy."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[r[x_key] for r in rows], y=[r["accuracy"] for r in rows],
        mode="lines", name="Accuracy", line=dict(color=PALETTE["accent"], width=2.4),
    ))
    fig.add_trace(go.Scatter(
        x=[r[x_key] for r in rows], y=[r["macro_f1"] for r in rows],
        mode="lines", name="Macro F1", line=dict(color=PALETTE["accent_2"], width=2.4),
    ))
    fig.add_trace(go.Scatter(
        x=[r[x_key] for r in rows], y=[r["draw_recall"] for r in rows],
        mode="lines", name="Draw recall", line=dict(color=DRAW_COLOR, width=1.8,
                                                     dash="dot"),
    ))
    fig.update_xaxes(title="Draw threshold" if x_key == "threshold" else "Draw margin")
    fig.update_yaxes(title="Score", tickformat=".0%", range=[0, 1])
    return plotly_layout(fig, height=320)


def per_class_reliability_figure(rows: list[dict]) -> go.Figure:
    colours = {"Away win": AWAY_COLOR, "Draw": DRAW_COLOR, "Home win": HOME_COLOR}
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Perfect",
        line=dict(color="rgba(255,255,255,0.22)", dash="dash", width=1),
        showlegend=False,
    ))
    for name in ("Away win", "Draw", "Home win"):
        subset = [r for r in rows if r["class"] == name]
        if not subset:
            continue
        fig.add_trace(go.Scatter(
            x=[r["mean_probability"] for r in subset],
            y=[r["observed_rate"] for r in subset],
            mode="lines+markers", name=name,
            line=dict(color=colours[name], width=2),
            marker=dict(size=5),
        ))
    fig.update_xaxes(title="Predicted probability", range=[0, 1], tickformat=".0%")
    fig.update_yaxes(title="Observed frequency", range=[0, 1], tickformat=".0%")
    return plotly_layout(fig, height=340)


def walk_forward_figure(rows: list[dict], variants: tuple[str, ...] = ("production", "market")) -> go.Figure:
    fig = go.Figure()
    colours = {"production": PALETTE["accent"], "market": PALETTE["accent_2"],
               "no_odds": PALETTE["good"]}
    for variant in variants:
        present = [r for r in rows if r.get(variant)]
        if not present:
            continue
        fig.add_trace(go.Scatter(
            x=[r["season"] for r in present],
            y=[r[variant]["accuracy"] for r in present],
            mode="lines+markers", name=variant.replace("_", " ").title(),
            line=dict(color=colours.get(variant, PALETTE["muted"]), width=2.2),
        ))
    fig.update_xaxes(title="Evaluation season")
    fig.update_yaxes(title="Accuracy", tickformat=".0%")
    return plotly_layout(fig, height=320)
