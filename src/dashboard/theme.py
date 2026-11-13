"""Visual design system for the EPL Intelligence dashboard.

Dark analytics theme: deep navy canvas, translucent panels, violet/electric-blue
accents. Everything is expressed as CSS variables so pages stay consistent.
"""
from __future__ import annotations

import hashlib

PALETTE = {
    "bg": "#070B14",
    "bg_soft": "#0B1120",
    "panel": "#111A2E",
    "panel_2": "#16203A",
    "border": "#1F2C4A",
    "text": "#E9EEF9",
    "muted": "#8C99B4",
    "accent": "#7C5CFF",
    "accent_2": "#22D3EE",
    "home": "#7C5CFF",
    "draw": "#FFB547",
    "away": "#22D3EE",
    "good": "#3BE58C",
    "bad": "#FF5C7A",
    "warn": "#FFB547",
}

HOME_COLOR = PALETTE["home"]
DRAW_COLOR = PALETTE["draw"]
AWAY_COLOR = PALETTE["away"]

# Distinct, readable crest gradients (index chosen by a stable hash).
CREST_GRADIENTS = [
    ("#7C5CFF", "#4C2FD6"),
    ("#22D3EE", "#0E8FA8"),
    ("#3BE58C", "#159C63"),
    ("#FFB547", "#D97A10"),
    ("#FF5C7A", "#C02848"),
    ("#5B8DEF", "#2A54B8"),
    ("#B78CFF", "#7550D6"),
    ("#2DD4BF", "#0F8F80"),
    ("#F472B6", "#BE2D74"),
    ("#A3E635", "#6BA312"),
]


def team_gradient(team: str) -> tuple[str, str]:
    digest = hashlib.md5(team.encode("utf-8")).hexdigest()
    return CREST_GRADIENTS[int(digest[:2], 16) % len(CREST_GRADIENTS)]


CSS = f"""
<style>
:root {{
  --bg: {PALETTE['bg']};
  --bg-soft: {PALETTE['bg_soft']};
  --panel: {PALETTE['panel']};
  --panel-2: {PALETTE['panel_2']};
  --border: {PALETTE['border']};
  --text: {PALETTE['text']};
  --muted: {PALETTE['muted']};
  --accent: {PALETTE['accent']};
  --accent-2: {PALETTE['accent_2']};
  --home: {PALETTE['home']};
  --draw: {PALETTE['draw']};
  --away: {PALETTE['away']};
  --good: {PALETTE['good']};
  --bad: {PALETTE['bad']};
}}

html, body, [class*="css"] {{
  font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
}}

.stApp {{
  background:
    radial-gradient(1100px 600px at 8% -12%, rgba(124, 92, 255, 0.20), transparent 60%),
    radial-gradient(900px 520px at 96% -6%, rgba(34, 211, 238, 0.13), transparent 62%),
    linear-gradient(180deg, #070B14 0%, #05080F 100%);
  color: var(--text);
}}

.block-container {{ padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1380px; }}

#MainMenu, footer, header [data-testid="stStatusWidget"] {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stDecoration"] {{ display: none; }}

/* ---------------- sidebar ---------------- */
section[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, #0A1020 0%, #070B14 100%);
  border-right: 1px solid var(--border);
}}
section[data-testid="stSidebar"] .block-container {{ padding-top: 1.1rem; }}

.side-brand {{
  display: flex; align-items: center; gap: 0.65rem; margin-bottom: 0.2rem;
}}
.side-mark {{
  width: 38px; height: 38px; border-radius: 11px;
  background: linear-gradient(135deg, var(--accent), #4C2FD6);
  display: flex; align-items: center; justify-content: center;
  font-weight: 800; font-size: 0.95rem; color: #fff; letter-spacing: 0.02em;
  box-shadow: 0 8px 22px rgba(124, 92, 255, 0.38);
}}
.side-title {{ font-size: 1.02rem; font-weight: 750; letter-spacing: -0.01em; line-height: 1.1; }}
.side-sub {{ font-size: 0.68rem; color: var(--muted); letter-spacing: 0.14em; text-transform: uppercase; }}
.side-meta {{
  margin-top: 0.9rem; padding: 0.7rem 0.8rem; border: 1px solid var(--border);
  border-radius: 12px; background: rgba(17, 26, 46, 0.65); font-size: 0.72rem;
  color: var(--muted); line-height: 1.55;
}}
.side-meta b {{ color: var(--text); font-weight: 650; }}
.dot {{
  display: inline-block; width: 7px; height: 7px; border-radius: 50%;
  background: var(--good); margin-right: 6px; box-shadow: 0 0 10px var(--good);
}}
.dot.stale {{ background: var(--draw); box-shadow: 0 0 10px var(--draw); }}
.dot.bad {{ background: var(--bad); box-shadow: 0 0 10px var(--bad); }}

/* sidebar radio -> nav */
section[data-testid="stSidebar"] div[role="radiogroup"] {{ gap: 0.15rem; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label {{
  padding: 0.48rem 0.7rem; border-radius: 10px; transition: background 0.15s ease;
  font-size: 0.9rem;
}}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{ background: rgba(124, 92, 255, 0.12); }}
section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"],
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{
  background: linear-gradient(90deg, rgba(124,92,255,0.24), rgba(124,92,255,0.04));
  border-left: 2px solid var(--accent);
}}

/* ---------------- typography ---------------- */
.page-eyebrow {{
  font-size: 0.68rem; letter-spacing: 0.2em; text-transform: uppercase;
  color: var(--accent-2); font-weight: 700; margin-bottom: 0.35rem;
}}
.page-title {{
  font-size: 2.0rem; font-weight: 800; letter-spacing: -0.025em; line-height: 1.1;
  margin: 0 0 0.3rem 0; color: #FFFFFF;
}}
.page-sub {{ color: var(--muted); font-size: 0.92rem; margin-bottom: 1.2rem; max-width: 850px; }}
.section-title {{
  font-size: 0.76rem; letter-spacing: 0.16em; text-transform: uppercase;
  color: var(--muted); font-weight: 700; margin: 1.5rem 0 0.7rem 0;
}}

/* ---------------- cards ---------------- */
.kpi {{
  background: linear-gradient(160deg, rgba(22,32,58,0.95), rgba(13,20,36,0.9));
  border: 1px solid var(--border); border-radius: 16px; padding: 1rem 1.05rem;
  height: 100%; position: relative; overflow: hidden;
}}
.kpi::after {{
  content: ""; position: absolute; inset: 0 0 auto 0; height: 2px;
  background: linear-gradient(90deg, var(--accent), var(--accent-2));
  opacity: 0.85;
}}
.kpi-label {{ font-size: 0.68rem; letter-spacing: 0.13em; text-transform: uppercase; color: var(--muted); font-weight: 650; }}
.kpi-value {{ font-size: 1.62rem; font-weight: 780; letter-spacing: -0.02em; margin-top: 0.28rem; color: #fff; }}
.kpi-delta {{ font-size: 0.73rem; color: var(--muted); margin-top: 0.2rem; }}
.kpi-delta.up {{ color: var(--good); }}
.kpi-delta.down {{ color: var(--bad); }}

.panel {{
  background: linear-gradient(160deg, rgba(17,26,46,0.92), rgba(10,16,29,0.92));
  border: 1px solid var(--border); border-radius: 16px; padding: 1rem 1.15rem;
}}

/* ---------------- match card ---------------- */
.mc {{
  background: linear-gradient(165deg, rgba(22,32,58,0.92) 0%, rgba(11,17,32,0.95) 100%);
  border: 1px solid var(--border); border-radius: 16px; padding: 0.85rem 0.95rem 0.9rem;
  margin-bottom: 0.85rem; transition: border-color 0.18s ease, transform 0.18s ease;
}}
.mc:hover {{ border-color: rgba(124,92,255,0.55); transform: translateY(-1px); }}
.mc-top {{ display: flex; justify-content: space-between; align-items: center; font-size: 0.7rem; color: var(--muted); }}
.mc-badge {{
  background: rgba(124,92,255,0.14); color: #C9BCFF; border: 1px solid rgba(124,92,255,0.32);
  padding: 0.1rem 0.45rem; border-radius: 999px; font-size: 0.64rem; letter-spacing: 0.08em; font-weight: 700;
}}
.mc-teams {{ display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; gap: 0.55rem; margin: 0.6rem 0 0.7rem; }}
.mc-team {{ display: flex; align-items: center; gap: 0.55rem; min-width: 0; }}
.mc-team.away {{ flex-direction: row-reverse; text-align: right; }}
.mc-team span {{ font-size: 0.9rem; font-weight: 650; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.crest {{
  width: 30px; height: 30px; border-radius: 9px; flex: 0 0 auto;
  display: flex; align-items: center; justify-content: center;
  font-size: 0.58rem; font-weight: 800; color: #fff; letter-spacing: 0.03em;
  background: linear-gradient(140deg, var(--c1), var(--c2));
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.25), 0 4px 12px rgba(0,0,0,0.35);
}}
.mc-pred {{ display: flex; flex-direction: column; align-items: center; gap: 0.22rem; }}
.pill {{ padding: 0.16rem 0.5rem; border-radius: 999px; font-size: 0.64rem; font-weight: 800; letter-spacing: 0.07em; }}
.pill-H {{ background: rgba(124,92,255,0.18); color: #C4B5FF; border: 1px solid rgba(124,92,255,0.4); }}
.pill-D {{ background: rgba(255,181,71,0.16); color: #FFD79A; border: 1px solid rgba(255,181,71,0.38); }}
.pill-A {{ background: rgba(34,211,238,0.16); color: #A5EEFB; border: 1px solid rgba(34,211,238,0.38); }}
.mc-conf {{ font-size: 0.62rem; color: var(--muted); }}

.pbar {{ display: flex; height: 22px; border-radius: 8px; overflow: hidden; background: #0A111F; }}
.pbar div {{
  display: flex; align-items: center; justify-content: center;
  font-size: 0.63rem; font-weight: 750; color: rgba(255,255,255,0.94);
  min-width: 2px; transition: width 0.3s ease;
}}
.pbar .h {{ background: linear-gradient(90deg, #6A48F0, #8B6BFF); }}
.pbar .d {{ background: linear-gradient(90deg, #E09A2B, #FFC46B); color: #2A1A00; }}
.pbar .a {{ background: linear-gradient(90deg, #17B6D0, #45E0F8); color: #052229; }}
.plabel {{ display: flex; justify-content: space-between; font-size: 0.62rem; color: var(--muted); margin-top: 0.3rem; }}

/* ---------------- small bits ---------------- */
.tag {{
  display: inline-block; padding: 0.14rem 0.5rem; border-radius: 999px;
  font-size: 0.66rem; font-weight: 650; letter-spacing: 0.04em;
  background: rgba(124,92,255,0.12); color: #C9BCFF; border: 1px solid rgba(124,92,255,0.28);
  margin-right: 0.28rem;
}}
.tag.ok {{ background: rgba(59,229,140,0.12); color: #9BF7C6; border-color: rgba(59,229,140,0.3); }}
.tag.warn {{ background: rgba(255,181,71,0.12); color: #FFD79A; border-color: rgba(255,181,71,0.3); }}
.tag.bad {{ background: rgba(255,92,122,0.12); color: #FFB3C1; border-color: rgba(255,92,122,0.3); }}
.tag.muted {{ background: rgba(140,153,180,0.1); color: var(--muted); border-color: rgba(140,153,180,0.22); }}

.note {{
  border-left: 3px solid var(--accent); background: rgba(124,92,255,0.07);
  padding: 0.65rem 0.85rem; border-radius: 0 10px 10px 0; font-size: 0.8rem;
  color: #C9D2E6; margin: 0.6rem 0;
}}
.note.warn {{ border-left-color: var(--draw); background: rgba(255,181,71,0.07); }}
.note.bad {{ border-left-color: var(--bad); background: rgba(255,92,122,0.07); }}

.big-prob {{ display: flex; gap: 0.65rem; }}
.bp-item {{
  flex: 1; text-align: center; padding: 0.75rem 0.4rem; border-radius: 14px;
  border: 1px solid var(--border); background: rgba(17,26,46,0.7);
}}
.bp-item .v {{ font-size: 1.45rem; font-weight: 800; letter-spacing: -0.02em; }}
.bp-item .l {{ font-size: 0.63rem; letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted); margin-top: 0.15rem; }}
.bp-h .v {{ color: #A98CFF; }} .bp-d .v {{ color: #FFC46B; }} .bp-a .v {{ color: #6DE3F7; }}
.bp-item.win {{ border-color: rgba(124,92,255,0.6); box-shadow: 0 0 0 1px rgba(124,92,255,0.25), 0 8px 26px rgba(124,92,255,0.16); }}

.cmp-row {{ display: grid; grid-template-columns: 88px 1fr 88px; align-items: center; gap: 0.6rem; margin-bottom: 0.42rem; }}
.cmp-label {{ font-size: 0.7rem; color: var(--muted); text-align: center; }}
.cmp-val {{ font-size: 0.82rem; font-weight: 700; }}
.cmp-track {{ height: 7px; background: #0A111F; border-radius: 99px; overflow: hidden; display: flex; }}
.cmp-track .l {{ background: linear-gradient(90deg, #6A48F0, #8B6BFF); }}
.cmp-track .r {{ background: linear-gradient(90deg, #45E0F8, #17B6D0); margin-left: auto; }}

.empty {{
  border: 1px dashed var(--border); border-radius: 16px; padding: 2.2rem 1rem;
  text-align: center; color: var(--muted); background: rgba(17,26,46,0.35);
}}
.empty b {{ display: block; color: var(--text); font-size: 1rem; margin-bottom: 0.3rem; }}

/* ---------------- streamlit widget polish ---------------- */
.stButton > button {{
  border-radius: 10px; border: 1px solid rgba(124,92,255,0.45);
  background: linear-gradient(135deg, rgba(124,92,255,0.85), rgba(76,47,214,0.85));
  color: #fff; font-weight: 650; padding: 0.45rem 1.1rem;
}}
.stButton > button:hover {{ border-color: var(--accent-2); filter: brightness(1.08); }}
div[data-baseweb="select"] > div, .stDateInput > div > div, .stTextInput input {{
  background: rgba(17,26,46,0.9) !important; border-color: var(--border) !important;
  color: var(--text) !important; border-radius: 10px !important;
}}
[data-testid="stDataFrame"] {{ border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }}
[data-testid="stExpander"] {{
  border: 1px solid var(--border); border-radius: 12px; background: rgba(17,26,46,0.5);
}}
h1, h2, h3, h4 {{ color: #fff; letter-spacing: -0.02em; }}
hr {{ border-color: var(--border); }}
[data-testid="stMetricValue"] {{ color: #fff; }}
</style>
"""


def inject(st) -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def plotly_layout(fig, height: int = 320, legend: bool = True):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", color=PALETTE["text"], size=12),
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        hoverlabel=dict(bgcolor=PALETTE["panel_2"], font_size=12,
                        bordercolor=PALETTE["border"]),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zeroline=False,
                     linecolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zeroline=False,
                     linecolor="rgba(255,255,255,0.08)")
    return fig
