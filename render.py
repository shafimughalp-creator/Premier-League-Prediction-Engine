"""HTML renderers — a line-for-line port of the prototype's JS render functions.

Each function returns the same markup and class names as `reference/index.html`.
The only intentional change is the href format: the prototype's hash routes
(`#/week`) become Streamlit query routes (`?page=week`).
"""

from __future__ import annotations

from datetime import date
from urllib.parse import quote

from data import CLUB, DATA, LOGOS

SH = "M4 4H44V30C44 42 34 49 24 53C14 49 4 42 4 30Z"
PAT = {
    "v": '<rect x="12" width="8" height="56"/><rect x="28" width="8" height="56"/>',
    "h": '<rect y="16" width="48" height="12"/><rect y="40" width="48" height="12"/>',
    "half": '<rect x="24" width="24" height="56"/>',
    "diag": '<polygon points="0,14 0,28 48,50 48,36"/>',
    "chev": '<path d="M0 22L24 38L48 22V32L24 48L0 32Z"/>',
    "band": '<rect y="22" width="48" height="10"/>',
    "top": '<rect width="48" height="15"/>',
    "cross": '<rect x="20" width="8" height="56"/><rect y="24" width="48" height="8"/>',
}
LBL = {"home": "Home win", "draw": "Draw", "away": "Away win"}
MISS = "\u2014<i data-miss></i>"
ORD = ["st", "nd", "rd"]
_WORDS = {
    0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
    6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten", 11: "Eleven",
    12: "Twelve", 13: "Thirteen", 14: "Fourteen", 15: "Fifteen",
    16: "Sixteen", 17: "Seventeen", 18: "Eighteen", 19: "Nineteen", 20: "Twenty",
}


def word(n: int) -> str:
    return _WORDS.get(n, str(n))


def short_date() -> str:
    """'Saturday 21 Nov' from the fixture date."""
    if not DATA["fixtures"]:
        return "\u2014"
    d = date.fromisoformat(DATA["fixtures"][0]["date"])
    return f"{d.strftime('%A')} {d.day} {d.strftime('%b')}"


def cutoff_date() -> str:
    """'8 November 2026' from meta.dataCutoff."""
    value = DATA["meta"].get("dataCutoff")
    if not value:
        return "\u2014"
    d = date.fromisoformat(value)
    return f"{d.day} {d.strftime('%B')} {d.year}"


def kickoff() -> str:
    return DATA["fixtures"][0]["time"] if DATA["fixtures"] else "\u2014"


def season() -> str:
    return DATA["meta"]["season"]


def matchweek():
    return DATA["meta"].get("matchweek") or "\u2014"


def esc(value) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def T(team_name):
    return next(t for t in DATA["table"] if t["team"] == team_name)


def pts5(form) -> int:
    return sum(3 if x == "W" else 1 if x == "D" else 0 for x in form)


def gd(n) -> str:
    if n > 0:
        return "+" + str(n)
    if n < 0:
        return "\u2212" + str(abs(n))
    return "0"


def v(value) -> str:
    return MISS if value is None else esc(value)


def nm(team_name) -> str:
    return esc(DATA["full"].get(team_name, team_name))


def badge(short: str, size: int) -> str:
    c = CLUB[short]
    height = round(size * 1.17)
    return (
        f'<svg class="crest" width="{size}" height="{height}" viewBox="0 0 48 56" aria-hidden="true">'
        f'<g clip-path="url(#sh)"><rect width="48" height="56" fill="{c[0]}"/>'
        f'<g fill="{c[1]}">{PAT[c[2]]}</g>'
        f'<rect width="48" height="28" fill="url(#shine)"/></g>'
        f'<path d="{SH}" fill="none" stroke="rgba(255,255,255,.55)" stroke-width="2"/>'
        f'<text x="24" y="34" text-anchor="middle" font-family="Inter,sans-serif" font-size="14" '
        f'font-weight="800" fill="#fff" stroke="rgba(0,0,0,.65)" stroke-width="3" '
        f'paint-order="stroke">{short}</text></svg>'
    )


def crest(team_name, size: int = 34) -> str:
    t = T(team_name)
    url = LOGOS.get(t["short"])
    if url:
        # The inline onerror from the prototype is stripped by Streamlit's
        # sanitizer; static/app.js binds the same fallback on load instead.
        return (
            f'<img class="crest" src="{esc(url)}" data-c="{t["short"]}" data-s="{size}" '
            f'width="{size}" height="{size}" alt="" referrerpolicy="no-referrer">'
        )
    # Streamlit's sanitizer removes inline <svg>. Render a correctly sized
    # placeholder and let static/app.js insert the same built-in badge.
    return (
        f'<span class="crest-slot" data-c="{t["short"]}" data-s="{size}" aria-hidden="true" '
        f'style="display:block;flex:none;width:{size}px;height:{round(size * 1.17)}px"></span>'
    )


def chips(form) -> str:
    inner = "".join(f'<b class="{x}">{x}</b>' for x in form)
    return f'<span class="fc" aria-label="Last five: {", ".join(form)}">{inner}</span>'


def bar(chances, big: bool = False) -> str:
    c = chances
    cls = "bar big" if big else "bar"
    return (
        f'<div class="{cls}" role="img" aria-label="Home win {c["home"]}%, draw {c["draw"]}%, '
        f'away win {c["away"]}%">'
        f'<i class="h" style="flex:{c["home"]}"></i>'
        f'<i class="d" style="flex:{c["draw"]}"></i>'
        f'<i class="a" style="flex:{c["away"]}"></i></div>'
        f'<div class="pcs"><span data-count="{c["home"]}" data-suf="%">{c["home"]}%</span>'
        f'<span data-count="{c["draw"]}" data-suf="%">{c["draw"]}%</span>'
        f'<span data-count="{c["away"]}" data-suf="%">{c["away"]}%</span></div>'
    )


def match_url(fixture) -> str:
    return "?page=match&id=" + quote(fixture["id"], safe="")


def row(fixture) -> str:
    f = fixture
    return (
        f'<a class="fx" href="{match_url(f)}" data-pick="{f["pick"]}" '
        f'aria-label="{esc(f["home"])} v {esc(f["away"])}">'
        f'<span class="time">{esc(f["time"])}</span>'
        f'<div class="tm r th"><span>{esc(f["home"])}</span>{crest(f["home"])}</div>'
        f'<div class="tb">{bar(f["chances"])}</div>'
        f'<div class="tm ta">{crest(f["away"])}<span>{esc(f["away"])}</span></div>'
        f'<span class="tag {f["pick"]}">{LBL[f["pick"]]}</span></a>'
    )


def head(eye: str, title: str, sub: str) -> str:
    return (
        f'<header class="head"><div class="eye">{eye}</div><h1>{title}</h1>'
        f'<p class="sub-t">{sub}</p></header>'
    )


# --------------------------------------------------------------------------
# screens
# --------------------------------------------------------------------------


def overview() -> str:
    f0 = next((f for f in DATA["fixtures"] if f["home"] == "Liverpool"), None)
    if f0 is None and DATA["fixtures"]:
        f0 = DATA["fixtures"][0]
    S = DATA["seasonNumbers"]
    top = DATA["table"][:5] + DATA["table"][-3:]

    mini_rows = []
    for i, t in enumerate(top):
        if i == 5:
            mini_rows.append(
                '<tr><td colspan="4" class="mut" style="text-align:center;padding:2px">\u00b7 \u00b7 \u00b7</td></tr>'
            )
        mini_rows.append(
            f'<tr data-team="{esc(t["team"])}" tabindex="0" role="button">'
            f'<td class="l" style="padding-right:6px">{t["position"]}</td>'
            f'<td class="l">{crest(t["team"], 24)}{esc(t["team"])}</td>'
            f'<td>{t["played"]}</td><td><b>{t["points"]}</b></td></tr>'
        )

    hero = ""
    if f0 is not None:
        hero = (
            '<section class="hero"><div class="hx"><div class="vs">'
            + crest(f0["home"], 84)
            + '<span class="vsl">v</span>'
            + crest(f0["away"], 84)
            + '</div><div><div class="eye">Featured fixture</div>'
            + f"<h2>{nm(f0['home'])} v {nm(f0['away'])}</h2>"
            + f'<p class="meta">{DATA["meta"]["dateLong"]} \u00b7 {f0["time"]} kick-off \u00b7 Premier League</p></div></div>'
            + bar(f0["chances"], True)
            + f'<a class="btn" href="{match_url(f0)}">See the full read</a></section>'
        )

    return (
        head(
            f"Matchweek {matchweek()} \u00b7 {season()}",
            "Welcome back",
            f"{word(len(DATA['fixtures']))} Premier League fixtures this weekend, with the chances for each result and the form behind them.",
        )
        + hero
        + f'<div class="grid"><section><h3 class="t">Matchweek {matchweek()} fixtures <a href="?page=week">View all</a></h3>'
        + "".join(row(f) for f in DATA["fixtures"])
        + "</section>"
        + '<div><div class="card"><h3 class="t">The table <a href="?page=table">Full table</a></h3>'
        + '<div class="tw"><table class="mini"><thead><tr><th class="l">#</th><th class="l">Team</th>'
        + "<th>P</th><th>Pts</th></tr></thead><tbody>"
        + "".join(mini_rows)
        + "</tbody></table></div></div>"
        + '<div class="card"><h3 class="t">Season so far</h3><div class="stats">'
        + f'<div class="stat"><b data-count="{S["goalsPerGame"]}">{S["goalsPerGame"]}</b><span>Goals per game</span></div>'
        + f'<div class="stat"><b data-count="{S["homeWins"]}">{S["homeWins"]}</b><span>Home wins</span></div>'
        + f'<div class="stat"><b data-count="{S["draws"]}">{S["draws"]}</b><span>Draws</span></div>'
        + f'<div class="stat"><b data-count="{S["awayWins"]}">{S["awayWins"]}</b><span>Away wins</span></div>'
        + "</div></div></div></div>"
    )


def week_list(filter_key: str = "all") -> str:
    shown = [f for f in DATA["fixtures"] if filter_key == "all" or f["pick"] == filter_key]
    if shown:
        return "".join(row(f) for f in shown)
    return '<p class="mut" style="padding:20px 0">No fixtures lean this way this week.</p>'


def week(filter_key: str = "all") -> str:
    def count(key):
        return len(DATA["fixtures"]) if key == "all" else sum(1 for f in DATA["fixtures"] if f["pick"] == key)

    chips_html = "".join(
        f'<button class="chip{" on" if filter_key == key else ""}" data-f="{key}" '
        f'aria-pressed="{"true" if filter_key == key else "false"}">{label}'
        f"<span>{count(key)}</span></button>"
        for key, label in [
            ("all", "All"),
            ("home", "Home wins"),
            ("draw", "Draws"),
            ("away", "Away wins"),
        ]
    )
    return (
        head(
            f"Matchweek {matchweek()} \u00b7 {season()}",
            f"Matchweek {matchweek()}",
            DATA["meta"]["dateLong"] + f" \u00b7 all kick-offs {kickoff()}",
        )
        + f'<div class="chips" role="group" aria-label="Filter fixtures">{chips_html}</div>'
        + f'<div id="wl">{week_list(filter_key)}</div>'
        + '<p class="note">Chances are rounded to whole percentages and always add up to 100.</p>'
    )


def match(fixture_id: str) -> str:
    f = next((x for x in DATA["fixtures"] if x["id"] == fixture_id or x.get("fixtureId") == fixture_id), None)
    if f is None:
        return '<p>Fixture not found. <a href="?page=week" style="text-decoration:underline">Back to fixtures</a></p>'

    H = T(f["home"])
    A = T(f["away"])
    r = f["rates"] or {}
    c = f["chances"]

    hp = r.get("homeForm5Ppg")
    if hp is None:
        hp = f"{pts5(H['form']) / 5:.1f}"
    ap = r.get("awayForm5Ppg")
    if ap is None:
        ap = f"{pts5(A['form']) / 5:.1f}"
    xg = f"roughly {r['xg'][0]} \u2013 {r['xg'][1]}" if r.get("xg") else MISS

    def cc(key, label, note):
        pick = f["pick"] == key
        return (
            f'<div class="cc{" pick" if pick else ""}">'
            + ('<span class="badge">Our pick</span>' if pick else "")
            + f'<div class="lb">{label}</div>'
            + f'<div class="pc" data-count="{c[key]}" data-suf="%">{c[key]}%</div>'
            + f"<small>{note}</small></div>"
        )

    read = [
        f"{nm(f['home'])} have taken {v(hp)} points a game across their last five; {nm(f['away'])} have taken {v(ap)}.",
        f"At home, {nm(f['home'])} average {v(r.get('homeHomePpg'))} points a game. On the road, {nm(f['away'])} average {v(r.get('awayAwayPpg'))}.",
        f"{nm(f['home'])} are scoring {v(r.get('homeGf5'))} and conceding {v(r.get('homeGa5'))} a game over that spell; "
        f"{nm(f['away'])} are scoring {v(r.get('awayGf5'))} and conceding {v(r.get('awayGa5'))}.",
        f"In the last five meetings, {nm(f['home'])} have taken {v(r.get('h2hPpg'))} points a game, with "
        + (MISS if r.get("h2hLevelPct") is None else str(r["h2hLevelPct"]) + "%")
        + " finishing level.",
        f"The scoring rates point to {xg} on the night.",
    ]

    def fm(t, is_home, gf, ga, ppg, loc):
        scored = MISS if gf is None else f"{gf} / {ga}"
        return (
            f"<h4>{crest(t['team'], 28)}{esc(t['team'])}</h4>"
            f"<span>Last five</span><span class=\"tr\">{chips(t['form'])}</span>"
            f"<span>Points per game (last five)</span><span class=\"tr\">{v(ppg)}</span>"
            f"<span>Scored / conceded per game</span><span class=\"tr\">{scored}</span>"
            f"<span>{'Home' if is_home else 'Away'} record (last five, pts per game)</span>"
            f"<span class=\"tr\">{v(loc)}</span>"
        )

    if f["headToHead"]:
        h2h = '<div class="tw"><table><tbody>' + "".join(
            f'<tr><td class="l mut">{esc(m["date"])}</td>'
            f'<td class="l"><span class="tm">{crest(m["home"], 22)}{esc(m["home"])}</span></td>'
            f'<td><b>{m["homeGoals"]}\u2013{m["awayGoals"]}</b></td>'
            f'<td class="l"><span class="tm">{crest(m["away"], 22)}{esc(m["away"])}</span></td>'
            f'<td class="mut">{esc(m["season"])}</td></tr>'
            for m in f["headToHead"]
        ) + "</tbody></table></div>"
    else:
        h2h = f'<p class="mut">Last five meetings: {MISS}</p>'

    def sr(label, a, b):
        return f'<tr><td class="l mut">{label}</td><td><b>{a}</b></td><td><b>{b}</b></td></tr>'

    home_rank = f"{H['position']}{ORD[H['position'] - 1] if H['position'] <= 3 else 'th'}"
    away_rank = f"{A['position']}{ORD[A['position'] - 1] if A['position'] <= 3 else 'th'}"

    return (
        '<button class="back" data-back>\u2190 Back to fixtures</button>'
        '<div class="mh"><div class="mt">'
        + crest(f["home"], 64)
        + f"<div><strong>{nm(f['home'])}</strong><small>{home_rank} \u00b7 {H['points']} pts</small></div></div>"
        + f'<div class="mc"><b>Matchweek {matchweek()}</b><br>{DATA["meta"]["dateLong"]}<br>{f["time"]} kick-off</div>'
        + f'<div class="mt r"><div><strong>{nm(f["away"])}</strong><small>{away_rank} \u00b7 {A["points"]} pts</small></div>'
        + crest(f["away"], 64)
        + "</div></div>"
        + f'<div class="cc3">{cc("home", nm(f["home"]) + " win", "Win in 90 minutes")}'
        + f'{cc("draw", "Draw", "Points shared")}'
        + f'{cc("away", nm(f["away"]) + " win", "Win in 90 minutes")}</div>'
        + f'<div class="card" style="margin-bottom:18px"><h3 class="t">The chances</h3>{bar(c, True)}</div>'
        + '<div class="g2"><div class="card"><h3 class="t">The read</h3><ul class="read">'
        + "".join(f"<li>{x}</li>" for x in read)
        + '</ul></div><div class="card"><h3 class="t">Form</h3><div class="fm">'
        + fm(H, True, r.get("homeGf5"), r.get("homeGa5"), hp, r.get("homeHomePpg"))
        + fm(A, False, r.get("awayGf5"), r.get("awayGa5"), ap, r.get("awayAwayPpg"))
        + '</div></div><div class="card"><h3 class="t">Head to head</h3>'
        + h2h
        + '</div><div class="card"><h3 class="t">Season so far</h3><div class="tw"><table><thead><tr>'
        + f'<th class="l"></th><th>{crest(f["home"], 26)}</th><th>{crest(f["away"], 26)}</th>'
        + "</tr></thead><tbody>"
        + sr("Position", H["position"], A["position"])
        + sr("Points", H["points"], A["points"])
        + sr("Goals for", H["gf"], A["gf"])
        + sr("Goals against", H["ga"], A["ga"])
        + sr("Goal difference", gd(H["gd"]), gd(A["gd"]))
        + "</tbody></table></div></div></div>"
    )


def table() -> str:
    rows = []
    for t in DATA["table"]:
        cls = "t4" if t["position"] <= 4 else ("bt" if t["position"] >= 18 else "")
        rows.append(
            f'<tr data-team="{esc(t["team"])}" tabindex="0" role="button" class="{cls}">'
            f'<td class="l">{t["position"]}</td>'
            f'<td class="l"><span class="tm">{crest(t["team"], 28)}{esc(t["team"])}</span></td>'
            f'<td>{t["played"]}</td><td>{t["won"]}</td><td>{t["drawn"]}</td><td>{t["lost"]}</td>'
            f'<td>{t["gf"]}</td><td>{t["ga"]}</td><td>{gd(t["gd"])}</td>'
            f'<td class="pts">{t["points"]}</td><td>{chips(t["form"])}</td></tr>'
        )
    played = DATA["table"][0]["played"] if DATA["table"] else 0
    return (
        head(f"{season()} \u00b7 After {played} matches", "League table", "Select a club for its season summary.")
        + '<div class="card"><div class="tw"><table><thead><tr><th class="l">#</th><th class="l">Team</th>'
        + "<th>P</th><th>W</th><th>D</th><th>L</th><th>GF</th><th>GA</th><th>GD</th><th>Pts</th><th>Last 5</th>"
        + "</tr></thead><tbody>"
        + "".join(rows)
        + '</tbody></table></div><div class="key"><span><i style="background:var(--cyan)"></i>Top four</span>'
        + '<span><i style="background:var(--pink)"></i>Bottom three</span></div></div>'
    )


# --------------------------------------------------------------------------
# results — prediction accuracy, scored against official final results
# --------------------------------------------------------------------------
_PICK = {"H": "home", "D": "draw", "A": "away"}


def _when(value) -> str:
    """'2026-11-14T16:26' -> '2026-11-14 16:26 UTC' (honest, unstylised)."""
    if not value:
        return MISS
    return esc(str(value)[:16].replace("T", " ") + " UTC")


def results_page(mw=None) -> str:
    perf = DATA.get("performance") or {}
    weeks = {k: v for k, v in (perf.get("matchweeks") or {}).items() if k != "unknown"}
    season_totals = perf.get("season") or {}
    keys = sorted(weeks, key=lambda k: int(k))
    selected = str(mw) if (mw is not None and str(mw) in weeks) else None
    if selected is None and keys:
        scored = [k for k in keys if weeks[k].get("evaluated")]
        completed = [k for k in keys if weeks[k].get("completed")]
        selected = (scored or completed or keys)[-1]

    def stat(value, label):
        shown = "\u2014" if value is None else value
        return f'<div class="stat"><b>{shown}</b><span>{label}</span></div>'

    summary = (
        '<div class="card"><h3 class="t">Season so far</h3><div class="stats">'
        + stat(season_totals.get("correct", 0), "Correct")
        + stat(season_totals.get("incorrect", 0), "Incorrect")
        + stat(season_totals.get("accuracy"), "Accuracy %")
        + stat(season_totals.get("evaluated", 0), "Evaluated")
        + '</div><div class="key"><span><i style="background:var(--green)"></i>'
        + f'{season_totals.get("correct", 0)} correct / {season_totals.get("evaluated", 0)} evaluated</span>'
        + f'<span>Pending: {season_totals.get("pending", 0)}</span>'
        + f'<span>No eligible prediction: {season_totals.get("missing_prediction", 0)}</span>'
        + "</div></div>"
    )

    if keys:
        chips = "".join(
            f'<a class="chip{" on" if k == selected else ""}" href="?page=results&amp;mw={k}">'
            f'MW {k}<span>{weeks[k]["correct"]}/{weeks[k]["evaluated"]}</span></a>'
            for k in keys
        )
        chips_html = f'<div class="chips" role="group" aria-label="Matchweek">{chips}</div>'
    else:
        chips_html = ""

    if selected is None:
        week_html = (
            '<div class="card"><p class="mut">No scored predictions yet \u2014 matches '
            "appear here automatically once official final results are in.</p></div>"
        )
    else:
        week = weeks[selected]
        rows_html = []
        for r in (DATA.get("results") or []):
            if str(r.get("matchweek")) != selected:
                continue
            pick = r.get("predicted")
            pick_tag = (
                f'<span class="tag {_PICK[pick]}">{LBL[_PICK[pick]]}</span>'
                if pick else '<span class="tag draw">No snapshot</span>'
            )
            correct = r.get("correct")
            verdict = (
                '<span class="tag ok">Correct</span>' if correct
                else '<span class="tag bad">Incorrect</span>' if correct is not None
                else MISS
            )
            rows_html.append(
                f'<tr><td class="l mut">{esc(r["date"])}</td>'
                f'<td class="l"><span class="tm">{crest(r["home"], 22)}{esc(r["home"])}</span></td>'
                f'<td><b>{r["homeGoals"]}\u2013{r["awayGoals"]}</b></td>'
                f'<td class="l"><span class="tm">{crest(r["away"], 22)}{esc(r["away"])}</span></td>'
                f"<td>{pick_tag}</td>"
                f'<td class="l mut">{LBL[_PICK.get(r["outcome"], r["outcome"])] if r.get("outcome") in _PICK else esc(r.get("outcome") or "\u2014")}</td>'
                f"<td>{verdict}</td></tr>"
            )
        table_html = (
            '<div class="tw"><table><thead><tr><th class="l">Date</th><th class="l">Home</th>'
            "<th>Score</th><th class=\"l\">Away</th><th>Prediction</th><th>Outcome</th><th>Verdict</th>"
            "</tr></thead><tbody>"
            + ("".join(rows_html) or '<tr><td colspan="7" class="mut">No completed matches this matchweek.</td></tr>')
            + "</tbody></table></div>"
        )
        week_html = (
            '<div class="card"><h3 class="t">Matchweek ' + esc(selected) + " performance</h3>"
            + f'<p class="note" style="margin:0 0 14px"><b>{week["correct"]} / {week["evaluated"]}</b> correct'
            + (f" \u00b7 {week['accuracy']}% accuracy" if week.get("accuracy") is not None else "")
            + "</p><div class=\"stats\">"
            + stat(week.get("correct", 0), "Correct")
            + stat(week.get("incorrect", 0), "Incorrect")
            + stat(week.get("evaluated", 0), "Evaluated")
            + stat(week.get("pending", 0), "Pending")
            + '</div><div class="key">'
            + f'<span>Pending: {week.get("pending", 0)}</span>'
            + (f'<span>Cancelled: {week["cancelled"]}</span>' if week.get("cancelled") else "")
            + f'<span>No eligible prediction: {week.get("missing_prediction", 0)}</span>'
            + f'<span>Coverage: {week.get("coverage") if week.get("coverage") is not None else "\u2014"}%</span>'
            + "</div></div>"
            + f'<div class="card"><h3 class="t">Matchweek {esc(selected)} match by match</h3>{table_html}</div>'
        )

    fresh = DATA.get("freshness") or {}

    def fresh_row(label, value):
        return f"<span>{label}</span><span class=\"tr\">{value}</span>"

    freshness_html = (
        '<div class="card"><h3 class="t">Data freshness</h3><div class="fm">'
        + fresh_row("Last fixture synchronisation", _when((fresh.get("fixtures") or {}).get("last_sync_at")))
        + fresh_row("Last prediction generation", _when((fresh.get("predictions") or {}).get("last_run")))
        + fresh_row("Last result synchronisation", _when((fresh.get("results") or {}).get("last_sync_at")))
        + fresh_row("Last evaluation update", _when((fresh.get("evaluation") or {}).get("last_run")))
        + "</div><p class=\"note\">All timestamps are UTC. Pending matches are never counted as "
        "incorrect; completed matches without a saved pre-match prediction are listed separately.</p></div>"
    )

    return (
        head(
            f"{season()} \u00b7 Prediction tracking",
            "Results",
            "Pre-match predictions scored against official final results.",
        )
        + summary
        + chips_html
        + week_html
        + freshness_html
    )


# --------------------------------------------------------------------------
# shell
# --------------------------------------------------------------------------

NAV = [
    ("ov", "top", "?page=overview", "Premier League"),
    ("ov2", "sub", "?page=overview", "Overview"),
    ("week", "sub", "?page=week", f"Matchweek {matchweek()}"),
    ("table", "sub", "?page=table", "Table"),
    ("results", "sub", "?page=results", "Results"),
]


def _nav_class(route_key: str, data_r: str) -> str:
    on = data_r == route_key or (route_key == "ov2" and data_r == "ov")
    return " on" if on else ""


def shell(page: str = "overview", fixture_id: str | None = None, filter_key: str = "all",
          mw=None) -> tuple[str, str]:
    """Return (html, document_title) for the requested route."""
    if page == "match":
        route_key = "week"
        title = "Match"
        body = match(fixture_id or "")
    elif page == "week":
        route_key = "week"
        title = f"Matchweek {matchweek()}"
        body = week(filter_key)
    elif page == "table":
        route_key = "table"
        title = "Table"
        body = table()
    elif page == "results":
        route_key = "results"
        title = "Results"
        body = results_page(mw)
    else:
        route_key = "ov2"
        title = "Overview"
        body = overview()

    nav_html = "".join(
        f'<a class="{cls}{_nav_class(route_key, data_r).strip() and _nav_class(route_key, data_r)}" '
        f'href="{href}" data-r="{data_r}">{label}</a>'
        for data_r, cls, href, label in NAV
    )

    html = (
        '<a class="skip" href="#main">Skip to content</a>'
        '<aside class="side" id="side" aria-label="Main navigation">'
        '<div class="brand"><span class="mark" data-mark aria-hidden="true"></span>EPL Intelligence</div>'
        '<div class="sec">Football</div>'
        f'<nav class="nav" id="nav" aria-label="Sections">{nav_html}</nav>'
        '<div class="sidebot"><div class="wk"><b>This weekend</b>'
        + f"{len(DATA['fixtures'])} fixtures \u00b7 {short_date()} \u00b7 {kickoff()}</div>"
        + f'<p class="fresh" id="fresh">Results up to {cutoff_date()}</p></div></aside>'
        '<div class="sbd" id="sbd"></div>'
        '<main id="main" tabindex="-1">'
        '<button class="menu" id="menuBtn" aria-controls="side" aria-expanded="false">\u2630 Menu</button>'
        f'<div id="view">{body}</div>'
        + f'<p class="foot">Chances are the project\'s own match ratings for {season()} \u2014 a read on the fixture, '
        "not betting odds and not a guarantee.<br>Form, table and head-to-head use real Premier League results "
        + f'up to {cutoff_date()}.</p></main>'
        '<div class="bd" id="bd"></div>'
        '<aside class="drawer" id="drawer" role="dialog" aria-modal="true" aria-label="Team summary"></aside>'
    )
    return html, f"{title} \u00b7 EPL Intelligence"
