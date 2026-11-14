"""Build the five HTML prototypes from one real data payload.

Reads existing project artifacts only (no retraining, no new predictions):
  data/predictions.json
  data/processed/matches.csv
  data/fixtures/upcoming.csv

Writes:
  prototypes/_data.json          shared payload (for inspection)
  prototypes/0X-*.html           one self-contained file per prototype

Each template contains the token /*__DATA__*/ inside a <script> tag; the build
replaces it with `window.EPL_DATA = {...};`.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import display_name, short_name  # noqa: E402

HERE = Path(__file__).resolve().parent
SRC = HERE / "_src"
SEASON = "2026/27"

COMPACT = {
    "Brighton & Hove Albion": "Brighton",
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Nottingham Forest": "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves",
}


def compact(name: str) -> str:
    return COMPACT.get(name, name)


def round_chances(home: float, draw: float, away: float) -> dict:
    raw = [home * 100, draw * 100, away * 100]
    floors = [int(x) for x in raw]
    for i in sorted(range(3), key=lambda i: raw[i] - floors[i], reverse=True)[: 100 - sum(floors)]:
        floors[i] += 1
    return {"home": floors[0], "draw": floors[1], "away": floors[2]}


def venue_record(rows: pd.DataFrame, team: str, venue: str) -> dict:
    if venue == "home":
        sub = rows[rows["home_team"] == team]
        gf, ga = sub["home_goals"], sub["away_goals"]
    else:
        sub = rows[rows["away_team"] == team]
        gf, ga = sub["away_goals"], sub["home_goals"]
    won = int((gf > ga).sum())
    drawn = int((gf == ga).sum())
    lost = int((gf < ga).sum())
    return {
        "played": int(len(sub)),
        "won": won,
        "drawn": drawn,
        "lost": lost,
        "gf": int(gf.sum()),
        "ga": int(ga.sum()),
        "points": won * 3 + drawn,
    }


def form(rows: pd.DataFrame, team: str, limit: int = 5) -> list[dict]:
    sub = rows[(rows["home_team"] == team) | (rows["away_team"] == team)].sort_values(["date", "match_id"])
    out = []
    for _, r in sub.tail(limit).iloc[::-1].iterrows():
        is_home = r["home_team"] == team
        gf = int(r["home_goals"] if is_home else r["away_goals"])
        ga = int(r["away_goals"] if is_home else r["home_goals"])
        opp = r["away_team"] if is_home else r["home_team"]
        out.append(
            {
                "date": str(r["date"])[:10],
                "opponent": opp,
                "opponent_display": compact(display_name(opp)),
                "venue": "H" if is_home else "A",
                "gf": gf,
                "ga": ga,
                "result": "W" if gf > ga else ("D" if gf == ga else "L"),
            }
        )
    return out


def last5_points(rows: list[dict]) -> int:
    return sum(3 if r["result"] == "W" else 1 if r["result"] == "D" else 0 for r in rows)


def main() -> int:
    preds = json.loads((ROOT / "data/predictions.json").read_text(encoding="utf-8"))
    matches = pd.read_csv(ROOT / "data/processed/matches.csv", parse_dates=["date"])
    fixtures = pd.read_csv(ROOT / "data/fixtures/upcoming.csv")

    season_rows = matches[matches["season"] == SEASON].sort_values(["date", "match_id"])

    agg: dict[str, dict] = defaultdict(lambda: {"won": 0, "drawn": 0, "lost": 0, "gf": 0, "ga": 0, "played": 0})
    for _, r in season_rows.iterrows():
        h, a, hg, ag = r["home_team"], r["away_team"], int(r["home_goals"]), int(r["away_goals"])
        for t in (h, a):
            agg[t]["played"] += 1
        agg[h]["gf"] += hg
        agg[h]["ga"] += ag
        agg[a]["gf"] += ag
        agg[a]["ga"] += hg
        if hg > ag:
            agg[h]["won"] += 1
            agg[a]["lost"] += 1
        elif hg < ag:
            agg[a]["won"] += 1
            agg[h]["lost"] += 1
        else:
            agg[h]["drawn"] += 1
            agg[a]["drawn"] += 1

    table = []
    for team, s in agg.items():
        table.append(
            {
                "team": team,
                "display": compact(display_name(team)),
                "full": display_name(team),
                "short": short_name(team),
                "played": s["played"],
                "won": s["won"],
                "drawn": s["drawn"],
                "lost": s["lost"],
                "gf": s["gf"],
                "ga": s["ga"],
                "gd": s["gf"] - s["ga"],
                "points": s["won"] * 3 + s["drawn"],
            }
        )
    table.sort(key=lambda t: (-t["points"], -t["gd"], -t["gf"], t["team"]))
    for i, row in enumerate(table, start=1):
        row["position"] = i
        row["form"] = form(season_rows, row["team"])
        row["last5Points"] = last5_points(row["form"])
        row["home"] = venue_record(season_rows, row["team"], "home")
        row["away"] = venue_record(season_rows, row["team"], "away")

    by_team = {t["team"]: t for t in table}

    total_goals = int(season_rows["home_goals"].sum() + season_rows["away_goals"].sum())
    played = int(len(season_rows))
    season_numbers = {
        "matchesPlayed": played,
        "goals": total_goals,
        "goalsPerGame": round(total_goals / played, 2) if played else 0,
        "homeWins": int((season_rows["result"] == "H").sum()),
        "draws": int((season_rows["result"] == "D").sum()),
        "awayWins": int((season_rows["result"] == "A").sum()),
        "bothScored": int(((season_rows["home_goals"] > 0) & (season_rows["away_goals"] > 0)).sum()),
        "cleanSheets": int((season_rows["home_goals"] == 0).sum() + (season_rows["away_goals"] == 0).sum()),
    }

    kickoff = {
        (r["home_team"], r["away_team"]): (str(r["date"])[:10], r["time"])
        for _, r in fixtures[fixtures["season"] == SEASON].iterrows()
    }

    out_fixtures = []
    for f in preds["fixtures"]:
        home, away = f["home_team"], f["away_team"]
        ctx = f.get("context") or {}
        chances = round_chances(f["prob_home"], f["prob_draw"], f["prob_away"])
        pick = max(chances, key=chances.get)
        date, time = kickoff.get((home, away), (str(f.get("date", ""))[:10], f.get("time", "")))

        h2h_rows = matches[
            ((matches["home_team"] == home) & (matches["away_team"] == away))
            | ((matches["home_team"] == away) & (matches["away_team"] == home))
        ].sort_values(["date", "match_id"])
        meetings = []
        for _, r in h2h_rows.tail(5).iloc[::-1].iterrows():
            meetings.append(
                {
                    "date": str(r["date"])[:10],
                    "season": r["season"],
                    "home": compact(display_name(r["home_team"])),
                    "away": compact(display_name(r["away_team"])),
                    "homeGoals": int(r["home_goals"]),
                    "awayGoals": int(r["away_goals"]),
                }
            )

        out_fixtures.append(
            {
                "id": f["match_id"],
                "date": date,
                "time": time,
                "matchweek": int(f.get("matchweek") or 11),
                "home": home,
                "away": away,
                "homeDisplay": compact(display_name(home)),
                "awayDisplay": compact(display_name(away)),
                "homeFull": display_name(home),
                "awayFull": display_name(away),
                "homeShort": short_name(home),
                "awayShort": short_name(away),
                "chances": chances,
                "pick": pick,
                "rates": {
                    "homeForm5Ppg": ctx.get("home_form5_ppg"),
                    "awayForm5Ppg": ctx.get("away_form5_ppg"),
                    "homeVenue5Ppg": ctx.get("home_venue5_ppg"),
                    "awayVenue5Ppg": ctx.get("away_venue5_ppg"),
                    "homeForm5Gf": ctx.get("home_form5_gf"),
                    "awayForm5Gf": ctx.get("away_form5_gf"),
                    "homeForm5Ga": ctx.get("home_form5_ga"),
                    "awayForm5Ga": ctx.get("away_form5_ga"),
                    "h2hHomePpg": ctx.get("h2h_ppg"),
                    "h2hDrawRate": ctx.get("h2h_draw_rate"),
                    "expectHomeGoals": ctx.get("expected_goals_home"),
                    "expectAwayGoals": ctx.get("expected_goals_away"),
                },
                "headToHead": meetings,
            }
        )
    out_fixtures.sort(key=lambda x: (x["date"], x["time"], x["home"]))

    payload = {
        "meta": {
            "season": SEASON,
            "matchweek": 11,
            "fixtureDate": "2026-11-21",
            "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataCutoff": preds.get("data_cutoff"),
            "chancesUpdated": preds.get("generated_at"),
            "note": "Chances come from the project's own match ratings for 2026/27.",
        },
        "seasonNumbers": season_numbers,
        "table": table,
        "fixtures": out_fixtures,
    }

    (HERE / "_data.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    blob = "window.EPL_DATA = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";"
    built = 0
    for template in sorted(SRC.glob("*.html")):
        html = template.read_text(encoding="utf-8")
        if "/*__DATA__*/" not in html:
            print(f"skip {template.name}: no data token")
            continue
        out = HERE / template.name
        out.write_text(html.replace("/*__DATA__*/", blob), encoding="utf-8")
        built += 1
        print(f"built {out.name} ({out.stat().st_size // 1024} KB)")

    print(f"teams={len(table)} fixtures={len(out_fixtures)} matches={played} built={built}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
