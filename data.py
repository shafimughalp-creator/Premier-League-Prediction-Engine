"""DATA for the EPL Intelligence match centre.

Every value here is copied from the approved prototype (`reference/index.html`,
the `DATA` object in its script) — same shape, same numbers, nothing invented.
The Python backend can later replace this module with an API response of the
same shape; the render functions only read this structure.
"""

from __future__ import annotations


def _row(position, team, short, won, drawn, lost, gf, ga, points, form):
    return {
        "position": position,
        "team": team,
        "short": short,
        "played": 10,
        "won": won,
        "drawn": drawn,
        "lost": lost,
        "gf": gf,
        "ga": ga,
        "gd": gf - ga,
        "points": points,
        "form": list(form),
    }


def _fixture(home, away, home_chance, draw_chance, away_chance, pick, extra=None):
    fixture = {
        "id": "2026-11-21_" + home + "_" + away,
        "home": home,
        "away": away,
        "date": "2026-11-21",
        "time": "15:00",
        "chances": {"home": home_chance, "draw": draw_chance, "away": away_chance},
        "pick": pick,
        "rates": None,
        "headToHead": None,
    }
    if extra:
        fixture.update(extra)
    return fixture


# Club badge identity: primary colour, secondary colour, pattern key.
CLUB = {
    "ARS": ["#ef0107", "#fff", "chev"],
    "BHA": ["#0057b8", "#fff", "v"],
    "MCI": ["#6cabdd", "#1c2c5b", "band"],
    "LIV": ["#c8102e", "#f6eb61", "top"],
    "CHE": ["#034694", "#fff", "cross"],
    "EVE": ["#003399", "#fff", "diag"],
    "BOU": ["#da291c", "#111", "v"],
    "BRE": ["#e30613", "#fff", "v"],
    "LEE": ["#1d428a", "#ffcd00", "h"],
    "AVL": ["#670e36", "#95bfe5", "half"],
    "TOT": ["#132257", "#fff", "chev"],
    "NEW": ["#111", "#fff", "v"],
    "HUL": ["#f5a12d", "#111", "v"],
    "IPS": ["#0044aa", "#fff", "top"],
    "FUL": ["#f2f2f2", "#111", "half"],
    "NFO": ["#dd0000", "#fff", "diag"],
    "COV": ["#2aa3d6", "#1c5b84", "cross"],
    "MUN": ["#da291c", "#fbe122", "chev"],
    "SUN": ["#eb172b", "#fff", "half"],
    "CRY": ["#1b458f", "#c4122e", "v"],
}

# Official badges from the Premier League image host. Coventry has no id and
# always uses the built-in SVG badge.
_BADGE_IDS = {
    "ARS": 3,
    "AVL": 7,
    "BOU": 91,
    "BRE": 94,
    "BHA": 36,
    "CHE": 8,
    "CRY": 31,
    "EVE": 11,
    "FUL": 54,
    "HUL": 88,
    "IPS": 40,
    "LEE": 2,
    "LIV": 14,
    "MCI": 43,
    "MUN": 1,
    "NEW": 4,
    "NFO": 17,
    "SUN": 56,
    "TOT": 6,
}
LOGOS = {
    short: f"https://resources.premierleague.com/premierleague/badges/70/t{badge_id}.png"
    for short, badge_id in _BADGE_IDS.items()
}

DATA = {
    "meta": {
        "season": "2026/27",
        "matchweek": 11,
        "dataCutoff": "2026-11-08",
        "dateLong": "Saturday, 21 November 2026",
    },
    "seasonNumbers": {
        "matchesPlayed": 100,
        "goals": 273,
        "goalsPerGame": 2.73,
        "homeWins": 42,
        "draws": 27,
        "awayWins": 31,
        "bothScored": 49,
        "cleanSheets": 57,
    },
    "full": {"Man United": "Manchester United", "Man City": "Manchester City"},
    "table": [
        _row(1, "Arsenal", "ARS", 9, 0, 1, 15, 5, 27, "WWWWW"),
        _row(2, "Brighton", "BHA", 7, 1, 2, 26, 7, 22, "WWLWW"),
        _row(3, "Man City", "MCI", 7, 0, 3, 21, 13, 21, "LLLWW"),
        _row(4, "Liverpool", "LIV", 5, 3, 2, 14, 10, 18, "WLWWL"),
        _row(5, "Chelsea", "CHE", 5, 1, 4, 19, 16, 16, "LWLWW"),
        _row(6, "Everton", "EVE", 4, 4, 2, 12, 10, 16, "WWLLD"),
        _row(7, "Bournemouth", "BOU", 4, 3, 3, 16, 14, 15, "WWWWL"),
        _row(8, "Brentford", "BRE", 3, 5, 2, 16, 11, 14, "LWDLD"),
        _row(9, "Leeds", "LEE", 3, 5, 2, 10, 8, 14, "WLDDL"),
        _row(10, "Aston Villa", "AVL", 3, 4, 3, 14, 17, 13, "DDWWD"),
        _row(11, "Tottenham", "TOT", 3, 3, 4, 10, 11, 12, "LWWWD"),
        _row(12, "Newcastle", "NEW", 3, 3, 4, 13, 16, 12, "DLWLL"),
        _row(13, "Hull", "HUL", 2, 5, 3, 11, 11, 11, "LDDLD"),
        _row(14, "Ipswich", "IPS", 3, 2, 5, 15, 20, 11, "LDDLW"),
        _row(15, "Fulham", "FUL", 2, 4, 4, 10, 12, 10, "DDWWL"),
        _row(16, "Nott'm Forest", "NFO", 2, 3, 5, 9, 13, 9, "WLDLL"),
        _row(17, "Coventry", "COV", 3, 0, 7, 10, 19, 9, "LWLLW"),
        _row(18, "Man United", "MUN", 1, 5, 4, 14, 19, 8, "DLLDD"),
        _row(19, "Sunderland", "SUN", 2, 2, 6, 8, 17, 8, "WLDLL"),
        _row(20, "Crystal Palace", "CRY", 2, 1, 7, 10, 24, 7, "LLLLW"),
    ],
    "fixtures": [
        _fixture("Aston Villa", "Sunderland", 49, 27, 24, "home"),
        _fixture("Bournemouth", "Nott'm Forest", 46, 31, 23, "home"),
        _fixture("Brentford", "Everton", 40, 25, 35, "home"),
        _fixture("Chelsea", "Leeds", 42, 28, 30, "home"),
        _fixture("Coventry", "Crystal Palace", 49, 25, 26, "home"),
        _fixture("Hull", "Brighton", 19, 20, 61, "away"),
        _fixture(
            "Liverpool",
            "Man United",
            44,
            29,
            27,
            "home",
            {
                "rates": {
                    "homeForm5Ppg": 1.8,
                    "awayForm5Ppg": 0.6,
                    "homeHomePpg": 1,
                    "awayAwayPpg": 0.6,
                    "homeGf5": 1.4,
                    "homeGa5": 1.2,
                    "awayGf5": 1.2,
                    "awayGa5": 2.2,
                    "h2hPpg": 1,
                    "h2hLevelPct": 40,
                    "xg": [1.3, 1.1],
                },
                "headToHead": [
                    {
                        "date": "2026-05-03",
                        "home": "Man United",
                        "away": "Liverpool",
                        "homeGoals": 3,
                        "awayGoals": 2,
                        "season": "2025/26",
                    },
                    {
                        "date": "2025-10-19",
                        "home": "Liverpool",
                        "away": "Man United",
                        "homeGoals": 1,
                        "awayGoals": 2,
                        "season": "2025/26",
                    },
                    {
                        "date": "2025-01-05",
                        "home": "Liverpool",
                        "away": "Man United",
                        "homeGoals": 2,
                        "awayGoals": 2,
                        "season": "2024/25",
                    },
                    {
                        "date": "2024-09-01",
                        "home": "Man United",
                        "away": "Liverpool",
                        "homeGoals": 0,
                        "awayGoals": 3,
                        "season": "2024/25",
                    },
                    {
                        "date": "2024-04-07",
                        "home": "Man United",
                        "away": "Liverpool",
                        "homeGoals": 2,
                        "awayGoals": 2,
                        "season": "2023/24",
                    },
                ],
            },
        ),
        _fixture("Man City", "Fulham", 72, 16, 12, "home"),
        _fixture("Newcastle", "Arsenal", 17, 22, 61, "away"),
        _fixture("Tottenham", "Ipswich", 62, 20, 18, "home"),
    ],
    "club": CLUB,
    "logos": LOGOS,
}
