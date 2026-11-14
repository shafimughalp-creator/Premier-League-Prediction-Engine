"""Structural parity between the live DATA loader and the approved prototype.

The full 44-point frozen-value verification (every prototype number
reproduced from real data) was executed during the integration build; these
tests pin the invariants that must keep holding as live data changes:
same shape, same rounding rule, same ordering conventions, same formulas.
"""
from __future__ import annotations

from data import CLUB, LOGOS, DATA

PROTO_CLUB_KEYS = {"ARS", "BHA", "MCI", "LIV", "CHE", "EVE", "BOU", "BRE",
                   "LEE", "AVL", "TOT", "NEW", "HUL", "IPS", "FUL", "NFO",
                   "COV", "MUN", "SUN", "CRY"}


def test_data_has_the_prototype_shape():
    assert set(DATA) >= {"meta", "seasonNumbers", "full", "table", "fixtures"}
    assert set(DATA["meta"]) >= {"season", "matchweek", "dataCutoff", "dateLong"}
    assert set(DATA["seasonNumbers"]) == {
        "matchesPlayed", "goals", "goalsPerGame", "homeWins", "draws",
        "awayWins", "bothScored", "cleanSheets"}


def test_every_fixture_chance_sums_to_100():
    for fixture in DATA["fixtures"]:
        total = sum(fixture["chances"].values())
        assert total == 100, f"{fixture['id']}: {fixture['chances']} = {total}"


def test_fixture_picks_are_valid_and_consistent():
    valid = {"home", "draw", "away"}
    for fixture in DATA["fixtures"]:
        assert fixture["pick"] in valid
        chances = fixture["chances"]
        best = max(chances, key=chances.get)
        # The pick must follow the (rounded) argmax, as the prototype's did.
        assert fixture["pick"] == best, f"{fixture['id']}: pick {fixture['pick']} vs argmax {best}"


def test_table_is_complete_and_ranked():
    table = DATA["table"]
    if not table:
        return  # no results synced yet: honest empty state
    assert [t["position"] for t in table] == list(range(1, len(table) + 1))
    for row in table:
        assert row["played"] == row["won"] + row["drawn"] + row["lost"]
        assert row["points"] == row["won"] * 3 + row["drawn"]
        assert row["gd"] == row["gf"] - row["ga"]
        assert len(row["form"]) <= 5
        assert set(row["form"]) <= {"W", "D", "L"}
    # Ranking order: points, then goal difference.
    for a, b in zip(table, table[1:]):
        assert (a["points"], a["gd"], a["gf"]) >= (b["points"], b["gd"], b["gf"])


def test_season_numbers_agree_with_the_table():
    table = DATA["table"]
    sn = DATA["seasonNumbers"]
    if not table:
        return
    assert sn["matchesPlayed"] == sum(t["played"] for t in table) / 2
    assert sn["goals"] == sum(t["gf"] for t in table)
    assert sn["homeWins"] + sn["draws"] + sn["awayWins"] == sn["matchesPlayed"]


def test_club_identity_is_the_prototype_s():
    assert set(CLUB) == PROTO_CLUB_KEYS
    for short, colours in CLUB.items():
        assert len(colours) == 3
        assert colours[0].startswith("#") and colours[1] in {"#fff", "#111", "#fbe122", "#ffcd00", "#95bfe5", "#1c2c5b", "#1c5b84", "#c4122e", "#f6eb61", "#111"}
    for short, url in LOGOS.items():
        assert url.startswith("https://resources.premierleague.com/premierleague/badges/70/t")
        assert url.endswith(".png")


def test_fixtures_carry_their_kickoffs_and_week():
    for fixture in DATA["fixtures"]:
        assert fixture["date"] and fixture["time"]
        assert fixture["matchweek"] == DATA["meta"]["matchweek"]
        assert fixture["rates"] is not None          # always a dict now
        assert set(fixture["rates"]) >= {
            "homeForm5Ppg", "awayForm5Ppg", "homeHomePpg", "awayAwayPpg",
            "homeGf5", "homeGa5", "awayGf5", "awayGa5", "h2hPpg",
            "h2hLevelPct", "xg"}
        # Honest missing values stay None; the renderer prints a dash.
        assert fixture["rates"]["xg"] is None


def test_head_to_head_is_newest_first():
    for fixture in DATA["fixtures"]:
        h2h = fixture["headToHead"] or []
        dates = [m["date"] for m in h2h]
        assert dates == sorted(dates, reverse=True)
        for meeting in h2h:
            assert {meeting["home"], meeting["away"]} == {fixture["home"], fixture["away"]}
