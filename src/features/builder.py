"""Leakage-safe pre-match feature engineering.

The builder is a chronological state machine. For every date D it first
featurises all matches on D using state that contains only matches from dates
strictly before D, then updates the state with D's results. A feature vector can
therefore never contain its own result, a concurrent result, or any future
information.

`src/data/validation.py` proves this empirically.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd

from src import config

EPS = 1e-9
MAX_GOALS = 10


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _new_state() -> dict:
    return {
        "elo": config.ELO_START,
        "history": [],
        "season": None,
        "season_games": 0,
        "season_points": 0,
        "season_gf": 0,
        "season_ga": 0,
        "last_date": None,
        "last_season": None,
        "promoted": 0,
        "prev_ppg": np.nan,
        "prev_gd_pg": np.nan,
        "prev_gf_pg": np.nan,
        "prev_ga_pg": np.nan,
    }


def _shrunk(total: float, n: int, prior: float, strength: float = 4.0) -> float:
    return (total + strength * prior) / (n + strength)


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _window(history: list[dict], n: int, home_only: bool | None = None) -> list[dict]:
    rows = history
    if home_only is True:
        rows = [m for m in rows if m["home"]]
    elif home_only is False:
        rows = [m for m in rows if not m["home"]]
    return rows[-n:]


def _form_stats(history: list[dict], n: int, home_only: bool | None = None) -> dict:
    rows = _window(history, n, home_only)
    if not rows:
        return {
            "ppg": np.nan, "win_rate": np.nan, "draw_rate": np.nan, "loss_rate": np.nan,
            "gf": np.nan, "ga": np.nan, "gd": np.nan, "shots": np.nan, "sot": np.nan,
            "corners": np.nan, "fouls": np.nan, "cards": np.nan, "over25": np.nan,
            "btts": np.nan, "clean_sheet": np.nan, "failed_score": np.nan,
        }
    points = [m["points"] for m in rows]
    gf = [m["gf"] for m in rows]
    ga = [m["ga"] for m in rows]
    shots = [m.get("shots", np.nan) for m in rows]
    sot = [m.get("sot", np.nan) for m in rows]
    corners = [m.get("corners", np.nan) for m in rows]
    fouls = [m.get("fouls", np.nan) for m in rows]
    cards = [m.get("cards", np.nan) for m in rows]
    return {
        "ppg": _mean(points),
        "win_rate": _mean([1.0 if p == 3 else 0.0 for p in points]),
        "draw_rate": _mean([1.0 if p == 1 else 0.0 for p in points]),
        "loss_rate": _mean([1.0 if p == 0 else 0.0 for p in points]),
        "gf": _mean(gf),
        "ga": _mean(ga),
        "gd": _mean([a - b for a, b in zip(gf, ga)]),
        "shots": _mean(shots),
        "sot": _mean(sot),
        "corners": _mean(corners),
        "fouls": _mean(fouls),
        "cards": _mean(cards),
        "over25": _mean([1.0 if a + b > 2.5 else 0.0 for a, b in zip(gf, ga)]),
        "btts": _mean([1.0 if a > 0 and b > 0 else 0.0 for a, b in zip(gf, ga)]),
        "clean_sheet": _mean([1.0 if b == 0 else 0.0 for b in ga]),
        "failed_score": _mean([1.0 if a == 0 else 0.0 for a in gf]),
    }


def _season_stats(state: dict) -> dict:
    n = state["season_games"]
    if n == 0:
        return {"ppg": np.nan, "gd_pg": np.nan, "gf_pg": np.nan, "ga_pg": np.nan}
    return {
        "ppg": state["season_points"] / n,
        "gd_pg": (state["season_gf"] - state["season_ga"]) / n,
        "gf_pg": state["season_gf"] / n,
        "ga_pg": state["season_ga"] / n,
    }


def _rest_and_congestion(state: dict, date: pd.Timestamp) -> tuple[float, float, float]:
    if state["last_date"] is None:
        return np.nan, np.nan, np.nan
    rest = (date - state["last_date"]).days
    recent = [m for m in state["history"] if (date - m["date"]).days <= 28]
    last14 = sum(1 for m in recent if (date - m["date"]).days <= 14)
    return float(rest), float(len(recent)), float(last14)


def _poisson_1x2(expected_home: float, expected_away: float) -> tuple[float, float, float]:
    if not np.isfinite(expected_home) or not np.isfinite(expected_away):
        return np.nan, np.nan, np.nan
    expected_home = float(np.clip(expected_home, 0.05, 6.0))
    expected_away = float(np.clip(expected_away, 0.05, 6.0))
    home_pmf = np.array([math.exp(-expected_home) * expected_home ** k / math.factorial(k)
                         for k in range(MAX_GOALS + 1)])
    away_pmf = np.array([math.exp(-expected_away) * expected_away ** k / math.factorial(k)
                         for k in range(MAX_GOALS + 1)])
    matrix = np.outer(home_pmf, away_pmf)
    home = float(np.tril(matrix, -1).sum())
    draw = float(np.trace(matrix))
    away = float(np.triu(matrix, 1).sum())
    total = home + draw + away
    return home / total, draw / total, away / total


def _market_features(row: pd.Series) -> dict:
    out = {}
    for col in ("mkt_home", "mkt_draw", "mkt_away", "mkt_overround", "mkt_disp_home",
                "mkt_disp_draw", "mkt_disp_away", "mkt_books", "mkt_balance",
                "mkt_favourite", "mkt_entropy", "odds_home", "odds_draw", "odds_away"):
        out[col] = float(row[col]) if col in row and pd.notna(row[col]) else np.nan
    return out


# --------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------
class FeatureBuilder:
    def __init__(self) -> None:
        self.states: dict[str, dict] = defaultdict(_new_state)
        self.h2h: dict[tuple[str, str], list[dict]] = defaultdict(list)
        self.league_home_goals = 20 * 1.55
        self.league_away_goals = 20 * 1.25
        self.league_matches = 20
        self.first_season = config.SEASONS[next(iter(config.SEASONS))]

    # ---- pre-match snapshot -------------------------------------------------
    def _team_block(self, team: str, opponent: str, date: pd.Timestamp,
                    season: str, is_home: bool) -> dict:
        state = self.states[team]
        prefix = "h" if is_home else "a"
        out: dict[str, float] = {}

        out[f"{prefix}_elo"] = state["elo"]
        out[f"{prefix}_promoted"] = float(state["promoted"])
        out[f"{prefix}_prev_ppg"] = state["prev_ppg"]
        out[f"{prefix}_prev_gd_pg"] = state["prev_gd_pg"]
        out[f"{prefix}_prev_gf_pg"] = state["prev_gf_pg"]
        out[f"{prefix}_prev_ga_pg"] = state["prev_ga_pg"]
        out[f"{prefix}_season_game"] = float(state["season_games"] + 1)
        out[f"{prefix}_season_progress"] = float((state["season_games"] + 1) / 38.0)
        rest, congestion, last14 = _rest_and_congestion(state, date)
        out[f"{prefix}_rest_days"] = rest
        out[f"{prefix}_matches_28d"] = congestion
        out[f"{prefix}_matches_14d"] = last14

        for n in config.ROLL_WINDOWS:
            stats = _form_stats(state["history"], n)
            for key, value in stats.items():
                out[f"{prefix}_form{n}_{key}"] = value

        # Home form for the home side, away form for the away side.
        side_stats = _form_stats(state["history"], 5, home_only=is_home)
        for key, value in side_stats.items():
            out[f"{prefix}_venue5_{key}"] = value

        season_stats = _season_stats(state)
        for key, value in season_stats.items():
            out[f"{prefix}_season_{key}"] = value

        # Shrunk attacking / defensive ratings (goals per match vs league prior).
        home_games = [m for m in state["history"] if m["home"]]
        away_games = [m for m in state["history"] if not m["home"]]
        lg_home = self.league_home_goals / max(self.league_matches, 1)
        lg_away = self.league_away_goals / max(self.league_matches, 1)
        att_home = _shrunk(sum(m["gf"] for m in home_games), len(home_games), lg_home)
        def_home = _shrunk(sum(m["ga"] for m in home_games), len(home_games), lg_away)
        att_away = _shrunk(sum(m["gf"] for m in away_games), len(away_games), lg_away)
        def_away = _shrunk(sum(m["ga"] for m in away_games), len(away_games), lg_home)
        out[f"{prefix}_attack_home"] = att_home / max(lg_home, EPS)
        out[f"{prefix}_defence_home"] = def_home / max(lg_away, EPS)
        out[f"{prefix}_attack_away"] = att_away / max(lg_away, EPS)
        out[f"{prefix}_defence_away"] = def_away / max(lg_home, EPS)
        return out

    def _features_for(self, row: pd.Series) -> dict:
        date = pd.Timestamp(row["date"])
        season = str(row["season_code"])
        home, away = row["home_team"], row["away_team"]
        hs, as_ = self.states[home], self.states[away]

        # Season transition: regression to the mean, reset seasonal counters.
        for team, state in ((home, hs), (away, as_)):
            if state["last_season"] is not None and state["last_season"] != season:
                if state["season_games"] > 0:
                    state["prev_ppg"] = state["season_points"] / state["season_games"]
                    state["prev_gd_pg"] = (state["season_gf"] - state["season_ga"]) / state["season_games"]
                    state["prev_gf_pg"] = state["season_gf"] / state["season_games"]
                    state["prev_ga_pg"] = state["season_ga"] / state["season_games"]
                else:
                    state["prev_ppg"] = np.nan
                    state["prev_gd_pg"] = np.nan
                    state["prev_gf_pg"] = np.nan
                    state["prev_ga_pg"] = np.nan
                if state["last_season"] != _previous_season(season):
                    state["prev_ppg"] = np.nan
                    state["prev_gd_pg"] = np.nan
                    state["prev_gf_pg"] = np.nan
                    state["prev_ga_pg"] = np.nan
                state["elo"] += (config.ELO_START - state["elo"]) * config.ELO_SEASON_REGRESSION
                state["season_games"] = 0
                state["season_points"] = 0
                state["season_gf"] = 0
                state["season_ga"] = 0
                state["promoted"] = 0 if state["last_season"] == _previous_season(season) else 1
            elif state["last_season"] is None:
                state["promoted"] = 0 if season == self.first_season else 1
                state["prev_ppg"] = np.nan
                state["prev_gd_pg"] = np.nan
                state["prev_gf_pg"] = np.nan
                state["prev_ga_pg"] = np.nan
            state["last_season"] = season

        feat: dict[str, float] = {}
        feat.update(self._team_block(home, away, date, season, True))
        feat.update(self._team_block(away, home, date, season, False))

        feat["elo_diff"] = feat["h_elo"] - feat["a_elo"]
        feat["abs_elo_diff"] = abs(feat["elo_diff"])
        feat["elo_closeness"] = 1.0 / (1.0 + abs(feat["elo_diff"]) / 100.0)
        feat["elo_expected_home"] = 1.0 / (
            1.0 + 10 ** (-((feat["h_elo"] + config.ELO_HOME_ADVANTAGE - feat["a_elo"]) / 400.0))
        )
        feat["elo_expected_away"] = 1.0 - feat["elo_expected_home"]
        feat["diff_prev_ppg"] = feat["h_prev_ppg"] - feat["a_prev_ppg"]
        feat["diff_prev_gd_pg"] = feat["h_prev_gd_pg"] - feat["a_prev_gd_pg"]

        for n in config.ROLL_WINDOWS:
            for key in ("ppg", "gf", "ga", "gd", "win_rate", "draw_rate", "loss_rate",
                        "sot", "shots", "corners"):
                feat[f"diff_form{n}_{key}"] = (
                    feat[f"h_form{n}_{key}"] - feat[f"a_form{n}_{key}"])
        for key in ("ppg", "gf", "ga", "gd", "win_rate", "clean_sheet", "over25"):
            feat[f"diff_venue5_{key}"] = feat[f"h_venue5_{key}"] - feat[f"a_venue5_{key}"]
        feat["diff_season_ppg"] = feat["h_season_ppg"] - feat["a_season_ppg"]
        feat["diff_season_gd_pg"] = feat["h_season_gd_pg"] - feat["a_season_gd_pg"]

        # Poisson expected goals from venue-specific strength ratings.
        lg_home = self.league_home_goals / max(self.league_matches, 1)
        lg_away = self.league_away_goals / max(self.league_matches, 1)
        exp_home = feat["h_attack_home"] * feat["a_defence_away"] * lg_home
        exp_away = feat["a_attack_away"] * feat["h_defence_home"] * lg_away
        feat["pois_exp_home"] = exp_home
        feat["pois_exp_away"] = exp_away
        feat["pois_exp_total"] = exp_home + exp_away
        feat["pois_exp_diff"] = exp_home - exp_away
        p_home, p_draw, p_away = _poisson_1x2(exp_home, exp_away)
        feat["pois_p_home"], feat["pois_p_draw"], feat["pois_p_away"] = p_home, p_draw, p_away

        # Head-to-head, from the current home team's perspective.
        meetings = self.h2h[tuple(sorted((home, away)))][-config.H2H_WINDOW:]
        if meetings:
            points = []
            totals = []
            for m in meetings:
                if m["home"] == home:
                    points.append(3 if m["hg"] > m["ag"] else 1 if m["hg"] == m["ag"] else 0)
                else:
                    points.append(3 if m["ag"] > m["hg"] else 1 if m["ag"] == m["hg"] else 0)
                totals.append(m["hg"] + m["ag"])
            feat["h2h_ppg"] = float(np.mean(points))
            feat["h2h_draw_rate"] = float(np.mean([1.0 if m["hg"] == m["ag"] else 0.0
                                                   for m in meetings]))
            feat["h2h_avg_total"] = float(np.mean(totals))
            feat["h2h_n"] = float(len(meetings))
        else:
            feat["h2h_ppg"] = np.nan
            feat["h2h_draw_rate"] = np.nan
            feat["h2h_avg_total"] = np.nan
            feat["h2h_n"] = 0.0

        feat.update(_market_features(row))
        return feat

    # ---- state update -------------------------------------------------------
    def _update(self, row: pd.Series) -> None:
        date = pd.Timestamp(row["date"])
        home, away = row["home_team"], row["away_team"]
        hg, ag = int(row["home_goals"]), int(row["away_goals"])
        result = row["result"]
        hs, as_ = self.states[home], self.states[away]

        expected_home = 1.0 / (
            1.0 + 10 ** (-((hs["elo"] + config.ELO_HOME_ADVANTAGE - as_["elo"]) / 400.0))
        )
        actual_home = 1.0 if result == "H" else 0.5 if result == "D" else 0.0
        goal_multiplier = 1.0 + 0.5 * max(0, abs(hg - ag) - 1)
        delta = config.ELO_K * goal_multiplier * (actual_home - expected_home)
        hs["elo"] += delta
        as_["elo"] -= delta

        home_points = 3 if result == "H" else 1 if result == "D" else 0
        away_points = 3 if result == "A" else 1 if result == "D" else 0

        for team, state, gf, ga, points, is_home, shots, sot, corners, fouls, cards in (
            (home, hs, hg, ag, home_points, True, row.get("home_shots"), row.get("home_sot"),
             row.get("home_corners"), row.get("home_fouls"),
             (row.get("home_yellows", 0) or 0) + (row.get("home_reds", 0) or 0)),
            (away, as_, ag, hg, away_points, False, row.get("away_shots"), row.get("away_sot"),
             row.get("away_corners"), row.get("away_fouls"),
             (row.get("away_yellows", 0) or 0) + (row.get("away_reds", 0) or 0)),
        ):
            state["history"].append({
                "date": date, "gf": gf, "ga": ga, "points": points, "home": is_home,
                "shots": shots, "sot": sot, "corners": corners, "fouls": fouls, "cards": cards,
            })
            state["history"] = state["history"][-25:]
            state["season_games"] += 1
            state["season_points"] += points
            state["season_gf"] += gf
            state["season_ga"] += ga
            state["last_date"] = date

        self.h2h[tuple(sorted((home, away)))].append({
            "date": date, "home": home, "away": away, "hg": hg, "ag": ag,
        })
        self.h2h[tuple(sorted((home, away)))] = self.h2h[tuple(sorted((home, away)))][-10:]

        self.league_home_goals += hg
        self.league_away_goals += ag
        self.league_matches += 1

    # ---- public API ---------------------------------------------------------
    def build(self, matches: pd.DataFrame) -> pd.DataFrame:
        matches = matches.sort_values(["date", "time", "home_team"], kind="mergesort")
        records = []
        for date, group in matches.groupby("date", sort=True):
            group = group.sort_values(["time", "home_team"], kind="mergesort")
            built = []
            for _, row in group.iterrows():
                feat = self._features_for(row)
                feat.update({
                    "_date": pd.Timestamp(row["date"]),
                    "_time": row.get("time", "15:00"),
                    "_season": row["season"],
                    "_season_code": str(row["season_code"]),
                    "_home_team": row["home_team"],
                    "_away_team": row["away_team"],
                    "_label": config.LABEL_TO_INDEX[row["result"]],
                    "_result": row["result"],
                    "_home_goals": int(row["home_goals"]),
                    "_away_goals": int(row["away_goals"]),
                })
                built.append(feat)
            records.extend(built)
            # Update only after every match on this date has been featurised.
            for _, row in group.iterrows():
                self._update(row)
        return pd.DataFrame(records).reset_index(drop=True)

    def build_fixtures(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        if fixtures is None or len(fixtures) == 0:
            return pd.DataFrame()
        rows = []
        for _, row in fixtures.sort_values(["date", "time"]).iterrows():
            date = pd.Timestamp(row["date"])
            season = config.CURRENT_SEASON
            # Treat a fixture as belonging to the season of its date.
            season_code = _season_code_for_date(date)
            feat = self._features_for(pd.Series({
                "date": date, "season_code": season_code, "home_team": row["home_team"],
                "away_team": row["away_team"], **{k: row.get(k, np.nan) for k in (
                    "mkt_home", "mkt_draw", "mkt_away", "mkt_overround", "mkt_disp_home",
                    "mkt_disp_draw", "mkt_disp_away", "mkt_books", "mkt_balance",
                    "mkt_favourite", "mkt_entropy", "odds_home", "odds_draw", "odds_away")},
            }))
            feat.update({
                "_date": date,
                "_time": row.get("time", "15:00"),
                "_season": config.SEASONS.get(season_code, season),
                "_season_code": season_code,
                "_home_team": row["home_team"],
                "_away_team": row["away_team"],
                "_source": row.get("source", ""),
                "matchweek": row.get("matchweek", np.nan),
            })
            rows.append(feat)
        return pd.DataFrame(rows).reset_index(drop=True)


def _previous_season(season_code: str) -> str | None:
    codes = list(config.SEASONS)
    try:
        idx = codes.index(str(season_code))
    except ValueError:
        return None
    return codes[idx - 1] if idx > 0 else None


def _season_code_for_date(date: pd.Timestamp) -> str:
    start = date.year if date.month >= 7 else date.year - 1
    for code, label in config.SEASONS.items():
        if label == f"{start}/{str(start + 1)[-2:]}":
            return code
    return config.CURRENT_SEASON


# --------------------------------------------------------------------------
# Feature columns
# --------------------------------------------------------------------------
NON_FEATURE = {
    "_date", "_time", "_season", "_season_code", "_home_team", "_away_team",
    "_label", "_result", "_home_goals", "_away_goals", "_source", "matchweek",
}


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric feature columns, stable and sorted."""
    cols = []
    for col in df.columns:
        if col in NON_FEATURE:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) or df[col].dtype == object:
            try:
                pd.to_numeric(df[col])
                cols.append(col)
            except (TypeError, ValueError):
                continue
    return sorted(cols)


def odds_free_columns(columns: list[str]) -> list[str]:
    return [c for c in columns if not c.startswith(("mkt_", "odds_"))]
