"""Data integrity checks and empirical leakage proofs.

Two independent guarantees are produced here:

1. **Integrity** - the canonical match table is internally consistent
   (unique fixtures, valid scores, coherent result codes, known teams).

2. **No target leakage** - proven by pertubation, not by assertion:

   * *Own-result test*: flipping the score of match *i* must not change any
     feature of match *i*. This rules out the target and any same-match
     post-match statistic entering the feature vector.
   * *Future test*: truncating the dataset at date *D* must leave the features
     of every match played on *D* byte-identical. This rules out any
     information from later dates.

Both proofs rebuild the feature matrix from scratch, so they exercise the real
production code path rather than a reimplementation.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src import config
from src.features.builder import FeatureBuilder, feature_columns

log = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "match_id", "date", "home_team", "away_team", "home_goals", "away_goals", "result",
]
LEAKAGE_JSON = config.QUALITY_DIR / "leakage_report.json"


# --------------------------------------------------------------------------
# Integrity
# --------------------------------------------------------------------------
def validate_matches(matches: pd.DataFrame) -> dict:
    issues: list[str] = []
    checks: dict[str, bool] = {}

    missing = [c for c in REQUIRED_COLUMNS if c not in matches.columns]
    checks["required_columns"] = not missing
    if missing:
        issues.append(f"missing columns: {missing}")

    if missing:
        return {"ok": False, "checks": checks, "issues": issues}

    dates = pd.to_datetime(matches["date"], errors="coerce")
    checks["valid_dates"] = bool(dates.notna().all())
    if not checks["valid_dates"]:
        issues.append(f"{int(dates.isna().sum())} unparseable dates")

    checks["distinct_teams"] = bool((matches["home_team"] != matches["away_team"]).all())
    if not checks["distinct_teams"]:
        issues.append("a team is listed against itself")

    goals_ok = ((matches["home_goals"] >= 0) & (matches["away_goals"] >= 0)
                & (matches["home_goals"] <= 15) & (matches["away_goals"] <= 15))
    checks["goal_range"] = bool(goals_ok.all())
    if not checks["goal_range"]:
        issues.append(f"{int((~goals_ok).sum())} implausible scorelines")

    expected = np.where(matches["home_goals"] > matches["away_goals"], "H",
                        np.where(matches["home_goals"] < matches["away_goals"], "A", "D"))
    coherent = np.asarray(matches["result"]) == expected
    checks["result_matches_score"] = bool(coherent.all())
    if not checks["result_matches_score"]:
        issues.append(f"{int((~coherent).sum())} result codes disagree with the score")

    duplicates = int(matches.duplicated(["date", "home_team", "away_team"]).sum())
    checks["no_duplicate_fixtures"] = duplicates == 0
    if duplicates:
        issues.append(f"{duplicates} duplicate fixtures")

    checks["result_codes"] = bool(matches["result"].isin(["H", "D", "A"]).all())
    if not checks["result_codes"]:
        issues.append("result column contains values outside H/D/A")

    monotonic = dates.is_monotonic_increasing
    checks["sorted_chronologically"] = bool(monotonic)

    teams = set(matches["home_team"]) | set(matches["away_team"])
    checks["team_names_canonical"] = bool(
        all(t == config.TEAM_ALIASES.get(t.lower(), t) or t in config.TEAM_ALIASES.values()
            for t in teams))
    if not checks["team_names_canonical"]:
        issues.append("some team names are not canonical")

    return {
        "ok": all(checks.values()),
        "checks": checks,
        "issues": issues,
        "n_matches": int(len(matches)),
        "n_teams": int(len(teams)),
        "first_date": str(dates.min().date()) if dates.notna().any() else None,
        "last_date": str(dates.max().date()) if dates.notna().any() else None,
    }


def validate_fixtures(fixtures: pd.DataFrame, matches: pd.DataFrame) -> dict:
    issues: list[str] = []
    checks: dict[str, bool] = {}
    if fixtures is None or len(fixtures) == 0:
        return {"ok": True, "checks": {"has_fixtures": False},
                "issues": ["no upcoming fixtures resolved"], "n": 0}

    known = set(matches["home_team"]) | set(matches["away_team"])
    checks["known_teams"] = bool(set(fixtures["home_team"]) | set(fixtures["away_team"]) <= known)
    if not checks["known_teams"]:
        unknown = (set(fixtures["home_team"]) | set(fixtures["away_team"])) - known
        issues.append(f"unknown teams: {sorted(unknown)}")

    checks["distinct_teams"] = bool((fixtures["home_team"] != fixtures["away_team"]).all())
    if not checks["distinct_teams"]:
        issues.append("fixture with identical home and away team")

    cutoff = pd.Timestamp(matches["date"].max())
    dates = pd.to_datetime(fixtures["date"])
    checks["in_the_future"] = bool((dates > cutoff).all())
    if not checks["in_the_future"]:
        issues.append("some fixtures are not after the last completed match")

    duplicates = int(fixtures.duplicated(["date", "home_team", "away_team"]).sum())
    checks["no_duplicates"] = duplicates == 0
    if duplicates:
        issues.append(f"{duplicates} duplicate fixtures")

    return {"ok": all(checks.values()), "checks": checks, "issues": issues, "n": int(len(fixtures))}


# --------------------------------------------------------------------------
# Leakage proofs
# --------------------------------------------------------------------------
def _fingerprint(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return np.round(frame[columns].to_numpy(dtype=float), 10)


def _features_up_to(matches: pd.DataFrame, last_date: pd.Timestamp) -> pd.DataFrame:
    subset = matches[matches["date"] <= last_date]
    return FeatureBuilder().build(subset)


def own_result_test(matches: pd.DataFrame, sample_match_ids: list[str] | None = None) -> dict:
    """Flip a match's own result; every feature of that match must stay identical."""
    matches = matches.sort_values(["date", "time"], kind="mergesort").reset_index(drop=True)
    candidates = matches.index
    if sample_match_ids is None:
        picks = [candidates[0], candidates[len(candidates) // 2], candidates[-1]]
    else:
        picks = [int(matches.index[matches["match_id"] == mid][0]) for mid in sample_match_ids]

    results = []
    for idx in picks:
        row = matches.loc[idx]
        subset = matches[matches["date"] <= row["date"]].reset_index(drop=True)
        position = int(subset.index[(subset["match_id"] == row["match_id"])][0])

        baseline = FeatureBuilder().build(subset)
        columns = [c for c in feature_columns(baseline) if c in baseline.columns]
        before = _fingerprint(baseline.iloc[[position]], columns)

        tampered = subset.copy()
        home_goals, away_goals = int(row["home_goals"]), int(row["away_goals"])
        # Choose a scoreline that flips the outcome.
        if home_goals > away_goals:
            tampered.loc[position, ["home_goals", "away_goals", "result"]] = [0, 3, "A"]
        elif home_goals < away_goals:
            tampered.loc[position, ["home_goals", "away_goals", "result"]] = [3, 0, "H"]
        else:
            tampered.loc[position, ["home_goals", "away_goals", "result"]] = [4, 0, "H"]

        after_features = FeatureBuilder().build(tampered)
        after = _fingerprint(after_features.iloc[[position]], columns)

        both_nan = np.isnan(before) & np.isnan(after)
        equal = bool(np.all((before == after) | both_nan))
        results.append({
            "match_id": str(row["match_id"]),
            "date": str(row["date"].date()),
            "n_features": len(columns),
            "identical": equal,
        })
        if not equal:
            diff = np.where(~((before == after) | both_nan))[1]
            results[-1]["changed_features"] = [columns[i] for i in diff[:10]]

    return {
        "test": "own_result_independence",
        "passed": all(r["identical"] for r in results),
        "cases": results,
    }


def future_independence_test(matches: pd.DataFrame, dates: list[str] | None = None) -> dict:
    """Truncating later data must not change features of an earlier date."""
    matches = matches.sort_values(["date", "time"], kind="mergesort").reset_index(drop=True)
    if dates is None:
        season_matches = matches[matches["season"] == "2019/20"]
        target = pd.Timestamp(season_matches["date"].iloc[len(season_matches) // 2])
        dates = [str(target.date())]

    full = FeatureBuilder().build(matches)
    cases = []
    for date_text in dates:
        date = pd.Timestamp(date_text)
        truncated = _features_up_to(matches, date)
        columns = [c for c in feature_columns(full) if c in full.columns]
        left = full[full["_date"] == date].reset_index(drop=True)
        right = truncated[truncated["_date"] == date].reset_index(drop=True)
        same_rows = len(left) == len(right) and len(left) > 0
        identical = False
        if same_rows:
            a = left[columns].to_numpy(dtype=float)
            b = right[columns].to_numpy(dtype=float)
            identical = bool(np.all((a == b) | (np.isnan(a) & np.isnan(b))))
        cases.append({
            "date": date_text,
            "matches_on_date": int(len(left)),
            "rows_match": bool(same_rows),
            "identical": identical,
        })
    return {"test": "future_independence", "passed": all(c["identical"] for c in cases),
            "cases": cases}


def chronological_order_test(matches: pd.DataFrame) -> dict:
    """The first ever match must start from a pristine state."""
    first = matches.sort_values(["date", "time"]).iloc[0]
    frame = FeatureBuilder().build(matches.sort_values(["date", "time"]))
    row = frame.iloc[0]
    checks = {
        "first_match_elo_is_start": bool(
            abs(float(row["h_elo"]) - config.ELO_START) < 1e-6
            and abs(float(row["a_elo"]) - config.ELO_START) < 1e-6),
        "first_match_has_no_history": bool(
            pd.isna(row["h_form5_ppg"]) and pd.isna(row["a_form5_ppg"])),
        "first_match_season_game_one": bool(
            float(row["h_season_game"]) == 1.0 and float(row["a_season_game"]) == 1.0),
    }
    return {
        "test": "chronological_cold_start",
        "passed": all(checks.values()),
        "checks": checks,
        "match": str(first["match_id"]),
    }


def run_all(matches: pd.DataFrame, sample_match_ids: list[str] | None = None,
            write: bool = True) -> dict:
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "integrity": validate_matches(matches),
        "own_result": own_result_test(matches, sample_match_ids),
        "future_independence": future_independence_test(matches),
        "cold_start": chronological_order_test(matches),
    }
    report["passed"] = all(report[k]["passed"] if "passed" in report[k] else report[k]["ok"]
                           for k in ("integrity", "own_result", "future_independence", "cold_start"))
    if write:
        config.QUALITY_DIR.mkdir(parents=True, exist_ok=True)
        LEAKAGE_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from src.data.processing import load_processed
    result = run_all(load_processed())
    print(json.dumps(result, indent=2, default=str)[:4000])
    print("\nALL CHECKS PASSED:", result["passed"])
