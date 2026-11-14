"""Clean, validate and persist the canonical match database.

The processed table is the single source of truth for features and models.
Raw files are never modified.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src import config
from src.data import acquisition

log = logging.getLogger(__name__)

BOOKIE_TRIPLES = (
    ("B365H", "B365D", "B365A"),
    ("BWH", "BWD", "BWA"),
    ("IWH", "IWD", "IWA"),
    ("PSH", "PSD", "PSA"),
    ("WHH", "WHD", "WHA"),
    ("VCH", "VCD", "VCA"),
    ("BFDH", "BFDD", "BFDA"),
)

STAT_MAP = {
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_sot",
    "AST": "away_sot",
    "HC": "home_corners",
    "AC": "away_corners",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HY": "home_yellows",
    "AY": "away_yellows",
    "HR": "home_reds",
    "AR": "away_reds",
}


def normalize_team(name: object) -> str:
    """Map any source spelling to the canonical football-data name."""
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ""
    text = str(name).strip()
    if not text:
        return ""
    key = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    key = re.sub(r"\s+", " ", key).strip()
    # Strip common suffixes before lookup.
    for suffix in (" afc", " fc", " football club"):
        if key.endswith(suffix):
            key = key[: -len(suffix)].strip()
    if key in config.TEAM_ALIASES:
        return config.TEAM_ALIASES[key]
    # Try without a leading AFC.
    if key.startswith("afc "):
        alt = key[4:].strip()
        if alt in config.TEAM_ALIASES:
            return config.TEAM_ALIASES[alt]
    return text


def _numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan
    return df


def _odds_features(df: pd.DataFrame) -> pd.DataFrame:
    """Consensus market probabilities, overround and dispersion."""
    out = pd.DataFrame(index=df.index)
    probs = []
    overrounds = []
    for cols in BOOKIE_TRIPLES:
        h, d, a = (df[c] if c in df.columns else pd.Series(np.nan, index=df.index)
                   for c in cols)
        valid = h.notna() & d.notna() & a.notna() & (h > 1) & (d > 1) & (a > 1)
        inv_h, inv_d, inv_a = 1.0 / h, 1.0 / d, 1.0 / a
        total = inv_h + inv_d + inv_a
        p = pd.DataFrame({
            "h": (inv_h / total).where(valid),
            "d": (inv_d / total).where(valid),
            "a": (inv_a / total).where(valid),
        })
        label = cols[0].replace("H", "").lower()
        out[f"{label}_p_home"] = p["h"]
        out[f"{label}_p_draw"] = p["d"]
        out[f"{label}_p_away"] = p["a"]
        out[f"{label}_overround"] = total.where(valid)
        probs.append(p)
        overrounds.append(total.where(valid))

    # Average consensus odds are a useful fallback when individual books are absent.
    if all(c in df.columns for c in ("AvgH", "AvgD", "AvgA")):
        h, d, a = df["AvgH"], df["AvgD"], df["AvgA"]
        valid = h.notna() & d.notna() & a.notna() & (h > 1) & (d > 1) & (a > 1)
        total = 1 / h + 1 / d + 1 / a
        avg_p = pd.DataFrame({
            "h": (1 / h / total).where(valid),
            "d": (1 / d / total).where(valid),
            "a": (1 / a / total).where(valid),
        })
        out["avg_p_home"] = avg_p["h"]
        out["avg_p_draw"] = avg_p["d"]
        out["avg_p_away"] = avg_p["a"]
        out["avg_overround"] = total.where(valid)
        probs.append(avg_p)
        overrounds.append(total.where(valid))

    # Maximum (best-price) consensus, where available.
    if all(c in df.columns for c in ("MaxH", "MaxD", "MaxA")):
        h, d, a = df["MaxH"], df["MaxD"], df["MaxA"]
        valid = h.notna() & d.notna() & a.notna() & (h > 1) & (d > 1) & (a > 1)
        total = 1 / h + 1 / d + 1 / a
        max_p = pd.DataFrame({
            "h": (1 / h / total).where(valid),
            "d": (1 / d / total).where(valid),
            "a": (1 / a / total).where(valid),
        })
        out["max_p_home"] = max_p["h"]
        out["max_p_draw"] = max_p["d"]
        out["max_p_away"] = max_p["a"]
        out["max_overround"] = total.where(valid)
        probs.append(max_p)
        overrounds.append(total.where(valid))

    stack = np.stack([p.to_numpy(dtype=float) for p in probs])  # books x rows x 3
    with np.errstate(invalid="ignore"):
        mean_p = np.nanmean(stack, axis=0)
        std_p = np.nanstd(stack, axis=0)
    count = np.sum(np.isfinite(stack).all(axis=2), axis=0)

    out["mkt_home"] = mean_p[:, 0]
    out["mkt_draw"] = mean_p[:, 1]
    out["mkt_away"] = mean_p[:, 2]
    out["mkt_disp_home"] = std_p[:, 0]
    out["mkt_disp_draw"] = std_p[:, 1]
    out["mkt_disp_away"] = std_p[:, 2]
    out["mkt_overround"] = np.nanmean(
        np.stack([o.to_numpy(dtype=float) for o in overrounds]), axis=0)
    out["mkt_books"] = count
    out["mkt_balance"] = out["mkt_home"] - out["mkt_away"]
    out["mkt_favourite"] = out[["mkt_home", "mkt_draw", "mkt_away"]].max(axis=1)
    out["mkt_entropy"] = -(
        out[["mkt_home", "mkt_draw", "mkt_away"]]
        .clip(lower=1e-9)
        .apply(lambda r: np.sum(r * np.log(r)), axis=1)
    )
    out["odds_home"] = 1.0 / out["mkt_home"]
    out["odds_draw"] = 1.0 / out["mkt_draw"]
    out["odds_away"] = 1.0 / out["mkt_away"]
    return out


def _parse_dates(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, dayfirst=True, format="mixed", errors="coerce")
    except (TypeError, ValueError):
        return pd.to_datetime(series, dayfirst=True, errors="coerce")


def load_matches(refresh: bool = False, force: bool = False) -> pd.DataFrame:
    """Load, clean and cache all historical matches."""
    acquisition.ensure_cache()
    if refresh:
        acquisition.refresh(force=force)

    frames = []
    for code in config.SEASONS:
        path = acquisition.season_path(code)
        if not path.exists():
            log.warning("Missing cached season %s; skipping", code)
            continue
        raw = pd.read_csv(path, encoding="utf-8-sig", low_memory=False).copy()
        raw["season_code"] = code
        frames.append(raw)
    if not frames:
        raise RuntimeError("No season files available; run acquisition.refresh(force=True)")

    df = pd.concat(frames, ignore_index=True)
    df = df.copy()  # consolidate blocks before column-wise assignment
    df = _numeric(df, ["FTHG", "FTAG", "HTHG", "HTAG", *STAT_MAP.keys(),
                       *[c for triple in BOOKIE_TRIPLES for c in triple],
                       "AvgH", "AvgD", "AvgA", "MaxH", "MaxD", "MaxA"])
    df["date"] = _parse_dates(df["Date"])
    df["time"] = df.get("Time", pd.Series("15:00", index=df.index)).fillna("15:00").astype(str)
    df["home_team"] = df["HomeTeam"].map(normalize_team)
    df["away_team"] = df["AwayTeam"].map(normalize_team)
    df["result"] = df["FTR"].astype(str).str.strip().str.upper()

    before = len(df)
    df = df.dropna(subset=["date", "FTHG", "FTAG"])
    df = df[df["home_team"].ne("") & df["away_team"].ne("")]
    df = df[df["home_team"] != df["away_team"]]
    df = df[df["result"].isin(["H", "D", "A"])]
    df = df.drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")
    df = df.sort_values(["date", "time", "home_team"], kind="mergesort").reset_index(drop=True)

    clean = pd.DataFrame({
        "match_id": (df["date"].dt.strftime("%Y-%m-%d") + "_" + df["home_team"]
                     + "_" + df["away_team"]),
        "date": df["date"].dt.normalize(),
        "time": df["time"],
        "season_code": df["season_code"].astype(str),
        "season": df["season_code"].astype(str).map(config.SEASONS),
        "home_team": df["home_team"],
        "away_team": df["away_team"],
        "home_goals": df["FTHG"].astype(int),
        "away_goals": df["FTAG"].astype(int),
        "result": df["result"],
        "ht_home": df["HTHG"],
        "ht_away": df["HTAG"],
    })
    for src, dst in STAT_MAP.items():
        clean[dst] = df[src]
    clean = pd.concat([clean, _odds_features(df)], axis=1)
    clean["total_goals"] = clean["home_goals"] + clean["away_goals"]
    clean["goal_diff"] = clean["home_goals"] - clean["away_goals"]

    clean.to_csv(config.MATCHES_CSV, index=False)
    log.info("Processed %d matches (%d raw rows, %d dropped/duplicated)", len(clean), before, before - len(clean))
    return clean


def load_processed() -> pd.DataFrame:
    if not config.MATCHES_CSV.exists():
        return load_matches()
    df = pd.read_csv(config.MATCHES_CSV, parse_dates=["date"],
                     dtype={"season_code": str})
    return df.sort_values(["date", "time", "home_team"], kind="mergesort").reset_index(drop=True)


def _fixture_odds(row: pd.Series) -> dict:
    frame = pd.DataFrame([row])
    frame = _numeric(frame, [c for triple in BOOKIE_TRIPLES for c in triple]
                     + ["AvgH", "AvgD", "AvgA"])
    feats = _odds_features(frame).iloc[0].to_dict()
    return feats


def load_fixtures(refresh: bool = False, matches: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return the next matchweek of upcoming fixtures from the synced store.

    The store is maintained by :mod:`src.data.fixture_sync` (CLI, pipeline and
    the scheduled workflow). Prediction only reads it, so scoring a fixture
    never triggers a network call. ``refresh=True`` forces a live
    synchronisation first; a missing store is bootstrapped from the cached
    provider payloads.
    """
    from src.data import fixture_sync

    matches = matches if matches is not None else load_processed()
    if refresh or fixture_sync.load_store() is None:
        fixture_sync.sync_fixtures(fetch=refresh, matches=matches)
    return fixture_sync.upcoming(matches=matches)


def build_standings(matches: pd.DataFrame, season_code: str | None = None) -> pd.DataFrame:
    season_code = season_code or config.CURRENT_SEASON
    season = matches[matches["season_code"] == season_code]
    teams = sorted(set(season["home_team"]) | set(season["away_team"]))
    rows = []
    for team in teams:
        home = season[season["home_team"] == team]
        away = season[season["away_team"] == team]
        gf = int(home["home_goals"].sum() + away["away_goals"].sum())
        ga = int(home["away_goals"].sum() + away["home_goals"].sum())
        wins = int((home["result"] == "H").sum() + (away["result"] == "A").sum())
        draws = int((home["result"] == "D").sum() + (away["result"] == "D").sum())
        losses = int((home["result"] == "A").sum() + (away["result"] == "H").sum())
        played = wins + draws + losses
        rows.append({
            "team": team, "played": played, "won": wins, "drawn": draws, "lost": losses,
            "goals_for": gf, "goals_against": ga, "goal_diff": gf - ga,
            "points": 3 * wins + draws,
            "ppg": round((3 * wins + draws) / played, 3) if played else 0.0,
        })
    table = pd.DataFrame(rows).sort_values(
        ["points", "goal_diff", "goals_for", "team"], ascending=[False, False, False, True]
    ).reset_index(drop=True)
    table.insert(0, "position", np.arange(1, len(table) + 1))
    table.insert(0, "season", config.SEASONS.get(season_code, season_code))
    table.to_csv(config.STANDINGS_CSV, index=False)
    return table


def data_quality_report(matches: pd.DataFrame) -> dict:
    complete = matches.groupby("season_code").size()
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_matches": int(len(matches)),
        "first_date": str(matches["date"].min().date()),
        "last_date": str(matches["date"].max().date()),
        "n_teams": int(len(set(matches["home_team"]) | set(matches["away_team"]))),
        "seasons": {code: int(n) for code, n in complete.items()},
        "duplicate_fixtures": int(matches.duplicated(["date", "home_team", "away_team"]).sum()),
        "invalid_results": int((~matches["result"].isin(["H", "D", "A"])).sum()),
        "odds_coverage": round(float(matches["mkt_home"].notna().mean()), 4),
        "shots_coverage": round(float(matches["home_shots"].notna().mean()), 4),
        "missing_by_column": {
            c: int(matches[c].isna().sum())
            for c in ("mkt_home", "home_shots", "home_sot", "home_corners", "home_fouls")
        },
        "outcome_distribution": {
            k: round(float(v), 4)
            for k, v in matches["result"].value_counts(normalize=True).items()
        },
    }
    config.DATA_QUALITY_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    m = load_matches()
    print(data_quality_report(m))
    print(build_standings(m).head(8).to_string(index=False))
