"""Expanding-window walk-forward evaluation.

Each origin uses only the past for fitting:

    train      : every season before the inner-validation season
    validation : the season immediately before the evaluation season
    evaluation : the next season

Hyper-parameters, ensemble weights, Poisson/market blends and calibration are
chosen on the inner-validation season. The evaluation season is never touched
during fitting.

The last two evaluation seasons (2024/25, 2025/26) are reported as historical
benchmark evidence only; they are not used to select a challenger.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config
from src.data.processing import load_matches
from src.features.builder import FeatureBuilder, feature_columns, odds_free_columns
from src.models.evaluation import evaluate
from src.models.training import CANDIDATES, _select_and_score
from src.research import diagnostics as diag

RESEARCH_DIR = config.REPORTS_DIR / "research"
PRED_DIR = RESEARCH_DIR / "walk_forward_predictions"

# A deliberately small, diverse candidate set. Running all nine at every
# origin would multiply compute without adding a distinct hypothesis.
CANDIDATE_NAMES = ("random_forest", "lgbm_regularised", "logistic")
CANDIDATES_SUBSET = [spec for spec in CANDIDATES if spec["name"] in CANDIDATE_NAMES]

# Selection-grade origins stop before the historical test seasons.
SELECTION_SEASONS = ("2019/20", "2020/21", "2021/22", "2022/23", "2023/24")
BENCHMARK_SEASONS = ("2024/25", "2025/26")


def _extract(metrics: dict) -> dict:
    return {
        "n": metrics["n"],
        "accuracy": metrics["accuracy"],
        "log_loss": metrics["log_loss"],
        "brier": metrics["brier"],
        "macro_f1": metrics["macro_f1"],
        "draw_recall": metrics["per_class"]["Draw"]["recall"],
        "draws_predicted": metrics["prediction_distribution"]["Draw"],
    }


def _with_predictions(frame: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    out = frame.copy()
    for i, suffix in enumerate(("away", "draw", "home")):
        out[f"pred_{suffix}"] = proba[:, i]
    return out


def _apply_policy(frame: pd.DataFrame, policy: dict | None) -> np.ndarray:
    proba = diag.probabilities(frame, "pred")
    if not policy:
        return proba.argmax(axis=1)
    if policy["policy"] == "threshold":
        return diag.choose(proba[:, 1], proba[:, 2], proba[:, 0],
                           threshold=policy["threshold"])
    return diag.choose(proba[:, 1], proba[:, 2], proba[:, 0],
                       margin=policy["margin"])


def _summary(rows: list[dict], variant: str, seasons: tuple[str, ...]) -> dict:
    subset = [row for row in rows if row["season"] in seasons]
    if not subset:
        return {}
    values = {
        metric: np.array([row[variant][metric] for row in subset], dtype=float)
        for metric in ("accuracy", "log_loss", "brier", "macro_f1", "draw_recall")
    }
    return {
        "origins": len(subset),
        "seasons": [row["season"] for row in subset],
        "mean": {k: float(v.mean()) for k, v in values.items()},
        "std": {k: float(v.std(ddof=1)) if len(v) > 1 else 0.0 for k, v in values.items()},
        "min": {k: float(v.min()) for k, v in values.items()},
        "max": {k: float(v.max()) for k, v in values.items()},
    }


def main() -> None:
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    matches = load_matches(refresh=False)
    features = FeatureBuilder().build(matches)
    columns = feature_columns(features)
    no_odds_columns = odds_free_columns(columns)
    features["_season"] = features["_season"].astype(str)

    seasons = list(config.SEASONS.values())
    eval_seasons = [s for s in seasons if s >= "2019/20" and s <= "2025/26"]

    rows = []
    for season in eval_seasons:
        index = seasons.index(season)
        train_seasons = set(seasons[:index - 1])
        val_season = seasons[index - 1]
        train = features[features["_season"].isin(train_seasons)]
        val = features[features["_season"] == val_season]
        test = features[features["_season"] == season]
        if len(train) < 1000 or len(val) < 300 or len(test) < 300:
            print(f"skip {season}: train={len(train)} val={len(val)} test={len(test)}")
            continue

        empty = features.iloc[0:0]
        print(f"origin {season}: train<={seasons[index - 2]} ({len(train)}), "
              f"val={val_season} ({len(val)}), eval={season} ({len(test)})")
        full = _select_and_score(train, val, test, empty, columns, "full",
                                 candidates=CANDIDATES_SUBSET)
        no_odds = _select_and_score(train, val, test, empty, no_odds_columns, "no_odds",
                                    candidates=CANDIDATES_SUBSET)

        y_eval = test["_label"].to_numpy(int)
        market = test[["mkt_away", "mkt_draw", "mkt_home"]].to_numpy(dtype=float)
        market_metrics = evaluate(y_eval, market)
        elo_metrics = full["test_baselines"].get("elo_logistic", {})

        # Calibration comparison at this origin, fitted on the inner validation
        # season only and applied forward.
        y_val = val["_label"].to_numpy(int)
        val_precal = full["val_proba_precal"]
        eval_precal = full["test_proba_precal"]
        temperature = diag.fit_temperature(y_val, val_precal)
        theta = diag.fit_vector_scaling(y_val, val_precal)
        temperature_metrics = evaluate(
            y_eval, diag.apply_temperature(eval_precal, temperature))
        vector_metrics = evaluate(
            y_eval, diag.apply_vector_scaling(eval_precal, theta))

        # Draw-aware decision rule, selected on the inner validation season and
        # applied unchanged to the evaluation season. This is the out-of-sample
        # stability check for the threshold policy.
        val_frame = _with_predictions(val, full["val_proba"])
        eval_frame = _with_predictions(test, full["test_proba"])
        policy = diag.select_draw_policy(val_frame, "pred")
        selected_policy = policy.get("selected")
        eval_argmax = diag.decision_metrics(
            diag.labels(eval_frame), diag.probabilities(eval_frame).argmax(axis=1))
        eval_policy = diag.decision_metrics(
            diag.labels(eval_frame), _apply_policy(eval_frame, selected_policy))

        row = {
            "season": season,
            "train_n": int(len(train)),
            "train_through": seasons[index - 2],
            "validation_season": val_season,
            "n": int(len(test)),
            "selected_model": full["selected"],
            "production_model": full["production_desc"],
            "no_odds_model": no_odds["selected"],
            "market_beta": full["market_beta"],
            "poisson_alpha": full["poisson_alpha"],
            "production": _extract(full["test_metrics"]),
            "no_odds": _extract(no_odds["test_metrics"]),
            "market": _extract(market_metrics),
            "elo": _extract(elo_metrics) if elo_metrics else None,
            "temperature_calibration": _extract(temperature_metrics),
            "vector_calibration": _extract(vector_metrics),
            "temperature": temperature,
            "vector_theta": [float(x) for x in theta],
            "draw_policy": {
                "selected": selected_policy,
                "validation_baseline": policy["baseline"],
                "eval_argmax": eval_argmax,
                "eval_policy": eval_policy,
                "accuracy_change": eval_policy["accuracy"] - eval_argmax["accuracy"],
                "macro_f1_change": eval_policy["macro_f1"] - eval_argmax["macro_f1"],
            },
        }
        rows.append(row)

        out = test[["_date", "_season", "_home_team", "_away_team",
                    "_home_goals", "_away_goals", "_result"]].copy()
        for prefix, proba in (("production", full["test_proba"]),
                              ("no_odds", no_odds["test_proba"]),
                              ("market", market),
                              ("precal", eval_precal)):
            out[f"{prefix}_away"] = proba[:, 0]
            out[f"{prefix}_draw"] = proba[:, 1]
            out[f"{prefix}_home"] = proba[:, 2]
        out.to_csv(PRED_DIR / f"{season.replace('/', '-')}.csv", index=False)

    selection_rows = [row for row in rows if row["season"] in SELECTION_SEASONS]
    draw_changes = {
        "accuracy": [row["draw_policy"]["accuracy_change"] for row in selection_rows],
        "macro_f1": [row["draw_policy"]["macro_f1_change"] for row in selection_rows],
        "policies_selected": sum(1 for row in selection_rows
                                 if row["draw_policy"]["selected"]),
        "origins": len(selection_rows),
    }
    draw_changes["mean_accuracy_change"] = float(np.mean(draw_changes["accuracy"])) if selection_rows else 0.0
    draw_changes["mean_macro_f1_change"] = float(np.mean(draw_changes["macro_f1"])) if selection_rows else 0.0
    draw_changes["origins_with_macro_gain"] = int(sum(
        1 for value in draw_changes["macro_f1"] if value > 0))

    payload = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "protocol": {
            "train": "all seasons before the inner-validation season",
            "validation": "season immediately before evaluation",
            "evaluation": "next season, never used for fitting",
            "candidates": list(CANDIDATE_NAMES),
            "selection_origins": list(SELECTION_SEASONS),
            "benchmark_only_origins": list(BENCHMARK_SEASONS),
        },
        "origins": rows,
        "draw_policy_stability": draw_changes,
        "summary": {
            variant: {
                "selection_origins": _summary(rows, variant, SELECTION_SEASONS),
                "benchmark_origins": _summary(rows, variant, BENCHMARK_SEASONS),
            }
            for variant in ("production", "no_odds", "market", "elo",
                            "temperature_calibration", "vector_calibration")
        },
    }
    path = RESEARCH_DIR / "walk_forward.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {path}")
    for variant in ("production", "no_odds", "market", "elo"):
        summary = payload["summary"][variant]["selection_origins"]
        if summary:
            print(variant, "selection-origin mean:",
                  {k: round(v, 4) for k, v in summary["mean"].items()})
    print("draw policy stability:", {
        k: round(v, 4) if isinstance(v, float) else v
        for k, v in draw_changes.items() if k not in ("accuracy", "macro_f1")})


if __name__ == "__main__":
    main()
