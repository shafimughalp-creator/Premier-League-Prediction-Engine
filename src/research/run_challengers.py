"""Evaluate two justified challengers on the fixed chronological split.

1. A Dixon-Coles goal model (standalone and blended with the existing models).
2. A compact "core" feature set for the odds-free model, testing whether 223
   features dilute signal on 3,420 training matches.

Selection uses 2023/24 validation only. The 2024/25 + 2025/26 test seasons are
reported once as historical benchmark evidence.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config
from src.data.processing import load_matches
from src.features.builder import FeatureBuilder, feature_columns, odds_free_columns
from src.models.dixon_coles import DixonColesModel
from src.models.evaluation import evaluate
from src.models.training import CANDIDATES, _select_and_score
from src.research import diagnostics as diag

RESEARCH_DIR = config.REPORTS_DIR / "research"
CANDIDATE_NAMES = ("random_forest", "lgbm_regularised", "logistic")
CANDIDATES_SUBSET = [spec for spec in CANDIDATES if spec["name"] in CANDIDATE_NAMES]


def core_columns(no_odds: list[str]) -> list[str]:
    """A compact, hypothesis-driven subset of the odds-free features."""
    keep = {
        "h_elo", "a_elo", "elo_diff", "abs_elo_diff", "elo_closeness",
        "elo_expected_home", "elo_expected_away",
        "h_promoted", "a_promoted", "h_prev_ppg", "a_prev_ppg",
        "h_prev_gd_pg", "a_prev_gd_pg", "diff_prev_ppg", "diff_prev_gd_pg",
        "h_rest_days", "a_rest_days", "h_matches_14d", "a_matches_14d",
        "h_matches_28d", "a_matches_28d",
        "h_attack_home", "h_defence_home", "h_attack_away", "h_defence_away",
        "a_attack_home", "a_defence_home", "a_attack_away", "a_defence_away",
        "pois_exp_home", "pois_exp_away", "pois_exp_total", "pois_exp_diff",
        "pois_p_home", "pois_p_draw", "pois_p_away",
        "h2h_ppg", "h2h_draw_rate", "h2h_avg_total", "h2h_n",
        "h_season_ppg", "a_season_ppg", "diff_season_ppg",
        "h_season_gd_pg", "a_season_gd_pg", "diff_season_gd_pg",
        "h_season_game", "a_season_game", "h_season_progress", "a_season_progress",
    }
    for n in (3, 5, 10):
        for side in ("h", "a"):
            for key in ("ppg", "gd", "gf", "ga", "win_rate", "draw_rate",
                        "loss_rate", "sot", "shots"):
                keep.add(f"{side}_form{n}_{key}")
        for key in ("ppg", "gd", "gf", "ga", "win_rate", "draw_rate",
                    "loss_rate", "sot", "shots"):
            keep.add(f"diff_form{n}_{key}")
    for side in ("h", "a"):
        for key in ("ppg", "gd", "clean_sheet", "over25"):
            keep.add(f"{side}_venue5_{key}")
    for key in ("ppg", "gd", "clean_sheet", "over25"):
        keep.add(f"diff_venue5_{key}")
    return sorted(c for c in no_odds if c in keep)


def blend(model: np.ndarray, extra: np.ndarray, weight: float) -> np.ndarray:
    out = (1.0 - weight) * model + weight * extra
    return out / out.sum(axis=1, keepdims=True)


def _dc_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Map the feature builder's internal columns to plain match columns."""
    return frame.rename(columns={
        "_date": "date", "_home_team": "home_team", "_away_team": "away_team",
        "_home_goals": "home_goals", "_away_goals": "away_goals",
    })


def select_blend(y: np.ndarray, base: np.ndarray, extra: np.ndarray) -> dict:
    best = {"weight": 0.0, "log_loss": float("inf"), "accuracy": 0.0}
    for weight in np.linspace(0.0, 0.40, 41):
        proba = blend(base, extra, float(weight))
        metrics = evaluate(y, proba)
        if metrics["log_loss"] < best["log_loss"]:
            best = {
                "weight": float(weight),
                "log_loss": metrics["log_loss"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "brier": metrics["brier"],
            }
    return best


def main() -> None:
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    matches = load_matches(refresh=False)
    features = FeatureBuilder().build(matches)
    columns = feature_columns(features)
    no_odds = odds_free_columns(columns)
    core = core_columns(no_odds)

    train = features[features["_date"] < pd.Timestamp(config.TRAIN_END)]
    val = features[(features["_date"] >= pd.Timestamp(config.TRAIN_END))
                   & (features["_date"] < pd.Timestamp(config.VAL_END))]
    test = features[(features["_date"] >= pd.Timestamp(config.VAL_END))
                    & (features["_date"] < pd.Timestamp(config.TEST_END))]
    y_val = val["_label"].to_numpy(int)
    y_test = test["_label"].to_numpy(int)

    print(f"core feature set: {len(core)} of {len(no_odds)} odds-free columns")

    # ---------------- Dixon-Coles -------------------------------------
    dc = DixonColesModel().fit(_dc_frame(train))
    dc_val = dc.predict_frame(_dc_frame(val))
    dc_test = dc.predict_frame(_dc_frame(test))
    dc_val_metrics = evaluate(y_val, dc_val)
    dc_test_metrics = evaluate(y_test, dc_test)

    val_pred = pd.read_csv(config.EVAL_DIR / "validation_predictions.csv")
    test_pred = pd.read_csv(config.EVAL_DIR / "test_predictions.csv")
    no_odds_val = diag.probabilities(val_pred, "no_odds")
    no_odds_test = diag.probabilities(test_pred, "no_odds")
    production_val = diag.probabilities(val_pred, "pred")
    production_test = diag.probabilities(test_pred, "pred")

    dc_no_odds = select_blend(y_val, no_odds_val, dc_val)
    dc_no_odds_test = evaluate(
        y_test, blend(no_odds_test, dc_test, dc_no_odds["weight"]))
    dc_production = select_blend(y_val, production_val, dc_val)
    dc_production_test = evaluate(
        y_test, blend(production_test, dc_test, dc_production["weight"]))

    # ---------------- Core feature set --------------------------------
    empty = features.iloc[0:0]
    core_result = _select_and_score(train, val, test, empty, core, "no_odds",
                                    candidates=CANDIDATES_SUBSET)
    core_val_metrics = core_result["val_metrics"]
    core_test_metrics = core_result["test_metrics"]

    payload = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "selection_data": "2023/24 validation only",
        "test_data": "2024/25 + 2025/26 historical benchmark",
        "dixon_coles": {
            "configuration": {
                "xi": dc.xi,
                "rho": dc.rho,
                "home_advantage": dc.home_advantage,
                "intercept": dc.intercept,
                "teams": len(dc.teams),
            },
            "validation": dc_val_metrics,
            "test": dc_test_metrics,
            "blend_with_no_odds": {
                "validation_selection": dc_no_odds,
                "test": dc_no_odds_test,
            },
            "blend_with_production": {
                "validation_selection": dc_production,
                "test": dc_production_test,
            },
        },
        "core_feature_set": {
            "n_features": len(core),
            "features": core,
            "selected_model": core_result["selected"],
            "validation": core_val_metrics,
            "test": core_test_metrics,
            "temperature": core_result["temperature"],
            "poisson_alpha": core_result["poisson_alpha"],
        },
        "reference": {
            "no_odds_validation": evaluate(
                y_val, no_odds_val),
            "no_odds_test": evaluate(y_test, no_odds_test),
            "production_validation": evaluate(
                y_val, production_val),
            "production_test": evaluate(y_test, production_test),
        },
    }
    path = RESEARCH_DIR / "challengers.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {path}")

    print("Dixon-Coles test:", {k: round(dc_test_metrics[k], 4)
                                for k in ("accuracy", "log_loss", "brier", "macro_f1")})
    print("DC + no-odds blend (val-selected weight %.2f) test:"
          % dc_no_odds["weight"],
          {k: round(dc_no_odds_test[k], 4)
           for k in ("accuracy", "log_loss", "brier", "macro_f1")})
    print("Core feature set test:", {k: round(core_test_metrics[k], 4)
                                     for k in ("accuracy", "log_loss", "brier", "macro_f1")})
    print("Reference no-odds test:", {k: round(payload["reference"]["no_odds_test"][k], 4)
                                      for k in ("accuracy", "log_loss", "brier", "macro_f1")})


if __name__ == "__main__":
    main()
