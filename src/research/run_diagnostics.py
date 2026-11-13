"""Run draw, calibration and paired-comparison diagnostics.

Reads the validation and historical-test prediction files produced by
``python -m src.models.training``. Writes JSON reports under
``reports/research/``. Selection decisions use validation only; test metrics
are reported as historical benchmark evidence.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config
from src.research import diagnostics as diag

RESEARCH_DIR = config.REPORTS_DIR / "research"


def _load(name: str) -> pd.DataFrame:
    path = config.EVAL_DIR / name
    frame = pd.read_csv(path, parse_dates=["_date"])
    for prefix in ("pred", "market", "no_odds"):
        cols = [f"{prefix}_{suffix}" for suffix in diag.PROB_SUFFIXES]
        if not set(cols).issubset(frame.columns):
            raise KeyError(f"{path.name} is missing {prefix} probability columns")
        sums = frame[cols].sum(axis=1)
        if not np.allclose(sums, 1.0, atol=1e-6):
            raise ValueError(f"{path.name}: {prefix} probabilities do not sum to 1")
    return frame


def draw_analysis(val: pd.DataFrame, test: pd.DataFrame) -> dict:
    variants = {}
    for prefix, label in (("pred", "production"), ("market", "market"),
                          ("no_odds", "no_odds")):
        variants[label] = {
            "test_summary": diag.draw_summary(test, prefix),
            "test_deciles": diag.draw_deciles(test, prefix),
            "test_by_season": diag.draw_by_season(test, prefix),
        }
    variants["production"]["validation_summary"] = diag.draw_summary(val, "pred")
    variants["production"]["validation_deciles"] = diag.draw_deciles(val, "pred")

    threshold_val = diag.threshold_curve(val, "pred")
    threshold_test = diag.threshold_curve(test, "pred")
    margin_val = diag.margin_curve(val, "pred")
    margin_test = diag.margin_curve(test, "pred")
    policy = diag.select_draw_policy(val, "pred")

    selected = policy.get("selected")
    if selected:
        if selected["policy"] == "threshold":
            val_pred = diag.choose(
                diag.probabilities(val)[:, 1], diag.probabilities(val)[:, 2],
                diag.probabilities(val)[:, 0], threshold=selected["threshold"])
            test_pred = diag.choose(
                diag.probabilities(test)[:, 1], diag.probabilities(test)[:, 2],
                diag.probabilities(test)[:, 0], threshold=selected["threshold"])
        else:
            val_pred = diag.choose(
                diag.probabilities(val)[:, 1], diag.probabilities(val)[:, 2],
                diag.probabilities(val)[:, 0], margin=selected["margin"])
            test_pred = diag.choose(
                diag.probabilities(test)[:, 1], diag.probabilities(test)[:, 2],
                diag.probabilities(test)[:, 0], margin=selected["margin"])
        applied = {
            "policy": selected,
            "validation": diag.decision_metrics(diag.labels(val), val_pred),
            "test": diag.decision_metrics(diag.labels(test), test_pred),
            "test_argmax": diag.decision_metrics(diag.labels(test),
                                                 diag.probabilities(test).argmax(axis=1)),
        }
    else:
        applied = {
            "policy": None,
            "validation": diag.decision_metrics(diag.labels(val),
                                                diag.probabilities(val).argmax(axis=1)),
            "test": diag.decision_metrics(diag.labels(test),
                                          diag.probabilities(test).argmax(axis=1)),
            "test_argmax": diag.decision_metrics(diag.labels(test),
                                                 diag.probabilities(test).argmax(axis=1)),
        }

    return {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "selection_data": "2023/24 validation only",
        "test_data": "2024/25 + 2025/26 historical benchmark (already examined)",
        "variants": variants,
        "threshold_curve_validation": threshold_val,
        "threshold_curve_test": threshold_test,
        "margin_curve_validation": margin_val,
        "margin_curve_test": margin_test,
        "policy_selection": policy,
        "policy_applied": applied,
    }


def _calibrated(frame: pd.DataFrame, method: str, theta) -> np.ndarray:
    proba = diag.probabilities(frame, "precal")
    if method == "temperature":
        return diag.apply_temperature(proba, theta)
    if method == "vector":
        return diag.apply_vector_scaling(proba, theta)
    return proba


def calibration_analysis(val: pd.DataFrame, test: pd.DataFrame) -> dict:
    y_val, y_test = diag.labels(val), diag.labels(test)
    precal_val = diag.probabilities(val, "precal")
    precal_test = diag.probabilities(test, "precal")

    temperature = diag.fit_temperature(y_val, precal_val)
    theta = diag.fit_vector_scaling(y_val, precal_val)

    candidates = {
        "raw": {},
        "temperature": {"temperature": temperature},
        "vector_scaling": {"theta": [float(x) for x in theta]},
    }
    metrics = {}
    for name in candidates:
        val_proba = _calibrated(val, "raw" if name == "raw" else
                                "temperature" if name == "temperature" else "vector",
                                temperature if name == "temperature" else theta)
        test_proba = _calibrated(test, "raw" if name == "raw" else
                                 "temperature" if name == "temperature" else "vector",
                                 temperature if name == "temperature" else theta)
        metrics[name] = {
            "validation": diag.calibration_metrics(y_val, val_proba),
            "test": diag.calibration_metrics(y_test, test_proba),
            "test_reliability": diag.reliability_table(y_test, test_proba),
            "test_per_class_reliability": diag.per_class_reliability(y_test, test_proba),
        }

    raw_ll = metrics["raw"]["validation"]["log_loss"]
    temp_ll = metrics["temperature"]["validation"]["log_loss"]
    vector_ll = metrics["vector_scaling"]["validation"]["log_loss"]
    vector_ece = metrics["vector_scaling"]["validation"]["ece"]
    temp_ece = metrics["temperature"]["validation"]["ece"]

    # Pre-registered promotion gate: vector scaling must improve validation
    # log loss by >= 0.002 over temperature scaling without worsening ECE by
    # more than 0.002. Otherwise the simpler method is retained.
    promoted = (
        vector_ll <= temp_ll - 0.002
        and vector_ece <= temp_ece + 0.002
    )
    return {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "selection_data": "2023/24 validation only",
        "raw_validation_log_loss": raw_ll,
        "temperature_validation_log_loss": temp_ll,
        "vector_validation_log_loss": vector_ll,
        "promotion_gate": {
            "min_log_loss_gain": 0.002,
            "max_ece_worsening": 0.002,
            "vector_scaling_promoted": bool(promoted),
        },
        "selected_method": "vector_scaling" if promoted else "temperature",
        "candidates": candidates,
        "metrics": metrics,
    }


def paired_analysis(val: pd.DataFrame, test: pd.DataFrame) -> dict:
    return {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "test": {
            "production_vs_market": diag.paired_comparison(test, "pred", "market"),
            "no_odds_vs_market": diag.paired_comparison(test, "no_odds", "market"),
            "production_vs_no_odds": diag.paired_comparison(test, "pred", "no_odds"),
        },
        "validation": {
            "production_vs_market": diag.paired_comparison(val, "pred", "market"),
        },
    }


def main() -> None:
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    val = _load("validation_predictions.csv")
    test = _load("test_predictions.csv")

    draws = draw_analysis(val, test)
    calibration = calibration_analysis(val, test)
    paired = paired_analysis(val, test)

    for name, payload in (("draw_analysis", draws),
                          ("calibration_analysis", calibration),
                          ("paired_comparisons", paired)):
        path = RESEARCH_DIR / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {path}")

    summary = draws["variants"]["production"]["test_summary"]
    print("Draw summary (production, test):", {
        k: round(v, 4) if isinstance(v, float) else v
        for k, v in summary.items() if k != "quantiles_p_draw"
    })
    print("Policy selected on validation:", draws["policy_selection"]["selected"])
    print("Calibration selected:", calibration["selected_method"],
          "| vector promoted:", calibration["promotion_gate"]["vector_scaling_promoted"])
    print("Production vs market (test):",
          paired["test"]["production_vs_market"]["accuracy_difference"])


if __name__ == "__main__":
    main()
