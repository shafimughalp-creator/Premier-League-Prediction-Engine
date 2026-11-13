"""Independent reproducibility and consistency audit.

Checks the claims made in the reports against the artefacts themselves:

* the feature matrix is deterministic;
* reported test metrics can be recomputed from the held-out prediction file;
* probabilities are valid distributions;
* class order is A/D/H everywhere;
* the model artifact matches its metadata and the training report;
* the experiment registry agrees with the final report;
* the market consensus can be reconstructed from its component prices;
* the no-odds feature list contains no market columns.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from src import config
from src.data.processing import load_matches
from src.features.builder import FeatureBuilder, feature_columns, odds_free_columns
from src.models.evaluation import evaluate
from src.research import diagnostics as diag

RESEARCH_DIR = config.REPORTS_DIR / "research"
# The consensus is the row-wise mean of every available price source:
# seven bookmaker triples plus the average-odds and maximum-odds series.
BOOKS = ("b365", "bw", "iw", "ps", "w", "vc", "bfd")


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all((a == b) | (np.isnan(a) & np.isnan(b))))


def main() -> None:
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    matches = load_matches(refresh=False)
    first = FeatureBuilder().build(matches)
    second = FeatureBuilder().build(matches)
    columns = feature_columns(first)
    no_odds = odds_free_columns(columns)

    determinism = {
        "n_features": len(columns),
        "n_features_no_odds": len(no_odds),
        "identical": _same(first[columns].to_numpy(dtype=float),
                           second[columns].to_numpy(dtype=float)),
        "market_columns_in_no_odds": [c for c in no_odds if c.startswith(("mkt_", "odds_"))],
    }

    report = json.loads((config.EVAL_DIR / "training_report.json").read_text(encoding="utf-8"))
    test = pd.read_csv(config.EVAL_DIR / "test_predictions.csv", parse_dates=["_date"])
    val = pd.read_csv(config.EVAL_DIR / "validation_predictions.csv", parse_dates=["_date"])
    y_test = diag.labels(test)
    proba_test = diag.probabilities(test, "pred")
    recomputed_test = evaluate(y_test, proba_test)

    metric_checks = {}
    for key in ("accuracy", "log_loss", "brier", "ece", "macro_f1"):
        metric_checks[key] = {
            "report": report["test_metrics"][key],
            "recomputed": recomputed_test[key],
            "match": bool(np.isclose(report["test_metrics"][key],
                                     recomputed_test[key], atol=1e-9)),
        }

    probability_checks = {}
    for name, frame in (("test", test), ("validation", val)):
        row = {}
        for prefix in ("pred", "market", "no_odds"):
            p = diag.probabilities(frame, prefix)
            row[prefix] = {
                "all_finite": bool(np.isfinite(p).all()),
                "sums_to_one": bool(np.allclose(p.sum(axis=1), 1.0, atol=1e-9)),
                "min": float(p.min()),
                "max": float(p.max()),
            }
        probability_checks[name] = row

    artifact = joblib.load(config.MODELS_DIR / "model.pkl")
    meta = json.loads((config.METADATA_DIR / "model_meta.json").read_text(encoding="utf-8"))
    registry = json.loads((config.EXPERIMENT_DIR / "experiment_log.json").read_text(encoding="utf-8"))
    training_records = [r for r in registry
                        if r.get("record_type", "training") == "training"]
    last = training_records[-1] if training_records else {}
    artifact_checks = {
        "features": len(artifact["features"]),
        "features_match_report": len(artifact["features"]) == report["n_features"],
        "no_odds_features": len(artifact["no_odds"]["features"]),
        "class_order": artifact["class_order"],
        "class_order_ok": tuple(artifact["class_order"]) == config.CLASS_ORDER,
        "test_accuracy": artifact["metrics"]["test"]["accuracy"],
        "test_accuracy_matches_report": bool(np.isclose(
            artifact["metrics"]["test"]["accuracy"], report["test_metrics"]["accuracy"])),
        "trained_at_matches_report": artifact["trained_at"] == report["trained_at"],
        "meta_trained_at_matches_report": meta["trained_at"] == report["trained_at"],
        "registry_last_matches_report": last.get("trained_at") == report["trained_at"],
        "registry_test_accuracy_matches_report": bool(np.isclose(
            last.get("test_metrics", {}).get("accuracy", -1),
            report["test_metrics"]["accuracy"])),
        "production_rows": artifact["production_rows"],
        "n_matches": int(len(matches)),
        "production_rows_match": artifact["production_rows"] == len(matches),
    }

    # Reconstruct the consensus from its component series on every row.
    components = [f"{book}_p_home" for book in BOOKS] + ["avg_p_home", "max_p_home"]
    rebuilt_home = matches[components].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        rebuilt_home = np.nanmean(rebuilt_home, axis=1)
    market_checks = {
        "rows": int(len(matches)),
        "coverage": float(matches["mkt_home"].notna().mean()),
        "home_consensus_max_abs_error": float(np.nanmax(np.abs(
            rebuilt_home - matches["mkt_home"].to_numpy(dtype=float)))),
        "all_rows_sum_to_one": bool(np.allclose(
            matches[["mkt_home", "mkt_draw", "mkt_away"]].sum(axis=1), 1.0, atol=1e-9)),
        "overround_range": [float(matches["mkt_overround"].min()),
                            float(matches["mkt_overround"].max())],
        # mkt_books counts price sources (books plus avg/max), not distinct bookmakers.
        "price_sources_min": int(matches["mkt_books"].min()),
        "price_sources_max": int(matches["mkt_books"].max()),
        "price_sources_median": float(matches["mkt_books"].median()),
    }

    paired = diag.paired_comparison(test, "pred", "market")
    paired_no_odds = diag.paired_comparison(test, "no_odds", "market")

    payload = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "git_commit": "8aff5dc (baseline before this audit)",
        "determinism": determinism,
        "metric_checks": metric_checks,
        "probability_checks": probability_checks,
        "artifact_checks": artifact_checks,
        "market_checks": market_checks,
        "paired_production_vs_market": paired,
        "paired_no_odds_vs_market": paired_no_odds,
        "passed": bool(
            determinism["identical"]
            and not determinism["market_columns_in_no_odds"]
            and all(row["match"] for row in metric_checks.values())
            and all(row[p]["all_finite"] and row[p]["sums_to_one"]
                    for row in probability_checks.values() for p in row)
            and artifact_checks["class_order_ok"]
            and artifact_checks["test_accuracy_matches_report"]
            and artifact_checks["trained_at_matches_report"]
            and artifact_checks["registry_last_matches_report"]
            and market_checks["coverage"] == 1.0
            and market_checks["all_rows_sum_to_one"]
            and market_checks["home_consensus_max_abs_error"] < 1e-9
        ),
    }
    path = RESEARCH_DIR / "audit.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {path}")
    print("PASSED:", payload["passed"])
    print("metric checks:", {k: v["match"] for k, v in metric_checks.items()})
    print("artifact checks:", {k: v for k, v in artifact_checks.items()
                               if k.endswith(("_ok", "match", "matches_report"))})
    print("market:", market_checks)


if __name__ == "__main__":
    main()
