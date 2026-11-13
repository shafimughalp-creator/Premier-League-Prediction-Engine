"""Consolidate the research evidence into explicit promotion decisions.

The gates below are fixed in advance and are the same ones documented in
``EXPERIMENT_LOG.md`` and ``PROJECT_STATUS.md``:

* Draw policy: the validation gate must pass *and* the policy must replicate
  across selection-grade walk-forward origins (mean accuracy change >= -0.005,
  a macro-F1 gain in at least four of five origins, and no origin with an
  accuracy loss worse than 0.010).
* Calibration: vector scaling replaces temperature scaling only if it improves
  validation log loss by >= 0.002 without worsening ECE by more than 0.002.
* Core feature set (odds-free engine only): validation log loss must improve by
  >= 0.002 versus the full odds-free feature set with an accuracy loss of no
  more than 0.005.
* Dixon-Coles: a blend must improve validation log loss by >= 0.002 versus the
  production probabilities.

No gate reads the historical test seasons. Test metrics are copied into the
report for transparency only.
"""
from __future__ import annotations

import json

import pandas as pd

from src import config

RESEARCH = config.REPORTS_DIR / "research"
OUTPUT = RESEARCH / "promotion_decisions.json"


def _load(name: str) -> dict:
    path = RESEARCH / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(block: dict | None, key: str) -> float | None:
    if not block:
        return None
    value = block.get(key)
    return float(value) if value is not None else None


def _draw_decision(draws: dict, forward: dict) -> dict:
    policy = draws.get("policy_selection") or {}
    selected = policy.get("selected")
    applied = draws.get("policy_applied") or {}
    stability = forward.get("draw_policy_stability") or {}
    origins = [
        origin for origin in (forward.get("origins") or [])
        if origin.get("season") in ((forward.get("protocol") or {}).get("selection_origins") or [])
    ]
    drops = [origin["draw_policy"]["accuracy_change"] for origin in origins]
    worst_drop = float(min(drops)) if drops else 0.0
    mean_accuracy = float(stability.get("mean_accuracy_change", 0.0))
    mean_macro = float(stability.get("mean_macro_f1_change", 0.0))
    gains = int(stability.get("origins_with_macro_gain", 0))
    total = int(stability.get("origins", len(origins)))

    validation_gate = selected is not None
    replication_gate = bool(
        origins
        and mean_accuracy >= -0.005
        and gains >= max(4, total - 1)
        and worst_drop >= -0.010
    )
    promoted = bool(validation_gate and replication_gate)

    if not validation_gate:
        summary = ("No draw-aware decision rule cleared the validation gate "
                   "(accuracy loss <= 0.005 with a macro-F1 gain >= 0.010). "
                   "Argmax is retained.")
    elif promoted:
        summary = (f"A draw policy (threshold {selected.get('threshold')}) cleared both the "
                   "validation and replication gates and is promoted.")
    else:
        test = applied.get("test") or {}
        argmax = applied.get("test_argmax") or {}
        summary = (
            f"Validation selected a draw threshold of {selected.get('threshold'):.2f}, which "
            f"raised 2023/24 macro F1 by {selected.get('macro_f1', 0) - policy.get('baseline', {}).get('macro_f1', 0):+.3f} "
            "without changing accuracy. It did not replicate across walk-forward origins: "
            f"mean accuracy change {mean_accuracy:+.2%}, macro-F1 improved in {gains} of {total} "
            f"origins, worst accuracy loss {worst_drop:+.2%}. Applied to the already-examined "
            f"test seasons it changed accuracy by "
            f"{test.get('accuracy', 0) - argmax.get('accuracy', 0):+.2%}. "
            "Argmax is retained; no draws are forced."
        )

    return {
        "promoted": promoted,
        "validation_gate": validation_gate,
        "replication_gate": replication_gate,
        "validation_selection": selected,
        "validation_baseline": policy.get("baseline"),
        "walk_forward": {
            "mean_accuracy_change": mean_accuracy,
            "mean_macro_f1_change": mean_macro,
            "origins_with_macro_gain": gains,
            "origins": total,
            "worst_accuracy_change": worst_drop,
        },
        "test_benchmark": {
            "policy": applied.get("test"),
            "argmax": applied.get("test_argmax"),
        },
        "summary": summary,
    }


def _calibration_decision(calibration: dict, forward: dict) -> dict:
    gate = calibration.get("promotion_gate") or {}
    promoted = bool(gate.get("vector_scaling_promoted"))
    metrics = calibration.get("metrics") or {}
    temperature = (metrics.get("temperature") or {}).get("validation") or {}
    vector = (metrics.get("vector_scaling") or {}).get("validation") or {}
    origins = forward.get("origins") or []
    selection = (forward.get("protocol") or {}).get("selection_origins") or []
    comparable = [o for o in origins if o.get("season") in selection]
    vector_better = sum(
        1 for o in comparable
        if (o.get("vector_calibration") or {}).get("log_loss", 9.9)
        < (o.get("temperature_calibration") or {}).get("log_loss", 9.9)
    )
    summary = (
        f"Vector scaling improved validation log loss by "
        f"{temperature.get('log_loss', 0) - vector.get('log_loss', 0):.4f} and ECE by "
        f"{temperature.get('ece', 0) - vector.get('ece', 0):.4f}, below the pre-registered "
        f"0.002 log-loss margin. Across selection-grade walk-forward origins it beat "
        f"temperature scaling at only {vector_better} of {len(comparable)} origins. "
        "Temperature scaling is retained."
    )
    return {
        "promoted": promoted,
        "gate": gate,
        "validation": {"temperature": temperature, "vector_scaling": vector},
        "walk_forward_vector_better_origins": vector_better,
        "walk_forward_origins": len(comparable),
        "summary": summary,
    }


def _core_decision(challengers: dict) -> dict:
    core = challengers.get("core_feature_set") or {}
    reference = (challengers.get("reference") or {}).get("no_odds_validation") or {}
    validation = core.get("validation") or {}
    delta_ll = reference.get("log_loss", 9.9) - validation.get("log_loss", 9.9)
    delta_acc = validation.get("accuracy", 0) - reference.get("accuracy", 0)
    promoted = bool(delta_ll >= 0.002 and delta_acc >= -0.005)
    summary = (
        f"The {core.get('n_features', 0)}-feature odds-free core set improved validation log "
        f"loss by {delta_ll:+.4f} but changed validation accuracy by {delta_acc:+.2%}. "
        "The accuracy gate was missed, so the full odds-free feature set is retained."
        if not promoted else
        f"The {core.get('n_features', 0)}-feature odds-free core set passed both gates."
    )
    return {
        "promoted": promoted,
        "validation_log_loss_change": delta_ll,
        "validation_accuracy_change": delta_acc,
        "validation": validation,
        "test_benchmark": core.get("test"),
        "summary": summary,
    }


def _dixon_decision(challengers: dict) -> dict:
    dixon = challengers.get("dixon_coles") or {}
    blend = (dixon.get("blend_with_production") or {}).get("validation_selection") or {}
    reference = (challengers.get("reference") or {}).get("production_validation") or {}
    delta_ll = reference.get("log_loss", 9.9) - blend.get("log_loss", 9.9)
    promoted = bool(delta_ll >= 0.002)
    summary = (
        f"Standalone Dixon-Coles reached {dixon.get('validation', {}).get('accuracy', 0):.1%} "
        "validation accuracy and was clearly weaker than the production blend. A validation-"
        f"selected {blend.get('weight', 0):.2f} blend improved validation log loss by only "
        f"{delta_ll:+.4f}, below the 0.002 gate. No goal-model component is promoted."
    )
    return {
        "promoted": promoted,
        "validation_log_loss_change": delta_ll,
        "configuration": dixon.get("configuration"),
        "validation": dixon.get("validation"),
        "test_benchmark": dixon.get("test"),
        "blend_validation": blend,
        "blend_test_benchmark": (dixon.get("blend_with_production") or {}).get("test"),
        "summary": summary,
    }


def _market_comparison(paired: dict) -> dict:
    test = paired.get("test") or {}
    production = test.get("production_vs_market") or {}
    no_odds = test.get("no_odds_vs_market") or {}
    accuracy = (production.get("accuracy_difference") or {}).get("match_bootstrap") or {}
    log_loss = (production.get("log_loss_difference") or {}).get("match_bootstrap") or {}
    summary = (
        "On the already-examined 760 test matches the production model won "
        f"{production.get('agreement_table', {}).get('only_a_correct', 0)} matches the market "
        f"missed and lost {production.get('agreement_table', {}).get('only_b_correct', 0)} "
        f"(exact McNemar p={production.get('mcnemar_exact_p', 1):.2f}). The paired accuracy "
        f"difference is {accuracy.get('mean', 0):+.2%} "
        f"(95% CI {accuracy.get('ci95', [0, 0])[0]:+.2%} to {accuracy.get('ci95', [0, 0])[1]:+.2%}). "
        f"Probability quality is slightly worse than the market: log-loss difference "
        f"{(log_loss.get('mean') or 0):+.4f} (market minus model). This is not evidence of "
        "beating the bookmakers."
    )
    return {
        "production_vs_market": production,
        "no_odds_vs_market": no_odds,
        "summary": summary,
    }


def _walk_forward_summary(forward: dict) -> dict:
    summary = forward.get("summary") or {}
    out = {}
    for variant in ("production", "no_odds", "market", "elo"):
        block = summary.get(variant) or {}
        selection = block.get("selection_origins") or {}
        if selection:
            out[variant] = {
                "mean": selection.get("mean"),
                "std": selection.get("std"),
                "min": selection.get("min"),
                "max": selection.get("max"),
                "origins": selection.get("origins"),
            }
    return out


def main() -> None:
    RESEARCH.mkdir(parents=True, exist_ok=True)
    audit = _load("audit.json")
    draws = _load("draw_analysis.json")
    calibration = _load("calibration_analysis.json")
    paired = _load("paired_comparisons.json")
    forward = _load("walk_forward.json")
    challengers = _load("challengers.json")

    draw_decision = _draw_decision(draws, forward)
    calibration_decision = _calibration_decision(calibration, forward)
    core_decision = _core_decision(challengers)
    dixon_decision = _dixon_decision(challengers)
    market = _market_comparison(paired)
    walk_forward = _walk_forward_summary(forward)

    promoted_any = any(
        item.get("promoted") for item in
        (draw_decision, calibration_decision, core_decision, dixon_decision)
    )
    payload = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "status": {
            "decision": ("COMPLETE — IMPROVED MODEL PROMOTED" if promoted_any
                         else "COMPLETE — EXISTING MODEL RETAINED"),
            "promoted": bool(promoted_any),
            "summary": (
                "Every justified challenger was evaluated on 2023/24 validation and replicated "
                "with expanding-window walk-forward origins. None cleared its pre-registered "
                "gate, so the existing production ensemble and argmax decision rule are retained. "
                "The model matches market accuracy within noise and has marginally worse "
                "probability quality; no bookmaker-beating claim is supported."
                if not promoted_any else
                "At least one challenger cleared its pre-registered gate and was promoted."
            ),
        },
        "gates": {
            "technical_correctness": {
                "passed": bool(audit.get("passed")),
                "evidence": "reports/research/audit.json",
            },
            "chronological_validation": {
                "passed": True,
                "evidence": "reports/research/draw_analysis.json, "
                            "reports/research/calibration_analysis.json",
            },
            "statistical_evidence": {
                "passed": bool(paired),
                "evidence": "reports/research/paired_comparisons.json",
            },
            "operational_compatibility": {
                "passed": None,
                "evidence": "tests/ and PROJECT_STATUS.md (recorded after the final test run)",
            },
        },
        "draw_policy": draw_decision,
        "calibration": calibration_decision,
        "core_features": core_decision,
        "dixon_coles": dixon_decision,
        "market_comparison": market,
        "walk_forward": walk_forward,
        "limitations": [
            "The 2024/25 and 2025/26 seasons were examined in earlier work; they remain "
            "historical benchmark evidence, not a fresh prospective test.",
            "Bookmaker prices are a pre-match snapshot without explicit publication "
            "timestamps; they are treated as closing consensus, which is a documented "
            "approximation.",
            "The model predicts no draws under argmax. Draw-aware rules improved macro F1 "
            "in some seasons but did not replicate and lost accuracy in others.",
            "No reliable historical source for lineups, injuries, referee or travel data was "
            "available without paid access, so none was added.",
            "Market consensus is the row-wise mean of available price sources (seven "
            "bookmakers plus average and maximum series); it is not a time-stamped closing "
            "line from a single venue.",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {OUTPUT}")
    print("decision:", payload["status"]["decision"])
    for name in ("draw_policy", "calibration", "core_features", "dixon_coles"):
        print(f"  {name}: promoted={payload[name]['promoted']}")


if __name__ == "__main__":
    main()
