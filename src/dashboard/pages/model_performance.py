"""Page 6 - Model Performance."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard import components as ui
from src.dashboard import data
from src.dashboard.theme import PALETTE


def _fmt(value, pattern="{:.3f}"):
    try:
        return pattern.format(float(value))
    except (TypeError, ValueError):
        return "—"


def _ci_text(block: dict, pattern="{:+.2%}") -> str:
    try:
        mean = float(block.get("mean"))
        low, high = block.get("ci95", [None, None])
        return f"{pattern.format(mean)}  [{pattern.format(low)}, {pattern.format(high)}]"
    except (TypeError, ValueError):
        return "—"


def _season_table(frame: pd.DataFrame) -> pd.DataFrame:
    from sklearn.metrics import log_loss

    mapping = {"A": 0, "D": 1, "H": 2}
    rows = []
    for season, group in frame.groupby("season", sort=True):
        y = group["result"].map(mapping).to_numpy()
        proba = group[["prob_away", "prob_draw", "prob_home"]].to_numpy(dtype=float)
        market = group[["market_away", "market_draw", "market_home"]].to_numpy(dtype=float)
        predicted = proba.argmax(axis=1)
        market_pred = market.argmax(axis=1)
        draw_mask = y == 1
        rows.append({
            "Season": season,
            "Matches": int(len(group)),
            "Accuracy": float((predicted == y).mean()) * 100,
            "Market acc.": float((market_pred == y).mean()) * 100,
            "Log loss": float(log_loss(y, proba, labels=[0, 1, 2])),
            "Market log loss": float(log_loss(y, market, labels=[0, 1, 2])),
            "Draw recall": (float((predicted[draw_mask] == 1).mean()) * 100
                            if draw_mask.any() else 0.0),
        })
    return pd.DataFrame(rows)


def render() -> None:
    report = data.report()
    meta = data.meta()

    ui.page_header(
        "Model Performance · Evidence",
        "Model Performance",
        "Selection used only the 2023/24 validation season. The 2024/25 and 2025/26 seasons "
        "were scored exactly once, after every hyper-parameter, blend weight and calibration "
        "temperature had been fixed.",
    )

    if not report:
        ui.empty_state("No training report",
                       "Run <code>python -m src.models.training</code> to produce the metrics.")
        return

    test = report.get("test_metrics") or {}
    market = report.get("market_baseline") or {}
    no_odds = report.get("no_odds_metrics") or {}
    per_class = test.get("per_class") or {}
    ci = test.get("accuracy_ci95") or [None, None]
    ci_text = (f"95% CI {ci[0]:.1%} – {ci[1]:.1%}"
               if ci[0] is not None and ci[1] is not None else "95% CI unavailable")

    ui.kpi_row([
        {"label": "Accuracy", "value": f"{test.get('accuracy', 0):.1%}",
         "delta": ci_text,
         "direction": "up" if test.get("accuracy", 0) >= 0.5 else "down"},
        {"label": "Macro F1", "value": f"{test.get('macro_f1', 0):.3f}",
         "delta": "Unweighted across H / D / A"},
        {"label": "Log loss", "value": f"{test.get('log_loss', 0):.4f}",
         "delta": f"Market baseline {market.get('log_loss', 0):.4f}"},
        {"label": "Brier score", "value": f"{test.get('brier', 0):.4f}",
         "delta": "Lower is better · 3-class sum convention"},
        {"label": "Calibration (ECE)", "value": f"{test.get('ece', 0):.4f}",
         "delta": "10 equal-width confidence bins"},
        {"label": "Draw recall", "value": f"{per_class.get('Draw', {}).get('recall', 0):.1%}",
         "delta": f"{test.get('prediction_distribution', {}).get('Draw', 0)} draws predicted",
         "direction": "down"},
    ])

    st.markdown(
        '<div class="plabel" style="margin-top:.9rem;gap:.5rem;flex-wrap:wrap">'
        f'{ui.tag("Production: " + str(report.get("production_model", "—")))}'
        f'{ui.tag("Poisson α " + str(report.get("poisson_alpha", "—")), "muted")}'
        f'{ui.tag("Market β " + str(report.get("market_beta", "—")), "muted")}'
        f'{ui.tag("Temperature " + str(report.get("temperature", "—")), "muted")}'
        f'{ui.tag(f"{report.get("n_features", 0)} features", "muted")}'
        f'{ui.tag("Accepted" if report.get("accepted") else "Below threshold", "ok" if report.get("accepted") else "bad")}'
        "</div>", unsafe_allow_html=True)

    # ---------------- comparison ----------------
    left, right = st.columns([1.05, 1], gap="medium")
    with left:
        ui.section("Accuracy against baselines")
        baselines = report.get("test_baselines") or report.get("baselines") or {}
        rows = [("This model", test.get("accuracy"), PALETTE["accent"]),
                ("Market consensus", market.get("accuracy"), PALETTE["accent_2"]),
                ("No-odds model", no_odds.get("accuracy"), PALETTE["good"]),
                ("Elo + logistic", (baselines.get("elo_logistic") or {}).get("accuracy"),
                 PALETTE["draw"]),
                ("Always home win", (baselines.get("home_always") or {}).get("accuracy"),
                 "#8C99B4"),
                ("Class prior", (baselines.get("majority_prior") or {}).get("accuracy"),
                 "#5A6685")]
        rows = [r for r in rows if r[1] is not None]
        st.plotly_chart(ui.metric_bar([r[0] for r in rows], [r[1] for r in rows],
                                      [r[2] for r in rows], height=290),
                        key="mp_baselines")
        st.caption("Every baseline is scored on the same 760 held-out test matches " 
                   "as the production model.")

    with right:
        ui.section("Per-class performance")
        rows = []
        for label in ("Home win", "Draw", "Away win"):
            metrics = per_class.get(label, {})
            rows.append({
                "Class": label,
                "Precision": metrics.get("precision", 0),
                "Recall": metrics.get("recall", 0),
                "F1": metrics.get("f1", 0),
                "Support": metrics.get("support", 0),
            })
        frame = pd.DataFrame(rows)
        st.dataframe(
            frame, hide_index=True, height=210,
            column_config={
                "Precision": st.column_config.NumberColumn(format="%.3f"),
                "Recall": st.column_config.NumberColumn(format="%.3f"),
                "F1": st.column_config.NumberColumn(format="%.3f"),
                "Support": st.column_config.NumberColumn(width="small"),
            })
        st.caption(
            f"Balanced accuracy {test.get('balanced_accuracy', 0):.3f}. The draw class is the "
            "weak point of every 1X2 model, including the bookmaker consensus."
        )

    # ---------------- confusion + calibration ----------------
    left, right = st.columns([1, 1], gap="medium")
    with left:
        ui.section("Test confusion matrix")
        matrix = test.get("confusion_matrix")
        if matrix:
            st.plotly_chart(ui.confusion_figure(matrix), key="mp_confusion")
    with right:
        ui.section("Calibration curve")
        curve = report.get("calibration") or []
        st.plotly_chart(ui.calibration_figure(curve), key="mp_calibration")
        st.caption("Points on the diagonal mean predicted confidence matches observed accuracy.")

    # ---------------- candidate comparison ----------------
    ui.section("Candidate models (validation season only)")
    candidates = report.get("candidates") or []
    if candidates:
        frame = pd.DataFrame(candidates)
        frame = frame[frame.get("status", "ok") == "ok"] if "status" in frame else frame
        keep = [c for c in ("name", "kind", "val_log_loss", "val_log_loss_calibrated",
                            "val_accuracy", "val_macro_f1", "temperature", "best_iteration")
                if c in frame.columns]
        frame = frame[keep].sort_values("val_log_loss")
        if "val_accuracy" in frame.columns:
            frame["val_accuracy"] = (pd.to_numeric(frame["val_accuracy"], errors="coerce") * 100).round(1)
        st.dataframe(
            frame, hide_index=True, height=min(420, 42 + 32 * len(frame)),
            column_config={
                "name": st.column_config.TextColumn("Model", width="medium"),
                "kind": st.column_config.TextColumn("Family", width="small"),
                "val_log_loss": st.column_config.NumberColumn("Val log loss", format="%.5f"),
                "val_log_loss_calibrated": st.column_config.NumberColumn("Calibrated", format="%.5f"),
                "val_accuracy": st.column_config.NumberColumn("Val acc.", format="%.1f%%"),
                "val_macro_f1": st.column_config.NumberColumn("Val macro F1", format="%.3f"),
                "temperature": st.column_config.NumberColumn("T", format="%.2f", width="small"),
                "best_iteration": st.column_config.NumberColumn("Iters", width="small"),
            })
        st.caption("Ranked by raw validation log loss. The ensemble re-blends the strongest "
                   "candidates and is only promoted if it beats the best single model.")

    # ---------------- draw analysis ----------------
    draws = data.draw_analysis()
    decisions = data.decisions()
    if draws:
        ui.section("Draw prediction analysis")
        production = (draws.get("variants") or {}).get("production") or {}
        summary = production.get("test_summary") or {}
        policy = draws.get("policy_selection") or {}
        selected = policy.get("selected")
        applied = draws.get("policy_applied") or {}
        test_policy = applied.get("test") or {}
        test_argmax = applied.get("test_argmax") or {}

        ui.kpi_row([
            {"label": "Actual draw rate", "value": f"{summary.get('actual_draw_rate', 0):.1%}",
             "delta": "Historical test seasons"},
            {"label": "Mean p(draw)", "value": f"{summary.get('mean_p_draw', 0):.1%}",
             "delta": f"Median {summary.get('median_p_draw', 0):.1%}"},
            {"label": "Draws predicted", "value": f"{summary.get('draws_predicted', 0)}",
             "delta": f"of {summary.get('n', 0)} matches", "direction": "down"},
            {"label": "Draw recall", "value": f"{summary.get('draw_recall', 0):.1%}",
             "delta": "Argmax decision rule", "direction": "down"},
            {"label": "Validation threshold", "value": f"{selected['threshold']:.2f}" if selected else "None",
             "delta": "Selected on 2023/24 only" if selected else "Argmax retained"},
        ])

        left, right = st.columns([1.1, 1], gap="medium")
        with left:
            st.plotly_chart(ui.draw_decile_figure(production.get("test_deciles") or []),
                            key="mp_draw_deciles")
            st.caption(
                "Draw probabilities occupy a narrow band: the model is reasonably ordered but "
                "almost never makes the draw its highest-probability outcome."
            )
        with right:
            st.plotly_chart(
                ui.policy_curve_figure(draws.get("threshold_curve_validation") or []),
                key="mp_draw_policy")
            st.caption(
                "Decision-policy sweep on the 2023/24 validation season. Thresholds were not "
                "tuned on the historical test seasons."
            )

        draw_decision = (decisions.get("draw_policy") or {})
        if draw_decision:
            kind = "ok" if draw_decision.get("promoted") else "warn"
            ui.note(draw_decision.get("summary", ""), kind)
        elif selected:
            change = (test_policy.get("accuracy", 0) - test_argmax.get("accuracy", 0))
            st.caption(
                f"Validation selected a draw threshold of {selected['threshold']:.2f}. "
                f"Applied unchanged to the historical test seasons it changed accuracy by "
                f"{change:+.2%} and macro F1 by "
                f"{test_policy.get('macro_f1', 0) - test_argmax.get('macro_f1', 0):+.3f}. "
                "See the walk-forward stability check before treating this as an improvement."
            )

    # ---------------- paired market comparison ----------------
    paired = data.paired_comparisons()
    if paired:
        ui.section("Statistical comparison with the bookmaker consensus")
        prod = (paired.get("test") or {}).get("production_vs_market") or {}
        no_odds_paired = (paired.get("test") or {}).get("no_odds_vs_market") or {}
        if prod:
            table = prod.get("agreement_table") or {}
            left, right = st.columns([1, 1.15], gap="medium")
            with left:
                agreement = pd.DataFrame([
                    {"Agreement": "Both correct", "Matches": table.get("both_correct", 0)},
                    {"Agreement": "Production only", "Matches": table.get("only_a_correct", 0)},
                    {"Agreement": "Market only", "Matches": table.get("only_b_correct", 0)},
                    {"Agreement": "Both incorrect", "Matches": table.get("both_wrong", 0)},
                ])
                st.dataframe(agreement, hide_index=True, height=180,
                             column_config={"Matches": st.column_config.NumberColumn(width="small")})
                st.caption(
                    f"Exact McNemar p = {prod.get('mcnemar_exact_p', 1):.3f}. A two-match "
                    "accuracy edge over 760 matches is not statistically meaningful."
                )
            with right:
                acc_diff = (prod.get("accuracy_difference") or {}).get("match_bootstrap") or {}
                ll_diff = (prod.get("log_loss_difference") or {}).get("match_bootstrap") or {}
                ll_mean = ll_diff.get("mean")
                rows = pd.DataFrame([
                    {"Metric": "Accuracy (model − market)",
                     "Difference": _ci_text(acc_diff)},
                    {"Metric": "Log loss (model − market)",
                     "Difference": _ci_text({**ll_diff,
                                             "mean": -ll_mean if ll_mean is not None else None,
                                             "ci95": [-ll_diff["ci95"][1], -ll_diff["ci95"][0]]
                                             if ll_diff.get("ci95") else [None, None]})},
                ])
                st.dataframe(rows, hide_index=True, height=140)
                st.caption(
                    "Paired bootstrap 95% intervals (4,000 resamples). The production model "
                    "matches market accuracy but has slightly worse probability quality; the "
                    "interval for log loss excludes zero, so this is a real, small deficit."
                )
        if no_odds_paired:
            st.caption(
                "No-odds model versus market: accuracy difference "
                f"{_ci_text((no_odds_paired.get('accuracy_difference') or {}).get('match_bootstrap') or {})}, "
                "log-loss difference "
                f"{_ci_text((no_odds_paired.get('log_loss_difference') or {}).get('match_bootstrap') or {})} "
                "(market − no-odds; negative means the market is better)."
            )

    # ---------------- calibration + sharpness ----------------
    calibration = data.calibration_analysis()
    if calibration:
        ui.section("Calibration, sharpness and reliability")
        metrics = calibration.get("metrics") or {}
        rows = []
        for name, label in (("raw", "Uncalibrated"), ("temperature", "Temperature"),
                            ("vector_scaling", "Vector scaling")):
            block = metrics.get(name) or {}
            val = block.get("validation") or {}
            test_block = block.get("test") or {}
            sharp = test_block.get("sharpness") or {}
            rows.append({
                "Method": label,
                "Val log loss": val.get("log_loss"),
                "Test log loss": test_block.get("log_loss"),
                "Test ECE": test_block.get("ece"),
                "Mean confidence": sharp.get("mean_confidence"),
                "Test accuracy": (test_block.get("accuracy") * 100
                                  if test_block.get("accuracy") is not None else None),
            })
        st.dataframe(
            pd.DataFrame(rows), hide_index=True, height=170,
            column_config={
                "Val log loss": st.column_config.NumberColumn(format="%.4f"),
                "Test log loss": st.column_config.NumberColumn(format="%.4f"),
                "Test ECE": st.column_config.NumberColumn(format="%.4f"),
                "Mean confidence": st.column_config.NumberColumn(format="%.3f"),
                "Test accuracy": st.column_config.NumberColumn(format="%.1f%%"),
            })
        st.caption(
            f"Selected on validation: {calibration.get('selected_method', '—').replace('_', ' ')}. "
            "Vector scaling improved validation log loss by less than the pre-registered 0.002 "
            "margin, so the simpler temperature scaling was retained."
        )
        per_class = ((metrics.get("temperature") or {}).get("test_per_class_reliability") or [])
        if per_class:
            st.plotly_chart(ui.per_class_reliability_figure(per_class), key="mp_per_class_reliability")
            st.caption(
                "Per-class reliability on the historical test seasons. The draw curve is the "
                "weakest: its probabilities are close to the base rate but poorly resolved."
            )

    # ---------------- season by season ----------------
    history = data.test_predictions()
    if not history.empty:
        ui.section("Season-by-season historical performance")
        seasons = _season_table(history)
        st.dataframe(
            seasons, hide_index=True, height=min(220, 40 + 32 * len(seasons)),
            column_config={
                "Accuracy": st.column_config.NumberColumn(format="%.1f%%"),
                "Market acc.": st.column_config.NumberColumn(format="%.1f%%"),
                "Log loss": st.column_config.NumberColumn(format="%.4f"),
                "Market log loss": st.column_config.NumberColumn(format="%.4f"),
                "Draw recall": st.column_config.NumberColumn(format="%.1f%%"),
            })
        st.caption(
            "The 2024/25 and 2025/26 seasons have already been examined; they are reported as "
            "historical benchmark evidence, not as a fresh prospective test."
        )

    # ---------------- walk-forward ----------------
    forward = data.walk_forward()
    if forward:
        ui.section("Walk-forward validation")
        origins = forward.get("origins") or []
        if origins:
            rows = []
            for origin in origins:
                policy_row = origin.get("draw_policy") or {}
                rows.append({
                    "Evaluation season": origin.get("season"),
                    "Trained through": origin.get("train_through"),
                    "Model": origin.get("selected_model"),
                    "Accuracy": ((origin.get("production") or {}).get("accuracy") or 0) * 100,
                    "Market acc.": ((origin.get("market") or {}).get("accuracy") or 0) * 100,
                    "Log loss": (origin.get("production") or {}).get("log_loss"),
                    "Draw policy": "Yes" if policy_row.get("selected") else "Argmax",
                    "Δ accuracy": (policy_row.get("accuracy_change") or 0) * 100,
                    "Δ macro F1": policy_row.get("macro_f1_change"),
                })
            st.dataframe(
                pd.DataFrame(rows), hide_index=True, height=min(340, 40 + 32 * len(rows)),
                column_config={
                    "Accuracy": st.column_config.NumberColumn(format="%.1f%%"),
                    "Market acc.": st.column_config.NumberColumn(format="%.1f%%"),
                    "Log loss": st.column_config.NumberColumn(format="%.4f"),
                    "Δ accuracy": st.column_config.NumberColumn(format="%+.1f%%"),
                    "Δ macro F1": st.column_config.NumberColumn(format="%+.3f"),
                })
            st.plotly_chart(ui.walk_forward_figure(origins), key="mp_walk_forward")
            summary = (forward.get("summary") or {}).get("production") or {}
            selection = summary.get("selection_origins") or {}
            if selection:
                mean = selection.get("mean") or {}
                std = selection.get("std") or {}
                st.caption(
                    "Selection-grade origins (2019/20–2023/24): mean accuracy "
                    f"{mean.get('accuracy', 0):.1%} ± {std.get('accuracy', 0):.1%}, "
                    f"mean log loss {mean.get('log_loss', 0):.4f} ± {std.get('log_loss', 0):.4f}. "
                    "The 2024/25 and 2025/26 origins are shown for completeness only."
                )
            stability = forward.get("draw_policy_stability") or {}
            if stability:
                st.caption(
                    "Draw-policy stability across selection-grade origins: "
                    f"{stability.get('origins_with_macro_gain', 0)} of "
                    f"{stability.get('origins', 0)} origins improved macro F1; mean accuracy "
                    f"change {stability.get('mean_accuracy_change', 0):+.2%}, mean macro-F1 "
                    f"change {stability.get('mean_macro_f1_change', 0):+.3f}."
                )

    # ---------------- challengers ----------------
    challenger_payload = data.challengers()
    if challenger_payload:
        ui.section("Challenger experiments")
        reference = challenger_payload.get("reference") or {}
        dixon = challenger_payload.get("dixon_coles") or {}
        core = challenger_payload.get("core_feature_set") or {}
        dc_blend = (dixon.get("blend_with_production") or {}).get("validation_selection") or {}
        rows = []

        def _add(label, val_block, test_block, decision):
            val_acc = (val_block or {}).get("accuracy")
            test_acc = (test_block or {}).get("accuracy")
            rows.append({
                "Candidate": label,
                "Val accuracy": val_acc * 100 if val_acc is not None else None,
                "Val log loss": (val_block or {}).get("log_loss"),
                "Test accuracy": test_acc * 100 if test_acc is not None else None,
                "Test log loss": (test_block or {}).get("log_loss"),
                "Decision": decision,
            })

        _add("Production (baseline)", reference.get("production_validation"),
             reference.get("production_test"), "Retained")
        _add("No-odds model", reference.get("no_odds_validation"),
             reference.get("no_odds_test"), "Reference")
        _add("Dixon-Coles standalone", dixon.get("validation"), dixon.get("test"),
             "Rejected — weaker than baseline")
        _add(f"Dixon-Coles + production (w={dc_blend.get('weight', 0):.2f})",
             dc_blend, (dixon.get("blend_with_production") or {}).get("test"),
             "Rejected — negligible validation gain")
        _add(f"Core features ({core.get('n_features', 0)} odds-free)",
             core.get("validation"), core.get("test"),
             "Rejected — accuracy fell, log loss gain too small")
        st.dataframe(
            pd.DataFrame(rows), hide_index=True, height=min(300, 40 + 32 * len(rows)),
            column_config={
                "Val accuracy": st.column_config.NumberColumn(format="%.1f%%"),
                "Val log loss": st.column_config.NumberColumn(format="%.4f"),
                "Test accuracy": st.column_config.NumberColumn(format="%.1f%%"),
                "Test log loss": st.column_config.NumberColumn(format="%.4f"),
            })
        st.caption(
            "Selection used the 2023/24 validation season. Test columns are historical "
            "benchmark evidence only and did not influence any decision."
        )

    # ---------------- status and limitations ----------------
    if decisions:
        ui.section("Model status and limitations")
        status = decisions.get("status") or {}
        ui.note(status.get("summary", ""), "ok" if status.get("promoted") else "warn")
        limitations = decisions.get("limitations") or []
        if limitations:
            st.markdown("\n".join(f"- {item}" for item in limitations))

    # ---------------- ensemble + importance ----------------
    left, right = st.columns([1, 1.15], gap="medium")
    with left:
        ui.section("Ensemble composition")
        ensemble = report.get("ensemble") or {}
        names = ensemble.get("names") or []
        weights = ensemble.get("weights") or []
        if names and weights:
            rows = pd.DataFrame({"model": names, "weight": weights})
            st.dataframe(rows, hide_index=True, height=190,
                         column_config={"weight": st.column_config.NumberColumn(
                             "Weight", format="%.3f")})
            st.caption(f"Blend strategy: {ensemble.get('weighting', '—')}. "
                       f"Shrunk toward equal weights to limit validation overfitting.")
        else:
            st.caption("The selected pipeline is a single model, not an ensemble.")
    with right:
        ui.section("Most influential features")
        importance = report.get("top_features") or []
        if importance:
            st.plotly_chart(ui.importance_figure(importance, top=14), key="mp_importance")
        else:
            st.caption("Feature importance unavailable for this model family.")

    # ---------------- split + methodology ----------------
    ui.section("Evaluation protocol")
    split = report.get("split") or {}
    rows = []
    for name, label in (("train", "Training"), ("validation", "Validation"), ("test", "Test")):
        block = split.get(name) or {}
        rows.append({
            "Split": label,
            "Matches": block.get("n", 0),
            "From": block.get("start", "—"),
            "To": block.get("end", "—"),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True,
                 column_config={"Matches": st.column_config.NumberColumn(width="small")})

    ui.note(
        "No random split was used anywhere. Features are produced by a single chronological "
        "state machine that updates only after every match on a given date has been "
        "featurised, and the leakage proofs in Data Centre re-verify this by flipping "
        "individual results and truncating future data."
    )
    st.caption(
        f"Trained {meta.get('trained_at', report.get('trained_at', '—'))} · "
        f"data cutoff {meta.get('data_cutoff', report.get('data_cutoff', '—'))} · "
        f"Python {report.get('python', '—')}"
    )
