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
