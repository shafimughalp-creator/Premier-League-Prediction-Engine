"""Train, calibrate, compare and persist EPL outcome models.

Selection is strictly chronological:

* train      : matches before 2023-07-01 (2014/15 - 2022/23)
* validation : 2023/24
* test       : 2024/25 + 2025/26 (untouched until the final score)

The 2026/27 partial season is used only for the final production refit and for
current predictions.
"""
from __future__ import annotations

import json
import logging
import platform
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.data.processing import load_matches, data_quality_report, build_standings
from src.features.builder import FeatureBuilder, feature_columns, odds_free_columns
from src.models.ensemble import SoftVotingModel
from src.models.evaluation import (
    calibration_curve,
    evaluate,
    season_breakdown,
)

log = logging.getLogger(__name__)
CLASSES = [0, 1, 2]
ODDS_PREFIXES = ("mkt_", "odds_")


# --------------------------------------------------------------------------
# Calibration and blending
# --------------------------------------------------------------------------
def apply_temperature(proba: np.ndarray, temperature: float) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    if temperature is None or not np.isfinite(temperature) or temperature <= 0:
        return proba
    p = np.clip(proba, 1e-12, 1.0) ** (1.0 / temperature)
    return p / p.sum(axis=1, keepdims=True)


def fit_temperature(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    best_t, best_ll = 1.0, float(log_loss(y_true, proba, labels=CLASSES))
    for temperature in np.linspace(0.60, 1.80, 49):
        ll = float(log_loss(y_true, apply_temperature(proba, float(temperature)), labels=CLASSES))
        if ll < best_ll:
            best_t, best_ll = float(temperature), ll
    return best_t, best_ll


POISSON_COLUMNS = ("pois_p_away", "pois_p_draw", "pois_p_home")


def poisson_probabilities(df: pd.DataFrame) -> np.ndarray:
    if not set(POISSON_COLUMNS).issubset(df.columns):
        return np.full((len(df), 3), np.nan)
    return df[list(POISSON_COLUMNS)].to_numpy(dtype=float)


def blend_with_poisson(proba: np.ndarray, poisson: np.ndarray, alpha: float) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    if alpha is None or not np.isfinite(alpha) or alpha >= 1.0:
        return proba
    alpha = float(np.clip(alpha, 0.0, 1.0))
    out = alpha * proba + (1.0 - alpha) * poisson
    bad = ~np.isfinite(poisson).all(axis=1)
    if bad.any():
        out[bad] = proba[bad]
    totals = out.sum(axis=1, keepdims=True)
    totals[totals <= 0] = 1.0
    return out / totals


def fit_poisson_blend(y_true: np.ndarray, model_proba: np.ndarray,
                      poisson_proba: np.ndarray) -> tuple[float, float, float]:
    best = (1.0, 1.0, float("inf"))
    for alpha in np.linspace(0.0, 1.0, 21):
        blended = blend_with_poisson(model_proba, poisson_proba, float(alpha))
        temperature, ll = fit_temperature(y_true, blended)
        if ll < best[2]:
            best = (float(alpha), float(temperature), float(ll))
    return best


def fit_market_blend(y_true: np.ndarray, model_proba: np.ndarray,
                     market_proba: np.ndarray) -> tuple[float, float, float]:
    """Blend model probabilities with the bookmaker consensus.

    Returns ``(beta, temperature, calibrated_log_loss)`` where ``beta=1`` keeps
    the model only and ``beta=0`` uses the market only.
    """
    best = (1.0, 1.0, float("inf"))
    for beta in np.linspace(0.0, 1.0, 21):
        blended = beta * model_proba + (1.0 - beta) * market_proba
        temperature, ll = fit_temperature(y_true, blended)
        if ll < best[2]:
            best = (float(beta), float(temperature), float(ll))
    return best


# --------------------------------------------------------------------------
# Model zoo
# --------------------------------------------------------------------------
CANDIDATES = [
    {
        "name": "lgbm_regularised", "kind": "lightgbm",
        "params": dict(learning_rate=0.03, num_leaves=15, min_child_samples=40,
                       subsample=0.85, subsample_freq=1, colsample_bytree=0.60,
                       reg_lambda=5.0, reg_alpha=0.5),
        "n_estimators": 1500,
    },
    {
        "name": "lgbm_shallow", "kind": "lightgbm",
        "params": dict(learning_rate=0.05, num_leaves=7, min_child_samples=25,
                       subsample=0.95, subsample_freq=1, colsample_bytree=0.75,
                       reg_lambda=1.0),
        "n_estimators": 1200,
    },
    {
        "name": "lgbm_deep", "kind": "lightgbm",
        "params": dict(learning_rate=0.02, num_leaves=31, min_child_samples=60,
                       subsample=0.75, subsample_freq=1, colsample_bytree=0.50,
                       reg_lambda=12.0, reg_alpha=1.0),
        "n_estimators": 2000,
    },
    {
        "name": "xgb_regularised", "kind": "xgboost",
        "params": dict(learning_rate=0.03, max_depth=4, min_child_weight=8,
                       subsample=0.85, colsample_bytree=0.60, reg_lambda=6.0),
        "n_estimators": 1500,
    },
    {
        "name": "xgb_deep", "kind": "xgboost",
        "params": dict(learning_rate=0.02, max_depth=6, min_child_weight=12,
                       subsample=0.75, colsample_bytree=0.50, reg_lambda=12.0),
        "n_estimators": 2000,
    },
    {
        "name": "random_forest", "kind": "random_forest",
        "params": dict(max_depth=12, min_samples_leaf=10, max_features="sqrt"),
        "n_estimators": 600,
    },
    {
        "name": "extra_trees", "kind": "extra_trees",
        "params": dict(max_depth=14, min_samples_leaf=8, max_features="sqrt"),
        "n_estimators": 700,
    },
    {
        "name": "hist_gradient", "kind": "hist_gradient",
        "params": dict(max_iter=500, learning_rate=0.05, max_leaf_nodes=15,
                       l2_regularization=1.0, min_samples_leaf=20),
        "n_estimators": 500,
    },
    {
        "name": "logistic", "kind": "logistic",
        "params": dict(C=0.5),
        "n_estimators": 1,
    },
]


def _build(kind: str, params: dict, n_estimators: int):
    if kind == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            objective="multiclass", num_class=3, n_estimators=n_estimators,
            random_state=config.RANDOM_STATE, verbose=-1, n_jobs=config.N_JOBS,
            **params,
        )
    if kind == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            objective="multi:softprob", num_class=3, n_estimators=n_estimators,
            eval_metric="mlogloss", tree_method="hist",
            random_state=config.RANDOM_STATE, n_jobs=config.N_JOBS, **params,
        )
    if kind == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=n_estimators, random_state=config.RANDOM_STATE,
                n_jobs=config.N_JOBS, **params,
            )),
        ])
    if kind == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesClassifier(
                n_estimators=n_estimators, random_state=config.RANDOM_STATE,
                n_jobs=config.N_JOBS, **params,
            )),
        ])
    if kind == "hist_gradient":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                random_state=config.RANDOM_STATE, early_stopping=False, **params,
            )),
        ])
    if kind == "logistic":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, random_state=config.RANDOM_STATE,
                                         **params)),
        ])
    raise ValueError(f"Unknown model kind: {kind}")


def _fit_once(kind: str, params: dict, n_estimators: int, Xtr, ytr, Xv, yv):
    model = _build(kind, params, n_estimators)
    if kind == "lightgbm":
        import lightgbm as lgb
        model.fit(
            Xtr, ytr, eval_set=[(Xv, yv)], eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        best_iter = int(model.best_iteration_ or n_estimators)
    elif kind == "xgboost":
        model.set_params(early_stopping_rounds=100)
        model.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
        best_iter = int(getattr(model, "best_iteration", n_estimators) or n_estimators)
    else:
        model.fit(Xtr, ytr)
        best_iter = n_estimators
    return model, best_iter, model.predict_proba(Xv)


def _refit(kind: str, params: dict, X, y, n_estimators: int):
    model = _build(kind, params, n_estimators)
    model.fit(X, y)
    return model


def _prior_proba(y: np.ndarray, n: int) -> np.ndarray:
    counts = np.bincount(y, minlength=3).astype(float)
    return np.tile(counts / counts.sum(), (n, 1))


def _elo_baseline(Xtr, ytr, Xv):
    cols = ["elo_diff", "elo_expected_home", "h_elo", "a_elo"]
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=config.RANDOM_STATE)),
    ])
    pipe.fit(Xtr[cols], ytr)
    return pipe, pipe.predict_proba(Xv[cols])


def _optimize_weights(y: np.ndarray, probas: list[np.ndarray]) -> np.ndarray:
    k = len(probas)
    if k == 1:
        return np.array([1.0])
    stack = np.stack(probas)

    def objective(w):
        w = np.abs(np.asarray(w, dtype=float))
        w = w / max(w.sum(), 1e-12)
        return float(log_loss(y, np.tensordot(w, stack, axes=(0, 0)), labels=CLASSES))

    result = minimize(
        objective, np.ones(k) / k, method="SLSQP",
        bounds=[(0.0, 1.0)] * k,
        constraints={"type": "eq", "fun": lambda w: np.sum(np.abs(w)) - 1.0},
        options={"maxiter": 200, "ftol": 1e-8},
    )
    weights = np.abs(result.x)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        return np.ones(k) / k
    return weights / weights.sum()


def _feature_importance(model, features: list[str], top: int = 20) -> list[dict]:
    try:
        if isinstance(model, SoftVotingModel):
            model = model.models[0]
        if isinstance(model, Pipeline):
            model = model.named_steps.get("model", model)
        if hasattr(model, "feature_importances_"):
            importance = np.asarray(model.feature_importances_, dtype=float)
        elif hasattr(model, "booster_"):
            importance = np.asarray(model.booster_.feature_importance(importance_type="gain"),
                                    dtype=float)
        else:
            return []
        total = importance.sum() or 1.0
        order = np.argsort(importance)[::-1][:top]
        return [{"feature": features[i], "importance": round(float(importance[i] / total), 4)}
                for i in order if i < len(features)]
    except Exception:
        return []


# --------------------------------------------------------------------------
# Pipeline selection
# --------------------------------------------------------------------------
def _split(df: pd.DataFrame):
    train = df[df["_date"] < pd.Timestamp(config.TRAIN_END)]
    val = df[(df["_date"] >= pd.Timestamp(config.TRAIN_END))
             & (df["_date"] < pd.Timestamp(config.VAL_END))]
    test = df[(df["_date"] >= pd.Timestamp(config.VAL_END))
              & (df["_date"] < pd.Timestamp(config.TEST_END))]
    current = df[df["_date"] >= pd.Timestamp(config.TEST_END)]
    return train, val, test, current


def _prepare(df: pd.DataFrame, columns: list[str]):
    return df[columns].astype("float64"), df["_label"].to_numpy(dtype=int)


def _fit_candidates(train, val, columns, ytr, yv, candidates=None):
    Xtr, _ = _prepare(train, columns)
    Xv, _ = _prepare(val, columns)
    fitted = {}
    records = []
    specs = CANDIDATES if candidates is None else candidates
    for spec in specs:
        name = spec["name"]
        try:
            model, best_iter, val_proba = _fit_once(
                spec["kind"], spec["params"], spec["n_estimators"], Xtr, ytr, Xv, yv)
            temperature, cal_ll = fit_temperature(yv, val_proba)
            calibrated = apply_temperature(val_proba, temperature)
            metrics = evaluate(yv, calibrated)
            fitted[name] = {
                "model": model, "val_proba": val_proba, "best_iter": best_iter,
                "spec": spec, "temperature": temperature,
            }
            records.append({
                "name": name, "kind": spec["kind"], "status": "ok",
                "val_log_loss": round(float(log_loss(yv, val_proba, labels=CLASSES)), 5),
                "val_log_loss_calibrated": round(cal_ll, 5),
                "val_accuracy": round(metrics["accuracy"], 5),
                "val_macro_f1": round(metrics["macro_f1"], 5),
                "temperature": round(temperature, 4),
                "best_iteration": best_iter,
            })
            log.info("Candidate %-18s val_ll=%.5f cal=%.5f acc=%.4f iter=%d",
                     name, records[-1]["val_log_loss"], cal_ll, metrics["accuracy"], best_iter)
        except Exception as exc:
            log.warning("Candidate %s failed: %s", name, exc)
            records.append({"name": name, "kind": spec["kind"],
                            "status": f"failed: {exc}"})
    return fitted, records


def _build_ensemble(fitted: dict, yv: np.ndarray) -> dict | None:
    usable = {k: v for k, v in fitted.items() if v.get("val_proba") is not None}
    if len(usable) < 2:
        return None
    ranked = sorted(usable, key=lambda k: log_loss(yv, usable[k]["val_proba"], labels=CLASSES))
    top = ranked[:4]
    probas = [usable[k]["val_proba"] for k in top]
    weights = _optimize_weights(yv, probas)
    equal = np.ones(len(top)) / len(top)
    # Shrink toward equal weights to reduce validation overfitting.
    shrunk = 0.5 * weights + 0.5 * equal
    candidates = {"optimised": weights, "shrunk": shrunk, "equal": equal}
    best_name, best_w, best_ll = None, None, float("inf")
    for label, w in candidates.items():
        blend = np.tensordot(w, np.stack(probas), axes=(0, 0))
        ll = float(log_loss(yv, blend, labels=CLASSES))
        if ll < best_ll:
            best_name, best_w, best_ll = label, w, ll
    blend = np.tensordot(best_w, np.stack(probas), axes=(0, 0))
    temperature, cal_ll = fit_temperature(yv, blend)
    metrics = evaluate(yv, apply_temperature(blend, temperature))
    return {
        "names": top, "weights": [float(w) for w in best_w], "weighting": best_name,
        "val_proba": blend, "temperature": temperature,
        "val_log_loss": best_ll, "val_log_loss_calibrated": cal_ll,
        "val_accuracy": metrics["accuracy"], "val_macro_f1": metrics["macro_f1"],
    }


def _select_and_score(train, val, test, current, columns, label: str, candidates=None):
    ytr = train["_label"].to_numpy(int)
    yv = val["_label"].to_numpy(int)
    yt = test["_label"].to_numpy(int)
    Xt, _ = _prepare(test, columns)

    fitted, candidate_records = _fit_candidates(train, val, columns, ytr, yv, candidates)

    # Baselines
    prior_val = _prior_proba(ytr, len(val))
    prior_metrics = evaluate(yv, prior_val)
    home_proba = np.tile(np.array([0.01, 0.01, 0.98]), (len(val), 1))
    home_metrics = evaluate(yv, home_proba)
    elo_pipe, elo_T = None, 1.0
    try:
        Xtr_elo, _ = _prepare(train, columns)
        elo_pipe, elo_val = _elo_baseline(Xtr_elo, ytr, val[columns].astype("float64"))
        elo_T, _ = fit_temperature(yv, elo_val)
        elo_metrics = evaluate(yv, apply_temperature(elo_val, elo_T))
    except Exception as exc:
        log.warning("Elo baseline failed: %s", exc)
        elo_metrics = {"status": "failed"}

    baseline_records = {
        "majority_prior": prior_metrics,
        "home_always": home_metrics,
        "elo_logistic": elo_metrics,
    }

    # Best single model by calibrated validation log loss.
    if not fitted:
        raise RuntimeError(f"No model candidate fitted for {label}")
    best_name = min(fitted, key=lambda k: log_loss(
        yv, apply_temperature(fitted[k]["val_proba"], fitted[k]["temperature"]), labels=CLASSES))
    best = fitted[best_name]
    best_cal = apply_temperature(best["val_proba"], best["temperature"])
    best_ll = float(log_loss(yv, best_cal, labels=CLASSES))

    ensemble = _build_ensemble(fitted, yv)
    use_ensemble = ensemble is not None and ensemble["val_log_loss_calibrated"] < best_ll - 1e-5
    if use_ensemble:
        base_proba = ensemble["val_proba"]
        base_T = ensemble["temperature"]
        base_name = "ensemble"
        base_ll = ensemble["val_log_loss_calibrated"]
        log.info("Selected ensemble of %s (%s) val_ll=%.5f", ensemble["names"],
                 ensemble["weighting"], base_ll)
    else:
        base_proba = best["val_proba"]
        base_T = best["temperature"]
        base_name = best_name
        base_ll = best_ll
        log.info("Selected %s val_ll=%.5f", base_name, base_ll)

    # Poisson prior blend, tuned on validation only.
    pois_val = poisson_probabilities(val)
    alpha, temperature, blended_ll = fit_poisson_blend(yv, base_proba, pois_val)
    log.info("Poisson blend %s: alpha=%.2f T=%.3f val_ll %.5f -> %.5f",
             label, alpha, temperature, base_ll, blended_ll)
    blended_val = blend_with_poisson(base_proba, pois_val, alpha)

    # Market consensus blend for the odds-aware model, also validation-only.
    market_beta, market_temperature, market_ll = 1.0, temperature, blended_ll
    market_val = val[["mkt_away", "mkt_draw", "mkt_home"]].to_numpy(dtype=float)
    if label == "full" and np.isfinite(market_val).all():
        beta, mkt_T, mkt_ll = fit_market_blend(yv, blended_val, market_val)
        if mkt_ll < blended_ll - 0.001:
            market_beta, market_temperature, market_ll = beta, mkt_T, mkt_ll
            log.info("Market blend %s: beta=%.2f T=%.3f val_ll %.5f -> %.5f",
                     label, beta, mkt_T, blended_ll, mkt_ll)
        else:
            log.info("Market blend %s rejected (val_ll %.5f vs %.5f)",
                     label, mkt_ll, blended_ll)

    # Test probabilities from the validation-selected pipeline.
    if use_ensemble:
        base_models = [fitted[n]["model"] for n in ensemble["names"]]
        eval_model = SoftVotingModel(base_models, ensemble["weights"], ensemble["names"])
    else:
        eval_model = best["model"]
    test_proba_raw = eval_model.predict_proba(Xt)
    test_base = blend_with_poisson(test_proba_raw, poisson_probabilities(test), alpha)
    if market_beta < 1.0:
        market_test = test[["mkt_away", "mkt_draw", "mkt_home"]].to_numpy(dtype=float)
        test_base = market_beta * test_base + (1.0 - market_beta) * market_test
    test_proba = apply_temperature(test_base, market_temperature)
    test_metrics = evaluate(yt, test_proba)

    # Validation probabilities from the *same* frozen pipeline, so decision
    # rules and calibration can be studied without touching the test seasons.
    val_final = blended_val
    if market_beta < 1.0:
        val_final = market_beta * blended_val + (1.0 - market_beta) * market_val
    val_proba = apply_temperature(val_final, market_temperature)

    # The same baselines scored on the *test* rows, so the dashboard compares
    # like with like. The validation baselines above drive model selection; the
    # test baselines below are purely descriptive evidence.
    test_baselines = {
        "majority_prior": evaluate(yt, _prior_proba(ytr, len(test))),
        "home_always": evaluate(yt, np.tile(np.array([0.01, 0.01, 0.98]), (len(test), 1))),
    }
    if elo_pipe is not None:
        try:
            Xt_elo, _ = _prepare(test, columns)
            test_elo = elo_pipe.predict_proba(
                Xt_elo[["elo_diff", "elo_expected_home", "h_elo", "a_elo"]])
            test_baselines["elo_logistic"] = evaluate(
                yt, apply_temperature(test_elo, elo_T))
        except Exception as exc:
            log.warning("Test Elo baseline failed: %s", exc)

    # Production refit on every completed match (train + val + test + current).
    all_df = pd.concat([train, val, test, current], axis=0)
    Xall, yall = _prepare(all_df, columns)
    if use_ensemble:
        final_models = []
        for name in ensemble["names"]:
            spec = fitted[name]["spec"]
            n_final = max(60, int(round(fitted[name]["best_iter"] * 1.15)))
            final_models.append(_refit(spec["kind"], spec["params"], Xall, yall, n_final))
        production_model = SoftVotingModel(final_models, ensemble["weights"], ensemble["names"])
        production_desc = f"ensemble:{'+'.join(ensemble['names'])}"
    else:
        spec = best["spec"]
        n_final = max(60, int(round(best["best_iter"] * 1.15)))
        production_model = _refit(spec["kind"], spec["params"], Xall, yall, n_final)
        production_desc = best_name

    result = {
        "label": label,
        "features": columns,
        "model": production_model,
        "eval_model": eval_model,
        "selected": base_name,
        "production_desc": production_desc,
        "weights": ensemble["weights"] if use_ensemble else None,
        "ensemble_names": ensemble["names"] if use_ensemble else None,
        "temperature": market_temperature,
        "poisson_alpha": alpha,
        "market_beta": market_beta,
        "val_log_loss": market_ll,
        "val_metrics": evaluate(yv, val_proba),
        "test_metrics": test_metrics,
        "candidates": candidate_records,
        "baselines": baseline_records,
        "test_baselines": test_baselines,
        "ensemble": ensemble,
        "test_proba": test_proba,
        "test_proba_precal": test_base,
        "test_rows": test,
        "val_proba": val_proba,
        "val_proba_precal": val_final,
        "top_features": _feature_importance(production_model, columns),
        "n_estimators": (max(60, int(round(best["best_iter"] * 1.15)))
                         if not use_ensemble else None),
    }
    return result


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def train(seed_data: bool = True) -> dict:
    matches = load_matches(refresh=False) if seed_data else load_matches(refresh=True)
    builder = FeatureBuilder()
    features = builder.build(matches)
    columns = feature_columns(features)
    no_odds_columns = odds_free_columns(columns)
    train_df, val_df, test_df, current_df = _split(features)

    if len(val_df) < 200 or len(test_df) < 400:
        raise RuntimeError(f"Not enough validation/test rows: val={len(val_df)} test={len(test_df)}")

    log.info("Split: train=%d val=%d test=%d current=%d | features=%d (no-odds %d)",
             len(train_df), len(val_df), len(test_df), len(current_df),
             len(columns), len(no_odds_columns))

    full = _select_and_score(train_df, val_df, test_df, current_df, columns, "full")
    no_odds = _select_and_score(train_df, val_df, test_df, current_df, no_odds_columns, "no_odds")

    # Market baseline on the test rows that have a quoted price.
    market_metrics = None
    # evaluate() expects A/D/H class order, while the market columns are H/D/A.
    market = test_df[["mkt_away", "mkt_draw", "mkt_home"]].to_numpy(dtype=float)
    mask = np.isfinite(market).all(axis=1)
    if mask.sum() >= 50:
        market_metrics = evaluate(test_df["_label"].to_numpy(int)[mask], market[mask])
        market_metrics["note"] = "Bookmaker consensus, overround-normalised"

    market_val = val_df[["mkt_away", "mkt_draw", "mkt_home"]].to_numpy(dtype=float)
    val_mask = np.isfinite(market_val).all(axis=1)
    market_val_metrics = None
    if val_mask.sum() >= 50:
        market_val_metrics = evaluate(
            val_df["_label"].to_numpy(int)[val_mask], market_val[val_mask])
        market_val_metrics["note"] = "Bookmaker consensus, overround-normalised"

    accepted = (full["test_metrics"]["accuracy"] >= config.MIN_TEST_ACCURACY
                and full["test_metrics"]["log_loss"] <= config.MAX_TEST_LOGLOSS)

    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    data_cutoff = str(pd.Timestamp(features["_date"].max()).date())
    artifact = {
        "model": full["model"],
        "model_type": full["production_desc"],
        "config": full["selected"],
        "features": columns,
        "temperature": full["temperature"],
        "poisson_alpha": full["poisson_alpha"],
        "market_beta": full["market_beta"],
        "weights": full["weights"],
        "ensemble_names": full["ensemble_names"],
        "classes": list(config.CLASS_NAMES),
        "class_order": list(config.CLASS_ORDER),
        "trained_at": trained_at,
        "data_cutoff": data_cutoff,
        "train_end": config.TRAIN_END,
        "val_end": config.VAL_END,
        "production_rows": int(len(train_df) + len(val_df) + len(test_df) + len(current_df)),
        "metrics": {
            "validation": full["val_metrics"],
            "test": full["test_metrics"],
            "market_baseline": market_metrics,
            "market_baseline_validation": market_val_metrics,
            "no_odds": no_odds["test_metrics"],
        },
        "no_odds": {
            "model": no_odds["model"],
            "features": no_odds_columns,
            "temperature": no_odds["temperature"],
            "poisson_alpha": no_odds["poisson_alpha"],
            "config": no_odds["selected"],
            "weights": no_odds["weights"],
            "ensemble_names": no_odds["ensemble_names"],
            "metrics": no_odds["test_metrics"],
        },
        "accepted": bool(accepted),
        "python": platform.python_version(),
    }
    joblib.dump(artifact, config.MODELS_DIR / "model.pkl")

    report = {
        "trained_at": trained_at,
        "data_cutoff": data_cutoff,
        "selected_model": full["selected"],
        "production_model": full["production_desc"],
        "temperature": round(full["temperature"], 4),
        "poisson_alpha": round(full["poisson_alpha"], 4),
        "market_beta": round(full["market_beta"], 4),
        "n_features": len(columns),
        "n_features_no_odds": len(no_odds_columns),
        "no_odds_model": no_odds["selected"],
        "no_odds_temperature": round(no_odds["temperature"], 4),
        "no_odds_poisson_alpha": round(no_odds["poisson_alpha"], 4),
        "split": {
            "train": {"n": len(train_df), "start": str(train_df["_date"].min().date()),
                      "end": str(train_df["_date"].max().date())},
            "validation": {"n": len(val_df), "start": str(val_df["_date"].min().date()),
                           "end": str(val_df["_date"].max().date())},
            "test": {"n": len(test_df), "start": str(test_df["_date"].min().date()),
                     "end": str(test_df["_date"].max().date())},
        },
        "candidates": full["candidates"],
        "no_odds_candidates": no_odds["candidates"],
        "baselines": full["baselines"],
        "test_baselines": full["test_baselines"],
        "ensemble": {k: v for k, v in (full["ensemble"] or {}).items() if k != "val_proba"},
        "no_odds_ensemble": {k: v for k, v in (no_odds["ensemble"] or {}).items()
                             if k != "val_proba"},
        "test_metrics": full["test_metrics"],
        "validation_metrics": full["val_metrics"],
        "market_baseline": market_metrics,
        "market_baseline_validation": market_val_metrics,
        "no_odds_metrics": no_odds["test_metrics"],
        "season_breakdown": season_breakdown(test_df, test_df["_label"].to_numpy(int),
                                             full["test_proba"]),
        "calibration": calibration_curve(test_df["_label"].to_numpy(int), full["test_proba"]),
        "top_features": full["top_features"],
        "no_odds_top_features": no_odds["top_features"],
        "accepted": bool(accepted),
        "thresholds": {"min_accuracy": config.MIN_TEST_ACCURACY,
                       "max_log_loss": config.MAX_TEST_LOGLOSS},
        "data_quality": data_quality_report(matches),
    }

    (config.EVAL_DIR / "training_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    (config.METADATA_DIR / "model_meta.json").write_text(json.dumps({
        "trained_at": trained_at,
        "data_cutoff": data_cutoff,
        "model_type": full["production_desc"],
        "config": full["selected"],
        "n_features": len(columns),
        "temperature": round(full["temperature"], 4),
        "poisson_alpha": round(full["poisson_alpha"], 4),
        "market_beta": round(full["market_beta"], 4),
        "no_odds_poisson_alpha": round(no_odds["poisson_alpha"], 4),
        "test_log_loss": round(full["test_metrics"]["log_loss"], 4),
        "test_accuracy": round(full["test_metrics"]["accuracy"], 4),
        "validation_log_loss": round(full["val_metrics"]["log_loss"], 4),
        "validation_accuracy": round(full["val_metrics"]["accuracy"], 4),
        "test_brier": round(full["test_metrics"]["brier"], 4),
        "test_ece": round(full["test_metrics"]["ece"], 4),
        "market_log_loss": round(market_metrics["log_loss"], 4) if market_metrics else None,
        "market_accuracy": round(market_metrics["accuracy"], 4) if market_metrics else None,
        "no_odds_log_loss": round(no_odds["test_metrics"]["log_loss"], 4),
        "no_odds_accuracy": round(no_odds["test_metrics"]["accuracy"], 4),
        "accepted": bool(accepted),
        "split": report["split"],
        "top_features": full["top_features"][:10],
        "data_quality": report["data_quality"],
    }, indent=2), encoding="utf-8")

    # Test predictions for the historical-analysis dashboard.
    test_out = test_df[["_date", "_season", "_home_team", "_away_team",
                        "_home_goals", "_away_goals", "_result"]].copy()
    test_out["pred_away"] = full["test_proba"][:, 0]
    test_out["pred_draw"] = full["test_proba"][:, 1]
    test_out["pred_home"] = full["test_proba"][:, 2]
    test_out["predicted"] = [config.INDEX_TO_CODE[i] for i in full["test_proba"].argmax(axis=1)]
    test_out["confidence"] = full["test_proba"].max(axis=1)
    test_out["correct"] = (test_out["predicted"] == test_out["_result"]).astype(int)
    test_out["precal_away"] = full["test_proba_precal"][:, 0]
    test_out["precal_draw"] = full["test_proba_precal"][:, 1]
    test_out["precal_home"] = full["test_proba_precal"][:, 2]
    test_out["market_away"] = market[:, 0]
    test_out["market_draw"] = market[:, 1]
    test_out["market_home"] = market[:, 2]
    test_out["no_odds_away"] = no_odds["test_proba"][:, 0]
    test_out["no_odds_draw"] = no_odds["test_proba"][:, 1]
    test_out["no_odds_home"] = no_odds["test_proba"][:, 2]
    test_out["no_odds_predicted"] = [
        config.INDEX_TO_CODE[i] for i in no_odds["test_proba"].argmax(axis=1)]
    test_out["no_odds_correct"] = (
        test_out["no_odds_predicted"] == test_out["_result"]).astype(int)
    test_out.to_csv(config.EVAL_DIR / "test_predictions.csv", index=False)

    # Validation predictions: the only set used for decision-rule and
    # calibration selection. Kept separate from the historical test file.
    val_out = val_df[["_date", "_season", "_home_team", "_away_team",
                      "_home_goals", "_away_goals", "_result"]].copy()
    val_out["pred_away"] = full["val_proba"][:, 0]
    val_out["pred_draw"] = full["val_proba"][:, 1]
    val_out["pred_home"] = full["val_proba"][:, 2]
    val_out["predicted"] = [config.INDEX_TO_CODE[i] for i in full["val_proba"].argmax(axis=1)]
    val_out["confidence"] = full["val_proba"].max(axis=1)
    val_out["correct"] = (val_out["predicted"] == val_out["_result"]).astype(int)
    val_out["precal_away"] = full["val_proba_precal"][:, 0]
    val_out["precal_draw"] = full["val_proba_precal"][:, 1]
    val_out["precal_home"] = full["val_proba_precal"][:, 2]
    val_out["market_away"] = market_val[:, 0]
    val_out["market_draw"] = market_val[:, 1]
    val_out["market_home"] = market_val[:, 2]
    val_out["no_odds_away"] = no_odds["val_proba"][:, 0]
    val_out["no_odds_draw"] = no_odds["val_proba"][:, 1]
    val_out["no_odds_home"] = no_odds["val_proba"][:, 2]
    val_out.to_csv(config.EVAL_DIR / "validation_predictions.csv", index=False)

    # Experiment registry
    registry_path = config.EXPERIMENT_DIR / "experiment_log.json"
    registry = []
    if registry_path.exists():
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            registry = []
    registry.append({
        "record_type": "training",
        "trained_at": trained_at,
        "selected_model": full["selected"],
        "poisson_alpha": full["poisson_alpha"],
        "temperature": full["temperature"],
        "candidates": full["candidates"],
        "validation_metrics": full["val_metrics"],
        "test_metrics": full["test_metrics"],
        "no_odds_test_metrics": no_odds["test_metrics"],
    })
    # Keep every research record (append-only governance) but cap the training
    # run history so the file does not grow without bound. Order is preserved.
    training_idx = [i for i, r in enumerate(registry)
                    if r.get("record_type", "training") == "training"]
    drop = set(training_idx[:-20])
    registry = [r for i, r in enumerate(registry) if i not in drop]
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    build_standings(matches)
    log.info("Saved model -> %s", config.MODELS_DIR / "model.pkl")
    log.info("Test: acc=%.4f ll=%.4f macroF1=%.4f draw_recall=%.4f accepted=%s",
             full["test_metrics"]["accuracy"], full["test_metrics"]["log_loss"],
             full["test_metrics"]["macro_f1"],
             full["test_metrics"]["per_class"]["Draw"]["recall"], accepted)
    return report


def _main() -> None:  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = train()
    print(json.dumps({
        "selected_model": report["selected_model"],
        "production_model": report["production_model"],
        "split": report["split"],
        "test_metrics": report["test_metrics"],
        "market_baseline": report["market_baseline"],
        "no_odds_metrics": report["no_odds_metrics"],
        "accepted": report["accepted"],
    }, indent=2))


if __name__ == "__main__":  # pragma: no cover
    _main()
