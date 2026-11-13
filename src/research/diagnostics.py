"""Post-hoc diagnostics for probabilistic match-outcome models.

Everything here is analysis, not model selection, unless a caller explicitly
selects a policy on the validation frame. Class order is A/D/H (0/1/2).

The module is deliberately dependency-light: NumPy, pandas, SciPy and
scikit-learn only.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

CLASSES = [0, 1, 2]
CODE_TO_INDEX = {"A": 0, "D": 1, "H": 2}
PROB_SUFFIXES = ("away", "draw", "home")


# --------------------------------------------------------------------------
# Frame helpers
# --------------------------------------------------------------------------
def labels(frame: pd.DataFrame) -> np.ndarray:
    return frame["_result"].map(CODE_TO_INDEX).to_numpy(dtype=int)


def probabilities(frame: pd.DataFrame, prefix: str = "pred") -> np.ndarray:
    cols = [f"{prefix}_{suffix}" for suffix in PROB_SUFFIXES]
    return frame[cols].to_numpy(dtype=float)


def _onehot(y: np.ndarray) -> np.ndarray:
    out = np.zeros((len(y), 3), dtype=float)
    out[np.arange(len(y)), y] = 1.0
    return out


def _row_log_loss(y: np.ndarray, proba: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(proba[np.arange(len(y)), y], 1e-12, 1.0))


def _row_brier(y: np.ndarray, proba: np.ndarray) -> np.ndarray:
    return np.sum((proba - _onehot(y)) ** 2, axis=1)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def choose(p_draw: np.ndarray, p_home: np.ndarray, p_away: np.ndarray,
           threshold: float | None = None,
           margin: float | None = None) -> np.ndarray:
    """Three-class decision rule used by every draw experiment.

    * ``threshold``: draw if p(draw) >= threshold, otherwise the stronger
      non-draw side. This mirrors ``PredictionEngine.choose_outcome``.
    * ``margin``: draw if p(draw) is within ``margin`` of the stronger
      non-draw side. ``margin=0`` is equivalent to argmax.
    """
    non_draw = np.where(p_home >= p_away, 2, 0)
    if threshold is not None:
        return np.where(p_draw >= float(threshold), 1, non_draw)
    if margin is not None:
        return np.where(p_draw >= np.maximum(p_home, p_away) - float(margin), 1, non_draw)
    return np.argmax(np.stack([p_away, p_draw, p_home], axis=1), axis=1)


def decision_metrics(y: np.ndarray, prediction: np.ndarray) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y, prediction, labels=CLASSES, zero_division=0)
    return {
        "accuracy": float(np.mean(prediction == y)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "draw_precision": float(precision[1]),
        "draw_recall": float(recall[1]),
        "draw_f1": float(f1[1]),
        "home_recall": float(recall[2]),
        "away_recall": float(recall[0]),
        "draws_predicted": int((prediction == 1).sum()),
        "home_predicted": int((prediction == 2).sum()),
        "away_predicted": int((prediction == 0).sum()),
        "n": int(len(y)),
    }


# --------------------------------------------------------------------------
# Draw diagnostics
# --------------------------------------------------------------------------
def draw_summary(frame: pd.DataFrame, prefix: str = "pred") -> dict:
    proba = probabilities(frame, prefix)
    y = labels(frame)
    p_draw = proba[:, 1]
    strongest = np.maximum(proba[:, 0], proba[:, 2])
    margin = p_draw - strongest
    actual_draw = y == 1
    prediction = proba.argmax(axis=1)
    return {
        "n": int(len(frame)),
        "actual_draw_rate": float(actual_draw.mean()),
        "mean_p_draw": float(p_draw.mean()),
        "median_p_draw": float(np.median(p_draw)),
        "quantiles_p_draw": {
            str(q): round(float(np.quantile(p_draw, q)), 4)
            for q in (0.05, 0.25, 0.50, 0.75, 0.95)
        },
        "mean_p_draw_actual_draw": float(p_draw[actual_draw].mean()),
        "mean_p_draw_non_draw": float(p_draw[~actual_draw].mean()),
        "mean_margin_draw_minus_strongest": float(margin.mean()),
        "share_draw_is_argmax": float(np.mean(prediction == 1)),
        "draws_predicted": int((prediction == 1).sum()),
        "draw_recall": float(np.mean(prediction[actual_draw] == 1))
        if actual_draw.any() else 0.0,
        "accuracy": float(np.mean(prediction == y)),
        "mean_confidence": float(proba.max(axis=1).mean()),
        "mean_entropy": float(
            np.mean(-np.sum(proba * np.log(np.clip(proba, 1e-12, 1.0)), axis=1))),
    }


def draw_deciles(frame: pd.DataFrame, prefix: str = "pred", bins: int = 10) -> list[dict]:
    proba = probabilities(frame, prefix)
    y = labels(frame)
    p_draw = proba[:, 1]
    prediction = proba.argmax(axis=1)
    # Quantile bins keep every bucket populated even though draw probabilities
    # occupy a narrow band.
    bucket = pd.qcut(pd.Series(p_draw), q=bins, duplicates="drop")
    out = []
    for interval, idx in bucket.groupby(bucket, observed=True).groups.items():
        idx = np.asarray(idx)
        out.append({
            "bin": str(interval),
            "n": int(len(idx)),
            "mean_p_draw": round(float(p_draw[idx].mean()), 4),
            "actual_draw_rate": round(float((y[idx] == 1).mean()), 4),
            "draws_predicted": int((prediction[idx] == 1).sum()),
            "accuracy": round(float((prediction[idx] == y[idx]).mean()), 4),
        })
    return out


def draw_by_season(frame: pd.DataFrame, prefix: str = "pred") -> list[dict]:
    out = []
    for season, group in frame.groupby("_season", sort=True):
        summary = draw_summary(group, prefix)
        out.append({
            "season": season,
            "n": summary["n"],
            "actual_draw_rate": round(summary["actual_draw_rate"], 4),
            "mean_p_draw": round(summary["mean_p_draw"], 4),
            "draws_predicted": summary["draws_predicted"],
            "draw_recall": round(summary["draw_recall"], 4),
            "accuracy": round(summary["accuracy"], 4),
        })
    return out


def threshold_curve(frame: pd.DataFrame, prefix: str = "pred",
                    thresholds: Iterable[float] | None = None) -> list[dict]:
    proba = probabilities(frame, prefix)
    y = labels(frame)
    p_away, p_draw, p_home = proba[:, 0], proba[:, 1], proba[:, 2]
    if thresholds is None:
        thresholds = np.round(np.arange(0.18, 0.361, 0.01), 3)
    rows = []
    for threshold in thresholds:
        prediction = choose(p_draw, p_home, p_away, threshold=float(threshold))
        metrics = decision_metrics(y, prediction)
        metrics["threshold"] = float(threshold)
        rows.append(metrics)
    return rows


def margin_curve(frame: pd.DataFrame, prefix: str = "pred",
                 margins: Iterable[float] | None = None) -> list[dict]:
    proba = probabilities(frame, prefix)
    y = labels(frame)
    p_away, p_draw, p_home = proba[:, 0], proba[:, 1], proba[:, 2]
    if margins is None:
        margins = np.round(np.arange(0.0, 0.161, 0.01), 3)
    rows = []
    for margin in margins:
        prediction = choose(p_draw, p_home, p_away, margin=float(margin))
        metrics = decision_metrics(y, prediction)
        metrics["margin"] = float(margin)
        rows.append(metrics)
    return rows


def select_draw_policy(val_frame: pd.DataFrame, prefix: str = "pred",
                       max_accuracy_drop: float = 0.005,
                       min_macro_gain: float = 0.010) -> dict:
    """Select a draw-aware policy on validation only.

    Pre-registered rule: accuracy may fall by at most 0.5 percentage points
    and macro F1 must improve by at least 0.010. Among eligible policies the
    highest validation macro F1 wins; argmax is the fallback.
    """
    proba = probabilities(val_frame, prefix)
    y = labels(val_frame)
    baseline = decision_metrics(y, proba.argmax(axis=1))
    candidates = []
    for kind, rows in (("threshold", threshold_curve(val_frame, prefix)),
                       ("margin", margin_curve(val_frame, prefix))):
        for row in rows:
            row = dict(row, policy=kind)
            candidates.append(row)
    eligible = [
        row for row in candidates
        if row["accuracy"] >= baseline["accuracy"] - max_accuracy_drop
        and row["macro_f1"] >= baseline["macro_f1"] + min_macro_gain
    ]
    if not eligible:
        return {
            "selected": None,
            "baseline": baseline,
            "reason": (f"No policy improved validation macro F1 by >= {min_macro_gain:.3f} "
                       f"while keeping accuracy within {max_accuracy_drop:.3f}."),
            "best_candidate": max(candidates, key=lambda r: r["macro_f1"]) if candidates else None,
        }
    best = max(eligible, key=lambda r: (r["macro_f1"], r["accuracy"]))
    return {
        "selected": best,
        "baseline": baseline,
        "reason": "Validation macro-F1 gate met without material accuracy loss.",
        "eligible_count": len(eligible),
    }


# --------------------------------------------------------------------------
# Calibration and sharpness
# --------------------------------------------------------------------------
def reliability_table(y: np.ndarray, proba: np.ndarray, bins: int = 10) -> list[dict]:
    confidence = proba.max(axis=1)
    prediction = proba.argmax(axis=1)
    correct = (prediction == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if not mask.any():
            continue
        rows.append({
            "bin_low": round(float(low), 3),
            "bin_high": round(float(high), 3),
            "count": int(mask.sum()),
            "mean_confidence": float(confidence[mask].mean()),
            "observed_accuracy": float(correct[mask].mean()),
        })
    return rows


def per_class_reliability(y: np.ndarray, proba: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for k, name in enumerate(("Away win", "Draw", "Home win")):
        observed = (y == k).astype(float)
        for low, high in zip(edges[:-1], edges[1:]):
            mask = (proba[:, k] > low) & (proba[:, k] <= high)
            if not mask.any():
                continue
            rows.append({
                "class": name,
                "bin_low": round(float(low), 3),
                "bin_high": round(float(high), 3),
                "count": int(mask.sum()),
                "mean_probability": float(proba[mask, k].mean()),
                "observed_rate": float(observed[mask].mean()),
            })
    return rows


def expected_calibration_error(y: np.ndarray, proba: np.ndarray, bins: int = 10) -> float:
    return float(sum(
        row["count"] / len(y)
        * abs(row["observed_accuracy"] - row["mean_confidence"])
        for row in reliability_table(y, proba, bins)
    ))


def brier_decomposition(y: np.ndarray, proba: np.ndarray, bins: int = 10) -> dict:
    """Murphy decomposition of the multiclass Brier score.

    For each class the binary Brier score is decomposed into reliability,
    resolution and uncertainty; the multiclass score is the sum over classes.
    """
    total_rel = total_res = total_unc = 0.0
    per_class = {}
    for k, name in enumerate(("Away win", "Draw", "Home win")):
        p = proba[:, k]
        observed = (y == k).astype(float)
        base = float(observed.mean())
        reliability = resolution = 0.0
        edges = np.linspace(0.0, 1.0, bins + 1)
        for low, high in zip(edges[:-1], edges[1:]):
            mask = (p > low) & (p <= high)
            if not mask.any():
                continue
            weight = mask.mean()
            reliability += weight * (p[mask].mean() - observed[mask].mean()) ** 2
            resolution += weight * (observed[mask].mean() - base) ** 2
        uncertainty = base * (1.0 - base)
        per_class[name] = {
            "reliability": float(reliability),
            "resolution": float(resolution),
            "uncertainty": float(uncertainty),
            "brier": float(reliability - resolution + uncertainty),
        }
        total_rel += reliability
        total_res += resolution
        total_unc += uncertainty
    return {
        "per_class": per_class,
        "reliability": float(total_rel),
        "resolution": float(total_res),
        "uncertainty": float(total_unc),
        "brier_decomposed": float(total_rel - total_res + total_unc),
    }


def sharpness(proba: np.ndarray) -> dict:
    confidence = proba.max(axis=1)
    entropy = -np.sum(proba * np.log(np.clip(proba, 1e-12, 1.0)), axis=1)
    return {
        "mean_confidence": float(confidence.mean()),
        "median_confidence": float(np.median(confidence)),
        "share_above_0_5": float((confidence > 0.5).mean()),
        "mean_entropy": float(entropy.mean()),
        "normalised_entropy": float(entropy.mean() / np.log(3)),
    }


def calibration_metrics(y: np.ndarray, proba: np.ndarray) -> dict:
    return {
        "log_loss": float(log_loss(y, proba, labels=CLASSES)),
        "brier": float(np.mean(_row_brier(y, proba))),
        "ece": expected_calibration_error(y, proba),
        "accuracy": float(np.mean(proba.argmax(axis=1) == y)),
        "sharpness": sharpness(proba),
        "brier_decomposition": brier_decomposition(y, proba),
    }


def fit_vector_scaling(y: np.ndarray, proba: np.ndarray, l2: float = 1e-3) -> np.ndarray:
    """Fit class-specific scale and bias on log-probabilities.

    Returns the six parameters ``[scale(3), bias(3)]``. The transformation is
    ``softmax(scale_k * log(p_k) + bias_k)``.
    """
    logp = np.log(np.clip(proba, 1e-9, 1.0))

    def objective(theta: np.ndarray) -> float:
        scale, bias = theta[:3], theta[3:]
        calibrated = _softmax(logp * scale + bias)
        penalty = l2 * float(np.sum((scale - 1.0) ** 2) + np.sum(bias ** 2))
        return float(log_loss(y, calibrated, labels=CLASSES)) + penalty

    result = minimize(
        objective, np.concatenate([np.ones(3), np.zeros(3)]),
        method="L-BFGS-B",
        bounds=[(0.25, 3.0)] * 3 + [(-1.5, 1.5)] * 3,
    )
    return np.asarray(result.x, dtype=float)


def apply_vector_scaling(proba: np.ndarray, theta: np.ndarray) -> np.ndarray:
    logp = np.log(np.clip(proba, 1e-9, 1.0))
    return _softmax(logp * theta[:3] + theta[3:])


def fit_temperature(y: np.ndarray, proba: np.ndarray) -> float:
    best_t, best_ll = 1.0, float(log_loss(y, proba, labels=CLASSES))
    for temperature in np.linspace(0.60, 1.80, 49):
        p = np.clip(proba, 1e-12, 1.0) ** (1.0 / temperature)
        p = p / p.sum(axis=1, keepdims=True)
        ll = float(log_loss(y, p, labels=CLASSES))
        if ll < best_ll:
            best_t, best_ll = float(temperature), ll
    return best_t


def apply_temperature(proba: np.ndarray, temperature: float) -> np.ndarray:
    p = np.clip(proba, 1e-12, 1.0) ** (1.0 / float(temperature))
    return p / p.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------
# Paired comparisons
# --------------------------------------------------------------------------
def _bootstrap_mean(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> dict:
    n = len(values)
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        draws[i] = values[rng.integers(0, n, n)].mean()
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "mean": float(values.mean()),
        "ci95": [float(low), float(high)],
        "p_value": float(min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean()))),
    }


def _season_block_bootstrap(values: np.ndarray, seasons: np.ndarray,
                            n_boot: int, rng: np.random.Generator) -> dict:
    unique = np.array(sorted(set(seasons)))
    index = {season: np.where(seasons == season)[0] for season in unique}
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        sample = np.concatenate([values[index[s]] for s in chosen])
        draws[i] = sample.mean()
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "mean": float(values.mean()),
        "ci95": [float(low), float(high)],
        "p_value": float(min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean()))),
        "blocks": int(len(unique)),
    }


def paired_comparison(frame: pd.DataFrame, prefix_a: str = "pred",
                      prefix_b: str = "market", n_boot: int = 4000,
                      seed: int = 42) -> dict:
    y = labels(frame)
    pa, pb = probabilities(frame, prefix_a), probabilities(frame, prefix_b)
    pred_a, pred_b = pa.argmax(axis=1), pb.argmax(axis=1)
    correct_a, correct_b = pred_a == y, pred_b == y
    both = int(np.sum(correct_a & correct_b))
    only_a = int(np.sum(correct_a & ~correct_b))
    only_b = int(np.sum(~correct_a & correct_b))
    neither = int(np.sum(~correct_a & ~correct_b))

    if only_a + only_b:
        mcnemar_p = float(min(1.0, stats.binomtest(
            min(only_a, only_b), only_a + only_b, 0.5).pvalue * 2))
    else:
        mcnemar_p = 1.0

    diff_acc = correct_a.astype(float) - correct_b.astype(float)
    diff_ll = _row_log_loss(y, pb) - _row_log_loss(y, pa)
    diff_brier = _row_brier(y, pb) - _row_brier(y, pa)

    rng = np.random.default_rng(seed)
    seasons = frame["_season"].to_numpy()
    out = {
        "n": int(len(y)),
        "accuracy_a": float(correct_a.mean()),
        "accuracy_b": float(correct_b.mean()),
        "log_loss_a": float(log_loss(y, pa, labels=CLASSES)),
        "log_loss_b": float(log_loss(y, pb, labels=CLASSES)),
        "brier_a": float(np.mean(_row_brier(y, pa))),
        "brier_b": float(np.mean(_row_brier(y, pb))),
        "agreement_table": {
            "both_correct": both,
            "only_a_correct": only_a,
            "only_b_correct": only_b,
            "both_wrong": neither,
        },
        "mcnemar_exact_p": mcnemar_p,
    }
    for name, values in (("accuracy", diff_acc), ("log_loss", diff_ll),
                         ("brier", diff_brier)):
        out[f"{name}_difference"] = {
            "match_bootstrap": _bootstrap_mean(values, n_boot, rng),
            "season_block_bootstrap": _season_block_bootstrap(values, seasons, n_boot, rng),
        }
    return out


# --------------------------------------------------------------------------
# Walk-forward aggregation
# --------------------------------------------------------------------------
def summarise_walk_forward(rows: list[dict]) -> dict:
    if not rows:
        return {}
    metrics = ("accuracy", "log_loss", "brier", "macro_f1", "draw_recall")
    summary = {}
    for metric in metrics:
        values = np.array([row[metric] for row in rows], dtype=float)
        summary[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
            "by_season": {row["season"]: round(float(row[metric]), 4) for row in rows},
        }
    return summary
