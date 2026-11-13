"""Evaluation metrics for three-class match outcome models."""
from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

from src import config

CLASSES = [0, 1, 2]


def brier_multiclass(y_true: np.ndarray, proba: np.ndarray) -> float:
    onehot = np.zeros_like(proba, dtype=float)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, bins: int = 10) -> float:
    confidence = proba.max(axis=1)
    prediction = proba.argmax(axis=1)
    correct = (prediction == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if not mask.any():
            continue
        ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    p = correct / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def evaluate(y_true: np.ndarray, proba: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    prediction = proba.argmax(axis=1)
    cm = confusion_matrix(y_true, prediction, labels=CLASSES)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, prediction, labels=CLASSES, zero_division=0
    )
    correct = int((prediction == y_true).sum())
    low, high = wilson_interval(correct, len(y_true))
    return {
        "log_loss": float(log_loss(y_true, proba, labels=CLASSES)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "brier": brier_multiclass(y_true, proba),
        "ece": expected_calibration_error(y_true, proba),
        "correct": correct,
        "incorrect": int(len(y_true) - correct),
        "n": int(len(y_true)),
        "accuracy_ci95": [round(low, 4), round(high, 4)],
        "per_class": {
            config.CLASS_NAMES[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(3)
        },
        "confusion_matrix": cm.tolist(),
        "prediction_distribution": {
            config.CLASS_NAMES[i]: int((prediction == i).sum()) for i in range(3)
        },
    }


def calibration_curve(y_true: np.ndarray, proba: np.ndarray, bins: int = 10) -> list[dict]:
    confidence = proba.max(axis=1)
    prediction = proba.argmax(axis=1)
    correct = (prediction == y_true).astype(float)
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
            "mean_confidence": round(float(confidence[mask].mean()), 4),
            "observed_accuracy": round(float(correct[mask].mean()), 4),
        })
    return rows


def season_breakdown(df, y_true: np.ndarray, proba: np.ndarray) -> list[dict]:
    seasons = df["_season"].to_numpy()
    rows = []
    for season in sorted(set(seasons)):
        mask = seasons == season
        metrics = evaluate(y_true[mask], proba[mask])
        rows.append({
            "season": season,
            "n": metrics["n"],
            "accuracy": round(metrics["accuracy"], 4),
            "log_loss": round(metrics["log_loss"], 4),
            "macro_f1": round(metrics["macro_f1"], 4),
            "draw_recall": round(metrics["per_class"]["Draw"]["recall"], 4),
            "draw_predicted": metrics["prediction_distribution"]["Draw"],
        })
    return rows
