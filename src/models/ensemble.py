"""Probability ensemble helper.

Kept in its own module so joblib can unpickle the production artifact without
importing the whole training pipeline.
"""
from __future__ import annotations

import numpy as np


class SoftVotingModel:
    """Weighted average of fitted classifiers."""

    def __init__(self, models: list, weights: list[float], names: list[str] | None = None):
        if len(models) != len(weights):
            raise ValueError("models and weights must have the same length")
        total = float(np.sum(weights))
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        self.models = list(models)
        self.weights = [float(w) / total for w in weights]
        self.names = list(names or [f"model_{i}" for i in range(len(models))])

    def predict_proba(self, X) -> np.ndarray:
        proba = None
        for weight, model in zip(self.weights, self.models):
            p = np.asarray(model.predict_proba(X), dtype=float)
            proba = weight * p if proba is None else proba + weight * p
        if proba is None:
            raise RuntimeError("SoftVotingModel has no base models")
        totals = proba.sum(axis=1, keepdims=True)
        totals[totals <= 0] = 1.0
        return proba / totals

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)
