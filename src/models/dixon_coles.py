"""Dixon-Coles goal model with exponential time decay.

The model estimates a home advantage, a log mean-goal level, and an attack and
defence strength per club. Recent matches receive exponentially larger weight.
The Dixon-Coles correction adjusts the four low-score cells of the scoreline
matrix, where independent Poisson models are known to misfit.

This is a standalone probability model, not a feature generator: it can be
fitted on any historical slice and scored for a future fixture without seeing
that fixture's result.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

MAX_GOALS = 10


class DixonColesModel:
    def __init__(self, xi: float = 0.0018, max_goals: int = MAX_GOALS,
                 l2: float = 0.05, max_iter: int = 600) -> None:
        self.xi = float(xi)
        self.max_goals = int(max_goals)
        self.l2 = float(l2)
        self.max_iter = int(max_iter)
        self.teams: list[str] = []
        self.index: dict[str, int] = {}
        self.attack: np.ndarray | None = None
        self.defence: np.ndarray | None = None
        self.home_advantage = 0.0
        self.rho = 0.0
        self.intercept = 0.0
        self.reference_date: pd.Timestamp | None = None

    # ------------------------------------------------------------------
    def _design(self, matches: pd.DataFrame):
        home = matches["home_team"].map(self.index)
        away = matches["away_team"].map(self.index)
        if home.isna().any() or away.isna().any():
            raise ValueError("Unknown team in Dixon-Coles training data")
        dates = pd.to_datetime(matches["date"])
        reference = dates.max() if self.reference_date is None else self.reference_date
        age_days = (reference - dates).dt.days.to_numpy(dtype=float)
        weights = np.exp(-self.xi * np.maximum(age_days, 0.0))
        return (home.to_numpy(dtype=int), away.to_numpy(dtype=int),
                matches["home_goals"].to_numpy(dtype=float),
                matches["away_goals"].to_numpy(dtype=float), weights)

    def _unpack(self, theta: np.ndarray):
        n = len(self.teams)
        attack = theta[:n]
        defence = theta[n:2 * n]
        attack = attack - attack.mean()
        defence = defence - defence.mean()
        return attack, defence, float(theta[-2]), float(theta[-1])

    def _objective(self, theta: np.ndarray, home, away, hg, ag, weights) -> float:
        attack, defence, home_adv, rho = self._unpack(theta)
        log_home = (self.intercept + home_adv + attack[home] - defence[away])
        log_away = self.intercept + attack[away] - defence[home]
        lam = np.exp(np.clip(log_home, -4.0, 3.0))
        mu = np.exp(np.clip(log_away, -4.0, 3.0))

        log_tau = np.zeros_like(lam)
        for h, a in ((0, 0), (0, 1), (1, 0), (1, 1)):
            mask = (hg == h) & (ag == a)
            if not mask.any():
                continue
            if (h, a) == (0, 0):
                tau = 1.0 - lam[mask] * mu[mask] * rho
            elif (h, a) == (0, 1):
                tau = 1.0 + lam[mask] * rho
            elif (h, a) == (1, 0):
                tau = 1.0 + mu[mask] * rho
            else:
                tau = 1.0 - rho
            log_tau[mask] = np.log(np.clip(tau, 1e-9, None))

        loglik = (
            log_tau
            + hg * log_home - lam
            + ag * log_away - mu
            - gammaln(hg + 1.0) - gammaln(ag + 1.0)
        )
        penalty = self.l2 * (float(np.sum(attack ** 2)) + float(np.sum(defence ** 2))
                             + home_adv ** 2 + rho ** 2)
        return float(-np.sum(weights * loglik) + penalty)

    def fit(self, matches: pd.DataFrame, reference_date=None) -> "DixonColesModel":
        required = {"date", "home_team", "away_team", "home_goals", "away_goals"}
        missing = required - set(matches.columns)
        if missing:
            raise ValueError(f"Missing columns for Dixon-Coles fit: {sorted(missing)}")
        data = matches.dropna(subset=list(required)).copy()
        if reference_date is not None:
            # Enforce the cutoff inside the model as well as at the call site.
            data = data[pd.to_datetime(data["date"]) <= pd.Timestamp(reference_date)]
        if len(data) < 100:
            raise ValueError("Not enough matches to fit Dixon-Coles")
        self.teams = sorted(set(data["home_team"]) | set(data["away_team"]))
        self.index = {team: i for i, team in enumerate(self.teams)}
        self.reference_date = (pd.Timestamp(reference_date)
                               if reference_date is not None else None)
        home, away, hg, ag, weights = self._design(data)
        # Fix the league log-goal level at the weighted sample mean. This keeps
        # the attack/defence parameters identifiable and leaves home advantage
        # as the only mean-shift term.
        weighted_total = float(np.sum(weights * (hg + ag)) / max(np.sum(weights), 1e-9))
        self.intercept = float(np.log(max(weighted_total / 2.0, 0.2)))
        n = len(self.teams)
        theta0 = np.concatenate([np.zeros(n), np.zeros(n), [0.20], [-0.05]])
        result = minimize(
            self._objective, theta0, args=(home, away, hg, ag, weights),
            method="L-BFGS-B",
            bounds=[(-2.5, 2.5)] * (2 * n) + [(0.0, 1.0), (-0.20, 0.20)],
            options={"maxiter": self.max_iter, "ftol": 1e-9},
        )
        attack, defence, home_adv, rho = self._unpack(result.x)
        self.attack, self.defence = attack, defence
        self.home_advantage, self.rho = home_adv, rho
        return self

    # ------------------------------------------------------------------
    def _score_matrix(self, home_team: str, away_team: str) -> np.ndarray:
        if self.attack is None or self.defence is None:
            raise RuntimeError("Dixon-Coles model is not fitted")
        h = self.index.get(home_team)
        a = self.index.get(away_team)
        attack_h = self.attack[h] if h is not None else 0.0
        defence_h = self.defence[h] if h is not None else 0.0
        attack_a = self.attack[a] if a is not None else 0.0
        defence_a = self.defence[a] if a is not None else 0.0
        lam = math.exp(np.clip(self.intercept + self.home_advantage
                               + attack_h - defence_a, -4.0, 3.0))
        mu = math.exp(np.clip(self.intercept + attack_a - defence_h, -4.0, 3.0))
        goals = np.arange(self.max_goals + 1)
        home_pmf = np.exp(-lam) * lam ** goals / np.exp(gammaln(goals + 1.0))
        away_pmf = np.exp(-mu) * mu ** goals / np.exp(gammaln(goals + 1.0))
        matrix = np.outer(home_pmf, away_pmf)
        rho = self.rho
        matrix[0, 0] *= max(1.0 - lam * mu * rho, 1e-9)
        matrix[0, 1] *= max(1.0 + lam * rho, 1e-9)
        matrix[1, 0] *= max(1.0 + mu * rho, 1e-9)
        matrix[1, 1] *= max(1.0 - rho, 1e-9)
        return matrix / matrix.sum()

    def predict_proba(self, home_team: str, away_team: str) -> np.ndarray:
        """Return ``[away, draw, home]`` probabilities (internal class order)."""
        matrix = self._score_matrix(home_team, away_team)
        home = float(np.tril(matrix, -1).sum())
        draw = float(np.trace(matrix))
        away = float(np.triu(matrix, 1).sum())
        total = home + draw + away
        return np.array([away / total, draw / total, home / total])

    def predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        return np.stack([
            self.predict_proba(row.home_team, row.away_team)
            for row in frame.itertuples()
        ])
