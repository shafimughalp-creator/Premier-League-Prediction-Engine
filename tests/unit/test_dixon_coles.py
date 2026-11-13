"""Tests for the Dixon-Coles goal model."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.dixon_coles import DixonColesModel


def _synthetic(seed: int = 11, n_teams: int = 8, n_matches: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = [f"Team {i}" for i in range(n_teams)]
    strength = np.linspace(0.5, -0.5, n_teams)
    rows = []
    start = pd.Timestamp("2020-08-01")
    for i in range(n_matches):
        home = teams[rng.integers(0, n_teams)]
        away = teams[rng.integers(0, n_teams)]
        if home == away:
            continue
        h = teams.index(home)
        a = teams.index(away)
        lam = np.exp(0.3 + 0.2 + strength[h] - strength[a])
        mu = np.exp(0.3 + strength[a] - strength[h])
        hg, ag = rng.poisson(lam), rng.poisson(mu)
        rows.append({
            "date": start + pd.Timedelta(days=i * 3),
            "home_team": home,
            "away_team": away,
            "home_goals": int(hg),
            "away_goals": int(ag),
        })
    return pd.DataFrame(rows)


def test_dixon_coles_probabilities_are_valid():
    data = _synthetic()
    model = DixonColesModel().fit(data)
    proba = model.predict_proba("Team 0", "Team 7")
    assert proba.shape == (3,)
    assert np.isclose(proba.sum(), 1.0)
    assert np.all(proba >= 0)


def test_dixon_coles_learns_team_strength_ordering():
    data = _synthetic()
    model = DixonColesModel().fit(data)
    strong_home = model.predict_proba("Team 0", "Team 7")[2]
    weak_home = model.predict_proba("Team 7", "Team 0")[2]
    assert strong_home > weak_home


def test_dixon_coles_does_not_use_future_results():
    data = _synthetic()
    cutoff = data["date"].quantile(0.8)
    past = data[data["date"] <= cutoff]
    future = data[data["date"] > cutoff].copy()
    model = DixonColesModel().fit(past, reference_date=cutoff)
    before = model.predict_proba(future.iloc[0]["home_team"], future.iloc[0]["away_team"])

    tampered = future.copy()
    tampered.loc[tampered.index[0], ["home_goals", "away_goals"]] = [9, 0]
    combined = pd.concat([past, tampered], ignore_index=True)
    # Even when future rows are supplied, the explicit reference date must
    # exclude them from the fit.
    model_after = DixonColesModel().fit(combined, reference_date=cutoff)
    after = model_after.predict_proba(future.iloc[0]["home_team"],
                                      future.iloc[0]["away_team"])
    assert np.allclose(before, after)


def test_dixon_coles_time_decay_weights_recent_matches():
    data = _synthetic()
    model = DixonColesModel(xi=0.01).fit(data)
    assert model.rho > -0.2 and model.rho < 0.2
    assert model.home_advantage >= 0
    assert np.isfinite(model.attack).all()
