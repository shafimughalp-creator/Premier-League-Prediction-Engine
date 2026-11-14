"""Prediction engine for upcoming, manual and retrospective matches.

Probability pipeline (identical to the one validated during training):

    base model probabilities
      -> Poisson prior blend        (alpha, 1.0 = model only)
      -> bookmaker consensus blend  (beta,  1.0 = model only)
      -> temperature scaling        (T,     1.0 = no sharpening)

Every stage is applied in the order above; the artifact stores the three
hyper-parameters that were selected on the 2023/24 validation season only.

Two model variants are stored in the artifact:

* ``full``    - 237 features including bookmaker prices
* ``no_odds`` - 223 features, no betting market information

Odds-free fixtures are scored with the ``no_odds`` variant instead of feeding
missing market columns to a model that was trained with them, which would
silently shift its input distribution.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
import pandas as pd

from src import config
from src.data import processing
from src.features.builder import FeatureBuilder, feature_columns
from src.utilities.io import read_json, write_json_atomic

log = logging.getLogger(__name__)

PREDICTIONS_JSON = config.DATA_DIR / "predictions.json"
MARKET_COLUMNS = ("mkt_away", "mkt_draw", "mkt_home")
ODDS_COLUMNS = ("mkt_home", "mkt_draw", "mkt_away", "mkt_overround", "mkt_books")
MODE_UPCOMING = "upcoming"
MODE_MANUAL = "manual"
MODE_RETROSPECTIVE = "retrospective"


class PredictionError(ValueError):
    """Raised when a prediction request is not valid."""


# --------------------------------------------------------------------------
# Probability post-processing (mirrors src.models.training)
# --------------------------------------------------------------------------
def apply_temperature(proba: np.ndarray, temperature: float | None) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    if temperature is None or not np.isfinite(temperature) or temperature <= 0:
        return proba
    p = np.clip(proba, 1e-12, 1.0) ** (1.0 / float(temperature))
    return p / p.sum(axis=1, keepdims=True)


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


def blend_with_market(proba: np.ndarray, market: np.ndarray, beta: float) -> np.ndarray:
    """``beta=1`` keeps the model, ``beta=0`` uses the market only."""
    proba = np.asarray(proba, dtype=float)
    if beta is None or not np.isfinite(beta) or beta >= 1.0:
        return proba
    beta = float(np.clip(beta, 0.0, 1.0))
    usable = np.isfinite(market).all(axis=1)
    out = proba.copy()
    out[usable] = beta * proba[usable] + (1.0 - beta) * market[usable]
    totals = out.sum(axis=1, keepdims=True)
    totals[totals <= 0] = 1.0
    return out / totals


def choose_outcome(proba: np.ndarray, draw_threshold: float | None = None) -> np.ndarray:
    """Argmax by default; optional draw-aware rule for macro-F1 analysis.

    With a threshold the draw class wins whenever ``p_draw >= threshold``,
    otherwise the stronger of home/away wins. Thresholds are fitted on the
    validation season, never on the test seasons.
    """
    proba = np.asarray(proba, dtype=float)
    if draw_threshold is None:
        return proba.argmax(axis=1)
    choice = np.where(proba[:, 2] >= proba[:, 0], 2, 0)
    return np.where(proba[:, 1] >= float(draw_threshold), 1, choice)


# --------------------------------------------------------------------------
# Result objects
# --------------------------------------------------------------------------
@dataclass
class Prediction:
    match_id: str
    date: str
    time: str
    home_team: str
    away_team: str
    prob_home: float
    prob_draw: float
    prob_away: float
    predicted: str
    confidence: float
    mode: str
    model_version: str
    predicted_at: str
    model_variant: str = "full"
    has_odds: bool = False
    market_home: float | None = None
    market_draw: float | None = None
    market_away: float | None = None
    source: str = ""
    season: str = ""
    matchweek: float | None = None
    context: dict = field(default_factory=dict)
    fixture_id: str | None = None

    def as_dict(self) -> dict:
        return {
            "fixture_id": self.fixture_id,
            "match_id": self.match_id,
            "date": self.date,
            "time": self.time,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "home_display": config.display_name(self.home_team),
            "away_display": config.display_name(self.away_team),
            "home_short": config.short_name(self.home_team),
            "away_short": config.short_name(self.away_team),
            "prob_home": float(self.prob_home),
            "prob_draw": float(self.prob_draw),
            "prob_away": float(self.prob_away),
            "probabilities": {
                "home": float(self.prob_home),
                "draw": float(self.prob_draw),
                "away": float(self.prob_away),
            },
            "predicted": self.predicted,
            "predicted_label": config.CODE_TO_LABEL.get(self.predicted, self.predicted),
            "confidence": float(self.confidence),
            "confidence_band": confidence_band(self.confidence),
            "mode": self.mode,
            "model_version": self.model_version,
            "model_variant": self.model_variant,
            "predicted_at": self.predicted_at,
            "has_odds": bool(self.has_odds),
            "market": (
                {
                    "home": float(self.market_home),
                    "draw": float(self.market_draw),
                    "away": float(self.market_away),
                }
                if self.has_odds and self.market_home is not None
                else None
            ),
            "source": self.source,
            "season": self.season,
            "matchweek": None if self.matchweek is None or pd.isna(self.matchweek)
            else int(self.matchweek),
            "context": self.context,
        }


def confidence_band(confidence: float) -> str:
    if confidence >= 0.60:
        return "High"
    if confidence >= 0.45:
        return "Medium"
    return "Low"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _model_version(trained_at: str, n_features: int, model_type: str) -> str:
    digest = hashlib.sha1(f"{trained_at}|{n_features}|{model_type}".encode()).hexdigest()[:8]
    stamp = trained_at[:10].replace("-", "")
    return f"epl-{stamp}-{digest}"


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------
class PredictionEngine:
    def __init__(self, artifact_path: Path | None = None) -> None:
        self.artifact_path = Path(artifact_path or config.MODELS_DIR / "model.pkl")
        if not self.artifact_path.exists():
            raise FileNotFoundError(
                f"No trained model at {self.artifact_path}. Run: python -m src.models.training"
            )
        self.artifact = joblib.load(self.artifact_path)
        self.version = self.artifact.get("model_version") or _model_version(
            self.artifact.get("trained_at", "unknown"),
            len(self.artifact.get("features", [])),
            self.artifact.get("model_type", "unknown"),
        )
        self._matches: pd.DataFrame | None = None
        self._builder: FeatureBuilder | None = None
        self._builder_cutoff: pd.Timestamp | None = None
        self._features_cache: pd.DataFrame | None = None
        self._known_teams: set[str] | None = None

    # ---- data access -------------------------------------------------------
    @property
    def matches(self) -> pd.DataFrame:
        if self._matches is None:
            self._matches = processing.load_processed()
        return self._matches

    @property
    def data_cutoff(self) -> pd.Timestamp:
        return pd.Timestamp(self.artifact.get("data_cutoff", self.matches["date"].max()))

    @property
    def known_teams(self) -> list[str]:
        if self._known_teams is None:
            matches = self.matches
            self._known_teams = set(matches["home_team"]) | set(matches["away_team"])
        return sorted(self._known_teams)

    def _builder_before(self, cutoff: pd.Timestamp) -> FeatureBuilder:
        """A builder whose state contains only matches strictly before *cutoff*."""
        if self._builder is not None and self._builder_cutoff == cutoff:
            return self._builder
        history = self.matches[self.matches["date"] < cutoff]
        builder = FeatureBuilder()
        builder.build(history)
        self._builder = builder
        self._builder_cutoff = cutoff
        return builder

    def _full_builder(self) -> FeatureBuilder:
        """Builder warmed with every completed match (state = latest)."""
        if self._builder is not None and self._builder_cutoff is None:
            return self._builder
        builder = FeatureBuilder()
        builder.build(self.matches)
        self._builder = builder
        self._builder_cutoff = None
        return builder

    def _feature_frame(self, fixtures: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
        builder = self._builder_before(cutoff)
        return builder.build_fixtures(fixtures)

    def _all_features(self) -> pd.DataFrame:
        """Pre-match features for every completed match (computed once)."""
        if getattr(self, "_features_cache", None) is None:
            frame = FeatureBuilder().build(self.matches)
            frame["match_id"] = (frame["_date"].dt.strftime("%Y-%m-%d") + "_"
                                 + frame["_home_team"] + "_" + frame["_away_team"])
            self._features_cache = frame
        return self._features_cache

    # ---- probability pipeline ---------------------------------------------
    def _variant(self, use_odds: bool) -> dict:
        if use_odds:
            return {
                "name": "full",
                "model": self.artifact["model"],
                "features": self.artifact["features"],
                "alpha": float(self.artifact.get("poisson_alpha", 1.0)),
                "beta": float(self.artifact.get("market_beta", 1.0)),
                "temperature": float(self.artifact.get("temperature", 1.0)),
            }
        variant = self.artifact.get("no_odds") or {}
        if not variant.get("model"):
            # Graceful fallback: use the full model with market features absent.
            return self._variant(True) | {"name": "full_fallback"}
        return {
            "name": "no_odds",
            "model": variant["model"],
            "features": variant["features"],
            "alpha": float(variant.get("poisson_alpha", 1.0)),
            "beta": 1.0,
            "temperature": float(variant.get("temperature", 1.0)),
        }

    def _matrix(self, features: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
        missing = [c for c in columns if c not in features.columns]
        if missing:
            log.warning("%d model features missing from the frame: %s", len(missing), missing[:8])
        return features.reindex(columns=list(columns)).astype("float64")

    def predict_frame(self, features: pd.DataFrame, use_odds: bool = True) -> np.ndarray:
        """Return calibrated A/D/H probabilities for a built feature frame."""
        variant = self._variant(use_odds)
        X = self._matrix(features, variant["features"])
        raw = np.asarray(variant["model"].predict_proba(X), dtype=float)
        raw = raw / np.clip(raw.sum(axis=1, keepdims=True), 1e-12, None)

        poisson = features.reindex(
            columns=["pois_p_away", "pois_p_draw", "pois_p_home"]).to_numpy(dtype=float)
        out = blend_with_poisson(raw, poisson, variant["alpha"])

        if variant["beta"] < 1.0:
            market = features.reindex(columns=list(MARKET_COLUMNS)).to_numpy(dtype=float)
            out = blend_with_market(out, market, variant["beta"])

        out = apply_temperature(out, variant["temperature"])
        return out / np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)

    # ---- prediction builders ----------------------------------------------
    def _context(self, features: pd.DataFrame, index: int) -> dict:
        row = features.iloc[index]

        def pick(name: str):
            value = row.get(name)
            return None if value is None or pd.isna(value) else round(float(value), 4)

        return {
            "home_elo": pick("h_elo"),
            "away_elo": pick("a_elo"),
            "elo_diff": pick("elo_diff"),
            "home_form5_ppg": pick("h_form5_ppg"),
            "away_form5_ppg": pick("a_form5_ppg"),
            "home_venue5_ppg": pick("h_venue5_ppg"),
            "away_venue5_ppg": pick("a_venue5_ppg"),
            "home_form5_gf": pick("h_form5_gf"),
            "away_form5_gf": pick("a_form5_gf"),
            "home_form5_ga": pick("h_form5_ga"),
            "away_form5_ga": pick("a_form5_ga"),
            "home_season_ppg": pick("h_season_ppg"),
            "away_season_ppg": pick("a_season_ppg"),
            "poisson_home": pick("pois_p_home"),
            "poisson_draw": pick("pois_p_draw"),
            "poisson_away": pick("pois_p_away"),
            "expected_goals_home": pick("pois_exp_home"),
            "expected_goals_away": pick("pois_exp_away"),
            "home_rest_days": pick("h_rest_days"),
            "away_rest_days": pick("a_rest_days"),
            "home_matches_14d": pick("h_matches_14d"),
            "away_matches_14d": pick("a_matches_14d"),
            "h2h_ppg": pick("h2h_ppg"),
            "h2h_draw_rate": pick("h2h_draw_rate"),
            "market_entropy": pick("mkt_entropy"),
        }

    def _build_predictions(
        self,
        fixtures: pd.DataFrame,
        features: pd.DataFrame,
        mode: str,
        use_odds_override: bool | None = None,
    ) -> list[Prediction]:
        """Score an already-built feature frame.

        ``fixtures`` and ``features`` must have identical row order.
        """
        if fixtures is None or len(fixtures) == 0:
            return []
        fixtures = fixtures.reset_index(drop=True)
        features = features.reset_index(drop=True)
        if len(fixtures) != len(features):
            raise PredictionError(
                f"Feature builder returned {len(features)} rows for {len(fixtures)} fixtures")
        generated = _now()
        out: list[Prediction] = []

        has_odds = features.reindex(columns=list(ODDS_COLUMNS)).notna().all(axis=1).to_numpy()
        market_matrix = features.reindex(columns=list(MARKET_COLUMNS)).to_numpy(dtype=float)

        # Score the two variants in batches; each fixture then takes the row
        # from the variant that matches the information actually available.
        needs_odds = has_odds.copy()
        if use_odds_override is not None:
            needs_odds = needs_odds & bool(use_odds_override)
        proba_full = self.predict_frame(features, use_odds=True) if needs_odds.any() else None
        proba_no_odds = (self.predict_frame(features, use_odds=False)
                         if (~needs_odds).any() else None)

        for i in range(len(features)):
            use_odds = bool(needs_odds[i])
            proba = (proba_full if use_odds else proba_no_odds)[i]
            proba = np.asarray(proba, dtype=float)
            proba = proba / proba.sum()
            row = fixtures.iloc[i]
            market = market_matrix[i]
            has_market = bool(np.isfinite(market).all())
            out.append(Prediction(
                match_id=str(row.get("match_id") or f"{row['date']}_{row['home_team']}_{row['away_team']}"),
                date=str(pd.Timestamp(row["date"]).date()),
                time=str(row.get("time") or "15:00"),
                home_team=row["home_team"],
                away_team=row["away_team"],
                prob_home=float(proba[2]),
                prob_draw=float(proba[1]),
                prob_away=float(proba[0]),
                predicted=config.INDEX_TO_CODE[int(proba.argmax())],
                confidence=float(proba.max()),
                mode=mode,
                model_version=self.version,
                predicted_at=generated,
                model_variant="full" if use_odds else "no_odds",
                has_odds=has_market,
                market_home=float(market[2]) if has_market else None,
                market_draw=float(market[1]) if has_market else None,
                market_away=float(market[0]) if has_market else None,
                source=str(row.get("source") or ""),
                season=str(row.get("season") or config.SEASONS.get(config.CURRENT_SEASON, "")),
                matchweek=row.get("matchweek"),
                context=self._context(features, i),
                fixture_id=str(row.get("fixture_id")) if row.get("fixture_id") else None,
            ))
        out.sort(key=lambda p: (p.date, p.time, p.home_team))
        return out

    # ---- public API --------------------------------------------------------
    def predict_upcoming(self, refresh: bool = False, save: bool = True) -> dict:
        fixtures = processing.load_fixtures(refresh=refresh, matches=self.matches)
        if len(fixtures) == 0:
            payload = self._payload([], MODE_UPCOMING)
            if save:
                write_json_atomic(PREDICTIONS_JSON, payload)
            return payload
        fixtures = fixtures.copy().reset_index(drop=True)
        fixtures["season"] = fixtures["date"].map(
            lambda d: config.SEASONS.get(_season_code(pd.Timestamp(d)), ""))
        features = self._full_builder().build_fixtures(fixtures)
        predictions = self._build_predictions(fixtures, features, MODE_UPCOMING)
        payload = self._payload(predictions, MODE_UPCOMING)
        if save:
            write_json_atomic(PREDICTIONS_JSON, payload)
        log.info("Predicted %d upcoming fixtures", len(predictions))
        return payload

    def predict_manual(
        self,
        home_team: str,
        away_team: str,
        date: str | pd.Timestamp | None = None,
        time: str = "15:00",
    ) -> Prediction:
        home = processing.normalize_team(home_team)
        away = processing.normalize_team(away_team)
        teams = set(self.known_teams)
        if not home or not away:
            raise PredictionError("Both teams are required")
        if home == away:
            raise PredictionError("A team cannot play itself")
        for team in (home, away):
            if team not in teams:
                raise PredictionError(f"Unknown team: {team}")
        if date is None:
            target = self.data_cutoff + pd.Timedelta(days=7)
        else:
            target = pd.Timestamp(date)
            if pd.isna(target):
                raise PredictionError("Invalid match date")
        target = target.normalize()

        season_code = _season_code(target)
        fixture = pd.DataFrame([{
            "date": target,
            "time": time,
            "home_team": home,
            "away_team": away,
            "source": "manual",
            "season": config.SEASONS.get(season_code, ""),
            "matchweek": np.nan,
            "match_id": f"{target.date()}_{home}_{away}",
        }])
        if target > self.data_cutoff:
            builder = self._full_builder()
        else:
            builder = self._builder_before(target)
        features = builder.build_fixtures(fixture)
        # Features are always point-in-time correct, but the production model
        # was fitted on every completed match. A manual request dated in the
        # past is therefore a retrospective simulation, not a live prediction.
        in_sample = bool(target <= self.data_cutoff)
        predictions = self._build_predictions(
            fixture, features, MODE_MANUAL if not in_sample else MODE_RETROSPECTIVE)
        prediction = predictions[0]
        prediction.context["in_sample"] = in_sample
        prediction.context["note"] = (
            "Retrospective: the production model was trained on this date range. "
            "Features still use only pre-match information."
            if in_sample else
            "Genuine pre-match prediction: the fixture has not been played in the dataset."
        )
        return prediction

    def predict_retrospective(self, season: str | None = None, limit: int = 380) -> dict:
        """Re-score completed matches with the production model.

        These are **retrospective simulations**, not predictions that existed
        before kickoff. The dashboard labels them accordingly.
        """
        matches = self.matches
        if season:
            matches = matches[matches["season"] == season]
        matches = matches.sort_values(["date", "time"]).tail(limit).copy()
        fixtures = pd.DataFrame({
            "date": matches["date"].to_numpy(),
            "time": matches["time"].to_numpy(),
            "home_team": matches["home_team"].to_numpy(),
            "away_team": matches["away_team"].to_numpy(),
            "season": matches["season"].to_numpy(),
            "matchweek": (matches["matchweek"].to_numpy()
                          if "matchweek" in matches.columns else np.nan),
            "match_id": (matches["date"].dt.strftime("%Y-%m-%d") + "_"
                         + matches["home_team"] + "_" + matches["away_team"]).to_numpy(),
            "source": "historical",
        })
        all_features = self._all_features().set_index("match_id")
        missing = [m for m in fixtures["match_id"] if m not in all_features.index]
        if missing:
            fixtures = fixtures[~fixtures["match_id"].isin(missing)].reset_index(drop=True)
        features = all_features.loc[fixtures["match_id"].tolist()].reset_index(drop=True)
        predictions = self._build_predictions(fixtures, features, MODE_RETROSPECTIVE)
        truth = matches.set_index("match_id")[["result", "home_goals", "away_goals"]]
        for prediction in predictions:
            if prediction.match_id in truth.index:
                row = truth.loc[prediction.match_id]
                prediction.context["actual_result"] = str(row["result"])
                prediction.context["actual_score"] = f"{int(row['home_goals'])}-{int(row['away_goals'])}"
                prediction.context["correct"] = bool(prediction.predicted == str(row["result"]))
                prediction.context["in_sample"] = True
        return self._payload(predictions, MODE_RETROSPECTIVE)

    # ---- payload -----------------------------------------------------------
    def _payload(self, predictions: Iterable[Prediction], mode: str) -> dict:
        rows = [p.as_dict() for p in predictions]
        counts = {"H": 0, "D": 0, "A": 0}
        for row in rows:
            counts[row["predicted"]] = counts.get(row["predicted"], 0) + 1
        return {
            "generated_at": _now(),
            "model_version": self.version,
            "model_type": self.artifact.get("model_type"),
            "mode": mode,
            "data_cutoff": str(self.data_cutoff.date()),
            "trained_at": self.artifact.get("trained_at"),
            "pipeline": {
                "poisson_alpha": float(self.artifact.get("poisson_alpha", 1.0)),
                "market_beta": float(self.artifact.get("market_beta", 1.0)),
                "temperature": float(self.artifact.get("temperature", 1.0)),
            },
            "n_fixtures": len(rows),
            "prediction_counts": counts,
            "fixtures": rows,
        }


def _season_code(date: pd.Timestamp) -> str:
    start = date.year if date.month >= 7 else date.year - 1
    label = f"{start}/{str(start + 1)[-2:]}"
    for code, season_label in config.SEASONS.items():
        if season_label == label:
            return code
    return config.CURRENT_SEASON


# --------------------------------------------------------------------------
# Convenience functions used by the dashboard and pipeline
# --------------------------------------------------------------------------
def load_predictions() -> dict:
    return read_json(PREDICTIONS_JSON, default={}) or {}


def generate_all(save: bool = True) -> dict:
    engine = PredictionEngine()
    upcoming = engine.predict_upcoming(save=save)
    return upcoming


def _main() -> None:  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    payload = generate_all()
    print(f"{payload['n_fixtures']} fixtures | model {payload['model_version']}")
    for row in payload["fixtures"]:
        print(f"{row['date']} {row['time']}  {row['home_display']:<26} vs "
              f"{row['away_display']:<26} {row['predicted']}  "
              f"H {row['prob_home']:.3f} D {row['prob_draw']:.3f} A {row['prob_away']:.3f}")


if __name__ == "__main__":  # pragma: no cover
    _main()
