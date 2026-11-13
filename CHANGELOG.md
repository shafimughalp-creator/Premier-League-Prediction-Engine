# Changelog

All notable changes to this project. Dates are UTC.

## [1.0.0] — 2026-11-13

### Added

- **Data layer**
  - Acquisition of 13 football-data.co.uk EPL seasons (2014/15 → 2026/27),
    the fixtures feed and the openfootball 2026/27 schedule.
  - Raw-byte caching with retry logic and an append-only update log.
  - Canonical team normalisation (36 clubs) and overround-normalised bookmaker
    consensus features from six providers.
  - Integrity validation and fixture validation.

- **Feature engineering**
  - Chronological state machine producing 237 pre-match features:
    Elo, rolling form (3/5/10), venue splits, season-to-date, previous-season
    form, promotion flags, rest/congestion, attack/defence ratings, Poisson
    goal probabilities, head-to-head and market consensus.
  - 223-feature odds-free variant selected automatically for unpriced fixtures.

- **Modelling**
  - Nine-candidate zoo (LightGBM ×3, XGBoost ×2, random forest, extra trees,
    histogram gradient boosting, multinomial logistic regression).
  - Validation-only temperature scaling, Poisson-prior blend and market blend.
  - Non-negative ensemble weight optimisation with shrinkage toward equal
    weights.
  - Strict chronological split: train < 2023-07-01, validation 2023/24,
    test 2024/25 + 2025/26, 2026/27 for the production refit only.
  - Full metrics: accuracy, balanced accuracy, macro F1, per-class
    precision/recall/F1/support, log loss, Brier, ECE, Wilson CI, confusion
    matrix, calibration curve, season breakdown.
  - Artifact versioning with a model version string, experiment registry and
    human-readable metadata.

- **Prediction**
  - Upcoming, manual and retrospective modes.
  - Point-in-time correct manual predictions; past dates are labelled
    retrospective with `context["in_sample"] = True`.
  - Atomic writes to `data/predictions.json`.

- **Validation**
  - Empirical leakage proofs by perturbation: own-result flip, future
    truncation, cold start, same-day isolation.
  - Full-history audit written to `reports/data_quality/leakage_report.json`.

- **Pipeline**
  - `refresh → validate → guarded retrain → predict` orchestration.
  - Retrain threshold (10 new matches), accuracy-regression guard (1 pp),
    automatic backup/restore of the previous artifact.
  - Durable `data/pipeline_state.json` with stage statuses and durations.

- **Dashboard**
  - Seven-page dark analytics app: Overview, Predictions, Match Centre,
    Manual Prediction, Historical Analysis, Model Performance, Data Centre.
  - Custom CSS design system, probability bars, match cards, radar charts,
    calibration and confusion figures.
  - Honest status object and graceful empty states.

- **Tests**
  - Unit tests for config, processing, features and metric definitions.
  - Leakage tests using the real builder.
  - Prediction-engine tests.
  - Integration tests for the pipeline and all seven dashboard pages via
    Streamlit `AppTest`.

- **Automation & docs**
  - `.github/workflows/test.yml` (unit/leakage then train + full suite).
  - `.github/workflows/update.yml` (daily refresh, guarded retrain, commit).
  - `requirements.txt`, `.gitignore`, README, architecture, data-source,
    status, experiment log and final report.

### Fixed

- `season_code` was round-tripping through CSV as `int64`, which silently set
  `promoted = 1` for every match and left every previous-season feature
  `NaN`. Coerced to `str` at build and load time; mean feature missingness
  fell from 4.85% to 1.66%.
- Market class order (`mkt_away, mkt_draw, mkt_home`) now matches the internal
  A/D/H convention in the blend and baseline code.
- Ensembling no longer tries to JSON-serialise NumPy probability arrays.
- Timezone-naive vs aware timestamp subtraction in the prediction engine.
- Pandas fragmentation warning during multi-season concatenation.
- Mixed date formats in older season files (`format="mixed"`).
- openfootball `score` field parsed whether it is a list or a string.
- Dashboard percentage columns now scale fractions before applying printf
  formats.
- Test predictions loader maps `pred_*` columns to `prob_*`.
- Baseline metrics used for the dashboard are now scored on the test rows
  (previously validation rows were compared with test metrics).
- A successful pipeline run now clears `error` and `traceback` left by an
  earlier failed run instead of leaving contradictory state in
  `data/pipeline_state.json`.
- Replaced the invalid `pytest -m ""` invocation with
  `pytest -o addopts="-q"`, which is how the slow full-history audit is
  included under pytest 9.
- `artifacts/metadata/*.json` is no longer git-ignored, so the update workflow
  can commit the model metadata alongside refreshed predictions.

### Notes

- CatBoost is not installed in this environment and is therefore not part of
  the candidate zoo.
- The 53–55% accuracy target was not reached; see `reports/evaluation/FINAL_REPORT.md`.
- The repository was initialised locally (initial commit `c6e77f5`). No remote
  is configured, so no push or live GitHub Actions run has occurred.
