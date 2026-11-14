# Changelog

All notable changes to this project. Dates are UTC.

## [1.2.0] — 2026-11-13

### Added

- **Persistent fixture synchronisation** (`src/data/fixture_sync.py`)
  - Parses the football-data `fixtures.csv` feed and the openfootball
    `football.json` schedule (the two providers the project already used).
  - Stable `fixture_id` (`"{season_code}:{home}:{away}"`) plus explicit
    `scheduled` / `rescheduled` / `postponed` / `cancelled` / `completed`
    statuses, `season_code`, `season`, `timezone`, `first_seen_at` and
    `updated_at` in `data/fixtures/upcoming.csv`.
  - Sidecar `data/fixtures/sync_state.json` records sources, counts and the
    last 60 runs.
  - Idempotent merge: a repeat run leaves the store byte-identical
    (`new=0 updated=0 unchanged=280`).
  - Fixtures absent from a provider response are retained; only explicit
    cancellation/postponement or a result in `matches.csv` removes them.
  - All-provider failure returns `ok: false`, exits `1` and preserves the
    store; one failed provider reports `partial`.
  - CLI: `python -m src.data.fixture_sync [--fetch] [--json]`.
- **`fixtures` pipeline stage** (`src/pipeline.py`), run after validation and
  before the guarded retrain/predict steps.
- **`.github/workflows/sync-fixtures.yml`**: daily 05:45 UTC + manual dispatch,
  Python 3.12, minimal `contents: write`, serialised with the prediction
  workflow, commits only when `upcoming.csv` changed.
- **31 fixture-sync tests** covering parsing, validation, dedupe, idempotency,
  reschedules, postponements, cancellations, history protection, unknown teams,
  provider failures, the upcoming view, pipeline delegation and workflow
  configuration. Full suite: **158 passed**.

### Fixed

- `_parse_date` parsed ISO dates with `dayfirst=True`, turning `2026-10-11`
  into 10 November (and `2026-10-12` into 10 December). That resurrected four
  already-completed Matchday 6 fixtures as "upcoming" and produced a 3-fixture
  view instead of Matchday 11's 10 fixtures. ISO is now parsed first, with
  day-first only as the football-data fallback.
- `acquisition.refresh()` silently returned cached files when `force=False`, so
  a scheduled run on a warm checkout never re-downloaded. It now always
  attempts the selected downloads and falls back to cache on failure.
- `processing.load_fixtures()` no longer rebuilds the fixture file from scratch
  (which could overwrite a valid store with an empty response); it delegates to
  the sync store and only bootstraps when the store is missing.

## [1.1.0] — 2026-11-13

### Added

- **Research evidence layer** (`src/research/`)
  - `run_audit.py`: determinism, metric recomputation, artifact/metadata/registry
    consistency, probability invariants, market-consensus reconstruction.
  - `diagnostics.py`: draw summaries and deciles, threshold/margin curves,
    validation-only policy selection, reliability tables, ECE, Brier
    decomposition, sharpness, temperature and vector scaling, exact McNemar,
    paired match bootstrap and season-block bootstrap.
  - `run_diagnostics.py`, `run_challengers.py`, `walk_forward.py`,
    `finalize.py` and JSON outputs under `reports/research/`.
  - `src/models/dixon_coles.py`: time-decayed Dixon-Coles goal model with an
    internal date cutoff and low-score correction.
- Validation probability export (`reports/evaluation/validation_predictions.csv`)
  and pre-calibration columns in both prediction files.
- Validation metrics in the training report, model artifact, metadata and
  experiment registry.
- Model Performance page: draw analysis, paired market comparison,
  calibration/sharpness, season table, walk-forward table, challenger table and
  an explicit status/limitations block.
- Tests for draw rules, calibration, paired statistics, promotion gates,
  Dixon-Coles leakage isolation, pipeline failure recovery and the new dashboard
  sections. Full suite: **127 passed**.

### Changed

- Experiment registry now records `record_type` and preserves research records
  while capping training-run history at 20.
- Dashboard research loaders degrade to empty states when a report is missing.
- Documentation reflects that `mkt_books` counts price sources (4–8, median 6),
  not distinct bookmakers, and that prices have no explicit publication
  timestamp.

### Research outcomes (no production change)

- Draw threshold 0.28 passed the validation gate but failed the walk-forward
  replication gate (mean −0.53 pp accuracy, macro gain in 3/5 origins) and was
  rejected.
- Vector scaling was rejected (0.0009 validation log-loss gain, gate 0.002).
- Dixon-Coles standalone and blended variants were rejected (0.0003 gain).
- The 143-feature odds-free core set was rejected (−2.63 pp validation accuracy).
- Paired test comparison: +2 model-only wins, 0 market-only, McNemar p = 1.00;
  accuracy difference +0.26 pp (95% CI 0.00 to +0.66); log loss 0.0058 worse
  than the market.
- Walk-forward selection-origin means: production 55.26%, market 55.53%,
  no-odds 52.63%.
- Decision: **COMPLETE — EXISTING MODEL RETAINED**.

## [1.0.0] — 2026-11-13

### Added

- **Data layer**
  - Acquisition of 13 football-data.co.uk EPL seasons (2014/15 → 2026/27),
    the fixtures feed and the openfootball 2026/27 schedule.
  - Raw-byte caching with retry logic and an append-only update log.
  - Canonical team normalisation (36 clubs) and overround-normalised bookmaker
    consensus features from seven providers (plus average and maximum series).
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
