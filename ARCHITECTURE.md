# Architecture

## System overview

```
                  ┌──────────────────────────────┐
                  │  football-data.co.uk (E0.csv)│
                  │  fixtures.csv / openfootball │
                  └──────────────┬───────────────┘
                                 │ raw bytes, cached unchanged
                                 ▼
   src/data/acquisition.py ──► data/raw/
                                 │
                                 ▼
   src/data/processing.py ──► data/processed/matches.csv
                                 │         (clean, normalised, odds derivation)
                                 ▼
   src/features/builder.py ──► 237 pre-match features (223 without odds)
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
      src/models/training   src/models/evaluation  src/data/validation
      (chronological        (metrics, calibration, (integrity + empirical
       selection, ensemble)  confusion matrices)   leakage proofs)
              │
              ▼
   artifacts/models/model.pkl ──► src/prediction/engine.py
              │                          │
              ▼                          ▼
   reports/evaluation/        data/predictions.json
   training_report.json                │
                                      ▼
                             app.py + src/dashboard/
```

## Directory layout

```
.
├── app.py                        Streamlit entry point (7 pages)
├── requirements.txt
├── pytest.ini
├── docs/                         long-form documentation
├── src/
│   ├── config.py                 paths, seasons, class order, aliases
│   ├── pipeline.py               refresh → validate → train → predict
│   ├── data/
│   │   ├── acquisition.py        download + cache raw payloads
│   │   ├── processing.py         clean matches, derive odds features, fixtures
│   │   └── validation.py         integrity checks + leakage proofs
│   ├── features/builder.py       chronological feature state machine
│   ├── models/
│   │   ├── training.py           candidate zoo, calibration, blends, artifact
│   │   ├── ensemble.py           soft-voting wrapper
│   │   ├── dixon_coles.py        time-decayed goal model (research)
│   │   └── evaluation.py         accuracy, F1, log loss, Brier, ECE, CI
│   ├── prediction/engine.py      upcoming / manual / retrospective scoring
│   ├── research/
│   │   ├── diagnostics.py        draw rules, calibration, paired statistics
│   │   ├── run_audit.py          reproducibility + market-consensus audit
│   │   ├── run_diagnostics.py    draw/calibration/paired reports
│   │   ├── run_challengers.py    Dixon-Coles + core-feature experiments
│   │   ├── walk_forward.py       expanding-window evaluation
│   │   └── finalize.py           pre-registered promotion gates
│   ├── utilities/io.py           atomic JSON read/write
│   └── dashboard/
│       ├── theme.py              CSS design system + Plotly layout
│       ├── data.py               cached loaders, honest health object
│       ├── components.py         reusable cards, bars, charts
│       └── pages/                overview, predictions, match_centre, manual,
│                                 historical, model_performance, data_centre
├── tests/
│   ├── unit/                     config, processing, features, metrics
│   ├── leakage/                  perturbation proofs
│   ├── prediction/               engine behaviour
│   └── integration/              pipeline + headless dashboard
├── data/
│   ├── raw/                      source files, never edited
│   ├── processed/                canonical tables + logs
│   └── fixtures/                 resolved upcoming fixtures
├── artifacts/
│   ├── models/model.pkl          production ensemble + no-odds variant
│   └── metadata/model_meta.json  human-readable run summary
├── reports/
│   ├── evaluation/               training report + held-out/validation predictions
│   ├── research/                 audit, diagnostics, walk-forward, decisions
│   ├── experiments/              append-only experiment registry
│   └── data_quality/             quality + leakage reports
└── .github/workflows/            test.yml, update.yml
```

## Data flow in detail

### 1. Acquisition (`src/data/acquisition.py`)

- Thirteen season CSVs and two fixture sources are downloaded with retries.
- Files are cached under `data/raw/` and validated before being written
  (`Div,` header check for season files, JSON parse for openfootball).
- A refresh failure never raises into the pipeline: the cached copy is used and
  the failure is recorded in `data/processed/update_log.json`.

### 2. Processing (`src/data/processing.py`)

- Team names are canonicalised through `config.TEAM_ALIASES`.
- `season_code` is forced to `str` at both build and load time. This matters:
  an integer round-trip through CSV silently broke season-transition logic.
- Bookmaker prices from seven providers (plus average and maximum series) are
  converted to overround-normalised probabilities; dispersion, price-source
  count, favourite and entropy are derived.
- Output columns are split into:
  - identity: `match_id`, `date`, `time`, `season_code`, `season`, teams
  - target: `home_goals`, `away_goals`, `result`
  - pre-match context used by features: historical stats and market fields

### 3. Feature engineering (`src/features/builder.py`)

The builder is a **chronological state machine**:

1. Sort matches by `(date, time, home_team)`.
2. Group by date.
3. Featurise **every** match on that date using state built only from earlier
   dates.
4. Only then update Elo, form windows, season totals, head-to-head, rest days
   and league scoring averages with that date's results.

Feature families (237 total):

| Family | Examples | Count |
|---|---|---|
| Elo | `h_elo`, `elo_diff`, `elo_expected_home` | 6 |
| Rolling form (3/5/10) | `h_form5_ppg`, `diff_form10_sot` | ~90 |
| Venue form | `h_venue5_ppg`, `diff_venue5_gd` | ~34 |
| Season-to-date | `h_season_ppg`, `diff_season_gd_pg` | ~12 |
| Previous season | `h_prev_ppg`, `h_promoted` | 12 |
| Rest / congestion | `h_rest_days`, `h_matches_14d` | 6 |
| Att/def strength | `h_attack_home`, `a_defence_away` | 8 |
| Poisson goal model | `pois_exp_home`, `pois_p_draw` | 7 |
| Head-to-head | `h2h_ppg`, `h2h_draw_rate` | 4 |
| Market | `mkt_home`, `mkt_entropy`, `odds_home` | 13 |

`odds_free_columns()` removes the 14 `mkt_*`/`odds_*` columns, yielding the
223-feature `no_odds` variant used for fixtures without prices.

### 4. Model selection (`src/models/training.py`)

Chronological split, fixed before any modelling:

| Split | Dates | Matches |
|---|---|---|
| Train | 2014-08-16 → 2023-05-28 | 3,420 |
| Validation | 2023-08-11 → 2024-05-19 | 380 |
| Test | 2024-08-16 → 2026-05-24 | 760 |
| Current (refit only) | 2026/27 partial | 100 |

Candidate families (9):

- LightGBM: regularised, deep, shallow
- XGBoost: regularised, deep
- Random forest, extra trees, histogram gradient boosting
- Multinomial logistic regression (scaled, imputed)

Each candidate is early-stopped on the validation season, temperature-scaled on
validation, and ranked by calibrated validation log loss. The top four are
blended by non-negative weights optimised on validation (with a shrink toward
equal weights). The best of {best single, ensemble} is carried forward, then:

1. Poisson-prior blend `alpha` tuned on validation,
2. bookmaker-consensus blend `beta` tuned on validation (full variant only),
3. temperature scaling `T` tuned on validation.

Only after all three are frozen is the test season scored once. The production
artifact is then refit on train + validation + test + current.

Validation and pre-calibration probabilities are exported to
`reports/evaluation/validation_predictions.csv` and `test_predictions.csv`.
These files are the only inputs to the research diagnostics, so no experiment
re-runs the training pipeline or touches the model artifact.

### 4b. Research and promotion gates (`src/research/`)

| Stage | Script | Output |
|---|---|---|
| Reproducibility audit | `run_audit.py` | `audit.json` |
| Draw / calibration / paired stats | `run_diagnostics.py` | `draw_analysis.json`, `calibration_analysis.json`, `paired_comparisons.json` |
| Challengers | `run_challengers.py` | `challengers.json` |
| Expanding-window evaluation | `walk_forward.py` | `walk_forward.json` + per-season prediction CSVs |
| Promotion decisions | `finalize.py` | `promotion_decisions.json` |

The gates are fixed before running the experiments. A draw policy must pass the
validation gate **and** replicate across selection-grade walk-forward origins
(mean accuracy change ≥ −0.5 pp, macro-F1 gain in at least four of five
origins, no origin worse than −1 pp). Calibration, core-feature and goal-model
challengers must improve validation log loss by at least 0.002 without an
unacceptable accuracy loss. The historical test seasons are copied into the
reports for transparency but never enter a gate.

### 5. Prediction (`src/prediction/engine.py`)

- **Upcoming**: fixtures resolved from the fixture feed or openfootball, scored
  with the full builder state. If no prices exist, the `no_odds` artifact is
  used rather than imputing market columns.
- **Manual**: any two known teams and a date. The builder is rebuilt from
  matches strictly before that date. Dates inside the training window are
  labelled `retrospective` with `context["in_sample"] = True`.
- **Retrospective**: re-scores completed matches and attaches the actual result
  so the dashboard can label them honestly.

All probabilities pass through the same blend pipeline and are renormalised.

## Leakage guarantees

`src/data/validation.py` proves the design empirically:

1. **Own-result independence** — flipping a match's scoreline leaves its own
   feature row unchanged (tested on first, middle and last matches).
2. **Future independence** — truncating the dataset at a date leaves that
   date's features byte-identical.
3. **Cold start** — the first match sees Elo 1500, no form, season game 1.
4. **Same-day isolation** — matches sharing a date are all featurised before
   any of them updates state.

Additional safeguards:

- No random train/test split anywhere.
- Target columns (`result`, goals) are excluded from `feature_columns()`.
- `test_predictions.csv` is written from the frozen test pass only.
- The manual mode never rebuilds state with future matches.

## Orchestration (`src/pipeline.py`)

```
refresh (network, degrade to cache)
  → process
  → validate integrity
  → retrain?  (only if ≥10 new matches; skip if cutoff unchanged)
      → backup model.pkl
      → train
      → promote, or restore backup on regression > 1 pp
  → predict upcoming
  → write data/pipeline_state.json
```

Stages and statuses are appended to `pipeline_state.json`, which the Data
Centre page renders. `--offline`, `--retrain {auto,always,never}` and
`--predict-only` are supported.

## Dashboard

A single Streamlit app with a radio-based nav and seven pages. Shared modules:

- `theme.py` — CSS variables, gradients, cards, probability bars.
- `data.py` — `st.cache_data(ttl=120)` loaders with graceful empty states.
- `components.py` — HTML match cards, probability donuts, calibration and
  confusion figures, radar charts.

The app never shows a raw traceback: each page render is wrapped and failures
surface as a readable error with technical detail behind an expander.

## Extension points

- Add a model: append a spec to `CANDIDATES` in `training.py`; it is scored and
  compared automatically.
- Add a feature: extend `_team_block` / `_features_for`; the leakage proofs and
  unit tests will immediately exercise it.
- Add a data source: implement a loader in `processing.py` that returns the
  canonical schema; nothing downstream changes.
- Change the evaluation window: edit `TRAIN_END` / `VAL_END` / `TEST_END` in
  `config.py`, then rerun the pipeline and the leakage audit.
