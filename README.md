# EPL Intelligence

A leakage-audited Premier League match-outcome platform: a chronological data
pipeline, a 237-feature pre-match model, a prediction engine with three modes,
and a seven-page football-analytics dashboard.

The project is built around one rule: **report what actually happened.** Every
metric below was produced by the code in this repository, on a test window that
was never used for model selection.

---

## Verified results

Test window: **2024/25 + 2025/26, 760 matches**, scored once after all choices
were frozen on the 2023/24 validation season.

| Model | Accuracy | Log loss | Brier | ECE | Macro F1 |
|---|---|---|---|---|---|
| **Production (market blend)** | **51.84%** | 0.9989 | 0.5991 | 0.0640 | 0.389 |
| Bookmaker consensus | 51.58% | 0.9931 | 0.5946 | 0.0321 | 0.386 |
| No-odds model | 50.39% | 1.0232 | 0.6126 | 0.0478 | 0.376 |
| Elo + logistic | 49.47% | 1.0162 | 0.6109 | 0.0675 | 0.369 |
| Always home win | 41.71% | 2.6928 | 1.1314 | 0.5629 | 0.196 |

- 95% Wilson CI for accuracy: **48.29% – 55.38%**
- Per class: home recall 81.4%, away recall 55.3%, **draw recall 0.0%**
- Season split: 2024/25 **54.47%**, 2025/26 **49.21%**

### The honest conclusion

- The 50% minimum is met; the **53–55% target is not demonstrated**.
- Validation accuracy was 59.5%, but that season was unusually predictable.
  The drop to 51.8% on unseen seasons is the realistic estimate.
- The validation-tuned blend assigns **95% weight to the bookmaker consensus**
  (`market_beta = 0.05`). The model therefore matches the market rather than
  beating it. This is documented rather than hidden.
- No argmax decision ever selects a draw. Draw recall is reported as zero; the
  output is not silently re-thresholded to inflate it.

Full detail: [`reports/evaluation/FINAL_REPORT.md`](reports/evaluation/FINAL_REPORT.md)
and [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md).

### Post-audit findings (2026-11-13)

An independent audit and controlled challenger programme was run after the
original release. Nothing below changed the production model.

| Check | Result |
|---|---|
| Reproducibility audit | PASS — metrics recomputed exactly from `test_predictions.csv`; artifact/metadata/registry agree |
| Paired market comparison | 392 both correct, 2 model-only, 0 market-only, 366 both wrong; McNemar p = 1.00 |
| Accuracy difference (model − market) | +0.26 pp, 95% CI 0.00 to +0.66 pp |
| Log-loss difference (model − market) | +0.0058 (model slightly worse) |
| Walk-forward, selection origins | production 55.26% ± 3.4 pp; market 55.53% ± 3.2 pp |
| Draw threshold (0.28, validation-selected) | rejected — mean −0.53 pp accuracy, macro gain in 3/5 origins |
| Vector scaling | rejected — 0.0009 validation log-loss gain (gate 0.002) |
| Dixon-Coles + production blend | rejected — 0.0003 validation log-loss gain |
| Core 143-feature odds-free set | rejected — −2.63 pp validation accuracy |

**Decision: COMPLETE — EXISTING MODEL RETAINED.** Evidence lives in
`reports/research/` and the Model Performance page.


---

## Quickstart

```bash
pip install -r requirements.txt
python -m src.pipeline --retrain always     # download, process, train, predict
streamlit run app.py                        # launch the dashboard
```

Day-to-day operation:

```bash
python -m src.pipeline                 # refresh + retrain only if needed
python -m src.pipeline --offline       # use cached raw data only
python -m src.pipeline --predict-only  # skip training
python -m src.data.validation          # integrity + leakage audit
python -m pytest                       # unit, leakage, prediction, integration
```

The first run downloads 13 seasons from football-data.co.uk and takes a few
minutes. Subsequent offline runs take about 20 seconds.

---

## Dashboard

Seven pages, all rendered headlessly in CI via Streamlit `AppTest`:

| Page | What it shows |
|---|---|
| **Overview** | KPI strip, next fixtures, latest results, model-vs-baseline chart, season trend, standings |
| **Predictions** | Matchweek/team filters, six sort modes, match cards, full probability table |
| **Match Centre** | Elo, form, venue splits, rest, head-to-head, radar profile, probability decomposition |
| **Manual Prediction** | Any two teams and a date; point-in-time features; retrospective warning |
| **Historical Analysis** | Held-out ledger, rolling accuracy, confusion matrix, highest-confidence misses |
| **Model Performance** | Metrics, calibration, per-class table, candidates, draw analysis, paired market statistics, walk-forward, challengers, limitations |
| **Data Centre** | Refresh control, sources, integrity + leakage audits, pipeline log, metadata |

Design: dark navy canvas, translucent panels, violet/electric-blue accents,
custom CSS, Plotly charts, no external images.

---

## How the model works

### Data

- **13 seasons** of football-data.co.uk results (2014/15 → 2026/27), 4,660
  matches, 36 clubs, 100% odds and match-stat coverage.
- Upcoming fixtures from the football-data feed when prices exist, otherwise
  the openfootball 2026/27 schedule.
- Raw payloads cached unchanged in `data/raw/`; processing is reproducible.

See [`DATA_SOURCES.md`](DATA_SOURCES.md).

### Features (237 / 223 without odds)

Elo, rolling form (3/5/10), venue-specific form, season-to-date, previous-season
form and promotion flags, rest and congestion, attack/defence ratings, Poisson
goal probabilities, head-to-head, and overround-normalised market consensus.
The odds-free variant drops the 14 market columns and is used when a fixture
has no published prices.

### Chronological guarantee

The feature builder is a state machine: for every date it featurises **all**
matches using state from strictly earlier dates, then updates Elo, form,
season totals and head-to-head with that date's results. There is no random
split anywhere.

`src/data/validation.py` proves it by perturbation:

1. Flip a match's own result → its own feature row is unchanged.
2. Truncate all later data → the earlier date's features are byte-identical.
3. The first match starts from Elo 1500, no form, season game 1.
4. Matches sharing a date cannot see each other's results.

The audit is written to `reports/data_quality/leakage_report.json`.

### Selection protocol

| Split | Dates | Rows | Use |
|---|---|---|---|
| Train | 2014-08-16 → 2023-05-28 | 3,420 | Fit candidates |
| Validation | 2023-08-11 → 2024-05-19 | 380 | Early stopping, temperature, blends, ensemble weights |
| Test | 2024-08-16 → 2026-05-24 | 760 | Scored once, never used for selection |
| Current | 2026/27 partial | 100 | Production refit and current predictions |

Nine candidates are compared (LightGBM ×3, XGBoost ×2, random forest, extra
trees, histogram gradient boosting, logistic regression). The top four form a
soft-voting ensemble whose weights are optimised on validation and shrunk
toward equal weights. Poisson and market blends plus temperature scaling are
then tuned on validation only.

---

## Prediction modes

1. **Upcoming** — the next round, scored with the full builder state. If no
   prices are published yet, the dedicated `no_odds` model is used rather than
   imputing market columns.
2. **Manual** — any two known teams and a date. Features use only matches
   strictly before that date. A date inside the training window is labelled
   `retrospective` and flagged `context["in_sample"] = true`.
3. **Retrospective** — re-scores completed matches and attaches the actual
   result, clearly labelled as a simulation rather than a pre-match call.

Outputs are written atomically to `data/predictions.json` with the model
version, generation timestamp, data cutoff and per-fixture context.

---

## Automation

- `.github/workflows/test.yml` — unit + leakage suites, then a full data build,
  training run, leakage audit and the complete test suite.
- `.github/workflows/update.yml` — daily at 06:15 UTC: refresh, guarded
  retrain (only after ≥10 new matches; reject a regression > 1 pp), re-score
  fixtures, and commit the refreshed tables and predictions.

Both workflows are **written but not yet observed running on GitHub**, because
this project is not currently a pushed repository. See `PROJECT_STATUS.md`.

---

## Tests

```bash
python -m pytest tests/unit tests/leakage      # fast
python -m pytest tests/prediction               # engine behaviour
python -m pytest tests/integration              # pipeline + dashboard AppTest
python -m pytest -o addopts="-q"                # include the slow full-history audit
```

The current suite contains **127 tests**. The leakage tests use the real
`FeatureBuilder`, not a reimplementation. The dashboard tests render every page
and exercise the manual-prediction workflow, including the same-team rejection
path. Pipeline tests simulate a refresh failure (cache fallback) and a training
failure (incumbent artifact restored byte-for-byte).

---

## Project layout

```
src/
  config.py            paths, seasons, class order, team aliases
  pipeline.py          refresh → validate → guarded retrain → predict
  data/                acquisition, processing, validation
  features/builder.py  chronological feature state machine
  models/              candidate zoo, ensemble, metrics, Dixon-Coles
  prediction/engine.py upcoming / manual / retrospective
  research/            audit, diagnostics, walk-forward, promotion gates
  dashboard/           theme, cached data, components, 7 pages
tests/                 unit, leakage, prediction, integration
data/                  raw cache, processed tables, fixtures
artifacts/             model.pkl, metadata
reports/               evaluation, research, experiments, data quality
```

More detail: [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## Limitations

- Draws are never selected by argmax (draw recall 0%). Validation-selected draw
  thresholds did not replicate across walk-forward origins and were rejected;
  no draws are forced to inflate recall.
- The model is 95% market consensus. Paired tests show it matches market
  accuracy within noise and has slightly worse probability quality.
- The 2024/25 and 2025/26 test seasons were examined in earlier work; a genuine
  prospective test requires future matches.
- The 2026/27 season is partial and used only for the production refit.
- No injury, lineup, referee or travel data: none can be reconstructed
  leakage-free across 13 seasons from the chosen public sources.
- CatBoost is unavailable in this environment and is omitted.
- GitHub automation is prepared but not yet verified on a live repository.

## Documentation

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — system design and data flow
- [`DATA_SOURCES.md`](DATA_SOURCES.md) — sources, season codes, coverage, limits
- [`PROJECT_STATUS.md`](PROJECT_STATUS.md) — verified vs prepared functionality
- [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) — every modelling decision and result
- [`CHANGELOG.md`](CHANGELOG.md) — implementation history
- [`reports/evaluation/FINAL_REPORT.md`](reports/evaluation/FINAL_REPORT.md) — full evaluation
