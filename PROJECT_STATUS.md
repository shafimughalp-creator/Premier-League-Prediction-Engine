# Project Status

**Last verified:** 2026-11-13
**Data cutoff:** 2026-11-08 (Matchweek 10 complete)
**Production model:** `ensemble:random_forest+lgbm_regularised+lgbm_deep+lgbm_shallow`
**Artifact:** `artifacts/models/model.pkl` (237 features; 223-feature no-odds variant)

This file distinguishes **verified** behaviour from **prepared** functionality.
Nothing is marked verified unless it was executed in this environment.

---

## Verified working

| Area | Evidence |
|---|---|
| Data acquisition | 13 season CSVs + fixtures + openfootball downloaded; 4,660 matches |
| Data processing | 36 clubs, 100% odds coverage, 100% match-stat coverage, 0 duplicates |
| Feature engineering | 237 features, 1.66% mean missingness, 223 without market columns |
| Chronological split | train 3,420 / validation 380 / test 760 / current 100 |
| Model training | 9 candidates compared; ensemble selected; artifact written (102 MB) |
| Test evaluation | 760 unseen matches, 51.84% accuracy, log loss 0.9989 |
| Leakage audit | all checks pass (own-result, future, cold start, integrity) |
| Prediction engine | upcoming 10 fixtures, manual, retrospective — all exercised |
| Dashboard | all 7 pages render headlessly through Streamlit `AppTest` |
| Pipeline | `python -m src.pipeline --offline --predict-only` exits 0 in ~20 s |
| Unit + leakage tests | 108 tests pass across unit, leakage, prediction and integration suites |
| Full suite | `python -m pytest -o addopts="-q"` → 108 passed in 230 s (includes slow full-history audit) |
| GitHub workflows | files written and syntactically complete (not yet run on GitHub) |
| Local version control | git repository initialized; initial commit `c6e77f5` (no remote configured) |

## Honest performance summary

| Model | Accuracy | Log loss | Brier | ECE |
|---|---|---|---|---|
| Production (market-blended) | **51.84%** | 0.9989 | 0.5991 | 0.0640 |
| Bookmaker consensus | 51.58% | 0.9931 | 0.5946 | 0.0321 |
| No-odds variant | 50.39% | 1.0232 | 0.6126 | 0.0478 |

- 95% CI for accuracy: **48.29% – 55.38%**.
- Macro F1: **0.3887**. Draw recall: **0.0** (no draws selected by argmax).
- The 50% minimum is met. The 53–55% target is **not** demonstrated.
- The selected blend puts **95% weight on the bookmaker consensus**
  (`market_beta = 0.05`), chosen on validation log loss. This is a finding, not
  a bug: with this feature set the independent model signal did not beat the
  closing market.

## Prepared but not yet proven in production

| Item | Status | What is needed |
|---|---|---|
| GitHub Actions | Workflows committed locally | Push to a GitHub repository and observe one green run |
| Daily automation | `update.yml` scheduled 06:15 UTC | Repository secrets/permissions; first scheduled execution |
| Live fixture prices | Full variant ready | football-data must publish prices for the next round |
| Auto-retrain guard | Implemented and unit-exercised | A real ≥10-match data increment to trigger promotion |
| Standings | Computed from processed matches | Verified for the current partial season |
| Draw-aware decision rule | API exists (`choose_outcome(threshold=...)`) | Validation-tuned threshold experiment; not promoted |

## Known limitations

1. **Draws are never predicted** by the argmax rule. This inflates home/away
   recall and zeroes draw recall. Forcing draws would trade accuracy for macro
   F1; that decision has not been made silently.
2. **Market dependence.** The production pipeline is 95% market consensus. It
   should be described as a calibrated market-plus-model blend, not as an
   independent forecasting edge.
3. **2026/27 is partial** (100 matches). It is used only for the production
   refit and current predictions, never for selection.
4. **No injury/lineup data.** The historical record does not support a
   leakage-free 13-season lineup dataset.
5. **CatBoost unavailable** in this environment; it is omitted rather than
   faked.

## Next actions (in priority order)

1. Push the repository and let `test.yml` run end to end.
2. Enable the daily `update.yml` schedule and confirm one committed refresh.
3. Run a validation-only draw-threshold study; promote only if it improves
   validation macro F1 without material accuracy loss.
4. Investigate features that could add signal beyond the closing market
   (lineup strength, travel, referee, rest asymmetry) with strict pre-match
   timestamps.
5. Add a proper probability-sharpness report (reliability + resolution) to the
   Model Performance page.
