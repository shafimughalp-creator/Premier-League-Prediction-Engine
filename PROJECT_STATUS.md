# Project Status

**Last verified:** 2026-11-15 (automation integration release)
**Data cutoff:** 2026-11-08 (Matchweek 10 complete)
**Production model:** `ensemble:random_forest+lgbm_regularised+lgbm_deep+lgbm_shallow`
**Artifact:** `artifacts/models/model.pkl` (237 features; 223-feature no-odds variant)
**Decision:** **COMPLETE — EXISTING MODEL RETAINED**

> **2026-11-15 update:** the automation system described in
> [`docs/automation.md`](docs/automation.md) is now implemented: immutable
> pre-match prediction snapshots, official results synchronisation,
> evaluation with matchweek/season accuracy, and a live `data.py` loader
> feeding the approved dashboard (which gained one Results nav entry — no
> redesign). The dashboard and test-suite rows below reflect that release.

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
| Reproducibility audit | `reports/research/audit.json` → `passed: true` |
| Leakage audit | all checks pass (own-result, future, cold start, integrity) |
| Draw diagnostics | `reports/research/draw_analysis.json` (validation + benchmark test) |
| Calibration study | `reports/research/calibration_analysis.json` |
| Paired market comparison | `reports/research/paired_comparisons.json` (McNemar + bootstrap) |
| Walk-forward evaluation | 7 chronological origins, `reports/research/walk_forward.json` |
| Challenger experiments | Dixon-Coles + core feature set, `reports/research/challengers.json` |
| Promotion decisions | `reports/research/promotion_decisions.json` |
| Prediction engine | upcoming 10 fixtures, manual, retrospective - all exercised |
| Fixture synchronisation | 280 stored fixtures, 10-fixture MW11 view; repeat run `new=0 updated=0`, byte-identical store |
| Prediction snapshots | `src/prediction/snapshots.py` + pipeline reconcile: 10 MW11 pre-match snapshots created, second run `unchanged=10` (idempotent); post-kickoff never overwritten |
| Results synchronisation | `src/results/sync.py`: 100/100 official 2026/27 results synced live from football-data + openfootball; idempotent rerun; matchweeks from provider rounds (6 filled via unambiguous pair fallback) |
| Evaluation & accuracy | `src/evaluation/evaluate.py`: matchweek + season aggregates, `accuracy = correct/evaluated`; MW1–10 = 100 `missing_prediction` (pre-automation, honest), MW11–38 pending, accuracy `null` |
| Live dashboard loader | `data.py` builds the prototype DATA shape from the automation stores (44/44 frozen-value parity checks vs the approved prototype); `refresh()` rebuilds in place on store change or 5-min TTL |
| Dashboard | approved match centre (`?page=overview/week/table/match`) plus new `?page=results[&mw=N]` page — all render headlessly through Streamlit `AppTest` with no exceptions |
| Pipeline | `python -m src.pipeline --offline --predict-only` exits 0 in ~20 s |
| Failure handling | refresh failure falls back to cache; training failure restores incumbent; sync failure preserves the store; missing stores degrade to honest empty states |
| Full suite | `python -m pytest -o addopts="-q"` → **211 passed** (14 snapshot + 15 results + 10 evaluation + 28 new integration tests included) |
| GitHub workflows | 4 files written, YAML-validated, shared `update` concurrency group (not yet run on GitHub — no remote) |
| Local version control | git repository; automation commits through `d4397bc`; no remote |

## Honest performance summary

| Model | Accuracy | Log loss | Brier | ECE |
|---|---|---|---|---|
| Production (market-blended) | **51.84%** | 0.9989 | 0.5991 | 0.0640 |
| Bookmaker consensus | 51.58% | 0.9931 | 0.5946 | 0.0321 |
| No-odds variant | 50.39% | 1.0232 | 0.6126 | 0.0478 |

- 95% CI for accuracy: **48.29% – 55.38%** (Wilson, 760 matches).
- Macro F1: **0.3887**. Draw recall: **0.0** (argmax never selects a draw).
- The 50% minimum is met. The 53–55% target is **not** demonstrated.
- The selected blend puts **95% weight on the bookmaker consensus**
  (`market_beta = 0.05`), chosen on validation log loss.
- Paired comparison: 392 matches correct for both systems, 2 only for the
  model, 0 only for the market, 366 both wrong. Exact McNemar p = 1.00.
  Accuracy difference +0.26 pp (95% CI 0.00 to +0.66). Log-loss difference
  −0.0058 (market minus model; the market is slightly better).

## Draw findings

- Actual draw rate on the test seasons: 25.9%. Mean predicted draw
  probability: 22.0%. The model is mildly under-confident on draws but never
  ranks a draw first (share of matches where draw is argmax: 0%).
- A validation-selected threshold of 0.28 raised 2023/24 macro F1 from 0.436 to
  0.463 at unchanged accuracy. It did **not** replicate: across five
  selection-grade walk-forward origins the mean accuracy change was −0.53 pp,
  macro F1 improved in only 3 of 5 origins, and the worst origin lost 2.63 pp
  accuracy. Applied to the already-examined test seasons it lost 1.45 pp.
- **Decision:** argmax retained. No draws are forced.

## Calibration findings

- Validation selected temperature scaling (`T = 0.85` in the production blend).
- Vector scaling improved validation log loss by only 0.0009 (gate: 0.002) and
  was worse than temperature scaling at 0 of 5 walk-forward origins.
- On the already-examined test seasons, uncalibrated probabilities score better
  than temperature-scaled ones (log loss 0.9935 vs 0.9989; ECE 0.038 vs 0.064).
  This is reported as a limitation, not used to re-select calibration.
- **Decision:** temperature scaling retained because it was selected on
  validation and the alternative failed its gate.

## Walk-forward findings

Selection-grade origins (2019/20–2023/24), mean ± standard deviation:

| Variant | Accuracy | Log loss | Macro F1 |
|---|---|---|---|
| Production | 55.26% ± 3.4 pp | 0.9619 | 0.408 |
| Market consensus | 55.53% ± 3.2 pp | 0.9578 | 0.411 |
| No-odds model | 52.63% ± 4.2 pp | 0.9840 | 0.393 |
| Elo + logistic | 53.63% ± 3.3 pp | 0.9781 | 0.395 |

The production blend tracks the market within noise. The validation-selected
market weight is unstable across origins (`beta` ranged 0.05–0.60), which is
expected when the independent signal is weak. The 2024/25 and 2025/26 origins
are reported for completeness only.

## Challengers evaluated and rejected

| Challenger | Validation result | Decision |
|---|---|---|
| Draw threshold / margin policies | +0.027 macro F1 at equal accuracy (2023/24) | Rejected — did not replicate |
| Vector scaling | −0.0009 log loss vs temperature | Rejected — below 0.002 gate |
| Dixon-Coles standalone | 56.84% accuracy, log loss 0.9364 | Rejected — weaker than production |
| Dixon-Coles + production blend | weight 0.08, −0.0003 log loss | Rejected — negligible gain |
| Core odds-free features (143) | −0.0045 log loss but −2.63 pp accuracy | Rejected — accuracy gate missed |

## Prepared but not yet proven in production

| Item | Status | What is needed |
|---|---|---|
| GitHub Actions | Workflows committed locally (`sync-fixtures`, `update`, `results`, `test`) | Push to a GitHub repository and observe one green run each |
| Daily automation | `sync-fixtures.yml` 05:45 UTC, `update.yml` 06:15 UTC, `results.yml` every 6 h at :23 | Repository permissions; first scheduled execution |
| Fixture workflow | Locally tested CLI + mocked tests | A real GitHub run with `contents: write` and branch protection allowing the bot push |
| Results & evaluation workflow | Live-provider tested locally (100 results, idempotent) | First real GitHub execution of `results.yml` |
| Prediction snapshots | Live-tested locally (create + idempotent reconcile) | First real GitHub execution of `update.yml` post-kickoff lock path |
| Live fixture prices | Full variant ready | football-data must publish prices for the next round |
| Auto-retrain guard | Implemented and integration-tested | A real ≥10-match data increment to trigger promotion |
| Prospective evaluation | 2026/27 snapshots accumulating | Let the season complete without re-tuning, then score once |

## Known limitations

1. **Draws are never predicted** by argmax. Validation-selected draw policies
   did not replicate out of sample, so none was promoted.
2. **Market dependence.** The production pipeline is 95% market consensus. It
   should be described as a calibrated market-plus-model blend, not as an
   independent forecasting edge.
3. **Test-set reuse.** 2024/25 and 2025/26 were examined in earlier work and
   remain historical benchmark evidence. Genuine prospective evidence requires
   future matches.
4. **Single-season validation.** Walk-forward evaluation mitigates but does not
   eliminate selection noise; the market weight varies widely by origin.
5. **Calibration transfer.** Temperature scaling helped validation but not the
   historical test seasons; the discrepancy is documented rather than tuned away.
6. **2026/27 is partial** (100 matches). It is used only for the production
   refit and current predictions, never for selection.
7. **No injury/lineup/travel/referee data.** No leakage-free historical source
   with reliable timestamps was available without paid access.
8. **CatBoost unavailable** in this environment; it is omitted rather than
   faked.
9. **Bookmaker prices** are a pre-match snapshot without explicit publication
   timestamps; they are treated as closing consensus.
10. **GitHub automation** is written but unverified on a live repository.
11. **Fixture provider gaps.** A provider can omit a fixture temporarily; the
    sync retains it rather than deleting it, so a stale fixture may linger until
    the provider republishes or a result appears in `matches.csv`.
12. **MW1–10 accuracy cannot be computed.** The automation (snapshots) started
    mid-season, so those 100 completed matches are reported as
    `missing_prediction`, never as correct or incorrect. Accuracy becomes
    meaningful from MW11 onward, where eligible snapshots exist.
13. **Current predictions are no-odds variants** because bookmaker prices for
    MW11 were not yet published at generation time; the eligible snapshot is
    frozen and will not be silently replaced if prices appear later.

## Next actions

1. Publish the repository (pending user approval): `gh auth login` device flow,
   then `gh repo create epl-intelligence --public --source . --push`.
2. Let `test.yml` run end to end; then trigger each data workflow once and
   verify job summaries and committed store diffs.
3. Deploy `app.py` on Streamlit Community Cloud and confirm the live app
   serves the latest committed stores.
4. Accumulate genuinely future matches and score them once with the frozen
   production model as a prospective test.
5. Only revisit new data sources if a timestamped, leakage-free historical
   source becomes available at acceptable cost.
