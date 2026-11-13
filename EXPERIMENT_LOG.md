# Experiment Log

A chronological, append-only record of modelling decisions. Every number below
was produced by `python -m src.models.training` in this environment. The machine
readable registry is `reports/experiments/experiment_log.json`.

## Protocol (fixed before any experiment)

| Split | Seasons | Dates | Rows |
|---|---|---|---|
| Train | 2014/15 – 2022/23 | 2014-08-16 → 2023-05-28 | 3,420 |
| Validation | 2023/24 | 2023-08-11 → 2024-05-19 | 380 |
| Test | 2024/25 – 2025/26 | 2024-08-16 → 2026-05-24 | 760 |
| Current | 2026/27 (partial) | 2026-08 → 2026-11-08 | 100 |

Rules:

1. The test seasons are scored once per pipeline, after all choices are frozen.
2. Hyper-parameters, ensemble weights, blend coefficients and temperature are
   chosen on validation only.
3. The current season is never used for selection.
4. A model is "accepted" only if test accuracy ≥ 50% and test log loss ≤ 1.10.

---

## E1 — Baseline ensemble, uniform temperature (2026-11-12 22:53 UTC)

- Selected: soft-voting ensemble, `alpha = 1.0`, `T = 0.90`, no market blend.
- Test: **50.13%** accuracy, log loss **1.0118**, Brier 0.6053.
- Draws predicted: 13 (recall 0.025).
- Observation: the pure model clears 50% but is less confident than the market
  and is materially worse on log loss.

## E2 — Add validation-tuned market blend (2026-11-12 23:09 UTC)

- Change: grid-search `beta` over the bookmaker consensus on validation.
- Selected: `beta = 0.05`, `T = 0.85`, `alpha = 1.0`.
- Test: **51.84%** accuracy, log loss **0.9989**, Brier 0.5991, ECE 0.0640.
- Draws predicted: 0 (recall 0.0).
- Observation: validation log loss fell from 0.9150 to 0.9058, so the blend was
  retained. The resulting predictor is 95% market consensus. It matches the
  market's accuracy (51.58%) almost exactly. Temperature scaling at 0.85
  sharpens the distribution enough that no draw is ever the argmax.

## E3 — Integrity rerun after the `season_code` fix (2026-11-13 00:58 UTC)

- Change: `season_code` coerced to `str`; `promoted` and all `prev_*` features
  became populated; mean feature missingness 4.85% → 1.66%.
- Test: identical to E2 (51.84%, 0.9989, 0.5991).
- Verification: removing the `prev_*` columns from the frozen test matrix changes
  44/760 predictions and moves probabilities by up to 0.105, proving the model
  uses the corrected features. The E2 artifact had already been trained on the
  corrected representation; the reported metrics are therefore stable rather
  than stale.

## E4 — Baseline comparability fix (2026-11-13, current run)

- Change: baseline metrics (`majority prior`, `always home`, `Elo + logistic`)
  are now computed on the **test** rows as well as validation. The dashboard
  previously compared validation baselines with test metrics.
- Model metrics are unchanged; `test_baselines` was added to the report.
- Test baselines: majority prior 41.71%, always-home 41.71%,
  Elo + logistic 49.47%. The production model (51.84%) beats all three on the
  same 760 rows.

---

## Candidate comparison (validation season, E4 run)

| Model | Family | Val log loss | Val acc. | Iters |
|---|---|---|---|---|
| random_forest | random forest | **0.91853** | 58.2% | 600 |
| lgbm_regularised | LightGBM | 0.91904 | 58.7% | 83 |
| lgbm_deep | LightGBM | 0.91905 | 60.0% | 193 |
| lgbm_shallow | LightGBM | 0.92154 | 58.7% | 51 |
| xgb_regularised | XGBoost | 0.92324 | 58.2% | 109 |
| xgb_deep | XGBoost | 0.92395 | 59.2% | 196 |
| extra_trees | extra trees | 0.92474 | 58.4% | 700 |
| logistic | multinomial logistic | 0.93794 | 58.2% | — |
| hist_gradient | histogram GB | 1.11169 | 55.8% | 500 |

Ensemble selected: `random_forest + lgbm_regularised + lgbm_deep + lgbm_shallow`
with optimised weights `[0.476, 0.193, 0.331, 0.000]`. Validation log loss
0.9168 raw, 0.9150 calibrated; validation accuracy 59.47%.

Note that validation accuracy (~59%) is much higher than test accuracy
(~52%). The 2023/24 season was unusually predictable; the drop is exactly why
the test seasons are never used for selection.

## Blends selected on validation

| Stage | Parameter | Validation log loss |
|---|---|---|
| Ensemble + temperature | `T = 0.90` | 0.91502 |
| Poisson prior | `alpha = 1.00` | 0.91502 (no change) |
| Market consensus | `beta = 0.05`, `T = 0.85` | **0.90576** |

## No-odds variant (same split)

| Item | Value |
|---|---|
| Features | 223 |
| Selected | `xgb_regularised + random_forest + lgbm_shallow + extra_trees` |
| Validation log loss | 0.93798 |
| Test accuracy | 50.39% |
| Test log loss | 1.0232 |
| Test Brier | 0.6126 |
| Test ECE | 0.0478 |

## Test outcome (both variants)

| Metric | Production | Market | No-odds |
|---|---|---|---|
| Accuracy | 51.84% | 51.58% | 50.39% |
| Balanced accuracy | 45.56% | 45.29% | 44.13% |
| Macro F1 | 0.3887 | 0.3862 | 0.3763 |
| Log loss | 0.9989 | 0.9931 | 1.0232 |
| Brier | 0.5991 | 0.5946 | 0.6126 |
| ECE | 0.0640 | 0.0321 | 0.0478 |
| Draw recall | 0.0 | 0.0 | 0.0 |

Season detail:

| Season | Accuracy | Log loss | Macro F1 |
|---|---|---|---|
| 2024/25 | 54.47% | 0.9749 | 0.4086 |
| 2025/26 | 49.21% | 1.0229 | 0.3676 |

## Rejected / not attempted

- **Forcing draws** by lowering the decision threshold: rejected for the
  production rule because it would trade accuracy for draw recall. It remains
  an available analysis (`choose_outcome(..., draw_threshold=...)`) and is the
  top candidate for the next experiment.
- **CatBoost**: not installed in this environment; omitted rather than reported
  as a failed candidate.
- **Stacking with a meta-learner**: too few validation rows (380) to fit a
  second-level model without overfitting; soft voting was preferred.
- **Random splits / k-fold on the full history**: prohibited by design because
  they leak future information across seasons.

## Next experiments (planned, not run)

1. Validation-tuned draw threshold, reported with accuracy, macro F1 and draw
   recall on test; promote only if validation improves.
2. Reliability/resolution decomposition to separate calibration from
   sharpness.
3. Pre-match lineup strength and travel features, with strict timestamp audits.
4. Probability-of-goals auxiliary model (Poisson regression) blended as a
   second prior.
5. Rolling-origin evaluation (walk-forward) as a robustness check on the fixed
   chronological split.

---

# Post-audit programme (2026-11-13)

Items 1, 2, 4 and 5 of the plan above were executed below. Item 3 was
investigated and rejected because no leakage-free historical source with
reliable timestamps was available. The production split, features, model
artifact and decision rule were **not** changed.

## E5 — Independent reproducibility audit

- **Hypothesis:** the reported metrics, artifact, metadata and registry are
  internally consistent, and the market consensus is reconstructible.
- **Changes:** validation/pre-calibration probability exports; `run_audit.py`.
- **Data boundaries:** unchanged (train 3,420 / validation 380 / test 760).
- **Features:** 237 full, 223 odds-free.
- **Result:** `reports/research/audit.json` → `passed: true`.
  - Feature matrix deterministic across two independent builds.
  - Accuracy, log loss, Brier, ECE and macro F1 recomputed exactly from
    `test_predictions.csv`.
  - Artifact class order `A/D/H`, feature counts, `trained_at` and test accuracy
    agree with the report, metadata and registry.
  - Every probability row finite and sums to 1.
  - Consensus reconstructed from seven bookmaker triples plus average and
    maximum series: maximum absolute error 0.0.
  - Price sources 4–8 per match (median 6); `mkt_books` counts sources, not
    bookmakers.
- **Leakage:** unchanged empirical proofs; no new feature was added.
- **Tests:** audit itself passed; full suite below.
- **Decision:** verified. Documentation corrected for `mkt_books` semantics.

## E6 — Draw decision policies

- **Hypothesis:** a validation-selected draw threshold or margin can raise
  macro F1 without materially losing accuracy, and the gain replicates.
- **Changes:** `diagnostics.py`, `run_diagnostics.py`; no production change.
- **Selection:** 2023/24 only. Thresholds 0.18–0.36, margins 0.00–0.16.
- **Result (validation):** threshold 0.28 raised macro F1 0.436 → 0.463 at
  unchanged 59.21% accuracy (11 draws predicted).
- **Replication (walk-forward, 5 selection-grade origins):**
  - mean accuracy change **−0.53 pp** (gate −0.50 pp);
  - macro-F1 gain in **3 of 5** origins (gate 4 of 5);
  - worst origin **−2.63 pp** (2020/21, margin 0.16 forced 131 draws).
- **Test benchmark:** −1.45 pp accuracy, +0.021 macro F1. Not used for the
  decision.
- **Leakage:** probabilities are frozen model outputs; no future data.
- **Tests:** decision-rule and gate tests added.
- **Decision:** **rejected**. Argmax retained; no draws are forced.

## E7 — Calibration comparison

- **Hypothesis:** class-specific vector scaling improves probability quality
  over the existing temperature scaling.
- **Selection:** 2023/24 only; gate = log-loss gain ≥ 0.002 with ECE worsening
  ≤ 0.002.
- **Result:** validation log loss 0.9058 (temperature) vs 0.9049 (vector):
  gain 0.0009, below the gate. Vector scaling was worse than temperature at
  0 of 5 selection-grade walk-forward origins.
- **Test benchmark:** temperature 0.9989 vs vector 0.9987 log loss.
- **Note:** raw (uncalibrated) probabilities score better on the already-
  examined test seasons (0.9935). This is reported as a limitation and was not
  used to re-select calibration.
- **Decision:** **rejected**. Temperature scaling retained.

## E8 — Challenger models

- **Hypothesis A:** a Dixon-Coles goal model adds information beyond the
  feature-based ensemble.
- **Hypothesis B:** a compact odds-free feature set reduces dilution on 3,420
  training matches.
- **Configuration:** Dixon-Coles with exponential decay `xi = 0.0018`,
  low-score correction `rho = 0.0221`, home advantage 0.178, internal date
  cutoff. Core set = 143 of 223 odds-free columns.
- **Result:**

| Candidate | Validation | Test (benchmark only) | Decision |
|---|---|---|---|
| Dixon-Coles standalone | 56.84% acc, 0.9364 LL | 46.84% acc, 1.0573 LL | Rejected |
| DC + production, w=0.08 | 0.90548 LL (−0.0003) | 51.45% acc, 1.0000 LL | Rejected |
| Core 143 features | 0.9331 LL (−0.0045), 57.89% acc | 50.13% acc, 1.0214 LL | Rejected |

- **Leakage:** Dixon-Coles enforces its cutoff internally; a test supplies
  future rows and confirms predictions are unchanged. No new data source.
- **Tests:** 4 Dixon-Coles tests plus gate tests.
- **Decision:** both **rejected**. The production feature set and model remain.

## E9 — Walk-forward evaluation

- **Hypothesis:** the fixed-split result is not an artefact of one validation
  season.
- **Protocol:** expanding window; train through season *n−2*, validate on
  *n−1*, evaluate on *n*. Origins 2019/20–2023/24 are selection-grade;
  2024/25–2025/26 are benchmark-only.
- **Candidates per origin:** random forest, LightGBM regularised, logistic
  regression (small diverse set to control compute).
- **Result (selection-origin mean):**

| Variant | Accuracy | Log loss | Macro F1 |
|---|---|---|---|
| Production | 55.26% ± 3.4 pp | 0.9619 | 0.408 |
| Market | 55.53% ± 3.2 pp | 0.9578 | 0.411 |
| No-odds | 52.63% ± 4.2 pp | 0.9840 | 0.393 |
| Elo + logistic | 53.63% ± 3.3 pp | 0.9781 | 0.395 |

- The selected market weight ranged 0.05–0.60 across origins. The production
  model never separated from the market.
- **Decision:** documented. No re-tuning on this basis.

## E10 — Promotion decision

| Challenger | Gate | Outcome |
|---|---|---|
| Draw threshold/margin | validation + replication | Rejected |
| Vector scaling | ≥0.002 validation log-loss gain | Rejected |
| Core feature set | ≥0.002 gain, ≤0.5 pp accuracy loss | Rejected |
| Dixon-Coles blend | ≥0.002 validation log-loss gain | Rejected |

**COMPLETE — EXISTING MODEL RETAINED.** Evidence:
`reports/research/promotion_decisions.json`.

## Post-audit verification

- `python -m pytest -o addopts="-q"` → **127 passed** in 304 s.
- Dashboard `AppTest` asserts the new Model Performance sections render.
- Pipeline tests simulate refresh failure (cache fallback) and training failure
  (incumbent artifact restored byte-for-byte).
- `python -m compileall src app.py` passes; Ruff is not installed.
