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
