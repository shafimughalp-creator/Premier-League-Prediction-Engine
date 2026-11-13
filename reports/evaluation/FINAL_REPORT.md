# Final Report

**Project:** EPL Intelligence — Premier League match-outcome prediction
**Date:** 2026-11-13
**Data cutoff:** 2026-11-08 (Matchweek 10 complete)
**Production model:** `ensemble:random_forest+lgbm_regularised+lgbm_deep+lgbm_shallow`
**Model version:** derived from `trained_at` and feature count; see `data/predictions.json`

---

## 1. Executive summary

A complete, reproducible platform was built: acquisition of 13 seasons, a
leakage-audited 237-feature pipeline, comparison of nine model families, a
calibrated ensemble with validation-tuned Poisson/market blends, a three-mode
prediction engine, a seven-page dashboard, automated tests, GitHub workflows and
operational documentation.

**Verified out-of-sample accuracy is 51.84%** on 760 matches from 2024/25 and
2025/26. This clears the 50% minimum but **does not reach the 53–55% primary
target or the 56–58% stretch target**. The result is reported as measured; no
test-set tuning, random splits or fabricated metrics were used.

The central finding is that the validation-tuned blend assigns 95% weight to
the bookmaker consensus. The model's independent signal does not beat the
closing market on this feature set. The platform is therefore best described
as a calibrated market-plus-model blend with honest evaluation, not as a
profitable forecasting edge.

---

## 2. Data

| Property | Value |
|---|---|
| Primary source | football-data.co.uk `E0.csv` per season |
| Seasons | 2014/15 → 2026/27 (13 codes) |
| Completed matches | 4,660 |
| Clubs | 36 |
| Date range | 2014-08-16 → 2026-11-08 |
| Odds coverage | 100% |
| Match-stat coverage | 100% (shots, shots on target, corners, fouls, cards) |
| Duplicate fixtures | 0 |
| Invalid result codes | 0 |
| Outcome distribution | Home 44.36%, Away 31.82%, Draw 23.82% |

Upcoming fixtures come from the football-data fixture feed when future EPL rows
exist, otherwise from the openfootball 2026/27 schedule. All raw payloads are
cached unchanged; refresh failures degrade to cache and are logged.

Season-level counts are exactly 380 for every completed season and 100 for the
in-progress 2026/27 season.

---

## 3. Feature engineering

237 numeric pre-match features (223 without market information):

| Family | Representative columns |
|---|---|
| Elo | `h_elo`, `a_elo`, `elo_diff`, `elo_expected_home`, `elo_closeness` |
| Rolling form (3/5/10) | `h_form5_ppg`, `diff_form10_sot`, win/draw/loss rates, goals, shots |
| Venue form | `h_venue5_ppg`, `diff_venue5_clean_sheet` |
| Season-to-date | `h_season_ppg`, `diff_season_gd_pg` |
| Previous season / promotion | `h_prev_ppg`, `h_promoted` |
| Rest and congestion | `h_rest_days`, `h_matches_14d`, `h_matches_28d` |
| Attack/defence strength | `h_attack_home`, `a_defence_away` (shrunk toward league prior) |
| Poisson goal model | `pois_exp_home`, `pois_p_home`, `pois_p_draw`, `pois_p_away` |
| Head-to-head | `h2h_ppg`, `h2h_draw_rate`, `h2h_avg_total` |
| Market consensus | `mkt_home`, `mkt_draw`, `mkt_away`, `mkt_overround`, `mkt_entropy`, `odds_*` |

Mean missingness across features is **1.66%**; the largest gaps are previous
season form for promoted clubs and head-to-head before a first meeting. Tree
models handle these natively; the logistic baseline uses median imputation.

### Leakage guarantees

The builder featurises all matches on a date before updating any state with
that date's results. This was proven empirically:

| Test | Case | Result |
|---|---|---|
| Own-result independence | 2014-08-16 Arsenal–Crystal Palace, 237 features | PASS |
| Own-result independence | 2020-10-24 Fulham–Crystal Palace | PASS |
| Own-result independence | 2026-11-08 Man United–Aston Villa | PASS |
| Future independence | Truncate after 2019-12-28 (7 matches that day) | PASS — byte-identical |
| Cold start | First-ever match | PASS — Elo 1500, no form, game 1 |
| Dataset integrity | 4,660 matches | PASS |

An additional unit test flips the final match's result only and confirms its
own feature row is unchanged. Same-day matches are tested for cross-isolation.
The full audit is reproducible with `python -m src.data.validation`.

---

## 4. Validation protocol

No random split was used. The boundaries were fixed before modelling:

| Split | Seasons | Dates | Rows |
|---|---|---|---|
| Train | 2014/15 – 2022/23 | 2014-08-16 → 2023-05-28 | 3,420 |
| Validation | 2023/24 | 2023-08-11 → 2024-05-19 | 380 |
| Test | 2024/25 – 2025/26 | 2024-08-16 → 2026-05-24 | 760 |
| Current | 2026/27 partial | 2026-08 → 2026-11-08 | 100 |

Early stopping, temperature, ensemble weights, Poisson `alpha` and market
`beta` were all selected on validation. The test seasons were scored once per
pipeline. The current season is used only for the final production refit.

---

## 5. Candidate models

Validation season, ranked by raw log loss:

| Model | Family | Val log loss | Val accuracy | Iterations |
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

CatBoost was unavailable in the environment and is omitted rather than reported
as a failed candidate.

### Selected pipeline

- Ensemble: `random_forest` 0.476, `lgbm_regularised` 0.193, `lgbm_deep` 0.331,
  `lgbm_shallow` 0.000 (weights optimised on validation, shrunk toward equal).
- Validation log loss: 0.9168 raw → 0.9150 calibrated (`T = 0.90`).
- Poisson blend: `alpha = 1.00` (no weight on the Poisson prior).
- Market blend: `beta = 0.05`, temperature 0.85 → validation log loss 0.9058.
- Production refit: all 4,660 matches, 15% more iterations than the
  validation-selected early-stopping point.

The no-odds variant selected a different ensemble
(`xgb_regularised + random_forest + lgbm_shallow + extra_trees`), validation
log loss 0.9380, with Elo and Poisson features dominating importance.

---

## 6. Out-of-sample results

### 6.1 Headline metrics (760 test matches)

| Metric | Production | Market | No-odds |
|---|---|---|---|
| Accuracy | **51.84%** | 51.58% | 50.39% |
| Correct / incorrect | 394 / 366 | 392 / 368 | 383 / 377 |
| Balanced accuracy | 45.56% | 45.29% | 44.13% |
| Macro F1 | 0.3887 | 0.3862 | 0.3763 |
| Log loss | 0.9989 | **0.9931** | 1.0232 |
| Brier (sum convention) | 0.5991 | **0.5946** | 0.6126 |
| ECE (10 bins) | 0.0640 | **0.0321** | 0.0478 |
| 95% CI | 48.29–55.38% | 48.03–55.12% | 46.85–53.94% |

### 6.2 Per-class performance (production)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Away win | 0.500 | 0.553 | 0.525 | 246 |
| Draw | 0.000 | 0.000 | 0.000 | 197 |
| Home win | 0.529 | 0.814 | 0.641 | 317 |

Predicted distribution: 272 away, **0 draw**, 488 home.

### 6.3 Confusion matrix (rows actual, columns predicted; order A/D/H)

|  | Pred A | Pred D | Pred H |
|---|---|---|---|
| **Actual A** | 136 | 0 | 110 |
| **Actual D** | 77 | 0 | 120 |
| **Actual H** | 59 | 0 | 258 |

### 6.4 Calibration

| Bin mean confidence | Observed accuracy | Count |
|---|---|---|
| 0.382 | 0.454 | 86 |
| 0.448 | 0.408 | 206 |
| 0.550 | 0.486 | 175 |
| 0.645 | 0.537 | 147 |
| 0.745 | 0.712 | 104 |
| 0.842 | 0.757 | 37 |
| 0.914 | 1.000 | 5 |

ECE 0.064. The model is reasonably calibrated in the middle of the
distribution but overconfident at the low end and underconfident at the high
end. The market consensus is better calibrated (ECE 0.032), which is expected
because the blend is 95% market.

### 6.5 Season stability

| Season | Matches | Accuracy | Log loss | Macro F1 | Draw recall |
|---|---|---|---|---|---|
| 2024/25 | 380 | 54.47% | 0.9749 | 0.4086 | 0.0 |
| 2025/26 | 380 | 49.21% | 1.0229 | 0.3676 | 0.0 |

The gap between seasons (5.3 pp) is within the sampling noise implied by the
confidence interval, but it is a reminder that a single-season estimate is
unstable.

### 6.6 Baselines on the same test rows

| Baseline | Accuracy | Log loss | Brier | Macro F1 |
|---|---|---|---|---|
| Majority prior | 41.71% | 1.0822 | 0.6557 | 0.196 |
| Always home win | 41.71% | 2.6928 | 1.1314 | 0.196 |
| Elo + logistic | 49.47% | 1.0162 | 0.6109 | 0.369 |
| **Production** | **51.84%** | **0.9989** | 0.5991 | **0.389** |

The production model beats every baseline on accuracy and macro F1, and beats
the Elo baseline on log loss and Brier. It does not beat the market on
probability quality.

---

## 7. Interpretation

1. **Accuracy target.** 51.84% clears 50% but misses 53–55%. The confidence
   interval includes 53%, so the result is consistent with a true skill of
   ~52–53%, but the point estimate does not demonstrate the target.
2. **Market dependence.** `beta = 0.05` means 95% of the probability mass is
   the bookmaker consensus. This was selected on validation log loss, and it is
   the correct choice under the stated protocol. It also means the "model" is
   not an independent edge.
3. **Draw blindness.** Temperature 0.85 sharpens the distribution so no draw is
   ever the argmax. The model still assigns draws non-trivial probability mass
   (mean draw probability is reported in the calibration curve), but the
   decision rule never selects them. Forcing draws would raise draw recall and
   macro F1 at the cost of accuracy; that trade-off was not made silently.
4. **Validation optimism.** Validation accuracy was 59.5%, far above test
   accuracy. This is a textbook example of why a single validation season is a
   selection tool, not an expected-performance estimate.

---

## 8. Prediction engine verification

| Capability | Verified behaviour |
|---|---|
| Upcoming | 10 fixtures for 2026-11-21 (Matchweek 11), all odds-free → `no_odds` variant |
| Probability invariant | Each fixture's three probabilities sum to 1 (full precision in JSON) |
| Manual | `Man City vs Liverpool` scored; names normalised; same-team and unknown-team requests rejected |
| Manual past date | 2025-03-15 labelled `retrospective`, `context["in_sample"] = true`, note attached |
| Manual future date | Labelled `manual`, `in_sample = false` |
| Retrospective | 2025/26 matches re-scored with actual result and correctness attached |
| Determinism | Repeated runs produce identical probabilities |
| Atomic output | `data/predictions.json` written via temp file + `os.replace` |

Example Matchweek 11 calls (no-odds model):

| Fixture | Call | P(H) | P(D) | P(A) |
|---|---|---|---|---|
| Aston Villa vs Sunderland | Home | 0.512 | 0.230 | 0.258 |
| Man City vs Fulham | Home | 0.689 | 0.161 | 0.150 |
| Newcastle vs Arsenal | Away | 0.218 | 0.216 | 0.566 |
| Tottenham vs Ipswich | Home | 0.618 | 0.194 | 0.188 |

These are genuine pre-match predictions: the fixtures are after the data
cutoff, and the model never saw them.

---

## 9. Dashboard verification

All seven pages rendered through Streamlit's headless `AppTest` with zero
exceptions:

| Page | Markdown blocks | Dataframes | Errors |
|---|---|---|---|
| Overview | 33 | 1 | 0 |
| Predictions | 20 | 1 | 0 |
| Match Centre | 26 | 1 | 0 |
| Manual Prediction | 8 | 0 | 0 |
| Historical Analysis | 27 | 3 | 0 |
| Model Performance | 23 | 4 | 0 |
| Data Centre | 24 | 5 | 0 |

The manual-prediction workflow is exercised end to end (select teams, click,
read probabilities) and the same-team rejection path is tested. Percentage
columns scale fractions correctly before printf formatting. The app wraps page
renders so a failure shows a readable error rather than a traceback.

---

## 10. Automation and operations

- **Pipeline:** `refresh → validate → guarded retrain → predict`, with stage
  statuses persisted to `data/pipeline_state.json`.
- **Retrain guard:** skipped unless ≥10 new matches; new model rejected and
  previous artifact restored if test accuracy regresses by more than 1 pp.
- **Offline mode:** verified, ~20 s, exits 0.
- **GitHub Actions:** `test.yml` (unit + leakage, then train + full suite) and
  `update.yml` (daily 06:15 UTC refresh, guarded retrain, commit, artifact
  upload) are written. **They have not been executed on GitHub**, because this
  directory is not a pushed repository. This is explicitly listed as prepared,
  not verified.

---

## 11. Requirements assessment

| Requirement | Status | Evidence |
|---|---|---|
| Complete platform | Met | Data, features, models, engine, dashboard, tests, docs |
| Accuracy > 50% | **Met** | 51.84% (CI 48.29–55.38%) |
| Accuracy 53–55% | **Not met** | Point estimate 51.84% |
| Accuracy 56–58% | **Not met** | Point estimate 51.84% |
| 10–15 reliable seasons | Met | 13 seasons, 4,660 matches |
| Raw/processed separation | Met | `data/raw`, `data/processed`, `data/fixtures` |
| No fabricated data | Met | All 4,660 rows traceable to source CSVs |
| Pre-match features only | **Proven** | Perturbation audit, 237 features |
| Strict chronological validation | Met | Train/val/test boundaries; no random splits |
| Baselines and multiple models | Met | 9 candidates + 3 baselines + market + no-odds |
| Full metric suite | Met | Accuracy, macro F1, per-class, draw recall, log loss, Brier, ECE, CI, confusion |
| Three prediction modes | Met | Upcoming, manual, retrospective |
| Retrospective labelling | Met | `mode` and `in_sample` flags |
| Premium 7-page dashboard | Met | All pages render headlessly |
| Continuous updates | **Prepared** | Workflows written; live schedule not yet observed |
| Tests | Met | Unit, leakage, prediction, integration |
| Versioning | Met | Model version string, metadata, experiment registry |
| Documentation | Met | README, architecture, sources, status, changelog, report |

---

## 12. Limitations and recommendations

### Limitations

1. Draw recall is zero under the argmax rule.
2. The production pipeline is 95% market consensus, so it does not demonstrate
   an independent edge.
3. One validation season (380 rows) is a noisy selection set; walk-forward
   validation would be more robust.
4. No lineup, injury, referee or travel data, because a leakage-free
   reconstruction across 13 seasons is not available from the chosen sources.
5. The 2026/27 season is partial and excluded from selection.
6. GitHub automation is written but unverified on a live repository.

### Recommended next steps

1. **Draw-threshold study (validation only).** Select a threshold on 2023/24,
   then report accuracy, macro F1 and draw recall on test. Promote only if
   validation improves without material accuracy loss.
2. **Walk-forward validation.** Re-run selection on rolling origins to quantify
   how much of the validation/test gap is noise.
3. **New pre-match features.** Lineup strength, travel distance, referee
   tendencies, and rest asymmetry — each with a timestamp audit.
4. **Reliability/resolution decomposition.** Separate calibration error from
   sharpness so the market-blend decision can be evaluated on skill, not just
   log loss.
5. **Push and observe CI.** Run `test.yml` and one scheduled `update.yml` on
   GitHub before claiming automation works.

---

## 13. Reproducibility

Environment: Python 3.14.7, pandas 3.1.1, numpy 2.5.4, scikit-learn 1.9.1,
LightGBM 4.7.0, XGBoost 3.4.1, Streamlit 1.67.0, plotly 7.1.0.

```bash
pip install -r requirements.txt
python -m src.pipeline --retrain always
python -m src.data.validation
python -m pytest -o addopts="-q"
streamlit run app.py
```

Key artefacts:

| Artefact | Path |
|---|---|
| Production model | `artifacts/models/model.pkl` |
| Model metadata | `artifacts/metadata/model_meta.json` |
| Training report | `reports/evaluation/training_report.json` |
| Held-out predictions | `reports/evaluation/test_predictions.csv` |
| Leakage audit | `reports/data_quality/leakage_report.json` |
| Experiment registry | `reports/experiments/experiment_log.json` |
| Current predictions | `data/predictions.json` |
| Pipeline state | `data/pipeline_state.json` |

All random seeds are fixed (`RANDOM_STATE = 42`). The training report records
the Python version and data cutoff. Re-running on the same cached raw files
reproduces the reported metrics exactly.
