# EPL Intelligence — Final Acceptance Report

**Date:** 2026-11-15
**Branch:** `main` (local, commit `e20d66b`) · **Remote:** none (nothing pushed)
**Production model:** `ensemble:random_forest+lgbm_regularised+lgbm_deep+lgbm_shallow`
**Artifact:** `artifacts/models/model.pkl` — unchanged, never overwritten

Execution-status legend used throughout:

| Label | Meaning |
|---|---|
| **[L]** | Executed and verified in this local environment |
| **[P]** | Live-provider tested locally (real network downloads from football-data.co.uk and openfootball) |
| **[G]** | Executed on GitHub Actions — **none**: no remote repository exists yet |

---

## Acceptance criteria — §15 mapping

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | New published fixtures synchronized automatically | **[L][P]** | `src/data/fixture_sync.py` reused (31 tests); live-provider sync downloaded both feeds → 280-fixture store, 10-fixture MW11 view; idempotent rerun byte-identical; `sync-fixtures.yml` daily 05:45 UTC ready |
| 2 | New fixtures receive predictions from the existing production model | **[L][P]** | `src/pipeline.py` `stage_predict` uses the unchanged engine/blend/temperature/argmax; 10 MW11 predictions generated with model `epl-20261113-1478e7d5`; no retraining occurred |
| 3 | Original pre-match predictions permanently preserved | **[L]** | `src/prediction/snapshots.py`: append-only `data/predictions/snapshots.json` keyed by stable `fixture_id`; eligible version frozen at latest pre-kickoff `prediction_created_at` (Europe/London, DST-correct); post-kickoff records locked; revisions retained with both timestamps; unknown-kickoff rows `locked_at_creation`. Live: first run `created=10`, second run `unchanged=10` |
| 4 | Completed results synchronized automatically | **[L][P]** | `src/results/sync.py` (15 tests): football-data authoritative, openfootball fallback with automatic source upgrade; live sync: **100/100** official 2026/27 results; kickoff passing is never treated as completion; `results.yml` every 6 h at :23 UTC ready |
| 5 | Predictions evaluated against actual results | **[L]** | `src/evaluation/evaluate.py` compares frozen eligible snapshots with final results into `data/evaluation/evaluations.json` with full audit trail; live: MW1–10 = 100 `missing_prediction` (pre-automation, honest), MW11–38 pending |
| 6 | Matchweek accuracy calculated correctly | **[L]** | Spec example pinned in tests: 10 scheduled / 9 evaluated / 7 correct / 2 incorrect / 1 pending → **77.78%**; `matchweeks.json` aggregates verified live |
| 7 | Season-to-date accuracy | **[L]** | Season aggregate in the same store, same formula over all evaluated fixtures; displayed on the Results page ("Season so far") |
| 8 | Pending matches never counted incorrect | **[L]** | Explicit tests; `accuracy = correct / evaluated` only; dashboard shows "Pending: N" separately |
| 9 | Completed matches without eligible predictions reported separately | **[L]** | `missing_prediction` counter per matchweek and season; MW1–10 honestly reports 10 each; never folded into accuracy |
| 10 | Repeated runs create no duplicates / double-counting | **[L][P]** | Every store is idempotent and written atomically (temp-file + rename); verified: fixture rerun `new=0 updated=0 unchanged=280`, snapshot rerun `unchanged=10`, results rerun no changes; workflows commit only on change |
| 11 | Dashboard shows latest fixture data | **[L]** | `data.py` live loader reads the automation stores; `refresh()` rebuilds DATA in place on store mtime change or 5-min TTL (cache invalidation test passes); AppTest payload carries the production prediction rows |
| 12 | Dashboard shows actual prediction probabilities | **[L]** | 44/44 frozen-value parity checks vs the approved prototype (chances 49/27/24 rule, largest-remainder rounding sum 100); production floats stored unrounded in `predictions.json` |
| 13 | Matchweek correct/incorrect + accuracy displayed | **[L]** | New `?page=results[&mw=N]` page: chips, stat cards, match-by-match verdicts with Correct/Incorrect tags; covered by AppTest + mocked-store tests |
| 14 | Season performance displayed | **[L]** | "Season so far" card: correct / incorrect / accuracy / evaluated / pending / no-eligible-prediction |
| 15 | Data freshness visible | **[L]** | `data/evaluation/freshness.json` (last fixture sync, prediction run, result sync, evaluation update) rendered as a UTC freshness card; fixture freshness kept distinct from prediction freshness |
| 16 | Existing dashboard design and functionality preserved | **[L]** | No redesign: existing pages untouched except empty-data guards; additive CSS only (`.tag.ok`, `.tag.bad`, 5th nav delay); one new nav entry in identical styling; structural parity tests (8) + AppTest suite (9) |
| 17 | Existing production model and historical evaluation preserved | **[L]** | No ensemble/weight/temperature/blend/argmax change; model artifact untouched; workflows declare `contents: write` only and never overwrite `artifacts/`; verified figures unchanged (accuracy 51.84%, log loss 0.9989, Brier 0.5991, ECE 0.0640, macro F1 0.3887, draw recall 0.0%, market 51.58%) |
| 18 | Relevant tests pass | **[L]** | Full suite: `python -m pytest -o addopts="-q"` → **211 passed, 0 failed** (28 new: 9 AppTest + 11 live-loader + 8 parity; plus 14 snapshot + 15 results + 10 evaluation tests) |
| 19 | GitHub Actions workflows ready for activation | **[L]** | 4 workflows (`sync-fixtures`, `update`, `results`, `test`) YAML-validated; Python 3.12 explicit; minimal permissions; shared `update` concurrency group prevents overlap; commit-on-change; job summaries; activation steps in `docs/automation.md` §11 |
| 20 | Report distinguishes execution levels | **[L]** | This document |

## What has NOT been executed on GitHub [G]

No remote repository exists. Consequently, none of the following has run
outside this machine:

1. `test.yml` on push.
2. The three scheduled workflows (`sync-fixtures`, `update`, `results`) —
   including their `GITHUB_TOKEN` commits and the first real cron executions.
3. Streamlit Community Cloud deployment and its redeploy-on-push behaviour.

Everything above is prepared, locally verified and live-provider tested;
activation requires the user-approved GitHub publish step
(`docs/automation.md` §11). Nothing has been pushed, published, deployed, or
had credentials changed.

## Known honest limitations

- **MW1–10 accuracy is permanently uncomputable** (snapshots began mid-season):
  reported as `missing_prediction`, never as incorrect. Meaningful accuracy
  starts at MW11.
- **Current predictions are no-odds variants** (bookmaker prices for MW11 were
  not published at generation time). Eligible snapshots are frozen; an
  authorized pre-kickoff revision would be retained alongside the original.
- **GitHub cron can be delayed**; all stores are idempotent, so late runs are
  correct — merely later.
- **GitHub Actions provenance.** All test evidence above was produced by local
  execution (Python 3.14 locally, 3.12 declared in workflows).

## Deliverables

| Artefact | Location |
|---|---|
| Automation documentation | `docs/automation.md` |
| Fixture-sync audit (Stage 1, still accurate) | `docs/fixture_automation_audit.md` |
| Status report | `PROJECT_STATUS.md` |
| Modules | `src/data/fixture_sync.py`, `src/prediction/snapshots.py`, `src/results/sync.py`, `src/evaluation/evaluate.py`, `src/pipeline.py`, `data.py`, `render.py`, `app.py` |
| Stores | `data/fixtures/`, `data/predictions/snapshots.json`, `data/results/results.json`, `data/evaluation/{evaluations,matchweeks,freshness}.json` |
| Workflows | `.github/workflows/{sync-fixtures,update,results,test}.yml` |
| Tests | 211-passing suite incl. `tests/integration/test_{dashboard,live_data,data_parity}.py` |
