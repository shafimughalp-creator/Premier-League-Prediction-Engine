# EPL Intelligence — Automation System

**Status:** implemented and locally verified (see "Execution status" at the end)
**Last updated:** 2026-11-15

This document describes the automation system that keeps EPL Intelligence
up to date with no manual intervention: fixture synchronization, immutable
pre-match predictions, official results, evaluation and matchweek accuracy,
and how all of it reaches the approved Streamlit dashboard.

---

## 1. Architecture

```
Published EPL fixtures
        |
        v
[A] sync-fixtures.yml  (daily 05:45 UTC)   src/data/fixture_sync.py
        |
        v
Fixture store  data/fixtures/upcoming.csv + sync_state.json
        |
        v
[B] update.yml  (daily 06:15 UTC)          src/pipeline.py
        |                                   src/prediction/engine.py
        v
Predictions  data/predictions.json  (display payload, regenerated)
        |
        v
Immutable snapshots  data/predictions/snapshots.json  (append-only)

Completed EPL matches
        |
        v
[C] results.yml  (every 6h at :23 UTC)     src/results/sync.py
        |
        v
Official results  data/results/results.json
        |
        v
Evaluation + aggregates + freshness        src/evaluation/evaluate.py
        |
        v
data/evaluation/{evaluations,matchweeks,freshness}.json
        |
        v
Streamlit dashboard (root app.py + data.py live loader)
```

All three data workflows share the GitHub Actions concurrency group
`update`, so they can never push at the same time. All writes are atomic
(temp-file + rename) and every store is idempotent: rerunning a workflow
with unchanged provider data leaves the files byte-identical and commits
nothing.

## 2. Data sources

| Source | Used for | Notes |
|---|---|---|
| football-data.co.uk `mmz4281/{code}/E0.csv` | Historical + current-season results, prices | Authoritative for final scores. Free, no registration. |
| football-data.co.uk `fixtures.csv` | Next-round fixtures with prices | |
| openfootball `football.json` (`en.1.json`) | Full-season schedule + matchweek rounds | Matchweeks come from the provider's `round` labels, never invented. |
| Local `data/processed/matches.csv` | Historical processed results (4,660 matches) | Refreshed by the existing pipeline. |

No credentials are required for any provider. Workflow C downloads fresh
copies on each run (`--fetch`) and keeps the cached copy when a download
fails, so a temporary provider outage never corrupts or empties a store.

## 3. Prediction snapshot policy (immutable pre-match records)

`data/predictions/snapshots.json` maps the stable fixture id
`"{season_code}:{home}:{away}"` to a record with a frozen **eligible**
pre-match prediction and an append-only `revisions[]` history.

* The pipeline creates a snapshot the first time a fixture is predicted.
* The **eligible** version is the latest one whose `prediction_created_at`
  is at or before kickoff (kickoff resolved from the provider's
  date + `Europe/London` time via `zoneinfo`, DST-correct).
* After kickoff the record is **locked**: regenerated predictions never
  silently replace the evaluated values.
* Authorized revisions (e.g. bookmaker prices are published and the full
  variant replaces the no-odds variant) are allowed **only before kickoff**;
  the superseded version is retained with both timestamps.
* Postponements/reschedules move `kickoff_at` and keep the prediction and
  its history. A postponed fixture whose provider publishes a new date
  returns to pre-match status. Cancelled fixtures are frozen.
* Fixtures without a usable kickoff time are snapshotted but
  `locked_at_creation` — never revised, because pre-matchness cannot be
  verified.
* Probabilities are validated (each in [0, 1], sum within 1e-6 of 1)
  before a version is accepted; invalid payloads are flagged, not stored.

The evaluated prediction is **always the frozen eligible snapshot**, never
the regenerated display payload.

## 4. Results synchronisation

`src/results/sync.py` stores only **official final results** (provider-
published scores) in `data/results/results.json`, keyed by the same stable
fixture id. Rules:

* A kickoff time passing is not evidence of completion; pending matches are
  simply absent and reported as pending by the evaluation layer.
* football-data is authoritative; openfootball `score.ft` is a fallback
  used only for fixtures football-data does not cover, with automatic
  source upgrade when the authoritative provider catches up.
* Scores and outcomes are validated; a provider-declared outcome that
  disagrees with the scores is flagged (`needs_review`), never guessed.
* Conflicting scores within one source are flagged, not resolved by guess.
* An authoritative score correction updates the record and appends the
  previous values to `corrections[]` with both timestamps (audit trail).
* Matchweeks come from the openfootball `round` labels; for fixtures whose
  kickoff was rescheduled after the schedule was published, an unambiguous
  team-pair lookup (every league pair meets exactly once per season) is
  used as fallback. Ambiguous pairs are excluded, not guessed.

## 5. Evaluation methodology and accuracy formula

`src/evaluation/evaluate.py` compares each final result against the frozen
eligible snapshot and stores one record per evaluated fixture
(`data/evaluation/evaluations.json`) containing fixture id, season,
matchweek, teams, predicted outcome, the three probabilities, actual
scores/outcome, correct/incorrect, prediction timestamp, result
confirmation timestamp, model id — plus a `history[]` audit trail whenever
a result correction or an authorized pre-kickoff revision forces a
recompute.

Matchweek and season aggregates (`data/evaluation/matchweeks.json`):

```
accuracy = correct / evaluated        (never divided by scheduled)
evaluated = completed fixtures with a valid eligible pre-match prediction
pending   = scheduled fixtures without an official result
missing_prediction = completed fixtures without an eligible snapshot
coverage  = evaluated / completed
```

* Pending matches and completed matches without a snapshot are **never**
  counted as incorrect; both are reported separately.
* With zero evaluated predictions the accuracy is `null` (an honest empty
  state), never 0%.
* A match belongs to the matchweek assigned by the authoritative provider
  schedule (carried on the store/snapshot/result row), not the calendar
  week it happened to be played.

## 6. GitHub Actions schedules (UTC)

| Workflow | File | Schedule | Purpose |
|---|---|---|---|
| A — fixture sync | `sync-fixtures.yml` | daily 05:45 UTC + dispatch | Update the fixture store (commit on change only). |
| B — predictions | `update.yml` | daily 06:15 UTC + dispatch | Refresh data, retrain guard, score fixtures, snapshot reconcile. |
| C — results | `results.yml` | every 6h at :23 (00/06/12/18) + dispatch | Fetch results, evaluate snapshots, rebuild aggregates + freshness. |
| tests | `test.yml` | on push/PR | Unit + integration suites. |

Shared concurrency group `update` serialises A/B/C. All use Python 3.12,
`contents: write` only, commit only when data actually changed, never
overwrite model artifacts, and write job summaries (per-matchweek accuracy
table for workflow C).

GitHub scheduled workflows can be delayed; every store is idempotent, so a
late run is always correct — merely later.

## 7. Streamlit integration (no redesign)

The approved dashboard (root `app.py` + `render.py` + `static/*`) is
unchanged visually. `data.py` is now a **live loader** that builds the exact
DATA object the approved prototype shipped:

* chances are the production model's probabilities, rounded with the
  largest-remainder method so they always sum to 100 (prototype rule);
* table, form (newest first), season numbers, head-to-head and per-fixture
  rates use real results with the prototype's exact formulas;
* a new `?page=results[&mw=<matchweek>]` route reuses the existing
  card/chip/stat/tag classes: season + matchweek accuracy, match-by-match
  verdicts, data freshness; the nav gains one "Results" entry;
* the sidebar "This weekend" card and the footer cutoff go live
  automatically because they already read DATA;
* `data.refresh()` rebuilds DATA **in place** (same object identity) when
  any store's mtime changes or a 5-minute TTL elapses, so reruns never
  serve stale data and never rebuild unnecessarily;
* missing stores degrade to honest empty states — the app never crashes
  and never invents data.

Deployment propagation: Streamlit Community Cloud redeploys the app when
the repository receives a push; workflows commit updated stores, so each
workflow run that changes data makes the change visible in the deployed
app after its redeploy.

## 8. Freshness

`data/evaluation/freshness.json` records, per source: last fixture sync
(`data/fixtures/sync_state.json`), last prediction run
(`data/pipeline_state.json`) + snapshot store update, last results sync,
last evaluation update. The results page shows all four (UTC) and
distinguishes fixture freshness from prediction freshness.

## 9. Failure recovery

* **Provider outage:** downloads keep the cached copy; stores are never
  emptied by absent provider data; the workflow exits non-zero only when no
  provider at all is usable.
* **Partial failure:** stores are written atomically after a full in-memory
  reconciliation; a crash leaves the previous complete store.
* **Bad provider data:** conflicting/invalid rows land in `needs_review`
  lists (reports and summaries), never silently in the stores.
* **Bad commit:** every automated commit only touches data directories
  (`data/fixtures`, `data/predictions`, `data/results`, `data/evaluation`,
  `data/processed`, `reports`); model artifacts are never overwritten.
* **Manual re-run:** every workflow has `workflow_dispatch`; all local
  commands below are safe to re-run.

## 10. Manual execution commands

```bash
python -m src.data.fixture_sync --fetch --json      # A: fixture sync
python -m src.pipeline --offline --predict-only     # B: predictions + snapshots (no network)
python -m src.pipeline --retrain auto               # B: full refresh
python -m src.results.sync --fetch --json           # C: results sync
python -m src.evaluation.evaluate --json            # C: evaluation + aggregates + freshness
python -m src.prediction.snapshots --json           # snapshot store summary
```

## 11. Deployment activation (pending user approval)

1. `gh auth login` (device code, scopes `repo,workflow`) — required because
   the repo ships workflow files.
2. `gh repo create epl-intelligence --public --source . --push`.
3. Confirm Actions permissions: "Allow all actions and reusable workflows"
   (default); workflows declare their own minimal permissions.
4. Deploy on <https://share.streamlit.io> → new app → pick the repo,
   branch `main`, main file `app.py`.
5. Verify the first real executions: run each workflow once via
   `gh workflow run sync-fixtures` / `update` / `results`, then check the
   job summaries and the committed store diffs.
6. Recovery: re-run the failed workflow; if a store is genuinely corrupted,
   restore the file from the previous commit (`git revert` of the data
   commit) and re-run.

Nothing has been pushed, published or deployed yet; no credentials were
changed.

## 12. Execution status

* **Locally executed and verified:** everything in this document except
  the GitHub-hosted runs — all stores, modules, workflows (validated
  YAML), the pipeline with snapshots, and the full test suite (211 tests).
* **Live-provider tested locally:** football-data + openfootball downloads
  and parsing against the real feeds (100/100 current-season results
  synced; fixture store 280 rows; idempotent reruns).
* **Not yet executed on GitHub:** the three scheduled workflows (no remote
  exists yet). Activation steps above are prepared but pending approval.
