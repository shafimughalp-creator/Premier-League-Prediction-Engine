# Fixture automation — architecture audit (Stage 1)

**Date:** 2026-11-13
**Scope:** how upcoming fixtures are stored, resolved and consumed today.

> **Note (2026-11-15):** this document describes the Stage-1 fixture
> synchronisation design, which was reused unchanged in the full automation
> build. For the complete system (prediction snapshots, results sync,
> evaluation, workflows, dashboard integration), see
> [`automation.md`](automation.md).

## Current data flow

```
football-data.co.uk/fixtures.csv ─┐
openfootball football.json ───────┴─> src/data/processing.load_fixtures()
                                          │  (rebuilds data/fixtures/upcoming.csv
                                          │   from scratch on every call)
                                          v
                                  src/prediction/engine.predict_upcoming()
                                          │  FeatureBuilder.build_fixtures()
                                          v
                                  data/predictions.json ─> Streamlit pages
```

- **Store:** `data/fixtures/upcoming.csv` (config `FIXTURES_CSV`). Columns:
  `date, time, home_team, away_team, source, round, mkt_*, matchweek, match_id`.
- **Producers:** `processing.load_fixtures()` — football-data fixture rows after the
  last completed match, otherwise the next openfootball matchday. The file is
  rewritten on every prediction call.
- **Consumers:** `PredictionEngine.predict_upcoming()` (feature builder +
  calibrated model), the dashboard Overview / Predictions / Match Centre pages,
  and `tests/integration/test_pipeline.py`.
- **Completed vs upcoming:** completed matches live in
  `data/processed/matches.csv` (results with scores). A fixture is "upcoming" if
  its date is strictly after `matches.date.max()`. There is no explicit status
  column today.
- **Team identity:** canonical football-data names via
  `processing.normalize_team()` and `config.TEAM_ALIASES`; short/display names in
  `config.SHORT_NAMES` / `DISPLAY_NAMES`.
- **Dates and times:** dates are parsed day-first and normalised to midnight;
  kickoff times are kept as the provider's `HH:MM` string (UK local time). No
  timezone conversion is performed anywhere, and there is no timezone column.
- **Identifiers:** `match_id = "{date}_{home}_{away}"` — derived, not stable
  across a rescheduled kickoff. No provider supplies a fixture ID.
- **Existing fetch mechanism:** `acquisition.download_fixtures()` and
  `download_openfootball()` cache raw bytes; `acquisition.refresh()` was meant to
  refresh them, but with `force=False` it returns the cached file without a
  network call, so a daily run on a warm cache never updates.
- **Risk to historical data / artifacts:** `load_fixtures` only reads raw feeds
  and writes the fixture CSV; it never touches `matches.csv`, the model artifact
  or evaluation reports. Prediction is skipped when the fixture frame is empty.

## Gaps this stage fixes

1. No persistent, idempotent fixture store: every call rebuilds from the feed,
   and an empty provider response overwrites a valid fixture file with nothing.
2. No stable fixture key, so a rescheduled match looks like a new fixture.
3. No status handling (scheduled / rescheduled / postponed / cancelled /
   completed) and no protection for fixtures that disappear temporarily.
4. The refresh path does not actually re-download on a warm cache.
5. No dedicated scheduled automation for fixtures.

## Design chosen

- New `src/data/fixture_sync.py` owns provider parsing, validation, merge and
  persistence. It reuses the two existing sources (no new provider, no scraping,
  no credentials) and the existing alias/normalisation code.
- The same file remains `data/fixtures/upcoming.csv` (existing consumers keep
  working) with added `fixture_id`, `status`, `season_code`, `season`,
  `timezone`, `first_seen_at`, `updated_at`. A sidecar
  `data/fixtures/sync_state.json` records source status, counts and timestamps.
- `fixture_id = "{season_code}:{home}:{away}"` — unique for a league fixture and
  stable across reschedules. `match_id` is still derived from the current date
  so the prediction engine contract is unchanged.
- `processing.load_fixtures()` delegates to the sync module (single code path).
- The pipeline gains a visible `fixtures` step; predictions keep using the
  existing engine and calibrated probabilities.
- A dedicated daily workflow (`.github/workflows/sync-fixtures.yml`) fetches and
  commits only fixture data; the existing `update.yml` continues to produce
  predictions and retrain when justified.

## Implementation outcome (2026-11-13)

### Files

| File | Change |
|---|---|
| `src/data/fixture_sync.py` | New: provider parsers, validation, stable IDs, merge/upsert, statuses, atomic store/state writes, upcoming view, CLI |
| `src/data/processing.py` | `load_fixtures()` now reads the sync store (single code path) |
| `src/data/acquisition.py` | `refresh()` re-downloads the selected sources instead of returning the cache |
| `src/pipeline.py` | New visible `fixtures` stage after validation |
| `src/config.py` | Added `FIXTURE_SYNC_STATE` |
| `.github/workflows/sync-fixtures.yml` | New: daily fixture-only sync + guarded commit |
| `tests/unit/test_fixture_sync.py` | New: 31 deterministic tests |
| `data/fixtures/upcoming.csv` | Regenerated with the new schema |
| `data/fixtures/sync_state.json` | New run metadata/history |

### Verification evidence

- **Cached sync**: 280 fixtures, 10-fixture Matchday 11 view, `status ok`,
  `rejected 0`.
- **Idempotency**: second run `new=0 updated=0 unchanged=280`,
  `store_changed false`, store SHA-256 identical before/after, exit `0`.
- **Live providers**: `--fetch` downloaded both feeds; `new=280`, `status ok`.
  A fresh-checkout simulation (empty raw cache, committed `matches.csv`) also
  downloaded both providers live and produced 280 fixtures / 10 upcoming; the
  immediate cached re-run was `new=0 updated=0`.
- **Pipeline**: `python -m src.pipeline --offline --retrain never --predict-only`
  exited `0` in 21.5 s, scored 10 fixtures with the unchanged model
  `epl-20261113-1478e7d5`, test accuracy 0.5184, no retraining.
- **Cross-platform determinism**: the store and state are written with explicit
  LF endings, so a Windows local run and a Linux CI run produce identical bytes.
- **Tests**: `python -m pytest -o addopts="-q"` → **158 passed** (127 existing
  + 31 new), including the slow leakage audit. After the final serialization
  change, the fixture and pipeline suites were re-run: 42 passed.
- **Not verified**: no real GitHub Actions execution (repository has no remote);
  workflow syntax is asserted by text tests because PyYAML is not installed.

### Bug found and fixed during verification

`_parse_date` used `dayfirst=True` for both providers. openfootball publishes
ISO dates, so `2026-10-11` was read as 10 November and `2026-10-12` as
10 December. Four already-completed Matchday 6 fixtures reappeared as upcoming
and the "next matchweek" became a 3-fixture view. ISO is now parsed first and
day-first is only the football-data fallback. A second guard rejects any
provider row whose `fixture_id` already exists in `matches.csv`.

### Activation on GitHub

1. Push the repository (including `.github/workflows/sync-fixtures.yml`).
2. In **Settings → Actions → General**, allow the workflow to read and write
   repository contents. If the default branch is protected, allow the
   `github-actions[bot]` app (or the `GITHUB_TOKEN`) to push, or the commit step
   will fail with a permission error.
3. Run **Actions → sync-fixtures → Run workflow** once to confirm a green run
   and a fixture commit (or a clean "no changes" run).
4. The schedule then runs daily at 05:45 UTC. GitHub cron is UTC, only runs on
   the default branch, and may be delayed during peak periods.

No secrets are required: both providers are public, unauthenticated HTTP feeds.
