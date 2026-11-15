"""End-to-end matchweek progression simulation (audit Part 10).

A deterministic, fully isolated replay of the production automation loop:

    provider fixtures -> predictions (real engine) -> frozen snapshots ->
    final results -> evaluation -> matchweek aggregates -> dashboard

Every store lives inside a temporary directory (the real ``data/`` artifacts
are never written); the only production input is the real ``matches.csv``
history, read-only, exactly like the pipeline uses it. The provider is
simulated with openfootball/football-data payloads written into a temporary
raw directory, and the clock is injected into every step.

Timeline (relative to the data cutoff / today, so it stays valid):

    T0  fixture sync publishes MW11 (10 fixtures) + MW12 (3 fixtures)
    T1  the engine predicts; 10 eligible pre-match snapshots are frozen
    R1  6 MW11 matches finish  ->  results + evaluation, MW11 stays active
    R2  3 more finish, 1 is rescheduled into January
        -> the lone outstanding MW11 fixture keeps MW11 active (no skip)
    R3  the rescheduled fixture finishes
        -> MW11 terminal, the view advances to MW12 automatically
    R4  one final score is corrected -> the verdict flips, nothing is
        double-counted, the frozen snapshot is untouched
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import pytest

import data as data_module
import render
from src.data import fixture_sync
from src.data import processing
from src.evaluation import evaluate
from src.prediction import snapshots
from src.prediction.engine import PredictionEngine
from src.results import sync as result_sync

SEASON = "2627"
REAL_LOAD_STORE = fixture_sync.load_store

MW11_COUNT = 10
MW12_COUNT = 3


# --------------------------------------------------------------------------
# Module scope: the real production engine (heavy) and the isolated stores
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sim(tmp_path_factory):
    engine = PredictionEngine()
    base = engine.matches.copy()
    assert len(base) > 4000, "expected the real processed history"

    root = tmp_path_factory.mktemp("progression")
    paths = {
        "RAW": root / "raw",
        "STORE": root / "data" / "fixtures" / "upcoming.csv",
        "STATE": root / "data" / "fixtures" / "sync_state.json",
        "SNAPSHOTS": root / "data" / "predictions" / "snapshots.json",
        "RESULTS": root / "data" / "results" / "results.json",
        "EVALUATIONS": root / "data" / "evaluation" / "evaluations.json",
        "MATCHWEEKS": root / "data" / "evaluation" / "matchweeks.json",
        "FRESHNESS": root / "data" / "evaluation" / "freshness.json",
        "PREDICTIONS": root / "data" / "predictions.json",
        "MATCHES_CSV": root / "data" / "processed" / "matches.csv",
    }
    paths["RAW"].mkdir(parents=True)

    mp = pytest.MonkeyPatch()
    # The fixture store is the temporary one; ``upcoming()`` and the
    # prediction engine therefore read exactly what the simulation synced.
    mp.setattr(fixture_sync, "load_store",
               lambda path=None: REAL_LOAD_STORE(paths["STORE"]))
    mp.setattr(processing, "load_fixtures",
               lambda refresh=False, matches=None:
               fixture_sync.upcoming(matches=base))
    # The dashboard reads only temporary stores (evaluations are resolved
    # through ROOT inside data.build).
    mp.setattr(data_module, "ROOT", root)
    for name, key in (("RESULTS", "RESULTS_JSON"), ("PREDICTIONS", "PREDICTIONS_JSON"),
                      ("MATCHWEEKS", "MATCHWEEKS_JSON"), ("FRESHNESS", "FRESHNESS_JSON"),
                      ("MATCHES_CSV", "MATCHES_CSV")):
        mp.setattr(data_module, key, paths[name])
    mp.setattr(data_module, "_STORES",
               (paths["RESULTS"], paths["PREDICTIONS"], paths["MATCHWEEKS"],
                paths["FRESHNESS"], paths["MATCHES_CSV"]))
    data_module._last_fingerprint = None
    data_module._last_built_at = 0.0
    data_module.DATA.clear()
    yield Simulation(engine, base, paths)
    mp.undo()
    data_module._last_fingerprint = None
    data_module._last_built_at = 0.0
    data_module.DATA.clear()


class Simulation:
    """The production loop, driven with temporary stores and an injected clock."""

    def __init__(self, engine: PredictionEngine, base: pd.DataFrame, paths: dict):
        self.engine = engine
        self.base = base
        self.paths = paths

        cutoff = pd.Timestamp(base["date"].max()).normalize()
        today = pd.Timestamp(datetime.now(timezone.utc).date())
        anchor = max(cutoff, today)          # everything happens after both
        self.anchor = anchor
        self.d11a = anchor + pd.Timedelta(days=7)    # MW11 Saturday
        self.d11b = anchor + pd.Timedelta(days=8)    # MW11 Sunday
        self.d12 = anchor + pd.Timedelta(days=14)    # MW12 Saturday
        self.resched = anchor + pd.Timedelta(days=45)  # the January replay

        # Injected clock instants (all UTC, all relative to the anchor so the
        # simulation stays valid whenever it runs).
        self.T0 = f"{(anchor + pd.Timedelta(days=2)).date()}T06:15:00+00:00"
        self.T1 = f"{(anchor + pd.Timedelta(days=3)).date()}T06:15:00+00:00"
        self.R1 = f"{self.d11a.date()}T18:00:00+00:00"   # after the Saturday kickoffs
        self.R2 = f"{self.d11b.date()}T18:00:00+00:00"   # after the Sunday kickoffs
        self.R3 = f"{self.resched.date()}T17:00:00+00:00"  # the replay is finished
        self.R4 = f"{(self.resched + pd.Timedelta(days=1)).date()}T09:00:00+00:00"

        # 13 unused ordered pairs between CURRENT-season teams (a season never
        # repeats one, so the store's completed-guard never fires on these
        # until they are actually played).
        season_rows = base[base["season_code"] == SEASON]
        teams = sorted(set(season_rows["home_team"]) | set(season_rows["away_team"]))
        used = set(zip(season_rows["home_team"], season_rows["away_team"]))
        pairs: list[tuple[str, str]] = []
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                if (teams[i], teams[j]) in used or (teams[j], teams[i]) in used:
                    continue
                pairs.append((teams[i], teams[j]))
                if len(pairs) == MW11_COUNT + MW12_COUNT:
                    break
            if len(pairs) == MW11_COUNT + MW12_COUNT:
                break
        assert len(pairs) == MW11_COUNT + MW12_COUNT
        self.mw11 = pairs[:MW11_COUNT]
        self.mw12 = pairs[MW11_COUNT:]

        # provider-assigned rounds, published dates and scores
        self.schedule = ([(h, a, self.d11a, 11) for h, a in self.mw11[:6]]
                         + [(h, a, self.d11b, 11) for h, a in self.mw11[6:]]
                         + [(h, a, self.d12, 12) for h, a in self.mw12])
        self.scores: dict[tuple[str, str], tuple[int, int]] = {}
        self.played_dates: dict[tuple[str, str], pd.Timestamp] = {}

        self.extended = base
        self.publish_provider()   # the provider publishes the season schedule

    # -- provider simulation ------------------------------------------------
    def publish_provider(self) -> None:
        entries = []
        for home, away, date, week in self.schedule:
            entry = {"team1": home, "team2": away,
                     "date": str(date.date()), "time": "15:00",
                     "round": f"Matchday {week}"}
            if (home, away) in self.scores:
                gh, ga = self.scores[(home, away)]
                entry["score"] = {"ft": [gh, ga]}
            entries.append(entry)
        path = self.paths["RAW"] / "openfootball_2026-27.json"
        path.write_text(json.dumps(
            {"name": "English Premier League 2026/27", "matches": entries}),
            encoding="utf-8")

    def publish_results_csv(self) -> None:
        rows = []
        for home, away, date, _week in self.schedule:
            if (home, away) not in self.scores:
                continue
            gh, ga = self.scores[(home, away)]
            rows.append({
                "Div": "E0", "Date": str(self.played_dates.get(
                    (home, away), date).date()),
                "Time": "15:00", "HomeTeam": home, "AwayTeam": away,
                "FTHG": gh, "FTAG": ga,
                "FTR": "H" if gh > ga else "D" if gh == ga else "A",
            })
        frame = pd.DataFrame(rows, columns=[
            "Div", "Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])
        frame.to_csv(self.paths["RAW"] / "E0_2627.csv", index=False)

    def play(self, home: str, away: str, gh: int, ga: int, date: pd.Timestamp) -> None:
        """A final score enters the authoritative feed and the results history."""
        self.scores[(home, away)] = (gh, ga)
        self.played_dates[(home, away)] = date
        row = {"date": date, "time": "15:00", "season_code": SEASON,
               "season": "2026/27", "home_team": home, "away_team": away,
               "home_goals": gh, "away_goals": ga,
               "result": "H" if gh > ga else "D" if gh == ga else "A"}
        self.extended = pd.concat(
            [self.extended, pd.DataFrame([row]).reindex(columns=self.base.columns)],
            ignore_index=True)
        self.publish_provider()
        self.publish_results_csv()

    def reschedule(self, home: str, away: str, date: pd.Timestamp) -> None:
        """The provider moves a not-yet-played fixture to a new kickoff date."""
        self.schedule = [
            (h, a, date if (h, a) == (home, away) else d, w)
            for h, a, d, w in self.schedule]
        self.publish_provider()

    def unplay(self, home: str, away: str) -> None:
        """Remove a scheduled score again (used for the waiting-state variant)."""
        self.scores.pop((home, away), None)
        self.played_dates.pop((home, away), None)
        self.publish_provider()

    # -- the real production steps -----------------------------------------
    def sync_fixtures(self, now: str) -> dict:
        return fixture_sync.sync_fixtures(
            fetch=False, matches=self.extended, store_path=self.paths["STORE"],
            state_path=self.paths["STATE"], raw_dir=self.paths["RAW"], now=now)

    def upcoming(self) -> pd.DataFrame:
        return fixture_sync.upcoming(matches=self.base)

    def predict(self) -> dict:
        return self.engine.predict_upcoming(save=False)

    def reconcile(self, payload: dict, now: str) -> dict:
        return snapshots.reconcile(
            REAL_LOAD_STORE(self.paths["STORE"]), payload["fixtures"],
            now=now, path=self.paths["SNAPSHOTS"])

    def results_sync(self, now: str) -> dict:
        return result_sync.sync(fetch=False, seasons=[SEASON], now=now,
                                raw_dir=self.paths["RAW"], path=self.paths["RESULTS"])

    def evaluate(self, now: str) -> dict:
        return evaluate.evaluate(now=now, snapshots_path=self.paths["SNAPSHOTS"],
                                 results_path=self.paths["RESULTS"],
                                 evaluations_path=self.paths["EVALUATIONS"])

    def aggregate(self, now: str) -> dict:
        return evaluate.aggregate(
            now=now, results_path=self.paths["RESULTS"],
            evaluations_path=self.paths["EVALUATIONS"],
            store_path=self.paths["MATCHWEEKS"],
            snapshots_path=self.paths["SNAPSHOTS"],
            fixture_store=REAL_LOAD_STORE(self.paths["STORE"]))

    def dashboard(self, payload: dict) -> dict:
        """Publish the prediction payload + history, then refresh the dashboard."""
        self.paths["PREDICTIONS"].write_text(json.dumps(payload), encoding="utf-8")
        columns = ["date", "time", "season_code", "season", "home_team",
                   "away_team", "home_goals", "away_goals", "result"]
        self.paths["MATCHES_CSV"].parent.mkdir(parents=True, exist_ok=True)
        self.extended[columns].to_csv(self.paths["MATCHES_CSV"], index=False)
        data_module.refresh(ttl_seconds=0.0)
        return data_module.DATA

    # -- convenience accessors ---------------------------------------------
    def week(self, aggregate: dict, number: int) -> dict:
        return aggregate["seasons"][SEASON]["matchweeks"][str(number)]

    def snap(self, home: str, away: str) -> dict:
        store = snapshots.load(self.paths["SNAPSHOTS"])["snapshots"]
        return store[f"{SEASON}:{home}:{away}"]


# --------------------------------------------------------------------------
# The 12-step simulation
# --------------------------------------------------------------------------
def test_matchweek_11_to_12_full_automation_simulation(sim: Simulation):
    fixture_ids = [f"{SEASON}:{h}:{a}" for h, a, *_ in sim.schedule]

    # -- 1. MW11 (10) + MW12 (3) are published and synchronised ------------
    report = sim.sync_fixtures(sim.T0)
    assert report["status"] == "ok"
    assert report["counts"]["new"] == MW11_COUNT + MW12_COUNT
    view = sim.upcoming()
    assert len(view) == MW11_COUNT
    assert view["matchweek"].eq(11).all()          # provider round decides

    # -- 2. predictions + frozen pre-match snapshots ------------------------
    payload_t1 = sim.predict()
    assert payload_t1["n_fixtures"] == MW11_COUNT
    recon = sim.reconcile(payload_t1, sim.T1)
    assert recon["created"] == MW11_COUNT
    assert recon["post_kickoff"] == 0
    for fixture_id in fixture_ids[:MW11_COUNT]:
        record = sim.snap(*fixture_id.split(":")[1:])
        assert record["status"] == "pre_match" and record["eligible"]
        # the actual creation instant is honest and before kickoff
        assert pd.Timestamp(record["eligible"]["prediction_created_at"]) \
            < pd.Timestamp(record["kickoff_at"])

    # -- 3-4. six MW11 matches finish: results + evaluation -----------------
    first_six = sim.mw11[:6]
    r1_scores = [(2, 0), (1, 1), (0, 1), (3, 1), (0, 0), (2, 2)]
    for (home, away), (gh, ga) in zip(first_six, r1_scores):
        sim.play(home, away, gh, ga, sim.d11a)
    sim.sync_fixtures(sim.R1)                     # provider marks them played
    res_report = sim.results_sync(sim.R1)
    assert res_report["new"] == 6
    eval_report = sim.evaluate(sim.R1)
    assert eval_report["new"] == 6 and eval_report["missing_predictions"] == 0
    agg = sim.aggregate(sim.R1)
    w11 = sim.week(agg, 11)
    assert w11["scheduled"] == 10 and w11["completed"] == 6 and w11["pending"] == 4
    expected_correct = sum(
        1 for (home, away), (gh, ga) in zip(first_six, r1_scores)
        if sim.snap(home, away)["eligible"]["predicted"]
        == ("H" if gh > ga else "D" if gh == ga else "A"))
    assert w11["correct"] == expected_correct and w11["evaluated"] == 6
    assert w11["accuracy"] == round(100.0 * expected_correct / 6, 2)

    # -- 5. the dashboard still shows MW11 as the active week ---------------
    fresh = sim.dashboard(payload_t1)
    assert fresh["meta"]["matchweek"] == 11
    assert fresh["meta"]["lastCompletedMatchweek"] is None
    assert fresh["meta"]["nextMatchweek"] == 12   # next round already published
    html, _ = render.shell("week")
    assert f"{MW11_COUNT} fixtures" in html       # sidebar follows the stores

    # -- 11 (early). one outstanding fixture is rescheduled into January ----
    last = sim.mw11[-1]
    for (home, away), (gh, ga) in zip(sim.mw11[6:9],
                                      [(1, 0), (1, 2), (2, 1)]):
        sim.play(home, away, gh, ga, sim.d11b)
    sim.reschedule(*last, date=sim.resched)
    sim.sync_fixtures(sim.R2)
    view = sim.upcoming()
    assert len(view) == 1                         # the lone rescheduled fixture
    assert view["matchweek"].eq(11).all()         # MW11 is NOT skipped
    assert pd.Timestamp(view.iloc[0]["date"]) == sim.resched
    recon = sim.reconcile(payload_t1, sim.R2)     # daily bookkeeping, same payload
    assert recon["status_updates"] == 1           # the kickoff instant moved
    record = sim.snap(*last)
    assert record["kickoff_at"] == snapshots.kickoff_instant(sim.resched, "15:00")
    assert record["eligible"] is not None         # pre-match prediction kept
    res_report = sim.results_sync(sim.R2)
    assert res_report["new"] == 3
    assert sim.evaluate(sim.R2)["new"] == 3
    agg = sim.aggregate(sim.R2)
    w11 = sim.week(agg, 11)
    assert w11["completed"] == 9 and w11["pending"] == 1 and w11["scheduled"] == 10
    fresh = sim.dashboard(payload_t1)
    assert fresh["meta"]["matchweek"] == 11       # still open, never skipped

    # -- 6-9. the last fixture finishes: MW11 terminal, automatic MW12 -------
    sim.play(last[0], last[1], 1, 1, sim.resched)
    sim.sync_fixtures(sim.R3)
    view = sim.upcoming()
    assert len(view) == MW12_COUNT
    assert view["matchweek"].eq(12).all()         # progression without edits
    payload_t3 = sim.predict()
    assert payload_t3["n_fixtures"] == MW12_COUNT
    recon = sim.reconcile(payload_t3, sim.R3)
    assert recon["created"] == MW12_COUNT
    assert sim.results_sync(sim.R3)["new"] == 1
    assert sim.evaluate(sim.R3)["new"] == 1
    agg = sim.aggregate(sim.R3)
    w11 = sim.week(agg, 11)
    assert w11["completed"] == 10 and w11["pending"] == 0
    w12 = sim.week(agg, 12)
    assert w12["scheduled"] == MW12_COUNT and w12["pending"] == MW12_COUNT
    fresh = sim.dashboard(payload_t3)
    assert fresh["meta"]["matchweek"] == 12       # advanced automatically
    assert fresh["meta"]["lastCompletedMatchweek"] == 11
    assert fresh["meta"]["nextMatchweek"] is None
    html, _ = render.shell("week")
    assert f"{MW12_COUNT} fixtures" in html       # sidebar shows MW12 now
    # -- 10. historical MW11 remains accessible ------------------------------
    html, _ = render.shell("results", mw=11)
    assert "Matchweek 11 performance" in html
    assert f'{w11["correct"]} / 10</b> correct' in html
    assert f'{round(100.0 * w11["correct"] / 10, 2)}% accuracy' in html
    season = agg["seasons"][SEASON]["season"]
    assert season["evaluated"] == 10 and season["completed"] == 10

    # -- idempotence: rerunning the loop changes nothing --------------------
    assert sim.evaluate(sim.R4)["unchanged"] == 10
    assert sim.aggregate(sim.R4)["changed"] is False

    # -- 12. a corrected final score flips one verdict, no double-count -----
    fid = f"{SEASON}:{first_six[0][0]}:{first_six[0][1]}"
    predicted = sim.snap(*first_six[0])["eligible"]["predicted"]
    correct_before = sim.week(sim.aggregate(sim.R4), 11)["correct"]
    sim.play(first_six[0][0], first_six[0][1], 0, 2, sim.d11a)   # 2-0 -> 0-2
    res_report = sim.results_sync(sim.R4)
    assert res_report["corrections"] == 1
    stored = result_sync.load(sim.paths["RESULTS"])["results"][fid]
    assert stored["corrections"][0]["previous_goals_home"] == 2
    assert stored["corrections"][0]["previous_goals_away"] == 0
    eval_report = sim.evaluate(sim.R4)
    assert eval_report["recomputed"] == 1 and eval_report["evaluated"] == 10
    record = evaluate.load_evaluations(sim.paths["EVALUATIONS"])["evaluations"][fid]
    assert record["history"][0]["reason"] == "result correction"
    assert record["history"][0]["previous_goals_home"] == 2
    expected = correct_before + (1 if predicted == "A"
                                 else -1 if predicted == "H" else 1)
    # a draw prediction flips to correct as well (0-2 is an away win)
    agg = sim.aggregate("R4")
    w11 = sim.week(agg, 11)
    assert w11["correct"] == expected
    assert w11["evaluated"] == 10 and w11["completed"] == 10     # no double-count
    assert w11["accuracy"] == round(100.0 * expected / 10, 2)
    season = agg["seasons"][SEASON]["season"]
    assert season["correct"] == expected and season["evaluated"] == 10
    frozen = sim.snap(*first_six[0])
    assert frozen["eligible"]["predicted"] == predicted          # snapshot untouched
    assert frozen["revisions"] == []
    fresh = sim.dashboard(payload_t3)
    html, _ = render.shell("results", mw=11)
    assert f"{expected} / 10</b> correct" in html                # dashboard updated


# --------------------------------------------------------------------------
# Honest waiting state: nothing published for the next matchweek
# --------------------------------------------------------------------------
def test_no_published_next_matchweek_is_an_honest_waiting_state(tmp_path, monkeypatch):
    store = tmp_path / "upcoming.csv"
    matches = pd.DataFrame({
        "date": ["2026-11-08"], "home_team": ["Arsenal"], "away_team": ["Chelsea"],
    })
    matches["date"] = pd.to_datetime(matches["date"])
    frame = pd.DataFrame([{
        "fixture_id": "2627:Arsenal:Chelsea", "season_code": "2627",
        "matchweek": 11, "date": "2026-11-07", "status": "scheduled",
        "home_team": "Arsenal", "away_team": "Chelsea",
    }])
    frame.to_csv(store, index=False)
    monkeypatch.setattr(fixture_sync, "load_store",
                        lambda path=None: REAL_LOAD_STORE(store))

    view = fixture_sync.upcoming(matches=matches)
    assert view.empty                             # nothing to play: no invented fixtures

    # The dashboard stays honest while the provider publishes nothing new.
    for name in ("RESULTS_JSON", "PREDICTIONS_JSON", "MATCHWEEKS_JSON",
                 "FRESHNESS_JSON", "MATCHES_CSV"):
        monkeypatch.setattr(data_module, name, tmp_path / name)
    monkeypatch.setattr(data_module, "ROOT", tmp_path)
    data_module._last_fingerprint = None
    data_module._last_built_at = 0.0
    data_module.DATA.clear()
    try:
        data_module.refresh(ttl_seconds=0.0)
        fresh = data_module.DATA
        assert fresh["meta"]["matchweek"] is None
        assert fresh["fixtures"] == []
        for page in ("overview", "week", "table", "results"):
            html, _ = render.shell(page)
            assert html
    finally:
        data_module._last_fingerprint = None
        data_module._last_built_at = 0.0
        data_module.DATA.clear()
