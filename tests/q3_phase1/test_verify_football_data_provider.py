import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "verify_football_data_provider.py"


def load_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("verify_football_data_provider", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def v4_match(match_id, home, away, utc, matchday=11, status="SCHEDULED", season_start="2026-08-01"):
    return {
        "id": match_id,
        "utcDate": utc,
        "status": status,
        "matchday": matchday,
        "homeTeam": {"name": home},
        "awayTeam": {"name": away},
        "season": {"startDate": season_start},
    }


def full_season_matches():
    """380 unique fixtures: 20-team double round robin, MW11 has target fixtures."""
    teams = [
        "Arsenal FC", "Aston Villa FC", "AFC Bournemouth", "Brentford FC", "Brighton & Hove Albion FC",
        "Chelsea FC", "Coventry City FC", "Crystal Palace FC", "Everton FC", "Fulham FC",
        "Hull City AFC", "Ipswich Town FC", "Leeds United FC", "Liverpool FC", "Manchester City FC",
        "Manchester United FC", "Newcastle United FC", "Nottingham Forest FC", "Sunderland AFC", "Tottenham Hotspur FC",
    ]
    mw11_pairs = [
        ("Manchester City FC", "Fulham FC"),
        ("AFC Bournemouth", "Nottingham Forest FC"),
        ("Aston Villa FC", "Sunderland AFC"),
        ("Chelsea FC", "Leeds United FC"),
        ("Coventry City FC", "Crystal Palace FC"),
        ("Tottenham Hotspur FC", "Ipswich Town FC"),
        ("Newcastle United FC", "Arsenal FC"),
        ("Hull City AFC", "Brighton & Hove Albion FC"),
        ("Liverpool FC", "Manchester United FC"),
        ("Brentford FC", "Everton FC"),
    ]

    # Circle method double round robin
    n = len(teams)
    arr = list(teams)
    rounds = []
    for _ in range(n - 1):
        pairs = []
        for i in range(n // 2):
            pairs.append((arr[i], arr[n - 1 - i]))
        rounds.append(pairs)
        arr = [arr[0]] + [arr[-1]] + arr[1:-1]
    rounds += [[(a, h) for h, a in r] for r in rounds]  # 38 rounds
    assert len(rounds) == 38 and all(len(r) == 10 for r in rounds)

    # Force MW11 (index 10) to be target pairs by swapping with locations where they appear.
    target_set = set(mw11_pairs)
    for md in range(38):
        if md == 10:
            continue
        for i, pair in enumerate(rounds[md]):
            if pair in target_set:
                rounds[md][i] = None
    old_md11 = rounds[10]
    rounds[10] = list(mw11_pairs)
    displaced = [p for p in old_md11 if p not in target_set]
    di = 0
    for md in range(38):
        if md == 10:
            continue
        for i, pair in enumerate(rounds[md]):
            if pair is None:
                rounds[md][i] = displaced[di]
                di += 1
    assert di == len(displaced)

    all_pairs = [p for r in rounds for p in r]
    assert len(all_pairs) == 380
    assert len(set(all_pairs)) == 380

    matches = []
    mid = 1
    for md_idx, pairs in enumerate(rounds, start=1):
        for h, a in pairs:
            if md_idx == 11:
                utc = "2026-11-21T15:00:00Z"
            else:
                day = 1 + ((md_idx * 7 + mid) % 28)
                month = 8 + ((md_idx - 1) // 4)
                month = month if month <= 12 else month - 12
                year = 2026 if month >= 8 else 2027
                utc = f"{year:04d}-{month:02d}-{day:02d}T15:00:00Z"
            matches.append(v4_match(mid, h, a, utc, matchday=md_idx))
            mid += 1
    return matches

def payload(matches):
    return {"matches": matches}


def reference_file(tmp_path, disagree=False, missing=False):
    fixtures = []
    pairs = [
        ("Man City", "Fulham", "2026-11-21T12:30:00Z", "2026-11-21", "12:30", "GMT"),
        ("Bournemouth", "Nott'm Forest", "2026-11-21T15:00:00Z", "2026-11-21", "15:00", "GMT"),
        ("Aston Villa", "Sunderland", "2026-11-21T15:00:00Z", "2026-11-21", "15:00", "GMT"),
        ("Chelsea", "Leeds", "2026-11-21T15:00:00Z", "2026-11-21", "15:00", "GMT"),
        ("Coventry", "Crystal Palace", "2026-11-21T15:00:00Z", "2026-11-21", "15:00", "GMT"),
        ("Tottenham", "Ipswich", "2026-11-21T15:00:00Z", "2026-11-21", "15:00", "GMT"),
        ("Newcastle", "Arsenal", "2026-11-21T17:30:00Z", "2026-11-21", "17:30", "GMT"),
        ("Hull", "Brighton", "2026-11-21T15:00:00Z", "2026-11-21", "15:00", "GMT"),
        ("Liverpool", "Man United", "2026-11-21T15:00:00Z", "2026-11-21", "15:00", "GMT"),
        ("Brentford", "Everton", "2026-11-21T15:00:00Z", "2026-11-21", "15:00", "GMT"),
    ]
    for h, a, utc, d, t, tz in pairs:
        if missing and (h == "Man City"):
            continue
        if disagree and h == "Man City":
            utc, t = "2026-11-21T15:00:00Z", "15:00"
        fixtures.append({
            "season": "2026/27", "matchday": 11, "home_team": h, "away_team": a,
            "official_fixture_id": f"PL-2627-MW11-{h}-{a}",
            "utc_kickoff": utc, "local_date": d, "local_time": t, "local_tz": tz,
            "source_url": "https://www.premierleague.com/fixtures", "verified_at": "2026-11-21",
        })
    p = tmp_path / "ref.json"
    p.write_text(json.dumps({"fixtures": fixtures}), encoding="utf-8")
    return p


def run_main(mod, monkeypatch, tmp_path, *, token="tok", http_result=None, ref_path=None, extra_args=None):
    summary = tmp_path / "summary.md"
    if token is None:
        monkeypatch.delenv("FOOTBALL_DATA_ORG_TOKEN", raising=False)
    else:
        monkeypatch.setenv("FOOTBALL_DATA_ORG_TOKEN", token)
    if http_result is not None:
        monkeypatch.setattr(mod, "http_get_json", lambda *a, **k: http_result)
    args = ["--summary-out", str(summary), "--official-reference", str(ref_path or (tmp_path / "missing.json"))]
    if extra_args:
        args.extend(extra_args)
    rc = mod.main(args)
    return rc, summary.read_text(encoding="utf-8") if summary.exists() else ""


def test_missing_token_prevents_http_request(monkeypatch, tmp_path):
    mod = load_module()
    called = {"n": 0}

    def fake(*a, **k):
        called["n"] += 1
        return mod.HttpResult(True, 200, payload={"matches": []})

    monkeypatch.setattr(mod, "http_get_json", fake)
    rc, summary = run_main(mod, monkeypatch, tmp_path, token=None)
    assert rc == 1
    assert called["n"] == 0
    assert "VERIFICATION BLOCKED" in summary
    assert "Token present: **False**" in summary


def test_successful_auth_valid_response(monkeypatch, tmp_path):
    mod = load_module()
    ref = reference_file(tmp_path)
    http = mod.HttpResult(True, 200, payload=payload(full_season_matches()))
    rc, summary = run_main(mod, monkeypatch, tmp_path, token="secret", http_result=http, ref_path=ref)
    assert rc == 0
    assert "PRIMARY PROVIDER AUTHENTICATES — SCHEDULE DISAGREEMENTS FOUND" in summary
    assert "secret" not in summary


@pytest.mark.parametrize("status,expected", [(401, "PRIMARY PROVIDER AUTHENTICATION FAILED"), (403, "PRIMARY PROVIDER AUTHENTICATION FAILED")])
def test_auth_failures(monkeypatch, tmp_path, status, expected):
    mod = load_module()
    http = mod.HttpResult(False, status, error=f"HTTP {status}")
    rc, summary = run_main(mod, monkeypatch, tmp_path, http_result=http)
    assert rc == 1
    assert expected in summary


def test_http_429(monkeypatch, tmp_path):
    mod = load_module()
    http = mod.HttpResult(False, 429, error="HTTP 429")
    rc, summary = run_main(mod, monkeypatch, tmp_path, http_result=http)
    assert rc == 1
    assert "VERIFICATION BLOCKED" in summary


def test_timeout(monkeypatch, tmp_path):
    mod = load_module()
    http = mod.HttpResult(False, None, error="TimeoutError")
    rc, summary = run_main(mod, monkeypatch, tmp_path, http_result=http)
    assert rc == 1
    assert "VERIFICATION BLOCKED" in summary


def test_invalid_json(monkeypatch, tmp_path):
    mod = load_module()
    http = mod.HttpResult(False, 200, error="Invalid JSON")
    rc, summary = run_main(mod, monkeypatch, tmp_path, http_result=http)
    assert rc == 1
    assert "VERIFICATION BLOCKED" in summary


def test_missing_matches_structure(monkeypatch, tmp_path):
    mod = load_module()
    http = mod.HttpResult(True, 200, payload={"not_matches": []})
    rc, summary = run_main(mod, monkeypatch, tmp_path, http_result=http)
    assert "Response valid (dict with matches list): **False**" in summary


def test_malformed_duplicate_season_matchday(monkeypatch, tmp_path):
    mod = load_module()
    matches = full_season_matches()
    matches.append("bad-record")
    matches.append(v4_match(9999, "Manchester City FC", "Fulham FC", "2026-11-21T15:00:00Z", matchday=11))  # duplicate identity
    http = mod.HttpResult(True, 200, payload=payload(matches))
    rc, summary = run_main(mod, monkeypatch, tmp_path, http_result=http)
    assert "Malformed records: **1**" in summary
    assert "Duplicate fixture identities: **1**" in summary


def test_missing_mw11_fixture(monkeypatch, tmp_path):
    mod = load_module()
    matches = [m for m in full_season_matches() if not (m.get("matchday") == 11 and m.get("homeTeam", {}).get("name") == "Manchester City FC")]
    ref = reference_file(tmp_path)
    http = mod.HttpResult(True, 200, payload=payload(matches))
    rc, summary = run_main(mod, monkeypatch, tmp_path, http_result=http, ref_path=ref)
    assert "PRIMARY PROVIDER AUTHENTICATES — COVERAGE INCOMPLETE" in summary
    assert "| Man City | Fulham | False |" in summary


def test_utc_to_europe_london_gmt_bst(monkeypatch, tmp_path):
    mod = load_module()
    # BST example: 2026-08-15T14:00:00Z -> 15:00 BST
    m_bst = v4_match(1, "Arsenal FC", "Chelsea FC", "2026-08-15T14:00:00Z", matchday=1)
    # GMT example: 2026-11-21T15:00:00Z -> 15:00 GMT
    m_gmt = v4_match(2, "Liverpool FC", "Everton FC", "2026-11-21T15:00:00Z", matchday=11)
    n1 = mod.normalize_match(m_bst)
    n2 = mod.normalize_match(m_gmt)
    assert n1["local_time"] == "15:00"
    assert n1["tz_name"] == "BST"
    assert n2["local_time"] == "15:00"
    assert n2["tz_name"] == "GMT"


def test_official_reference_missing_marks_unverified(monkeypatch, tmp_path):
    mod = load_module()
    http = mod.HttpResult(True, 200, payload=payload(full_season_matches()))
    rc, summary = run_main(mod, monkeypatch, tmp_path, http_result=http, ref_path=tmp_path / "nope.json")
    assert rc == 1
    assert "VERIFICATION BLOCKED" in summary
    assert "UNVERIFIED" in summary
    assert "PRIMARY PROVIDER VERIFIED" not in summary


def test_schedule_disagreement_detection(monkeypatch, tmp_path):
    mod = load_module()
    ref = reference_file(tmp_path, disagree=True)
    http = mod.HttpResult(True, 200, payload=payload(full_season_matches()))
    rc, summary = run_main(mod, monkeypatch, tmp_path, http_result=http, ref_path=ref)
    assert "PRIMARY PROVIDER AUTHENTICATES — SCHEDULE DISAGREEMENTS FOUND" in summary
    assert "DISAGREEMENT" in summary


def test_no_token_leakage(monkeypatch, tmp_path):
    mod = load_module()
    secret = "SUPER_SECRET_TOKEN_123"
    http = mod.HttpResult(False, 401, error=f"HTTP 401 for token {secret}")
    rc, summary = run_main(mod, monkeypatch, tmp_path, token=secret, http_result=http)
    assert secret not in summary


def test_no_production_writes(monkeypatch, tmp_path):
    mod = load_module()
    ref = reference_file(tmp_path)
    http = mod.HttpResult(True, 200, payload=payload(full_season_matches()))
    # run in tmp cwd; should not create data files
    monkeypatch.chdir(tmp_path)
    rc, summary = run_main(mod, monkeypatch, tmp_path, http_result=http, ref_path=ref)
    assert not (tmp_path / "data").exists()
    assert "secret" not in summary


def test_full_season_completeness_not_blind(monkeypatch, tmp_path):
    mod = load_module()
    matches = full_season_matches()
    # break completeness by duplicating one identity and removing another to keep count 380
    matches[0] = v4_match(1, "Manchester City FC", "Fulham FC", "2026-11-21T15:00:00Z", matchday=11)
    matches[1] = v4_match(2, "Manchester City FC", "Fulham FC", "2026-11-22T15:00:00Z", matchday=11)
    http = mod.HttpResult(True, 200, payload=payload(matches))
    analysis = mod.analyze(http.payload)
    assert analysis["season_2026_27_matches"] == 380
    assert analysis["coverage_complete"] is False


def test_coverage_incomplete_when_not_full(monkeypatch, tmp_path):
    mod = load_module()
    http = mod.HttpResult(True, 200, payload=payload(full_season_matches()[:100]))
    rc, summary = run_main(mod, monkeypatch, tmp_path, http_result=http, ref_path=reference_file(tmp_path))
    assert "PRIMARY PROVIDER AUTHENTICATES — COVERAGE INCOMPLETE" in summary


def test_status_mapping_and_missing_kickoff(monkeypatch, tmp_path):
    mod = load_module()
    m = v4_match(1, "Arsenal FC", "Chelsea FC", None, matchday=1, status="POSTPONED")
    n = mod.normalize_match(m)
    assert n["status_mapped"] == "postponed"
    assert n["utc_valid"] is False
    assert n["local_date"] is None
    assert mod._status_from_football_data_org("FINISHED") == "completed"
    assert mod._status_from_football_data_org("AWARDED") == "completed"
    assert mod._status_from_football_data_org("IN_PLAY") == "live"
    assert mod._status_from_football_data_org("PAUSED") == "paused"
    assert mod._status_from_football_data_org("SUSPENDED") == "suspended"
    assert mod._status_from_football_data_org("POSTPONED") == "postponed"
    assert mod._status_from_football_data_org("CANCELLED") == "cancelled"
    assert mod._status_from_football_data_org("CANCELED") == "cancelled"
    assert mod._status_from_football_data_org("SCHEDULED") == "scheduled"
    assert mod._status_from_football_data_org("TIMED") == "scheduled"
    # explicit: never classify IN_PLAY/PAUSED as completed
    assert mod._status_from_football_data_org("IN_PLAY") != "completed"
    assert mod._status_from_football_data_org("PAUSED") != "completed"


def test_malformed_team_objects(monkeypatch, tmp_path):
    mod = load_module()
    raw = {"id": 1, "utcDate": "2026-11-21T15:00:00Z", "status": "SCHEDULED", "matchday": 11,
           "homeTeam": {"name": "Liverpool FC"}, "awayTeam": {"shortName": "MUN"}, "season": {"startDate": "2026-08-01"}}
    n = mod.normalize_match(raw)
    assert n["home"] in {"Liverpool", "Liverpool FC"} or n["home"] is not None
    assert "away_raw" in n


def test_http_500_server_error(monkeypatch, tmp_path):
    mod = load_module()
    http = mod.HttpResult(False, 500, error="HTTP 500")
    rc, summary = run_main(mod, monkeypatch, tmp_path, http_result=http)
    assert rc == 1
    assert "VERIFICATION BLOCKED" in summary


def test_no_filesystem_writes_outside_summary(tmp_path, monkeypatch):
    mod = load_module()
    ref = reference_file(tmp_path)
    http = mod.HttpResult(True, 200, payload=payload(full_season_matches()))
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    rc, summary = run_main(mod, monkeypatch, tmp_path, token="secret", http_result=http, ref_path=ref)
    assert rc == 0
    created = [p for p in cwd.rglob("*") if p.is_file()]
    assert created == []


def test_parity_with_fixture_providers_when_available():
    """Guarded parity: skip when untracked production module is unavailable (remote main)."""
    try:
        from src.data import fixture_providers as fp
    except Exception:
        pytest.skip("src.data.fixture_providers unavailable on remote baseline")
    mod = load_module()
    assert mod._extract_team_name({"name": "Arsenal FC", "shortName": "ARS"}) == fp._extract_team_name({"name": "Arsenal FC", "shortName": "ARS"})
    # Parity for statuses where semantics agree. IN_PLAY/PAUSED/SUSPENDED were
    # intentionally corrected in verification script; production helper may differ.
    for st in ["SCHEDULED", "TIMED", "FINISHED", "POSTPONED", "CANCELLED"]:
        assert mod._status_from_football_data_org(st) == fp._status_from_football_data_org(st)
    # Document discrepancy if production helper still classifies live/paused as completed.
    if fp._status_from_football_data_org("IN_PLAY") == "completed":
        assert mod._status_from_football_data_org("IN_PLAY") == "live"
    if fp._status_from_football_data_org("PAUSED") == "completed":
        assert mod._status_from_football_data_org("PAUSED") == "paused"
    if fp._status_from_football_data_org("SUSPENDED") == "postponed":
        assert mod._status_from_football_data_org("SUSPENDED") == "suspended"
    for val in ["2026-11-21T12:30:00Z", "2026-08-15T14:00:00Z", "2026-11-21", None]:
        a = mod._parse_provider_datetime(val)
        b = fp._parse_provider_datetime(val)
        a_date = str(a[0].date()) if a[0] is not None and str(a[0]) != "NaT" else None
        b_date = str(b[0].date()) if b[0] is not None and str(b[0]) != "NaT" else None
        assert a_date == b_date
        assert a[1] == b[1]


def test_coverage_matchday_with_9_fixtures(monkeypatch, tmp_path):
    mod = load_module()
    matches = full_season_matches()
    # remove one fixture from matchday 5 (not MW11)
    removed = False
    for i, m in enumerate(matches):
        if m.get("matchday") == 5:
            matches.pop(i)
            removed = True
            break
    assert removed
    http = mod.HttpResult(True, 200, payload=payload(matches))
    rc, summary = run_main(mod, monkeypatch, tmp_path, http_result=http, ref_path=reference_file(tmp_path))
    assert "PRIMARY PROVIDER AUTHENTICATES — COVERAGE INCOMPLETE" in summary
    assert "matchday_fixture_counts" in summary.lower() or "Matchday fixture-count violations" in summary


def test_coverage_matchday_with_11_fixtures(monkeypatch, tmp_path):
    mod = load_module()
    matches = full_season_matches()
    # add extra expected-season fixture to matchday 5
    extra = v4_match(99999, "Arsenal FC", "Chelsea FC", "2027-01-05T15:00:00Z", matchday=5)
    matches.append(extra)
    http = mod.HttpResult(True, 200, payload=payload(matches))
    rc, summary = run_main(mod, monkeypatch, tmp_path, http_result=http, ref_path=reference_file(tmp_path))
    assert "PRIMARY PROVIDER AUTHENTICATES — COVERAGE INCOMPLETE" in summary


def test_coverage_missing_matchday(monkeypatch, tmp_path):
    mod = load_module()
    matches = [m for m in full_season_matches() if m.get("matchday") != 5]
    http = mod.HttpResult(True, 200, payload=payload(matches))
    rc, summary = run_main(mod, monkeypatch, tmp_path, http_result=http, ref_path=reference_file(tmp_path))
    assert "PRIMARY PROVIDER AUTHENTICATES — COVERAGE INCOMPLETE" in summary
    assert "Missing expected matchdays" in summary


def test_other_season_records_do_not_affect_expected_season(monkeypatch, tmp_path):
    mod = load_module()
    matches = full_season_matches()
    # add other-season records that would look odd if counted
    for i in range(20):
        matches.append(v4_match(80000 + i, "Arsenal FC", "Chelsea FC", "2025-08-16T15:00:00Z", matchday=1, season_start="2025-08-01"))
    http = mod.HttpResult(True, 200, payload=payload(matches))
    rc, summary = run_main(mod, monkeypatch, tmp_path, http_result=http, ref_path=reference_file(tmp_path))
    assert "2026/27 season matches: **380**" in summary
    assert "Matchday fixture-count violations (expected season): **None**" in summary
    assert "Duplicate fixture identities: **0**" in summary


def test_380_records_with_duplicate_and_absent_identity(monkeypatch, tmp_path):
    mod = load_module()
    matches = full_season_matches()
    # create duplicate identity while keeping count 380 by replacing another fixture
    # duplicate Man City vs Fulham identity in MW11
    dup = v4_match(90001, "Manchester City FC", "Fulham FC", "2026-11-21T15:00:00Z", matchday=11)
    # replace a non-MW11 fixture
    for i, m in enumerate(matches):
        if m.get("matchday") == 12:
            matches[i] = dup
            break
    http = mod.HttpResult(True, 200, payload=payload(matches))
    analysis = mod.analyze(http.payload)
    assert analysis["season_2026_27_matches"] == 380
    assert len(analysis["duplicate_identities"]) >= 1
    assert analysis["unique_identities"] < 380
    assert analysis["coverage_complete"] is False


def reference_file_valid_matching(tmp_path):
    """Official reference that exactly matches full_season_matches() MW11 UTC/local times."""
    fixtures = []
    pairs = [
        ("Man City", "Fulham"), ("Bournemouth", "Nott'm Forest"), ("Aston Villa", "Sunderland"),
        ("Chelsea", "Leeds"), ("Coventry", "Crystal Palace"), ("Tottenham", "Ipswich"),
        ("Newcastle", "Arsenal"), ("Hull", "Brighton"), ("Liverpool", "Man United"), ("Brentford", "Everton"),
    ]
    for h, a in pairs:
        fixtures.append({
            "season": "2026/27", "matchday": 11, "home_team": h, "away_team": a,
            "official_fixture_id": f"PL-2627-MW11-{h}-{a}",
            "utc_kickoff": "2026-11-21T15:00:00Z", "local_date": "2026-11-21", "local_time": "15:00", "local_tz": "GMT",
            "source_url": "https://www.premierleague.com/fixtures", "verified_at": "2026-11-22",
        })
    pth = tmp_path / "ref_valid.json"
    pth.write_text(json.dumps({"fixtures": fixtures}), encoding="utf-8")
    return pth


def test_reference_root_list_is_unverified(monkeypatch, tmp_path):
    mod = load_module()
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps([]), encoding="utf-8")
    http = mod.HttpResult(True, 200, payload=payload(full_season_matches()))
    rc, summary = run_main(mod, monkeypatch, tmp_path, http_result=http, ref_path=ref)
    assert "VERIFICATION BLOCKED" in summary
    assert "reference_available" not in summary.lower() or "Official reference available: **False**" in summary


def test_reference_9_fixtures_unverified(monkeypatch, tmp_path):
    mod = load_module()
    base = json.loads(reference_file_valid_matching(tmp_path).read_text(encoding="utf-8"))
    base["fixtures"] = base["fixtures"][:9]
    ref = tmp_path / "ref9.json"
    ref.write_text(json.dumps(base), encoding="utf-8")
    res = mod.load_official_reference(ref)
    assert res["available"] is False


def test_reference_11_fixtures_unverified(tmp_path):
    mod = load_module()
    base = json.loads(reference_file_valid_matching(tmp_path).read_text(encoding="utf-8"))
    extra = dict(base["fixtures"][0])
    extra["home_team"] = "Arsenal"
    extra["away_team"] = "Chelsea"
    extra["official_fixture_id"] = "EXTRA"
    base["fixtures"].append(extra)
    ref = tmp_path / "ref11.json"
    ref.write_text(json.dumps(base), encoding="utf-8")
    assert mod.load_official_reference(ref)["available"] is False


def test_reference_wrong_season_unverified(tmp_path):
    mod = load_module()
    base = json.loads(reference_file_valid_matching(tmp_path).read_text(encoding="utf-8"))
    base["fixtures"][0]["season"] = "2025/26"
    ref = tmp_path / "ref_bad_season.json"
    ref.write_text(json.dumps(base), encoding="utf-8")
    assert mod.load_official_reference(ref)["available"] is False


def test_reference_wrong_matchday_unverified(tmp_path):
    mod = load_module()
    base = json.loads(reference_file_valid_matching(tmp_path).read_text(encoding="utf-8"))
    base["fixtures"][0]["matchday"] = 10
    ref = tmp_path / "ref_bad_md.json"
    ref.write_text(json.dumps(base), encoding="utf-8")
    assert mod.load_official_reference(ref)["available"] is False


def test_reference_duplicate_identity_unverified(tmp_path):
    mod = load_module()
    base = json.loads(reference_file_valid_matching(tmp_path).read_text(encoding="utf-8"))
    base["fixtures"][1] = dict(base["fixtures"][0])
    base["fixtures"][1]["official_fixture_id"] = "DUP-ID"
    ref = tmp_path / "ref_dup.json"
    ref.write_text(json.dumps(base), encoding="utf-8")
    assert mod.load_official_reference(ref)["available"] is False


def test_reference_duplicate_official_id_unverified(tmp_path):
    mod = load_module()
    base = json.loads(reference_file_valid_matching(tmp_path).read_text(encoding="utf-8"))
    base["fixtures"][1]["official_fixture_id"] = base["fixtures"][0]["official_fixture_id"]
    ref = tmp_path / "ref_dup_official.json"
    ref.write_text(json.dumps(base), encoding="utf-8")
    assert mod.load_official_reference(ref)["available"] is False


def test_reference_invalid_utc_unverified(tmp_path):
    mod = load_module()
    base = json.loads(reference_file_valid_matching(tmp_path).read_text(encoding="utf-8"))
    base["fixtures"][0]["utc_kickoff"] = "not-a-date"
    ref = tmp_path / "ref_bad_utc.json"
    ref.write_text(json.dumps(base), encoding="utf-8")
    assert mod.load_official_reference(ref)["available"] is False


def test_provider_matchday_10_not_counted_as_mw11(monkeypatch, tmp_path):
    mod = load_module()
    matches = full_season_matches()
    # move Man City vs Fulham to matchday 10 (keep same teams/date)
    for m in matches:
        if m.get("matchday") == 11 and m["homeTeam"]["name"] == "Manchester City FC":
            m["matchday"] = 10
            break
    ref = reference_file_valid_matching(tmp_path)
    http = mod.HttpResult(True, 200, payload=payload(matches))
    rc, summary = run_main(mod, monkeypatch, tmp_path, http_result=http, ref_path=ref)
    assert "| Man City | Fulham | False |" in summary
    assert "PRIMARY PROVIDER VERIFIED" not in summary


def test_provider_matchday_12_not_counted_as_mw11(monkeypatch, tmp_path):
    mod = load_module()
    matches = full_season_matches()
    for m in matches:
        if m.get("matchday") == 11 and m["homeTeam"]["name"] == "Manchester City FC":
            m["matchday"] = 12
            break
    ref = reference_file_valid_matching(tmp_path)
    http = mod.HttpResult(True, 200, payload=payload(matches))
    rc, summary = run_main(mod, monkeypatch, tmp_path, http_result=http, ref_path=ref)
    assert "| Man City | Fulham | False |" in summary
    assert "PRIMARY PROVIDER VERIFIED" not in summary


def test_provider_missing_or_malformed_matchday_not_counted(monkeypatch, tmp_path):
    mod = load_module()
    matches = full_season_matches()
    for m in matches:
        if m.get("matchday") == 11 and m["homeTeam"]["name"] == "Manchester City FC":
            m["matchday"] = "TBD"
            break
    ref = reference_file_valid_matching(tmp_path)
    http = mod.HttpResult(True, 200, payload=payload(matches))
    rc, summary = run_main(mod, monkeypatch, tmp_path, http_result=http, ref_path=ref)
    assert "| Man City | Fulham | False |" in summary


def test_malformed_reference_never_produces_verified(monkeypatch, tmp_path):
    mod = load_module()
    # malformed root
    ref = tmp_path / "bad.json"
    ref.write_text(json.dumps([{"fixtures": []}]), encoding="utf-8")
    http = mod.HttpResult(True, 200, payload=payload(full_season_matches()))
    rc, summary = run_main(mod, monkeypatch, tmp_path, http_result=http, ref_path=ref)
    assert "PRIMARY PROVIDER VERIFIED" not in summary
    assert "VERIFICATION BLOCKED" in summary or "UNVERIFIED" in summary


def test_valid_reference_and_matching_mw11_can_verify(monkeypatch, tmp_path):
    mod = load_module()
    ref = reference_file_valid_matching(tmp_path)
    http = mod.HttpResult(True, 200, payload=payload(full_season_matches()))
    rc, summary = run_main(mod, monkeypatch, tmp_path, http_result=http, ref_path=ref)
    assert rc == 0
    assert "PRIMARY PROVIDER VERIFIED — SCHEDULE MATCHES OFFICIAL RECORD" in summary


def test_provider_matchday_int_11_accepted():
    mod = load_module()
    m = {"matchday": 11}
    assert mod._is_mw11_provider_record(m) is True


def test_provider_matchday_string_11_rejected():
    mod = load_module()
    assert mod._is_mw11_provider_record({"matchday": "11"}) is False


def test_provider_matchday_float_11_rejected():
    mod = load_module()
    assert mod._is_mw11_provider_record({"matchday": 11.0}) is False


def test_provider_matchday_bool_true_rejected():
    mod = load_module()
    assert mod._is_mw11_provider_record({"matchday": True}) is False


def test_official_reference_matchday_string_11_rejected(tmp_path):
    mod = load_module()
    base = json.loads(reference_file_valid_matching(tmp_path).read_text(encoding="utf-8"))
    base["fixtures"][0]["matchday"] = "11"
    ref = tmp_path / "ref_md_str.json"
    ref.write_text(json.dumps(base), encoding="utf-8")
    assert mod.load_official_reference(ref)["available"] is False


def test_official_reference_matchday_float_11_rejected(tmp_path):
    mod = load_module()
    base = json.loads(reference_file_valid_matching(tmp_path).read_text(encoding="utf-8"))
    base["fixtures"][0]["matchday"] = 11.0
    ref = tmp_path / "ref_md_float.json"
    ref.write_text(json.dumps(base), encoding="utf-8")
    assert mod.load_official_reference(ref)["available"] is False


def test_official_reference_matchday_bool_true_rejected(tmp_path):
    mod = load_module()
    base = json.loads(reference_file_valid_matching(tmp_path).read_text(encoding="utf-8"))
    base["fixtures"][0]["matchday"] = True
    ref = tmp_path / "ref_md_bool.json"
    ref.write_text(json.dumps(base), encoding="utf-8")
    assert mod.load_official_reference(ref)["available"] is False


def test_official_reference_matchday_int_11_accepted(tmp_path):
    mod = load_module()
    ref = reference_file_valid_matching(tmp_path)
    res = mod.load_official_reference(ref)
    assert res["available"] is True
    assert all(f["matchday"] == 11 and type(f["matchday"]) is int for f in res["fixtures"])
