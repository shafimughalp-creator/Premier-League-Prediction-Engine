#!/usr/bin/env python3
"""Isolated, read-only verification for football-data.org v4 EPL provider.

This script is diagnostic only. It does NOT:
- write fixture caches, fixture stores, sync state, history, handoffs, or predictions
- call sync_fixtures/collect_sources/fetch_football_data_org_v4(fetch=True)
- modify provider precedence or production data

It performs one authenticated GET to the configured v4 PL matches endpoint,
validates/parses response in memory, computes coverage metrics, inspects MW11,
and emits a sanitized Markdown summary.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# Prevent import-time project directory creation side effects.
os.environ.setdefault("EPL_INTELLIGENCE_TEST_ISOLATED", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import config  # noqa: E402
try:
    from src.data import processing as _processing  # noqa: E402
except Exception:  # pragma: no cover - remote baseline should provide processing
    _processing = None

CLASSIFICATION_VERIFIED = "PRIMARY PROVIDER VERIFIED — SCHEDULE MATCHES OFFICIAL RECORD"
CLASSIFICATION_DISAGREEMENTS = "PRIMARY PROVIDER AUTHENTICATES — SCHEDULE DISAGREEMENTS FOUND"
CLASSIFICATION_INCOMPLETE = "PRIMARY PROVIDER AUTHENTICATES — COVERAGE INCOMPLETE"
CLASSIFICATION_AUTH_FAILED = "PRIMARY PROVIDER AUTHENTICATION FAILED"
CLASSIFICATION_BLOCKED = "VERIFICATION BLOCKED"

EXPECTED_SEASON = "2026/27"
EXPECTED_SEASON_CODE = "2627"
EXPECTED_TOTAL_FIXTURES = 380
EXPECTED_MATCHDAYS = 38
MW11_MATCHDAY = 11

# Normalized project team names as expected after fixture_providers._normalize_team_name.
MW11_TARGETS = [
    ("Man City", "Fulham"),
    ("Bournemouth", "Nott'm Forest"),
    ("Aston Villa", "Sunderland"),
    ("Chelsea", "Leeds"),
    ("Coventry", "Crystal Palace"),
    ("Tottenham", "Ipswich"),
    ("Newcastle", "Arsenal"),
    ("Hull", "Brighton"),
    ("Liverpool", "Man United"),
    ("Brentford", "Everton"),
]

SECRET_ENV = "FOOTBALL_DATA_ORG_TOKEN"
USER_AGENT = "EPL-Intelligence-Verification/1.0"


@dataclass
class HttpResult:
    ok: bool
    status: int | None
    payload: Any = None
    error: str | None = None


def _redact(text: str | None, token: str | None) -> str:
    if not text:
        return ""
    out = str(text)
    if token:
        out = out.replace(token, "[REDACTED]")
    # redact common auth header patterns if ever present
    out = re.sub(r"(?i)(x-auth-token\s*[:=]\s*)([^\s,;]+)", r"\1[REDACTED]", out)
    out = re.sub(r"(?i)(authorization\s*[:=]\s*)([^\s,;]+)", r"\1[REDACTED]", out)
    return out


def build_url() -> str:
    base = getattr(config, "FOOTBALL_DATA_ORG_BASE", "https://api.football-data.org/v4")
    return f"{base.rstrip('/')}/competitions/PL/matches"


def http_get_json(url: str, token: str, timeout: int = 30) -> HttpResult:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "X-Auth-Token": token,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = getattr(resp, "status", 200)
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        # Do not expose headers.
        return HttpResult(ok=False, status=int(exc.code), error=f"HTTP {exc.code}")
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        return HttpResult(ok=False, status=None, error=f"{type(exc).__name__}")
    except Exception as exc:
        return HttpResult(ok=False, status=None, error=f"{type(exc).__name__}")

    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        return HttpResult(ok=False, status=status, error="Invalid JSON")
    return HttpResult(ok=True, status=status, payload=payload)


def _safe_str(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, float) and v != v:
        return None
    s = str(v).strip()
    return s or None


def _extract_team_name(team_obj: Any) -> Any:
    """Extract provider team name without stringifying structured objects."""
    if isinstance(team_obj, dict):
        name = team_obj.get("name")
        if name is None:
            name = team_obj.get("shortName") or team_obj.get("tla")
        return name
    return team_obj


def _normalize_team_name(name: Any) -> str | None:
    if name is None:
        return None
    try:
        if _processing is not None and hasattr(_processing, "normalize_team"):
            out = _processing.normalize_team(name)
            s = str(out).strip()
            return s or None
    except Exception:
        pass
    # Minimal fallback: preserve trimmed string.
    s = str(name).strip()
    return s or None


def _status_from_football_data_org(status: object) -> str | None:
    """Map football-data.org v4 match status to diagnostic categories.

    v4 semantics (distinct categories preserved for verification reporting):
    - SCHEDULED, TIMED -> scheduled
    - FINISHED, AWARDED -> completed
    - IN_PLAY -> live (never completed)
    - PAUSED -> paused (live/interrupted, never completed)
    - POSTPONED -> postponed
    - SUSPENDED -> suspended (not automatically postponed)
    - CANCELLED, CANCELED -> cancelled
    Unknown -> None (caller records as other).
    """
    raw = str(status or "").strip().upper()
    mapping = {
        "SCHEDULED": "scheduled",
        "TIMED": "scheduled",
        "POSTPONED": "postponed",
        "SUSPENDED": "suspended",
        "CANCELLED": "cancelled",
        "CANCELED": "cancelled",
        "AWARDED": "completed",
        "FINISHED": "completed",
        "IN_PLAY": "live",
        "PAUSED": "paused",
    }
    return mapping.get(raw)


def _parse_provider_datetime(value: object) -> tuple[Any, str | None, str]:
    """Parse provider datetime/date deterministically, UTC-aware -> Europe/London.

    Mirrors fixture_providers._parse_provider_datetime behavior relevant to verification:
    - date-only ISO => unknown time (no fabricated midnight kickoff)
    - datetime/ISO with time/offset/Z => convert to Europe/London
    - naive timestamps keep parsed date/time without tz conversion assumptions
    """
    tz = "Europe/London"
    if value is None:
        return None, None, tz
    try:
        import pandas as pd
    except Exception:
        return None, None, tz

    if isinstance(value, (pd.Timestamp, datetime)):
        ts = pd.Timestamp(value)
        try:
            if ts.tzinfo is not None:
                ts = ts.tz_convert("Europe/London")
        except Exception:
            pass
        date = pd.Timestamp(ts.date())
        time = ts.strftime("%H:%M")
        return date, time, tz

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat"}:
        return None, None, tz

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        parsed_date = pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")
        if pd.notna(parsed_date):
            return pd.Timestamp(parsed_date).normalize(), None, tz

    has_time_component = ("T" in text) or (":" in text)
    if has_time_component:
        try:
            ts = pd.to_datetime(text, utc=True, errors="raise")
            ts_local = ts.tz_convert("Europe/London")
            return pd.Timestamp(ts_local.date()), ts_local.strftime("%H:%M"), tz
        except Exception:
            pass
    try:
        parsed = pd.to_datetime(text, format="ISO8601", errors="coerce")
    except Exception:
        parsed = pd.NaT
    if pd.isna(parsed):
        try:
            parsed = pd.to_datetime(text, dayfirst=True, errors="coerce")
        except Exception:
            parsed = pd.NaT
    if pd.isna(parsed):
        return None, None, tz
    if not has_time_component:
        return pd.Timestamp(parsed).normalize(), None, tz
    return pd.Timestamp(parsed).normalize(), pd.Timestamp(parsed).strftime("%H:%M"), tz


def normalize_match(raw: dict[str, Any]) -> dict[str, Any]:
    home_raw = _extract_team_name(raw.get("homeTeam"))
    away_raw = _extract_team_name(raw.get("awayTeam"))
    home = _normalize_team_name(home_raw)
    away = _normalize_team_name(away_raw)
    status_raw = raw.get("status")
    status_mapped = _status_from_football_data_org(status_raw)
    utc_date = raw.get("utcDate")
    dt_utc = None
    dt_local = None
    local_date = None
    local_time = None
    tz_name = None
    utc_valid = False
    if utc_date is not None:
        try:
            import pandas as pd

            ts = pd.to_datetime(utc_date, utc=True, errors="coerce")
            if ts is not pd.NaT and not pd.isna(ts):
                utc_valid = True
                dt_utc = ts
                dt_local = ts.tz_convert("Europe/London")
                local_date = dt_local.strftime("%Y-%m-%d")
                local_time = dt_local.strftime("%H:%M")
                tz_name = dt_local.tzname()  # GMT/BST
        except Exception:
            utc_valid = False
    # also use project-equivalent parser for consistency check
    try:
        p_date, p_time, _ = _parse_provider_datetime(utc_date)
    except Exception:
        p_date, p_time = None, None

    matchday = raw.get("matchday")
    season = raw.get("season") if isinstance(raw.get("season"), dict) else {}
    season_start = season.get("startDate")
    season_label = None
    if isinstance(season_start, str) and len(season_start) >= 4 and season_start[:4].isdigit():
        start_year = int(season_start[:4])
        season_label = f"{start_year}/{str(start_year + 1)[-2:]}"
    return {
        "provider_id": raw.get("id"),
        "home_raw": home_raw,
        "away_raw": away_raw,
        "home": home,
        "away": away,
        "status_raw": status_raw,
        "status_mapped": status_mapped,
        "utcDate": utc_date,
        "utc_valid": utc_valid,
        "local_date": local_date,
        "local_time": local_time,
        "tz_name": tz_name,
        "parser_date": str(p_date.date()) if p_date is not None and str(p_date) != "NaT" else None,
        "parser_time": p_time,
        "matchday": matchday,
        "season_start": season_start,
        "season_label": season_label,
        "raw": raw,
    }


def identity(m: dict[str, Any]) -> str | None:
    season_code = EXPECTED_SEASON_CODE if m.get("season_label") == EXPECTED_SEASON else None
    home = m.get("home")
    away = m.get("away")
    if not season_code or not home or not away:
        return None
    return f"{season_code}:{home}:{away}"


def analyze(payload: dict[str, Any]) -> dict[str, Any]:
    matches_raw = payload.get("matches")
    result: dict[str, Any] = {
        "response_valid": isinstance(payload, dict) and isinstance(matches_raw, list),
        "total_matches": len(matches_raw) if isinstance(matches_raw, list) else 0,
        "season_2026_27_matches": 0,
        "unique_identities": 0,
        "duplicate_identities": [],
        "malformed_records": 0,
        "missing_team_names": 0,
        "missing_or_invalid_kickoff": 0,
        "matchday_distribution": {},  # expected season only, strict int matchdays
        "matchday_fixture_counts": {},  # expected season only, matchday -> count
        "distinct_matchdays": 0,
        "missing_expected_matchdays": [],
        "matchday_count_violations": [],
        "invalid_matchday_records": 0,
        "status_counts": {
            "scheduled": 0,
            "live": 0,
            "paused": 0,
            "postponed": 0,
            "suspended": 0,
            "cancelled": 0,
            "completed": 0,
            "other": 0,
        },
        "unknown_status_raw": [],
        "season_labels": {},
        "normalized_matches": [],
        "coverage_complete": False,
        "coverage_status": "COVERAGE UNVERIFIED",
    }
    if not result["response_valid"]:
        return result

    seen: dict[str, int] = {}
    for item in matches_raw:
        if not isinstance(item, dict):
            result["malformed_records"] += 1
            continue
        m = normalize_match(item)
        result["normalized_matches"].append(m)
        lbl = m.get("season_label") or "UNKNOWN"
        result["season_labels"][lbl] = result["season_labels"].get(lbl, 0) + 1
        in_expected_season = m.get("season_label") == EXPECTED_SEASON
        if in_expected_season:
            result["season_2026_27_matches"] += 1
        if not m.get("home") or not m.get("away"):
            result["missing_team_names"] += 1
        if not m.get("utc_valid"):
            result["missing_or_invalid_kickoff"] += 1
        sm = m.get("status_mapped")
        if sm in result["status_counts"]:
            result["status_counts"][sm] += 1
        else:
            result["status_counts"]["other"] += 1
            if m.get("status_raw"):
                result["unknown_status_raw"].append(str(m.get("status_raw")))
        md = m.get("matchday")
        if in_expected_season:
            # Strict: only actual Python int (bool rejected; bool is subclass of int).
            if type(md) is int:
                key = str(md)
                result["matchday_distribution"][key] = result["matchday_distribution"].get(key, 0) + 1
            else:
                result["invalid_matchday_records"] += 1
        ident = identity(m)
        if ident and in_expected_season:
            seen[ident] = seen.get(ident, 0) + 1

    result["duplicate_identities"] = sorted([k for k, v in seen.items() if v > 1])
    result["unique_identities"] = len(seen)
    mds = []
    for k, count in result["matchday_distribution"].items():
        try:
            md_int = int(k)
        except Exception:
            continue
        mds.append(md_int)
        result["matchday_fixture_counts"][str(md_int)] = int(count)
    result["distinct_matchdays"] = len(set(mds))
    expected = set(range(1, EXPECTED_MATCHDAYS + 1))
    result["missing_expected_matchdays"] = sorted(list(expected - set(mds)))
    violations = []
    for md in sorted(expected):
        count = int(result["matchday_distribution"].get(str(md), result["matchday_distribution"].get(md, 0)) or 0)
        if count != 10:
            violations.append({"matchday": md, "count": count, "expected": 10})
    result["matchday_count_violations"] = violations

    # Conservative completeness assessment
    complete = (
        result["season_2026_27_matches"] == EXPECTED_TOTAL_FIXTURES
        and result["unique_identities"] == EXPECTED_TOTAL_FIXTURES
        and len(result["duplicate_identities"]) == 0
        and result["malformed_records"] == 0
        and result["missing_team_names"] == 0
        and result["missing_or_invalid_kickoff"] == 0
        and result["distinct_matchdays"] == EXPECTED_MATCHDAYS
        and len(result["missing_expected_matchdays"]) == 0
        and len(result["matchday_count_violations"]) == 0
        and int(result.get("invalid_matchday_records") or 0) == 0
    )
    # Do not blindly declare complete just because count == 380; all conditions above required.
    result["coverage_complete"] = bool(complete)
    result["coverage_status"] = "COMPLETE" if complete else "COVERAGE UNVERIFIED / INCOMPLETE"
    return result


def _expected_mw11_identities() -> set[str]:
    return {f"{EXPECTED_SEASON_CODE}:{h}:{a}" for h, a in MW11_TARGETS}


# Narrow, explicit allowlist of approved official Premier League source hostnames.
# Basis: premierleague.com is the official Premier League website domain; www is its
# canonical public hostname. No other hostnames/subdomains are approved.
OFFICIAL_REFERENCE_ALLOWED_HOSTS = {"premierleague.com", "www.premierleague.com"}


def _is_approved_official_source_url(url: Any) -> bool:
    raw = str(url or "").strip()
    if not raw or any(ch.isspace() for ch in raw):
        return False
    try:
        parsed = urllib.parse.urlparse(raw)
    except Exception:
        return False
    if parsed.scheme.lower() != "https":
        return False
    if not parsed.netloc or parsed.params or parsed.username or parsed.password:
        return False
    hostname = (parsed.hostname or "").lower().strip(".")
    if hostname not in OFFICIAL_REFERENCE_ALLOWED_HOSTS:
        return False
    # Disallow ports and any netloc tricks; require exact approved hostname.
    if parsed.port is not None:
        return False
    if parsed.netloc.lower().strip(".") != hostname:
        return False
    return True


def _parse_verified_at(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        # Accept ISO-8601 with explicit timezone/offset. Reject naive timestamps.
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None or dt.utcoffset() is None:
        return None
    # basic sanity: impossible/invalid handled by fromisoformat; ensure convertible
    try:
        dt.astimezone(timezone.utc)
    except Exception:
        return None
    return dt


def load_official_reference(path: Path | None) -> dict[str, Any]:
    """Load and strictly validate official MW11 reference.

    Any malformed/invalid/out-of-scope/provenance-invalid reference is unavailable
    (UNVERIFIED), never an exception and never a verified result.
    """
    if path is None or not path.exists():
        return {"available": False, "reason": "missing"}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"available": False, "reason": "invalid_json"}
    if not isinstance(data, dict):
        return {"available": False, "reason": "invalid_root"}
    fixtures = data.get("fixtures")
    if not isinstance(fixtures, list) or not fixtures:
        return {"available": False, "reason": "invalid_structure"}
    if len(fixtures) != 10:
        return {"available": False, "reason": "invalid_fixture_count"}

    required = {
        "season", "matchday", "home_team", "away_team", "official_fixture_id",
        "utc_kickoff", "local_date", "local_time", "local_tz", "source_url", "verified_at",
    }
    expected_ids = _expected_mw11_identities()
    seen_ids: set[str] = set()
    seen_official_ids: set[str] = set()
    for f in fixtures:
        if not isinstance(f, dict) or not required.issubset(f.keys()):
            return {"available": False, "reason": "invalid_fixture_record"}
        if str(f.get("season") or "").strip() != EXPECTED_SEASON:
            return {"available": False, "reason": "invalid_season"}
        md = f.get("matchday")
        # Strict: actual int only (bool is subclass of int, so reject bool explicitly).
        if type(md) is not int or md != MW11_MATCHDAY:
            return {"available": False, "reason": "invalid_matchday"}
        home = str(f.get("home_team") or "").strip()
        away = str(f.get("away_team") or "").strip()
        ident = f"{EXPECTED_SEASON_CODE}:{home}:{away}"
        if ident not in expected_ids:
            return {"available": False, "reason": "unexpected_fixture_identity"}
        if ident in seen_ids:
            return {"available": False, "reason": "duplicate_fixture_identity"}
        seen_ids.add(ident)
        official_id = str(f.get("official_fixture_id") or "").strip()
        if not official_id:
            return {"available": False, "reason": "missing_official_fixture_id"}
        if official_id in seen_official_ids:
            return {"available": False, "reason": "duplicate_official_fixture_id"}
        seen_official_ids.add(official_id)

        # Official source URL provenance
        if not _is_approved_official_source_url(f.get("source_url")):
            return {"available": False, "reason": "invalid_source_url"}

        # Verification timestamp: valid, parseable, explicit tz/offset, not naive
        verified_at = _parse_verified_at(f.get("verified_at"))
        if verified_at is None:
            return {"available": False, "reason": "invalid_verified_at"}

        # UTC kickoff must be valid/parseable timezone-aware timestamp
        try:
            import pandas as pd
            utc_ts = pd.to_datetime(f.get("utc_kickoff"), utc=True, errors="coerce")
            if pd.isna(utc_ts):
                return {"available": False, "reason": "invalid_utc_kickoff"}
        except Exception:
            return {"available": False, "reason": "invalid_utc_kickoff"}

        # local_tz must be valid IANA timezone
        local_tz_name = str(f.get("local_tz") or "").strip()
        try:
            tz = ZoneInfo(local_tz_name)
        except Exception:
            return {"available": False, "reason": "invalid_local_tz"}

        # UTC->local conversion must exactly match local_date/local_time
        try:
            local_ts = utc_ts.tz_convert(tz)
            expected_date = local_ts.strftime("%Y-%m-%d")
            expected_time = local_ts.strftime("%H:%M")
            ref_date = str(f.get("local_date") or "").strip()
            ref_time = str(f.get("local_time") or "").strip()
            if ref_date != expected_date:
                return {"available": False, "reason": "local_date_mismatch"}
            if ref_time != expected_time:
                return {"available": False, "reason": "local_time_mismatch"}
        except Exception:
            return {"available": False, "reason": "local_time_mismatch"}

    if seen_ids != expected_ids:
        return {"available": False, "reason": "missing_expected_fixtures"}
    return {"available": True, "data": data, "fixtures": fixtures}


def _is_mw11_provider_record(m: dict[str, Any]) -> bool:
    """Accept only provider records explicitly in matchday 11; malformed/missing => False."""
    md = m.get("matchday")
    # Strict: actual int only (bool is subclass of int, so reject bool explicitly).
    if type(md) is not int:
        return False
    return md == MW11_MATCHDAY


def compare_mw11(normalized_matches: list[dict[str, Any]], ref: dict[str, Any]) -> dict[str, Any]:
    rows = []
    by_id: dict[str, dict[str, Any]] = {}
    for m in normalized_matches:
        if not _is_mw11_provider_record(m):
            continue
        ident = identity(m)
        if ident and ident in _expected_mw11_identities() and ident not in by_id:
            by_id[ident] = m

    ref_by_id = {}
    if ref.get("available"):
        for f in ref.get("fixtures", []):
            home = str(f.get("home_team") or "").strip()
            away = str(f.get("away_team") or "").strip()
            key = f"{EXPECTED_SEASON_CODE}:{home}:{away}"
            ref_by_id[key] = f

    all_found = True
    any_disagreement = False
    any_unverified = not ref.get("available")
    for home, away in MW11_TARGETS:
        key = f"{EXPECTED_SEASON_CODE}:{home}:{away}"
        m = by_id.get(key)
        r = ref_by_id.get(key)
        row = {
            "home": home,
            "away": away,
            "provider_fixture_id": m.get("provider_id") if m else None,
            "status_raw": m.get("status_raw") if m else None,
            "status_mapped": m.get("status_mapped") if m else None,
            "utc_kickoff": m.get("utcDate") if m else None,
            "local_date": m.get("local_date") if m else None,
            "local_time": m.get("local_time") if m else None,
            "tz_name": m.get("tz_name") if m else None,
            "matchday": m.get("matchday") if m else None,
            "found": m is not None,
            "official_comparison": "UNVERIFIED",
            "official_ref": None,
        }
        if m is None:
            all_found = False
        if r is None:
            row["official_comparison"] = "UNVERIFIED"
            any_unverified = True
        else:
            row["official_ref"] = {
                "utc_kickoff": r.get("utc_kickoff"),
                "local_date": r.get("local_date"),
                "local_time": r.get("local_time"),
                "local_tz": r.get("local_tz"),
                "source_url": r.get("source_url"),
                "verified_at": r.get("verified_at"),
            }
            if m is None:
                row["official_comparison"] = "UNVERIFIED"
                any_unverified = True
            else:
                # Compare by UTC instant and local date/time
                try:
                    import pandas as pd

                    p_utc = pd.to_datetime(m.get("utcDate"), utc=True, errors="coerce")
                    r_utc = pd.to_datetime(r.get("utc_kickoff"), utc=True, errors="coerce")
                    inst_match = (p_utc == r_utc) and not pd.isna(p_utc)
                    local_match = (
                        str(m.get("local_date")) == str(r.get("local_date"))
                        and str(m.get("local_time")) == str(r.get("local_time"))
                    )
                    if inst_match and local_match:
                        row["official_comparison"] = "MATCH"
                    else:
                        row["official_comparison"] = "DISAGREEMENT"
                        any_disagreement = True
                except Exception:
                    row["official_comparison"] = "UNVERIFIED"
                    any_unverified = True
        rows.append(row)

    return {
        "rows": rows,
        "all_found": all_found,
        "any_disagreement": any_disagreement,
        "any_unverified": any_unverified,
        "reference_available": bool(ref.get("available")),
        "reference_reason": ref.get("reason"),
    }


def decide(auth_ok: bool, response_valid: bool, analysis: dict[str, Any], mw11: dict[str, Any], token_present: bool, http_status: int | None) -> str:
    if not token_present:
        return CLASSIFICATION_BLOCKED
    if not auth_ok:
        if http_status in (401, 403):
            return CLASSIFICATION_AUTH_FAILED
        return CLASSIFICATION_BLOCKED
    if not response_valid:
        return CLASSIFICATION_BLOCKED
    # Coverage completeness must be verified; if not complete, do not award verified schedule match.
    if not analysis.get("coverage_complete"):
        return CLASSIFICATION_INCOMPLETE
    if not mw11.get("all_found"):
        return CLASSIFICATION_INCOMPLETE
    # Strongest classification requires valid, correctly scoped official reference and no unresolved comparison.
    if not mw11.get("reference_available") or mw11.get("any_unverified"):
        return CLASSIFICATION_BLOCKED
    # Explicit safeguard: exactly 10 expected MW11 rows all matched with no unresolved comparison.
    rows = mw11.get("rows") or []
    if len(rows) != 10 or any(not r.get("found") for r in rows):
        return CLASSIFICATION_INCOMPLETE
    if any(r.get("official_comparison") not in {"MATCH", "DISAGREEMENT"} for r in rows):
        return CLASSIFICATION_BLOCKED
    if mw11.get("any_disagreement") or any(r.get("official_comparison") == "DISAGREEMENT" for r in rows):
        return CLASSIFICATION_DISAGREEMENTS
    if all(r.get("official_comparison") == "MATCH" for r in rows):
        return CLASSIFICATION_VERIFIED
    return CLASSIFICATION_BLOCKED


def render_summary(report: dict[str, Any]) -> str:
    a = report["analysis"]
    mw11 = report["mw11"]
    lines = []
    lines.append("# Football-data.org v4 Provider Verification (Isolated)")
    lines.append("")
    lines.append(f"**Final classification:** `{report['classification']}`")
    lines.append("")
    lines.append("## A. Authentication result")
    lines.append(f"- Token present: **{report['token_present']}** (value not displayed)")
    lines.append(f"- Authenticated request attempted: **{report['request_attempted']}**")
    lines.append(f"- HTTP status: **{report['http_status']}**")
    lines.append(f"- Auth/result: **{report['auth_result']}**")
    if report.get("http_error"):
        lines.append(f"- Error: **{report['http_error']}**")
    lines.append("")
    lines.append("## B. HTTP/API response result")
    lines.append(f"- Response valid (dict with matches list): **{a.get('response_valid')}**")
    lines.append(f"- Total matches returned: **{a.get('total_matches')}**")
    lines.append(f"- Season labels observed: **{a.get('season_labels')}**")
    lines.append("")
    lines.append("## C. Season and fixture coverage")
    lines.append(f"- 2026/27 season matches: **{a.get('season_2026_27_matches')}** / {EXPECTED_TOTAL_FIXTURES} expected")
    lines.append(f"- Unique fixture identities (2026/27): **{a.get('unique_identities')}**")
    lines.append(f"- Duplicate fixture identities: **{len(a.get('duplicate_identities', []))}**")
    if a.get('duplicate_identities'):
        lines.append(f"- Duplicate IDs: `{', '.join(a.get('duplicate_identities')[:20])}`")
    lines.append(f"- Malformed records: **{a.get('malformed_records')}**")
    lines.append(f"- Missing team names: **{a.get('missing_team_names')}**")
    lines.append(f"- Missing/invalid kickoff timestamps: **{a.get('missing_or_invalid_kickoff')}**")
    lines.append(f"- Distinct matchdays represented: **{a.get('distinct_matchdays')}** / {EXPECTED_MATCHDAYS}")
    lines.append(f"- Missing expected matchdays: **{a.get('missing_expected_matchdays') or 'None'}**")
    lines.append(f"- Matchday fixture-count violations (expected season): **{a.get('matchday_count_violations') or 'None'}**")
    lines.append(f"- Invalid matchday records (expected season): **{a.get('invalid_matchday_records', 0)}**")
    lines.append(f"- Status counts (mapped): **{a.get('status_counts')}**")
    lines.append(f"- Coverage status: **{a.get('coverage_status')}**")
    lines.append(f"- Coverage complete (conservative): **{a.get('coverage_complete')}**")
    lines.append("")
    lines.append("## D. Matchweek 11 fixture-by-fixture comparison")
    lines.append(f"- Official reference available: **{mw11.get('reference_available')}** ({mw11.get('reference_reason')})")
    lines.append("")
    lines.append("| Home | Away | Found | Provider ID | Status | UTC kickoff | Europe/London date | Europe/London time | TZ | MD | Official comparison |")
    lines.append("|---|---|---:|---|---|---|---|---|---|---:|---|")
    for r in mw11.get("rows", []):
        lines.append(
            f"| {r['home']} | {r['away']} | {r['found']} | {r.get('provider_fixture_id')} | {r.get('status_raw')} | {r.get('utc_kickoff')} | {r.get('local_date')} | {r.get('local_time')} | {r.get('tz_name')} | {r.get('matchday')} | {r.get('official_comparison')} |"
        )
    lines.append("")
    lines.append("## E. Timezone conversion checks")
    lines.append("- Conversion uses `pandas.to_datetime(utcDate, utc=True).tz_convert('Europe/London')`, consistent with existing provider parser behavior.")
    lines.append("- `TZ` column reports Europe/London zone abbreviation (GMT/BST) at kickoff instant.")
    lines.append("- Provider status categories are reported distinctly: scheduled/live/paused/postponed/suspended/cancelled/completed/other; raw provider status is shown in MW11 rows.")
    lines.append("")
    lines.append("## F. OpenFootball comparison")
    lines.append("- Not included in this initial isolated verification run (read-only optional comparison not enabled).")
    lines.append("")
    lines.append("## G. Provider suitability decision")
    lines.append(f"- **{report['classification']}**")
    lines.append("")
    lines.append("## H. Limitations and unresolved risks")
    for lim in report.get("limitations", []):
        lines.append(f"- {lim}")
    lines.append("")
    lines.append("_Sanitized report: no token, no authorization headers, no raw API response body included._")
    return "\n".join(lines) + "\n"


def build_report(*, token_present: bool, request_attempted: bool, http: HttpResult | None, ref: dict[str, Any], token_for_redaction: str | None) -> dict[str, Any]:
    analysis = analyze(http.payload) if http and isinstance(http.payload, dict) else {
        "response_valid": False, "total_matches": 0, "season_2026_27_matches": 0, "unique_identities": 0,
        "duplicate_identities": [], "malformed_records": 0, "missing_team_names": 0, "missing_or_invalid_kickoff": 0,
        "matchday_distribution": {}, "distinct_matchdays": 0, "missing_expected_matchdays": list(range(1, 39)),
        "matchday_fixture_counts": {}, "matchday_count_violations": [{"matchday": md, "count": 0, "expected": 10} for md in range(1, 39)],
        "invalid_matchday_records": 0,
        "status_counts": {"scheduled": 0, "live": 0, "paused": 0, "postponed": 0, "suspended": 0, "cancelled": 0, "completed": 0, "other": 0},
        "unknown_status_raw": [], "season_labels": {}, "normalized_matches": [], "coverage_complete": False,
        "coverage_status": "COVERAGE UNVERIFIED / INCOMPLETE",
    }
    mw11 = compare_mw11(analysis.get("normalized_matches", []), ref)
    auth_ok = bool(http and http.ok)
    classification = decide(auth_ok, bool(analysis.get("response_valid")), analysis, mw11, token_present, http.status if http else None)
    auth_result = "not_attempted"
    if request_attempted:
        if auth_ok:
            auth_result = "authenticated"
        elif http and http.status in (401, 403):
            auth_result = "authentication_failed"
        else:
            auth_result = "request_failed"
    limitations = [
        "Live API behavior, rate limits, pagination, and full-season completeness cannot be fully proven from one response alone unless explicitly supported by provider metadata.",
        "Official schedule comparison is only as strong as the independently verified reference artifact; if missing/unverified, schedule match cannot be claimed.",
        "This verification is isolated and non-publishing; it does not modify production fixture stores, caches, sync state, predictions, or workflows.",
    ]
    if not ref.get("available"):
        limitations.append(f"Official MW11 reference unavailable/unverified ({ref.get('reason')}); schedule comparison marked UNVERIFIED.")
    report = {
        "classification": classification,
        "token_present": token_present,
        "request_attempted": request_attempted,
        "http_status": http.status if http else None,
        "http_error": _redact(http.error if http else None, token_for_redaction),
        "auth_result": auth_result,
        "analysis": {k: v for k, v in analysis.items() if k != "normalized_matches"},
        "mw11": mw11,
        "limitations": limitations,
    }
    # include normalized match count only for debug-safe internal use; not raw
    report["analysis"]["normalized_match_count"] = len(analysis.get("normalized_matches", []))
    return report


def write_summary(markdown: str, summary_out: str | None) -> None:
    if summary_out:
        p = Path(summary_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(markdown)
            if not markdown.endswith("\n"):
                f.write("\n")
    else:
        print(markdown)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Isolated football-data.org v4 provider verification")
    parser.add_argument("--summary-out", default=os.environ.get("GITHUB_STEP_SUMMARY"))
    parser.add_argument("--official-reference", default=str(REPO_ROOT / "tests/fixtures/epl_2026_27_mw11_official_reference.json"))
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true", help="Do not make HTTP request; only validate local/reference logic")
    args = parser.parse_args(argv)

    token = os.environ.get(SECRET_ENV)
    token_present = bool(token)

    ref = load_official_reference(Path(args.official_reference) if args.official_reference else None)

    http = None
    request_attempted = False
    if not token_present:
        report = build_report(token_present=False, request_attempted=False, http=None, ref=ref, token_for_redaction=None)
    elif args.dry_run:
        report = build_report(token_present=True, request_attempted=False, http=None, ref=ref, token_for_redaction=token)
        report["classification"] = CLASSIFICATION_BLOCKED
        report["limitations"].append("Dry-run mode: no HTTP request performed.")
    else:
        request_attempted = True
        url = build_url()
        http = http_get_json(url, token=token, timeout=args.timeout)
        report = build_report(token_present=True, request_attempted=True, http=http, ref=ref, token_for_redaction=token)

    md = render_summary(report)
    write_summary(md, args.summary_out)

    # Exit codes: 0 only for non-blocking informative classifications? Better non-zero on blocking/auth/invalid.
    cls = report["classification"]
    if cls in {CLASSIFICATION_BLOCKED, CLASSIFICATION_AUTH_FAILED}:
        return 1
    # disagreements/incomplete are valid diagnostic outcomes; return 0 so workflow shows report but can still mark status via summary.
    # However, they indicate non-verified; returning 0 is acceptable if summary is clear. Keep 0.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
