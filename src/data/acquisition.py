"""Download and cache raw EPL data.

Primary source: football-data.co.uk season CSV files (results, shots, corners,
cards, bookmaker odds). Upcoming fixtures come from the football-data fixture
feed when it contains future EPL rows, otherwise from the openfootball
published schedule. Raw bytes are cached unchanged and every retrieval is
recorded with a UTC timestamp.
"""
from __future__ import annotations

import io
import json
import logging
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src import config

log = logging.getLogger(__name__)

USER_AGENT = "EPL-Intelligence/1.0 (+https://github.com/)"
RETRIES = 3
TIMEOUT = 45


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def http_get(url: str, timeout: int = TIMEOUT) -> bytes:
    """GET with retries and a descriptive user agent."""
    last: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = exc
            log.warning("GET %s failed (%s/%s): %s", url, attempt, RETRIES, exc)
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"Could not download {url}: {last}")


def season_path(code: str) -> Path:
    return config.RAW_DIR / f"E0_{code}.csv"


def download_season(code: str, force: bool = False) -> Path:
    """Cache one football-data season CSV and return its local path."""
    path = season_path(code)
    if path.exists() and not force:
        return path
    url = f"{config.FOOTBALL_DATA_BASE}/{code}/E0.csv"
    raw = http_get(url)
    if b"Div," not in raw[:500]:
        raise RuntimeError(f"Unexpected payload for season {code}: {raw[:120]!r}")
    path.write_bytes(raw)
    log.info("Downloaded %s -> %s (%d bytes)", url, path.name, len(raw))
    return path


def read_season(code: str, force: bool = False) -> pd.DataFrame:
    path = download_season(code, force=force)
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def download_fixtures(force: bool = False) -> Path:
    path = config.RAW_DIR / "fixtures.csv"
    if path.exists() and not force:
        return path
    raw = http_get(config.FIXTURES_URL)
    path.write_bytes(raw)
    log.info("Downloaded fixtures feed -> %s", path.name)
    return path


def download_openfootball(season: str = "2026-27", force: bool = False) -> Path:
    path = config.RAW_DIR / f"openfootball_{season}.json"
    if path.exists() and not force:
        return path
    url = f"{config.OPENFOOTBALL_BASE}/{season}/en.1.json"
    raw = http_get(url)
    json.loads(raw.decode("utf-8"))  # validate before caching
    path.write_bytes(raw)
    log.info("Downloaded openfootball schedule -> %s", path.name)
    return path


def refresh(force: bool = False) -> dict:
    """Refresh the current + previous season and both fixture sources."""
    started = _now()
    seasons = ["2526", "2627"] if not force else list(config.SEASONS)
    status: dict[str, str] = {}
    for code in seasons:
        try:
            download_season(code, force=force)
            status[code] = "downloaded"
        except Exception as exc:  # keep the cache usable
            log.warning("Season %s refresh failed: %s", code, exc)
            status[code] = f"failed: {exc}" if not season_path(code).exists() else "cached"

    fixtures_status = "cached"
    try:
        download_fixtures(force=force)
        fixtures_status = "downloaded"
    except Exception as exc:
        log.warning("Fixtures refresh failed: %s", exc)
        fixtures_status = f"failed: {exc}" if (config.RAW_DIR / "fixtures.csv").exists() else "missing"

    openfootball_status = "cached"
    try:
        download_openfootball(force=force)
        openfootball_status = "downloaded"
    except Exception as exc:
        log.warning("openfootball refresh failed: %s", exc)
        openfootball_status = f"failed: {exc}"

    record = {
        "started_at": started,
        "finished_at": _now(),
        "seasons": status,
        "fixtures": fixtures_status,
        "openfootball": openfootball_status,
    }
    history: list[dict] = []
    if config.UPDATE_LOG.exists():
        try:
            history = json.loads(config.UPDATE_LOG.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            history = []
    history.append(record)
    config.UPDATE_LOG.write_text(json.dumps(history[-60:], indent=2), encoding="utf-8")
    return record


def latest_update() -> dict | None:
    if not config.UPDATE_LOG.exists():
        return None
    try:
        history = json.loads(config.UPDATE_LOG.read_text(encoding="utf-8"))
        return history[-1] if history else None
    except json.JSONDecodeError:
        return None


def ensure_cache() -> None:
    """Download every season once if the cache is empty."""
    missing = [c for c in config.SEASONS if not season_path(c).exists()]
    for code in missing:
        download_season(code)
    if not (config.RAW_DIR / "fixtures.csv").exists():
        try:
            download_fixtures()
        except Exception as exc:
            log.warning("Fixtures unavailable: %s", exc)
    try:
        download_openfootball()
    except Exception as exc:
        log.warning("openfootball unavailable: %s", exc)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(json.dumps(refresh(), indent=2))
