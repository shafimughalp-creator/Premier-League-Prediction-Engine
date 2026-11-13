"""Central configuration for EPL Intelligence.

All paths are derived from the repository root so the project is portable.
Class order is **A, D, H** throughout the ML code (0=away win, 1=draw,
2=home win). Presentation layers map this to H/D/A explicitly.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FIXTURES_DIR = DATA_DIR / "fixtures"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METADATA_DIR = ARTIFACTS_DIR / "metadata"
REPORTS_DIR = PROJECT_ROOT / "reports"
EVAL_DIR = REPORTS_DIR / "evaluation"
EXPERIMENT_DIR = REPORTS_DIR / "experiments"
QUALITY_DIR = REPORTS_DIR / "data_quality"

for _p in (RAW_DIR, PROCESSED_DIR, FIXTURES_DIR, MODELS_DIR, METADATA_DIR,
           EVAL_DIR, EXPERIMENT_DIR, QUALITY_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Data sources
# --------------------------------------------------------------------------
FOOTBALL_DATA_BASE = "https://www.football-data.co.uk/mmz4281"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
OPENFOOTBALL_BASE = "https://raw.githubusercontent.com/openfootball/football.json/master"

# football-data.co.uk uses a compact season code; 2020/21 is the odd one out.
SEASONS: dict[str, str] = {
    "1415": "2014/15",
    "1516": "2015/16",
    "1617": "2016/17",
    "1718": "2017/18",
    "1819": "2018/19",
    "1920": "2019/20",
    "2021": "2020/21",
    "2122": "2021/22",
    "2223": "2022/23",
    "2324": "2023/24",
    "2425": "2024/25",
    "2526": "2025/26",
    "2627": "2026/27",
}
CURRENT_SEASON = "2627"

MATCHES_CSV = PROCESSED_DIR / "matches.csv"
FIXTURES_CSV = FIXTURES_DIR / "upcoming.csv"
STANDINGS_CSV = PROCESSED_DIR / "standings.csv"
UPDATE_LOG = PROCESSED_DIR / "update_log.json"
DATA_QUALITY_JSON = QUALITY_DIR / "data_quality.json"

# --------------------------------------------------------------------------
# Chronological split (never change without re-running every evaluation)
# --------------------------------------------------------------------------
TRAIN_END = "2023-07-01"   # train: seasons 2014/15 - 2022/23
VAL_END = "2024-07-01"     # validation: 2023/24
TEST_END = "2026-07-01"    # test: 2024/25 + 2025/26
# 2026/27 is used only for the final production refit and current predictions.

# --------------------------------------------------------------------------
# Modelling
# --------------------------------------------------------------------------
RANDOM_STATE = 42
N_JOBS = 4
CLASS_ORDER = ("A", "D", "H")
CLASS_NAMES = ("Away win", "Draw", "Home win")
LABEL_TO_INDEX = {"A": 0, "D": 1, "H": 2}
INDEX_TO_CODE = ("A", "D", "H")
CODE_TO_LABEL = {"H": "Home win", "D": "Draw", "A": "Away win"}

ELO_START = 1500.0
ELO_K = 20.0
ELO_HOME_ADVANTAGE = 60.0
ELO_SEASON_REGRESSION = 0.25

ROLL_WINDOWS = (3, 5, 10)
H2H_WINDOW = 5

# Acceptance thresholds for the final report.
MIN_TEST_ACCURACY = 0.50
MAX_TEST_LOGLOSS = 1.10

# --------------------------------------------------------------------------
# Team normalisation
# --------------------------------------------------------------------------
# Canonical names follow football-data.co.uk. Aliases cover openfootball and
# common variants. Keys are lower-cased with punctuation removed.
TEAM_ALIASES: dict[str, str] = {
    "afc bournemouth": "Bournemouth",
    "bournemouth": "Bournemouth",
    "arsenal": "Arsenal",
    "arsenal fc": "Arsenal",
    "aston villa": "Aston Villa",
    "aston villa fc": "Aston Villa",
    "brentford": "Brentford",
    "brentford fc": "Brentford",
    "brighton": "Brighton",
    "brighton and hove albion": "Brighton",
    "brighton hove albion": "Brighton",
    "brighton & hove albion fc": "Brighton",
    "burnley": "Burnley",
    "burnley fc": "Burnley",
    "chelsea": "Chelsea",
    "chelsea fc": "Chelsea",
    "coventry": "Coventry",
    "coventry city": "Coventry",
    "coventry city fc": "Coventry",
    "crystal palace": "Crystal Palace",
    "crystal palace fc": "Crystal Palace",
    "everton": "Everton",
    "everton fc": "Everton",
    "fulham": "Fulham",
    "fulham fc": "Fulham",
    "hull": "Hull",
    "hull city": "Hull",
    "hull city afc": "Hull",
    "ipswich": "Ipswich",
    "ipswich town": "Ipswich",
    "ipswich town fc": "Ipswich",
    "leeds": "Leeds",
    "leeds united": "Leeds",
    "leeds united fc": "Leeds",
    "leicester": "Leicester",
    "leicester city": "Leicester",
    "liverpool": "Liverpool",
    "liverpool fc": "Liverpool",
    "man city": "Man City",
    "manchester city": "Man City",
    "manchester city fc": "Man City",
    "man united": "Man United",
    "manchester united": "Man United",
    "manchester united fc": "Man United",
    "newcastle": "Newcastle",
    "newcastle united": "Newcastle",
    "newcastle united fc": "Newcastle",
    "nottm forest": "Nott'm Forest",
    "nottingham forest": "Nott'm Forest",
    "nottingham forest fc": "Nott'm Forest",
    "sunderland": "Sunderland",
    "sunderland afc": "Sunderland",
    "tottenham": "Tottenham",
    "tottenham hotspur": "Tottenham",
    "tottenham hotspur fc": "Tottenham",
    "west ham": "West Ham",
    "west ham united": "West Ham",
    "west ham united fc": "West Ham",
    "wolves": "Wolves",
    "wolverhampton": "Wolves",
    "wolverhampton wanderers": "Wolves",
    "wolverhampton wanderers fc": "Wolves",
    "west brom": "West Brom",
    "west bromwich albion": "West Brom",
    "sheffield united": "Sheffield United",
    "sheffield utd": "Sheffield United",
    "norwich": "Norwich",
    "watford": "Watford",
    "swansea": "Swansea",
    "stoke": "Stoke",
    "qpr": "QPR",
    "queens park rangers": "QPR",
    "reading": "Reading",
    "middlesbrough": "Middlesbrough",
    "derby": "Derby",
    "bolton": "Bolton",
    "wigan": "Wigan",
    "blackpool": "Blackpool",
    "birmingham": "Birmingham",
    "blackburn": "Blackburn",
    "cardiff": "Cardiff",
    "charlton": "Charlton",
    "portsmouth": "Portsmouth",
    "fulham": "Fulham",
    "huddersfield": "Huddersfield",
    "luton": "Luton",
    "barnsley": "Barnsley",
    "southampton": "Southampton",
    "nottingham forest": "Nott'm Forest",
}

DISPLAY_NAMES: dict[str, str] = {
    "Nott'm Forest": "Nottingham Forest",
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Brighton": "Brighton & Hove Albion",
    "Wolves": "Wolverhampton Wanderers",
    "Spurs": "Tottenham Hotspur",
}

SHORT_NAMES: dict[str, str] = {
    "Nott'm Forest": "NFO",
    "Man City": "MCI",
    "Man United": "MUN",
    "Crystal Palace": "CRY",
    "Aston Villa": "AVL",
    "Sheffield United": "SHU",
    "West Ham": "WHU",
    "Newcastle": "NEW",
    "Tottenham": "TOT",
    "Brighton": "BHA",
    "Bournemouth": "BOU",
    "Brentford": "BRE",
    "Arsenal": "ARS",
    "Chelsea": "CHE",
    "Everton": "EVE",
    "Fulham": "FUL",
    "Liverpool": "LIV",
    "Burnley": "BUR",
    "Leeds": "LEE",
    "Leicester": "LEI",
    "Sunderland": "SUN",
    "Wolves": "WOL",
    "Hull": "HUL",
    "Ipswich": "IPS",
    "Coventry": "COV",
    "Southampton": "SOU",
    "Watford": "WAT",
    "Norwich": "NOR",
    "Swansea": "SWA",
    "Stoke": "STK",
    "West Brom": "WBA",
    "QPR": "QPR",
    "Cardiff": "CAR",
    "Huddersfield": "HUD",
    "Luton": "LUT",
}


def display_name(team: str) -> str:
    return DISPLAY_NAMES.get(team, team)


def short_name(team: str) -> str:
    if team in SHORT_NAMES:
        return SHORT_NAMES[team]
    cleaned = "".join(ch for ch in team if ch.isalnum())
    return (cleaned[:3] or team[:3]).upper()
