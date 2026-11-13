"""Configuration invariants: class order, aliases, naming."""
from __future__ import annotations

import pytest

from src import config
from src.data.processing import normalize_team


def test_class_order_is_away_draw_home():
    assert config.CLASS_ORDER == ("A", "D", "H")
    assert config.LABEL_TO_INDEX == {"A": 0, "D": 1, "H": 2}
    assert config.INDEX_TO_CODE == ("A", "D", "H")
    assert config.CLASS_NAMES == ("Away win", "Draw", "Home win")


def test_every_class_has_presentation_label():
    for code in ("H", "D", "A"):
        assert code in config.CODE_TO_LABEL


def test_split_dates_are_ordered():
    assert config.TRAIN_END < config.VAL_END < config.TEST_END


def test_seasons_are_thirteen_contiguous_years():
    labels = list(config.SEASONS.values())
    assert len(labels) == 13
    assert labels[0] == "2014/15"
    assert labels[-1] == "2026/27"
    for earlier, later in zip(labels, labels[1:]):
        start = int(earlier[:4])
        assert int(later[:4]) == start + 1


@pytest.mark.parametrize("raw,expected", [
    ("Manchester City", "Man City"),
    ("manchester city fc", "Man City"),
    ("Man City", "Man City"),
    ("Manchester United", "Man United"),
    ("Nottingham Forest", "Nott'm Forest"),
    ("Nott'm Forest", "Nott'm Forest"),
    ("Wolverhampton Wanderers", "Wolves"),
    ("Brighton & Hove Albion", "Brighton"),
    ("AFC Bournemouth", "Bournemouth"),
    ("Tottenham Hotspur", "Tottenham"),
    ("Newcastle United", "Newcastle"),
    ("West Ham United", "West Ham"),
    ("Sheffield United", "Sheffield United"),
])
def test_team_normalisation(raw, expected):
    assert normalize_team(raw) == expected


def test_normalisation_handles_junk():
    assert normalize_team(None) == ""
    assert normalize_team(float("nan")) == ""
    assert normalize_team("   ") == ""


def test_short_names_are_three_chars():
    for team in ("Arsenal", "Liverpool", "Nott'm Forest", "Crystal Palace"):
        assert len(config.short_name(team)) == 3


def test_display_names_expand_abbreviations():
    assert config.display_name("Man City") == "Manchester City"
    assert config.display_name("Nott'm Forest") == "Nottingham Forest"
    assert config.display_name("Arsenal") == "Arsenal"
