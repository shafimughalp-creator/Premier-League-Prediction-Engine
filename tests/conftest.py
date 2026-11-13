"""Shared pytest fixtures.

The suite intentionally exercises the real pipeline modules rather than
re-implementing logic, so a regression anywhere in the data or feature layers
surfaces here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import processing  # noqa: E402
from src.features.builder import FeatureBuilder, feature_columns  # noqa: E402


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def matches() -> pd.DataFrame:
    return processing.load_processed()


@pytest.fixture(scope="session")
def early_slice(matches: pd.DataFrame) -> pd.DataFrame:
    """First two seasons - keeps leakage rebuilds fast."""
    return matches[matches["date"] <= pd.Timestamp("2016-06-30")].copy()


@pytest.fixture(scope="session")
def features(matches: pd.DataFrame) -> pd.DataFrame:
    return FeatureBuilder().build(matches)


@pytest.fixture(scope="session")
def feature_names(features: pd.DataFrame) -> list[str]:
    return feature_columns(features)
