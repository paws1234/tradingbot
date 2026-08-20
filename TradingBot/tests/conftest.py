"""Shared fixtures for the TradingBot test suite."""

from collections.abc import Callable

import pytest

from app.config import Settings

REQUIRED_SETTINGS = {
    "oanda_api_key": "oanda-key",
    "oanda_account_id": "001-001-1234567-001",
    "deepseek_api_key": "deepseek-key",
    "mongodb_uri": "mongodb://localhost:27017",
    "finnhub_api_key": "finnhub-key",
}


@pytest.fixture
def make_settings() -> Callable[..., Settings]:
    """Factory for valid Settings — override any field with keyword args."""

    def _make(**overrides: object) -> Settings:
        return Settings(**{**REQUIRED_SETTINGS, **overrides})

    return _make
