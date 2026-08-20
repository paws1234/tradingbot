"""Shared fixtures for the TradingBot test suite."""

import os
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

# ``app.main`` builds the production app at import time (``app = create_app()``),
# which validates the full settings against the environment. Export dummy values
# (setdefault: real env wins) so any test can ``from app.main import create_app``
# without a committed .env.
for _name, _value in REQUIRED_SETTINGS.items():
    os.environ.setdefault(_name.upper(), str(_value))


@pytest.fixture
def make_settings() -> Callable[..., Settings]:
    """Factory for valid Settings — override any field with keyword args."""

    def _make(**overrides: object) -> Settings:
        return Settings(**{**REQUIRED_SETTINGS, **overrides})

    return _make
