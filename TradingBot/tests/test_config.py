"""Tests for app.config — env loading, CSV parsing, and fail-fast validation."""

import pytest
from pydantic import ValidationError

from app.config import OANDA_REST_URLS, OANDA_STREAM_URLS, Settings, get_settings

REQUIRED = {
    "oanda_api_key": "oanda-key",
    "oanda_account_id": "001-001-1234567-001",
    "deepseek_api_key": "deepseek-key",
    "mongodb_uri": "mongodb://localhost:27017",
    "finnhub_api_key": "finnhub-key",
}

ALL_STRATEGIES = ["asia_sweep", "ema_fvg", "atr_breakout", "mean_reversion"]


def make_settings(**overrides: object) -> Settings:
    return Settings(**{**REQUIRED, **overrides})


def test_defaults() -> None:
    s = make_settings()
    assert s.account_type == "practice"
    assert s.instruments == ["XAU_USD"]
    assert s.granularity == "M15"
    assert s.strategies == ALL_STRATEGIES
    assert s.min_confidence == 7
    assert s.daily_loss_limit_pct == 0.03
    assert s.risk_per_trade_pct == 0.01
    assert s.blackout_minutes == 30
    assert s.ema_period == 200
    assert s.rsi_oversold == 30.0
    assert s.rsi_overbought == 70.0
    assert s.atr_sl_mult == 2.0
    assert s.atr_tp_mult == 3.0
    assert s.deepseek_base_url == "https://api.deepseek.com"
    assert s.deepseek_model == "deepseek-chat"
    assert s.mongodb_db == "tradingbot"
    assert s.port == 8000


def test_oanda_urls_follow_account_type() -> None:
    practice = make_settings()
    assert practice.oanda_rest_url == OANDA_REST_URLS["practice"]
    assert practice.oanda_stream_url == OANDA_STREAM_URLS["practice"]
    live = make_settings(account_type="live")
    assert live.oanda_rest_url == OANDA_REST_URLS["live"]
    assert live.oanda_stream_url == OANDA_STREAM_URLS["live"]


def test_instruments_csv_parsing() -> None:
    s = make_settings(instruments="XAU_USD, eur_usd , GBP_USD,")
    assert s.instruments == ["XAU_USD", "EUR_USD", "GBP_USD"]


def test_strategies_csv_parsing() -> None:
    s = make_settings(strategies="asia_sweep, atr_breakout")
    assert s.strategies == ["asia_sweep", "atr_breakout"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("account_type", "paper"),
        ("granularity", "M7"),
        ("port", 0),
        ("port", 70000),
        ("blackout_minutes", -1),
        ("ema_period", 0),
        ("min_confidence", -1),
        ("min_confidence", 11),
        ("daily_loss_limit_pct", 0.0),
        ("daily_loss_limit_pct", 1.5),
        ("risk_per_trade_pct", -0.01),
        ("rsi_oversold", 120.0),
        ("atr_sl_mult", 0.0),
    ],
)
def test_invalid_values_rejected(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        make_settings(**{field: value})


def test_unknown_strategy_rejected() -> None:
    with pytest.raises(ValidationError):
        make_settings(strategies="asia_sweep,not_a_strategy")


def test_empty_csv_rejected() -> None:
    with pytest.raises(ValidationError):
        make_settings(instruments=" , , ")


@pytest.mark.parametrize("missing", sorted(REQUIRED))
def test_missing_secret_rejected(
    missing: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = {k: v for k, v in REQUIRED.items() if k != missing}
    # tests/conftest.py exports dummy env values so `app.main` can be imported;
    # clear the missing key's env too, else Settings() would find it there.
    monkeypatch.delenv(missing.upper(), raising=False)
    with pytest.raises(ValidationError):
        Settings(**values)


def test_loads_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    env = {
        "OANDA_API_KEY": "oanda-key",
        "OANDA_ACCOUNT_ID": "001-001-1234567-001",
        "ACCOUNT_TYPE": "live",
        "INSTRUMENTS": "XAU_USD,EUR_USD",
        "DEEPSEEK_API_KEY": "deepseek-key",
        "MONGODB_URI": "mongodb://localhost:27017",
        "FINNHUB_API_KEY": "finnhub-key",
        "STRATEGIES": "ema_fvg",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    s = Settings()
    assert s.account_type == "live"
    assert s.instruments == ["XAU_USD", "EUR_USD"]
    assert s.strategies == ["ema_fvg"]


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in REQUIRED.items():
        monkeypatch.setenv(key.upper(), value)
    get_settings.cache_clear()
    try:
        assert get_settings() is get_settings()
    finally:
        get_settings.cache_clear()
