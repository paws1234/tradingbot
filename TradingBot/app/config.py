"""Application configuration loaded from environment variables.

All API credentials are required and validated at startup — a missing or
malformed variable fails fast instead of surfacing mid-trade.
"""

from functools import lru_cache
from typing import Literal, Sequence

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Granularities accepted by OANDA v20 candles/pricing endpoints.
Granularity = Literal[
    "S5", "S10", "S15", "S30",
    "M1", "M2", "M4", "M5", "M10", "M15", "M30",
    "H1", "H2", "H3", "H4", "H6", "H8", "H12",
    "D", "W", "M",
]

# Strategy IDs from strategy.md §6.1 — signals.py builds its registry from this.
STRATEGY_IDS = {"asia_sweep", "ema_fvg", "atr_breakout", "mean_reversion"}

OANDA_REST_URLS = {
    "practice": "https://api-fxpractice.oanda.com",
    "live": "https://api-fxtrade.oanda.com",
}

OANDA_STREAM_URLS = {
    "practice": "https://stream-fxpractice.oanda.com",
    "live": "https://stream-fxtrade.oanda.com",
}


def parse_csv(value: str | Sequence[str]) -> list[str]:
    """Split a CSV env value into stripped, non-empty items."""
    if isinstance(value, str):
        value = value.split(",")
    items = [item.strip() for item in value]
    items = [item for item in items if item]
    if not items:
        raise ValueError("CSV value must contain at least one non-empty item")
    return items


class Settings(BaseSettings):
    """Validated settings, loaded from env vars (case-insensitive) or `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # INSTRUMENTS/STRATEGIES are comma-separated, not JSON — keep raw
        # strings so the CSV validators below receive them intact.
        enable_decoding=False,
    )

    # --- OANDA ---
    oanda_api_key: str
    oanda_account_id: str
    account_type: Literal["practice", "live"] = "practice"
    instruments: list[str] = "XAU_USD"
    granularity: Granularity = "M15"

    # --- DeepSeek ---
    deepseek_api_key: str
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    min_confidence: int = 7

    # --- MongoDB ---
    mongodb_uri: str
    mongodb_db: str = "tradingbot"

    # --- Finnhub ---
    finnhub_api_key: str

    # --- ForexFactory (Task 7) ---
    # Manual calendar override: JSON array of CalendarEvent dicts, e.g.
    # [{"title":"FOMC","time":"2026-08-20T18:00:00Z","currency":"USD","impact":"High"}].
    # Set when the site's layout breaks scraping (Plan.md risks).
    forexfactory_override: str | None = None

    # --- Strategies (strategy.md §6.1) ---
    strategies: list[str] = "asia_sweep,ema_fvg,atr_breakout,mean_reversion"

    # --- Filters / risk ---
    daily_loss_limit_pct: float = 0.03
    risk_per_trade_pct: float = 0.01
    blackout_minutes: int = Field(default=30, ge=0)
    ema_period: int = Field(default=200, gt=0)
    rsi_oversold: float = Field(default=30.0, ge=0, le=100)
    rsi_overbought: float = Field(default=70.0, ge=0, le=100)
    atr_sl_mult: float = Field(default=2.0, gt=0)
    atr_tp_mult: float = Field(default=3.0, gt=0)

    # --- Server ---
    port: int = Field(default=8000, ge=1, le=65535)

    @field_validator("instruments", mode="before")
    @classmethod
    def _instruments_csv(cls, value: str | Sequence[str]) -> list[str]:
        return [item.upper() for item in parse_csv(value)]

    @field_validator("strategies", mode="before")
    @classmethod
    def _strategies_csv(cls, value: str | Sequence[str]) -> list[str]:
        return parse_csv(value)

    @field_validator("strategies")
    @classmethod
    def _known_strategy_ids(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - STRATEGY_IDS)
        if unknown:
            raise ValueError(
                f"unknown strategy IDs {unknown}; valid: {sorted(STRATEGY_IDS)}"
            )
        return value

    @field_validator("min_confidence")
    @classmethod
    def _confidence_bounds(cls, value: int) -> int:
        if not 0 <= value <= 10:
            raise ValueError("min_confidence must be between 0 and 10")
        return value

    @field_validator("daily_loss_limit_pct", "risk_per_trade_pct")
    @classmethod
    def _pct_bounds(cls, value: float) -> float:
        if not 0 < value <= 1:
            raise ValueError("percentage settings must be in (0, 1]")
        return value

    @property
    def oanda_rest_url(self) -> str:
        """REST base URL for the configured account type."""
        return OANDA_REST_URLS[self.account_type]

    @property
    def oanda_stream_url(self) -> str:
        """Streaming base URL for the configured account type."""
        return OANDA_STREAM_URLS[self.account_type]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build (and cache) settings from the environment — the fail-fast gate."""
    return Settings()
