"""Pydantic schemas shared across the pipeline.

Field contracts come from:
- strategy.md §1.5 — Signal
- Plan.md "Pipeline stages" — TradeDecision, OrderResult
- OANDA v20 / Finnhub / ForexFactory wire formats — data models
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

StrategyId = Literal["asia_sweep", "ema_fvg", "atr_breakout", "mean_reversion"]
Side = Literal["BUY", "SELL"]
Impact = Literal["High", "Medium", "Low"]


class UTCModel(BaseModel):
    """Base model: every datetime field must be timezone-aware.

    All pipeline timestamps are UTC. A naive datetime silently breaks blackout
    window math and audit-log ordering, so reject it at construction.
    """

    @model_validator(mode="after")
    def _datetimes_aware(self) -> "UTCModel":
        for name, value in self:
            if isinstance(value, datetime) and value.tzinfo is None:
                raise ValueError(f"{name} must be a timezone-aware datetime")
        return self


class Candle(UTCModel):
    """Closed OHLCV bar (strategy.md §1.1)."""

    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None


class PriceTick(UTCModel):
    """One bid/ask tick from the OANDA pricing stream."""

    instrument: str
    time: datetime
    bid: float
    ask: float


class NewsItem(UTCModel):
    """Finnhub news headline, reduced to the fields the blackout filter needs."""

    headline: str
    published_at: datetime
    source: str
    url: str | None = None
    summary: str | None = None
    related: list[str] = Field(default_factory=list)


class CalendarEvent(UTCModel):
    """ForexFactory calendar event, normalized by app/data/forexfactory.py."""

    title: str
    time: datetime
    currency: str
    impact: Impact
    forecast: str | None = None
    previous: str | None = None


class Signal(UTCModel):
    """Strategy output (strategy.md §1.5). Always carries a numeric stop_loss."""

    strategy: StrategyId
    side: Side
    instrument: str
    entry: float
    stop_loss: float
    take_profit: float
    atr: float
    reason: str
    timestamp: datetime
    pending_ai_veto: bool = True

    @model_validator(mode="after")
    def _stops_bracket_entry(self) -> "Signal":
        # Strict brackets also guarantee |entry - stop_loss| > 0, so the
        # sizing formula `units = (balance x 0.01) / |entry - SL|` never
        # divides by zero.
        long_ok = self.side == "BUY" and self.stop_loss < self.entry < self.take_profit
        short_ok = self.side == "SELL" and self.stop_loss > self.entry > self.take_profit
        if not (long_ok or short_ok):
            raise ValueError(
                "stop_loss and take_profit must bracket entry on the correct side"
            )
        return self


class TradeDecision(UTCModel):
    """DeepSeek veto verdict (Plan.md Stage 3): {execute, confidence, reason}."""

    execute: bool
    confidence: int = Field(ge=0, le=10)
    reason: str


class OrderResult(UTCModel):
    """OANDA order outcome, for the trade_logs audit trail."""

    order_id: str
    status: str
    instrument: str
    units: str  # OANDA v20 returns units as a string
    price: float | None = None
    created_at: datetime
