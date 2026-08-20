"""Tests for app.models.schemas — field contracts and validation."""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app.models.schemas import (
    CalendarEvent,
    Candle,
    NewsItem,
    OrderResult,
    PriceTick,
    Signal,
    TradeDecision,
)

T0 = datetime(2026, 8, 18, 0, 0, tzinfo=timezone.utc)


def make_signal(**overrides: object) -> Signal:
    values = {
        "strategy": "asia_sweep",
        "side": "BUY",
        "instrument": "XAU_USD",
        "entry": 2500.0,
        "stop_loss": 2495.0,
        "take_profit": 2510.0,
        "atr": 2.1,
        "reason": "sweep_of_asia_low",
        "timestamp": T0,
        **overrides,
    }
    return Signal(**values)


def test_candle_matches_ohlcv_contract() -> None:
    c = Candle(time=T0, open=1.0, high=1.5, low=0.9, close=1.2)
    assert c.open == 1.0
    assert c.high == 1.5
    assert c.volume is None


def test_price_tick_fields() -> None:
    t = PriceTick(instrument="XAU_USD", time=T0, bid=2500.0, ask=2500.2)
    assert t.bid < t.ask


def test_news_item_defaults() -> None:
    n = NewsItem(headline="CPI beats", published_at=T0, source="reuters")
    assert n.url is None
    assert n.summary is None
    assert n.related == []


def test_calendar_event_rejects_unknown_impact() -> None:
    with pytest.raises(ValidationError):
        CalendarEvent(title="NFP", time=T0, currency="USD", impact="high")


def test_signal_matches_strategy_contract() -> None:
    s = make_signal()
    assert s.strategy == "asia_sweep"
    assert s.side == "BUY"
    assert s.pending_ai_veto is True


@pytest.mark.parametrize(
    "side,entry,stop_loss,take_profit",
    [
        ("BUY", 2500.0, 2505.0, 2510.0),  # SL above entry
        ("BUY", 2500.0, 2495.0, 2490.0),  # TP below entry
        ("SELL", 2500.0, 2495.0, 2490.0),  # SL below entry
        ("SELL", 2500.0, 2505.0, 2510.0),  # TP above entry
        ("BUY", 2500.0, 2500.0, 2510.0),  # SL == entry: zero-risk sizing
    ],
)
def test_signal_rejects_stops_on_wrong_side(
    side: str, entry: float, stop_loss: float, take_profit: float
) -> None:
    with pytest.raises(ValidationError):
        make_signal(
            side=side, entry=entry, stop_loss=stop_loss, take_profit=take_profit
        )


def test_signal_rejects_unknown_strategy() -> None:
    with pytest.raises(ValidationError):
        make_signal(strategy="momentum")


@pytest.mark.parametrize("confidence", [0, 10])
def test_trade_decision_confidence_boundaries(confidence: int) -> None:
    d = TradeDecision(execute=True, confidence=confidence, reason="ok")
    assert d.confidence == confidence


@pytest.mark.parametrize("confidence", [-1, 11])
def test_trade_decision_confidence_out_of_bounds(confidence: int) -> None:
    with pytest.raises(ValidationError):
        TradeDecision(execute=True, confidence=confidence, reason="bad")


def test_order_result_keeps_units_as_string() -> None:
    r = OrderResult(
        order_id="42",
        status="FILLED",
        instrument="XAU_USD",
        units="12",
        created_at=T0,
    )
    assert r.units == "12"
    assert r.price is None


@pytest.mark.parametrize(
    "model",
    [
        Candle(time=T0, open=1.0, high=1.1, low=0.9, close=1.0),
        PriceTick(instrument="XAU_USD", time=T0, bid=1.0, ask=1.1),
        NewsItem(headline="h", published_at=T0, source="s"),
        CalendarEvent(title="NFP", time=T0, currency="USD", impact="High"),
        make_signal(),
        OrderResult(
            order_id="1", status="FILLED", instrument="EUR_USD",
            units="1", created_at=T0,
        ),
    ],
)
def test_naive_datetimes_rejected(model: object) -> None:
    # Same fields with tzinfo stripped — must fail the UTCModel guard.
    naive = {
        name: value.replace(tzinfo=None) if isinstance(value, datetime) else value
        for name, value in model.model_dump().items()
    }
    with pytest.raises(ValidationError):
        type(model).model_validate(naive)


VALID_STRATEGIES = ["asia_sweep", "ema_fvg", "atr_breakout", "mean_reversion"]


@pytest.mark.parametrize("strategy", VALID_STRATEGIES)
def test_signal_accepts_all_strategy_ids(strategy: str) -> None:
    assert make_signal(strategy=strategy).strategy == strategy


def test_signal_valid_sell_construction() -> None:
    s = make_signal(
        side="SELL", entry=2500.0, stop_loss=2510.0, take_profit=2480.0
    )
    assert s.side == "SELL"


@pytest.mark.parametrize("field", ["entry", "stop_loss", "take_profit"])
def test_signal_rejects_nan_prices(field: str) -> None:
    # NaN comparisons are False, so the bracket validator must reject NaN
    # prices rather than letting them slip through as "valid" signals.
    with pytest.raises(ValidationError):
        make_signal(**{field: float("nan")})


def test_aware_non_utc_datetime_accepted() -> None:
    # The contract is "aware", not "UTC zone": correctly offset datetimes
    # from other sources must compare fine against UTC ones.
    offset = timezone(timedelta(hours=2))
    c = Candle(
        time=datetime(2026, 8, 18, 2, 0, tzinfo=offset),
        open=1.0, high=1.1, low=0.9, close=1.0,
    )
    assert c.time.utcoffset() == timedelta(hours=2)


def test_news_item_ignores_unknown_fields() -> None:
    # Finnhub payloads carry extra keys (id, image, category, ...); the model
    # must drop them rather than fail parsing.
    n = NewsItem(
        headline="ECB holds rates",
        published_at=T0,
        source="reuters",
        related=["EUR"],
        id=12345,
        image="https://example.com/x.png",
    )
    assert n.headline == "ECB holds rates"
    assert n.related == ["EUR"]
