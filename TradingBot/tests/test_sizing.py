"""Tests for app.strategy.sizing — Stage 4 position sizing (Plan.md Stage 4).

Covers the sizing hand-calc and the OANDA MARKET payload contract:

- ``compute_units``: ``(balance x risk_per_trade_pct) / |entry - stop_loss|``
  on known numbers, flooring so the stake never exceeds the risk budget,
  and the guard clauses (non-positive balance/risk, zero stop distance).
- ``build_market_order``: the ``{"order": {...}}`` spec that
  ``OandaClient.place_market_order`` transports — MARKET, signed units
  (positive BUY / negative SELL), ``FOK``, ``stopLossOnFill`` /
  ``takeProfitOnFill`` at the signal's hard stop and take profit, and
  ``None`` when no whole unit can be sized.

All functions are pure and synchronous — no fixtures or I/O needed.
"""

from datetime import datetime, timezone

from app.models.schemas import Signal
from app.strategy.sizing import build_market_order, compute_units

T0 = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)


def make_signal(side: str = "BUY", instrument: str = "XAU_USD") -> Signal:
    """A valid Signal with a 10.0 stop distance and SL/TP bracketing entry."""
    if side == "BUY":
        return Signal(
            strategy="ema_fvg",  # type: ignore[arg-type]
            side="BUY",
            instrument=instrument,
            entry=100.0,
            stop_loss=90.0,
            take_profit=110.0,
            atr=1.0,
            reason="test",
            timestamp=T0,
        )
    return Signal(
        strategy="ema_fvg",  # type: ignore[arg-type]
        side="SELL",
        instrument=instrument,
        entry=100.0,
        stop_loss=110.0,
        take_profit=90.0,
        atr=1.0,
        reason="test",
        timestamp=T0,
    )


# --- compute_units ---------------------------------------------------------


def test_compute_units_hand_calc() -> None:
    # Plan.md Stage 4: units = (balance x 0.01) / |entry - SL|
    #                    = (10000 x 0.01) / |100 - 90| = 100 / 10 = 10.
    assert compute_units(10000.0, 100.0, 90.0, 0.01) == 10


def test_compute_units_scales_with_balance() -> None:
    # Doubling the balance doubles the stake: 200 / 10 = 20 units.
    assert compute_units(20000.0, 100.0, 90.0, 0.01) == 20


def test_compute_units_scales_with_stop_distance() -> None:
    # Halving the stop distance doubles the stake: 100 / 5 = 20 units.
    assert compute_units(10000.0, 100.0, 95.0, 0.01) == 20


def test_compute_units_floors_to_never_exceed_risk() -> None:
    # 100 / 3 = 33.33… → 33 whole units, not 34 (34 would risk > 1%).
    assert compute_units(10000.0, 100.0, 97.0, 0.01) == 33


def test_compute_units_zero_balance() -> None:
    assert compute_units(0.0, 100.0, 90.0, 0.01) == 0


def test_compute_units_negative_balance() -> None:
    assert compute_units(-1000.0, 100.0, 90.0, 0.01) == 0


def test_compute_units_zero_risk() -> None:
    assert compute_units(10000.0, 100.0, 90.0, 0.0) == 0


def test_compute_units_zero_stop_distance() -> None:
    assert compute_units(10000.0, 100.0, 100.0, 0.01) == 0


# --- build_market_order ----------------------------------------------------


def test_build_market_order_buy_payload() -> None:
    spec = build_market_order(make_signal("BUY"), 10000.0, 0.01)
    assert spec is not None
    order = spec["order"]
    assert order["type"] == "MARKET"
    assert order["instrument"] == "XAU_USD"
    assert order["units"] == "10"  # positive for BUY
    assert order["timeInForce"] == "FOK"
    assert order["stopLossOnFill"] == {"price": "90.00000"}
    assert order["takeProfitOnFill"] == {"price": "110.00000"}


def test_build_market_order_sell_has_negative_units() -> None:
    spec = build_market_order(make_signal("SELL"), 10000.0, 0.01)
    assert spec is not None
    assert spec["order"]["units"] == "-10"
    assert spec["order"]["stopLossOnFill"] == {"price": "110.00000"}
    assert spec["order"]["takeProfitOnFill"] == {"price": "90.00000"}


def test_build_market_order_uses_signal_instrument() -> None:
    spec = build_market_order(make_signal("BUY", "EUR_USD"), 10000.0, 0.01)
    assert spec is not None
    assert spec["order"]["instrument"] == "EUR_USD"


def test_build_market_order_returns_none_when_no_whole_unit() -> None:
    # 50 x 0.01 / 10 = 0.05 → floors to 0 units: no sub-minimum order.
    spec = build_market_order(make_signal("BUY"), 50.0, 0.01)
    assert spec is None
