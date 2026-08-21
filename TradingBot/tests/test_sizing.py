"""Tests for app.strategy.sizing — Stage 4 position sizing (Plan.md Stage 4).

Covers the sizing hand-calc and the OANDA MARKET payload contract:

- ``compute_units``: ``(balance x risk_per_trade_pct) / |entry - stop_loss|``
  on known numbers, flooring so the stake never exceeds the risk budget,
  and the guard clauses (non-positive balance/risk, zero stop distance).
- ``build_market_order``: the ``{"order": {...}}`` spec that
  ``OandaClient.place_market_order`` transports — MARKET, signed units
  (positive BUY / negative SELL), ``FOK``, ``stopLossOnFill`` /
  ``takeProfitOnFill`` at the signal's hard stop and take profit, and
  ``None`` when no whole unit can be sized, when the size falls outside
  the instrument's [min, max] unit bounds (strategy.md §6.5), or when the
  order notional exceeds ``marginAvailable`` / the fallback notional cap.

All functions are pure and synchronous — no fixtures or I/O needed.
"""

from datetime import datetime, timezone

from app.models.schemas import Signal
from app.strategy.sizing import (
    INSTRUMENT_UNIT_LIMITS,
    build_market_order,
    compute_units,
)

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


# --- instrument min/max guards (strategy.md §6.5) --------------------------


def test_instrument_unit_limits_map() -> None:
    # Pins the config map Task 27 ships: three traded instruments only.
    assert INSTRUMENT_UNIT_LIMITS == {
        "XAU_USD": (10, 1_000_000),
        "EUR_USD": (1, 10_000_000),
        "GBP_USD": (1, 10_000_000),
    }


def test_build_market_order_returns_none_below_instrument_min() -> None:
    # XAU_USD min is 10 units; balance 5000 sizes 5 units → refused.
    # Notional 500 is inside the fallback cap, so the refusal is the min.
    spec = build_market_order(make_signal("BUY"), 5000.0, 0.01)
    assert spec is None


def test_build_market_order_returns_none_above_instrument_max() -> None:
    # XAU_USD max is 1,000,000 units; balance 1.5e9 sizes 1,500,000 → refused.
    # A generous margin isolates the instrument-max guard.
    spec = build_market_order(
        make_signal("BUY"), 1_500_000_000.0, 0.01, margin_available=1e12
    )
    assert spec is None


def test_build_market_order_fx_majors_within_limits() -> None:
    # 5 units is inside [1, 10,000,000] for both FX majors → not refused.
    for instrument in ("EUR_USD", "GBP_USD"):
        spec = build_market_order(make_signal("BUY", instrument), 5000.0, 0.01)
        assert spec is not None
        assert spec["order"]["units"] == "5"


# --- margin guard (strategy.md §6.5) ---------------------------------------


def test_build_market_order_returns_none_when_notional_exceeds_margin() -> None:
    # Balance 1e6 sizes 1000 units (within XAU bounds); notional 1000 x 100 =
    # 100,000 > marginAvailable 50,000 → refused.
    spec = build_market_order(
        make_signal("BUY"), 1_000_000.0, 0.01, margin_available=50_000.0
    )
    assert spec is None


def test_build_market_order_passes_when_margin_sufficient() -> None:
    # Same size, notional 100,000 ≤ marginAvailable 200,000 → allowed.
    spec = build_market_order(
        make_signal("BUY"), 1_000_000.0, 0.01, margin_available=200_000.0
    )
    assert spec is not None
    assert spec["order"]["units"] == "1000"


def test_build_market_order_margin_guard_applies_to_sell() -> None:
    # Shorts require margin too: SELL 1000 units has the same 100,000 notional.
    spec = build_market_order(
        make_signal("SELL"), 1_000_000.0, 0.01, margin_available=50_000.0
    )
    assert spec is None


def test_build_market_order_fallback_cap_when_margin_missing() -> None:
    # No margin_available → fallback cap 50,000: notional 100,000 → refused.
    spec = build_market_order(make_signal("BUY"), 1_000_000.0, 0.01)
    assert spec is None

    # Notional 10,000 (100 units) is inside the fallback cap → allowed.
    ok = build_market_order(make_signal("BUY"), 100_000.0, 0.01)
    assert ok is not None
    assert ok["order"]["units"] == "100"
