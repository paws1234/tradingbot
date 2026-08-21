"""Stage 4 position sizing (Plan.md Stage 4): 1% risk units, OANDA
rounding, and the MARKET order payload.

The engine (Task 15) feeds every approved signal through here before
dispatch:

1. :func:`compute_units` — ``units = (balance x risk_per_trade_pct) /
   |entry - stop_loss|`` (Plan.md Stage 4), floored to a whole number so
   the trade never risks more than the budget. The Signal validator
   guarantees ``|entry - stop_loss| > 0``; the guard here keeps the
   function safe to call standalone.
2. :func:`build_market_order` — wraps a :class:`~app.models.schemas.Signal`
   into the ``{"order": {...}}`` spec that
   :meth:`OandaClient.place_market_order
   <app.data.oanda.OandaClient.place_market_order>` transports: a MARKET
   order with signed units (positive BUY / negative SELL), ``FOK``, and
   ``stopLossOnFill``/``takeProfitOnFill`` priced at the signal's hard
   stop and take profit. Returns ``None`` when the balance cannot size
   even one whole unit, when the computed size falls outside the
   instrument's [min, max] unit bounds, or when the order's notional
   (units x entry) exceeds the account's ``marginAvailable`` (strategy.md
   §6.5) — the engine skips dispatch (fail-safe, never a sub-minimum or
   over-leveraged order).

Pure and synchronous with explicit inputs (no Settings), so tests drive it
directly and the engine calls it without I/O. The ``balance`` comes from
``OandaClient.get_account_summary`` as a decimal string — callers cast with
``float()`` before passing it in.
"""

from app.models.schemas import Signal

# OANDA v20 prices are decimal strings. 5 dp covers the trading precision of
# XAU and the FX majors; any excess precision is rounded by the API to the
# instrument's precision.
_PRICE_FORMAT = "{:.5f}"

# Static OANDA v20 instrument trade limits for the instruments this bot trades
# (strategy.md §6.5, Task 27). Fetching live instrument metadata from OANDA
# is explicitly out of scope, so these are pinned constants. Values follow the
# OANDA v20 instrument specs: metals trade in units of 10 minimum (XAU_USD),
# FX majors in units of 1 minimum; both cap at the API's per-order maximum.
INSTRUMENT_UNIT_LIMITS: dict[str, tuple[int, int]] = {
    "XAU_USD": (10, 1_000_000),
    "EUR_USD": (1, 10_000_000),
    "GBP_USD": (1, 10_000_000),
}

# Fallback notional (units x entry) cap applied when the account summary omits
# margin data. An unverifiable margin must never let an over-leveraged order
# through, so the guard errs to this conservative ceiling.
_NOTIONAL_CAP_FALLBACK = 50_000.0


def compute_units(
    balance: float,
    entry: float,
    stop_loss: float,
    risk_per_trade_pct: float,
) -> int:
    """Whole-number position size for a risk-fraction trade (positive).

    ``units = (balance x risk_per_trade_pct) / |entry - stop_loss|``
    (Plan.md Stage 4), floored — never rounded up — so the stake never
    exceeds the risk budget. A non-positive balance or risk fraction, or a
    zero stop distance, cannot size a trade and yields 0.
    """
    if balance <= 0 or risk_per_trade_pct <= 0:
        return 0
    stop_distance = abs(entry - stop_loss)
    if stop_distance == 0:
        return 0
    risk_amount = balance * risk_per_trade_pct
    return int(risk_amount / stop_distance)


def build_market_order(
    signal: Signal,
    balance: float,
    risk_per_trade_pct: float,
    margin_available: float | None = None,
) -> dict | None:
    """Build the ``{"order": {...}}`` MARKET spec for ``place_market_order``.

    ``units`` follows the OANDA sign convention: positive for BUY, negative
    for SELL. ``stopLossOnFill``/``takeProfitOnFill`` carry the signal's
    hard stop and take profit as decimal strings. Returns ``None`` — the
    engine skips dispatch rather than placing a bad order — when no whole
    unit can be sized, when the size falls outside the instrument's
    [min, max] unit bounds, or when the notional (units x entry) exceeds
    the account's ``marginAvailable`` (strategy.md §6.5; a missing margin
    value falls back to :data:`_NOTIONAL_CAP_FALLBACK`).
    """
    units = compute_units(balance, signal.entry, signal.stop_loss, risk_per_trade_pct)
    if units == 0:
        return None
    if not _within_instrument_limits(signal.instrument, units):
        return None
    if not _within_margin(signal, units, margin_available):
        return None
    signed_units = units if signal.side == "BUY" else -units
    return {
        "order": {
            "type": "MARKET",
            "instrument": signal.instrument,
            "units": str(signed_units),
            "timeInForce": "FOK",
            "stopLossOnFill": {"price": _PRICE_FORMAT.format(signal.stop_loss)},
            "takeProfitOnFill": {"price": _PRICE_FORMAT.format(signal.take_profit)},
        }
    }


def _within_instrument_limits(instrument: str, units: int) -> bool:
    """True when ``units`` falls inside the instrument's [min, max] bounds.

    Instruments without a configured limit have no guard and pass — the map
    only covers the three instruments this bot trades.
    """
    limits = INSTRUMENT_UNIT_LIMITS.get(instrument)
    if limits is None:
        return True
    minimum, maximum = limits
    return minimum <= units <= maximum


def _within_margin(
    signal: Signal, units: int, margin_available: float | None
) -> bool:
    """True when the order notional (units x entry) is affordable in margin.

    Uses ``marginAvailable`` from the OANDA account summary; when it is
    missing, the conservative :data:`_NOTIONAL_CAP_FALLBACK` applies so an
    unverifiable margin never lets an over-leveraged order through.
    """
    notional = units * signal.entry
    cap = margin_available if margin_available is not None else _NOTIONAL_CAP_FALLBACK
    return notional <= cap
