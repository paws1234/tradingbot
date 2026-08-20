"""Stage 2 local filters (Plan.md Stage 2): circuit breaker, news blackout,
and Finnhub headline relevance (contract: app/data/finnhub.py).

The engine (Task 15) runs these in order, before the DeepSeek gate:

1. **Circuit breaker** — once the day's realized loss reaches
   ``daily_loss_limit_pct`` (-3% default) trading halts for the rest of the
   day. The ``account_state`` doc carries the ``trading_halted`` flag and is
   reset at 00:00 UTC by the scheduler (Task 14).
2. **News blackout** — no trades inside ±``blackout_minutes`` of a high-impact
   calendar event. :func:`build_blackout_windows` turns the events into the
   windows the scheduler stores in ``daily_context["blackouts"]``; an empty
   window list means no block (accepted risk, app/data/forexfactory.py).
3. **News relevance** — whether a Finnhub headline touches a traded instrument
   (the strategy-neutral half of the news filter).

Everything is pure and synchronous: each filter takes explicit state (account
state dict, blackout windows, news item) and returns a verdict, so the engine
can call it without I/O and tests can drive it directly.
"""

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

from app.models.schemas import CalendarEvent, NewsItem

CIRCUIT_BREAKER_HALTED = "circuit_breaker_halted"
CIRCUIT_BREAKER_DAY_LOSS = "circuit_breaker_day_loss"
NEWS_BLACKOUT = "news_blackout"

# Natural-language aliases for the base side of common OANDA instruments
# (strategy.md §6.3). A headline rarely writes "XAU"; it says "gold". Keep the
# map to bases the recommended instruments use; an unknown base still matches
# by its raw code and any Finnhub `related` ticker that carries it.
_BASE_ALIASES: dict[str, tuple[str, ...]] = {
    "XAU": ("gold", "gld"),
    "XAG": ("silver",),
    "EUR": ("euro",),
    "GBP": ("pound", "sterling"),
    "JPY": ("yen",),
    "AUD": ("aussie", "australian dollar"),
    "NZD": ("kiwi", "new zealand dollar"),
    "CAD": ("loonie", "canadian dollar"),
    "CHF": ("swiss franc", "franc"),
}


def circuit_breaker(
    account_state: dict | None, daily_loss_limit_pct: float
) -> tuple[bool, str | None]:
    """(halted, reason) for the day-loss circuit breaker.

    Halts when the realized day loss reaches the limit:
    ``realized_pnl / day_start_balance <= -daily_loss_limit_pct``. Once
    ``trading_halted`` is set the breaker stays on for the day — only the
    scheduler's 00:00 UTC reset clears it.

    A missing account state (day not started) or an unusable day-start
    baseline means no loss can be measured, so the breaker is open: there is
    nothing to halt on. ``reason`` is ``None`` when trading is allowed.
    """
    if account_state is None:
        return False, None
    if account_state.get("trading_halted"):
        return True, CIRCUIT_BREAKER_HALTED
    day_start = account_state.get("day_start_balance")
    pnl = account_state.get("realized_pnl", 0.0)
    if day_start is None or day_start <= 0:
        return False, None
    if pnl / day_start <= -daily_loss_limit_pct:
        return True, CIRCUIT_BREAKER_DAY_LOSS
    return False, None


def build_blackout_windows(
    events: Sequence[CalendarEvent], blackout_minutes: int
) -> list[dict[str, Any]]:
    """Expand calendar events into ``±blackout_minutes`` UTC windows.

    Each window is ``{"title", "start", "end"}`` with timezone-aware UTC
    datetimes, sorted by start for readable audit output. The scheduler
    (Task 14) stores the result in ``daily_context["blackouts"]``;
    :func:`in_blackout` reads it back.
    """
    delta = timedelta(minutes=blackout_minutes)
    windows = [
        {
            "title": event.title,
            "start": event.time - delta,
            "end": event.time + delta,
        }
        for event in events
    ]
    windows.sort(key=lambda window: window["start"])
    return windows


def in_blackout(
    timestamp: datetime, windows: Sequence[dict[str, Any]]
) -> tuple[bool, str | None]:
    """(in_blackout, reason) for a signal timestamp against the windows.

    A signal is blocked when its timestamp falls inside any window (both
    edges inclusive). Empty or absent windows → no block — the accepted risk
    when the calendar is unavailable (app/data/forexfactory.py).
    """
    for window in windows:
        if window["start"] <= timestamp <= window["end"]:
            return True, NEWS_BLACKOUT
    return False, None


def is_news_relevant(news: NewsItem, instruments: Sequence[str]) -> bool:
    """Does a Finnhub headline touch any traded instrument?

    Matches, case-insensitively, the headline, summary, source and
    ``related`` tickers against each instrument's base code and its
    natural-language aliases. A quote-currency match is deliberately
    ignored — "USD" appears in nearly every macro headline and would make
    every item relevant.
    """
    haystack = " ".join(
        part for part in (news.headline, news.summary or "", news.source) if part
    ).lower()
    haystack += " " + " ".join(news.related).lower()
    for instrument in instruments:
        base = instrument.split("_", 1)[0].upper()
        tokens = _BASE_ALIASES.get(base, ()) + (base.lower(),)
        if any(token in haystack for token in tokens):
            return True
    return False
