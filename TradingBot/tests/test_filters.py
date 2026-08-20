"""Tests for app.strategy.filters — the Stage 2 local filters.

Covers the three filters and the verification points from tasks 10 and 20:

- circuit breaker: halts once the realized day loss reaches
  ``daily_loss_limit_pct`` (-3% default), inclusive of the boundary, and is
  sticky for the day.
- news blackout: ``build_blackout_windows`` expands events to ±N minutes;
  ``in_blackout`` blocks signals inside a window (edges inclusive) and lets
  signals outside pass.
- news relevance: a headline touching a traded instrument by code or
  natural-language alias is relevant; unrelated headlines are not.

All functions are pure and synchronous — no fixtures or I/O needed.
"""

from datetime import datetime, timedelta, timezone

import pytest

from app.models.schemas import CalendarEvent, NewsItem
from app.strategy.filters import (
    CIRCUIT_BREAKER_DAY_LOSS,
    CIRCUIT_BREAKER_HALTED,
    NEWS_BLACKOUT,
    build_blackout_windows,
    circuit_breaker,
    in_blackout,
    is_news_relevant,
)

T0 = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)


def event(title: str = "FOMC", at: datetime = T0) -> CalendarEvent:
    """A high-impact calendar event at `at` UTC."""
    return CalendarEvent(title=title, time=at, currency="USD", impact="High")


def news_item(
    headline: str = "Fed holds rates steady", related: list[str] | None = None
) -> NewsItem:
    """A Finnhub news item with an optional `related` ticker list."""
    return NewsItem(
        headline=headline,
        published_at=T0,
        source="Finnhub",
        related=related or [],
    )


# --- circuit breaker -------------------------------------------------------


def test_circuit_breaker_allows_trading_when_loss_below_limit() -> None:
    state = {"account_id": "a", "day_start_balance": 10000.0, "realized_pnl": -250.0}
    halted, reason = circuit_breaker(state, daily_loss_limit_pct=0.03)
    assert halted is False
    assert reason is None


def test_circuit_breaker_halts_at_exactly_minus_three_percent() -> None:
    # -300 / 10000 = -0.03, the boundary: "day loss ≤ -3%" includes the edge.
    state = {"account_id": "a", "day_start_balance": 10000.0, "realized_pnl": -300.0}
    halted, reason = circuit_breaker(state, daily_loss_limit_pct=0.03)
    assert halted is True
    assert reason == CIRCUIT_BREAKER_DAY_LOSS


def test_circuit_breaker_halts_past_the_limit() -> None:
    state = {"account_id": "a", "day_start_balance": 10000.0, "realized_pnl": -500.0}
    halted, _ = circuit_breaker(state, daily_loss_limit_pct=0.03)
    assert halted is True


def test_circuit_breaker_is_sticky_once_halted() -> None:
    # Even a recovered P&L must not reopen the breaker within the day; only
    # the scheduler's 00:00 UTC reset (Task 14) clears `trading_halted`.
    state = {
        "account_id": "a",
        "day_start_balance": 10000.0,
        "realized_pnl": -50.0,
        "trading_halted": True,
    }
    halted, reason = circuit_breaker(state, daily_loss_limit_pct=0.03)
    assert halted is True
    assert reason == CIRCUIT_BREAKER_HALTED


def test_circuit_breaker_open_without_account_state() -> None:
    # No account_state doc yet (day not started) → nothing to halt on.
    halted, reason = circuit_breaker(None, daily_loss_limit_pct=0.03)
    assert halted is False
    assert reason is None


def test_circuit_breaker_open_with_unusable_baseline() -> None:
    # No day-start balance recorded → the loss cannot be measured.
    state = {"account_id": "a", "realized_pnl": -500.0}
    halted, reason = circuit_breaker(state, daily_loss_limit_pct=0.03)
    assert halted is False
    assert reason is None


# --- news blackout ---------------------------------------------------------


def test_build_blackout_windows_flanks_event_by_blackout_minutes() -> None:
    windows = build_blackout_windows([event(at=T0)], blackout_minutes=30)
    assert len(windows) == 1
    assert windows[0]["title"] == "FOMC"
    assert windows[0]["start"] == T0 - timedelta(minutes=30)
    assert windows[0]["end"] == T0 + timedelta(minutes=30)


def test_build_blackout_windows_sorts_by_start() -> None:
    later = event(title="CPI", at=T0 + timedelta(hours=1))
    earlier = event(title="FOMC", at=T0)
    windows = build_blackout_windows([later, earlier], blackout_minutes=15)
    assert [window["title"] for window in windows] == ["FOMC", "CPI"]


def test_in_blackout_blocks_inside_window() -> None:
    windows = build_blackout_windows([event(at=T0)], blackout_minutes=30)
    blocked, reason = in_blackout(T0 + timedelta(minutes=10), windows)
    assert blocked is True
    assert reason == NEWS_BLACKOUT


@pytest.mark.parametrize("offset_minutes", [-30, 0, 30])
def test_in_blackout_edges_are_inclusive(offset_minutes: int) -> None:
    windows = build_blackout_windows([event(at=T0)], blackout_minutes=30)
    blocked, _ = in_blackout(T0 + timedelta(minutes=offset_minutes), windows)
    assert blocked is True


def test_in_blackout_passes_outside_window() -> None:
    windows = build_blackout_windows([event(at=T0)], blackout_minutes=30)
    blocked, reason = in_blackout(T0 + timedelta(minutes=31), windows)
    assert blocked is False
    assert reason is None


def test_in_blackout_passes_in_the_gap_between_windows() -> None:
    windows = build_blackout_windows(
        [event(at=T0), event(title="CPI", at=T0 + timedelta(hours=2))],
        blackout_minutes=30,
    )
    blocked, reason = in_blackout(T0 + timedelta(hours=1), windows)
    assert blocked is False
    assert reason is None


def test_in_blackout_passes_without_windows() -> None:
    # Empty calendar → no windows → nothing to block (accepted risk, Plan.md).
    blocked, reason = in_blackout(T0, [])
    assert blocked is False
    assert reason is None


# --- news relevance --------------------------------------------------------


def test_news_relevant_by_natural_language_alias() -> None:
    item = news_item("Gold surges as dollar weakens after Fed decision")
    assert is_news_relevant(item, ["XAU_USD"]) is True


def test_news_relevant_by_base_code_ticker() -> None:
    item = news_item("Commodities rally", related=["XAU"])
    assert is_news_relevant(item, ["XAU_USD"]) is True


def test_news_relevant_case_insensitive() -> None:
    item = news_item("GOLD futures hit a record high")
    assert is_news_relevant(item, ["XAU_USD"]) is True


def test_news_relevant_to_any_traded_instrument() -> None:
    item = news_item("Euro zone inflation beats expectations")
    assert is_news_relevant(item, ["XAU_USD", "EUR_USD"]) is True


def test_news_irrelevant_to_other_instruments() -> None:
    item = news_item("Tech stocks close higher on earnings")
    assert is_news_relevant(item, ["XAU_USD", "EUR_USD"]) is False


def test_news_irrelevant_with_no_instruments() -> None:
    item = news_item("Gold surges on safe-haven demand")
    assert is_news_relevant(item, []) is False
