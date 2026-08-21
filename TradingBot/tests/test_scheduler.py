"""Tests for app.core.scheduler — the daily 00:00 UTC context job.

Drives ``DailyContextScheduler.build_daily_context`` directly with fake
calendar / OANDA / store — no network, MongoDB, or waiting for midnight —
and checks the cron wiring with a recording scheduler.

Verification points (task 14):

- context build: high-impact events, ±blackout windows, macro bias and a
  UTC-stamped ``created_at`` land in ``daily_context`` keyed by day.
- account reset: fresh OANDA balance becomes ``day_start_balance`` with
  ``realized_pnl = 0`` and ``trading_halted = False``.
- degradation: an empty calendar yields empty windows/bias; a balance fetch
  failure skips the reset but still writes the context.
- scheduling: the cron trigger fires at 00:00 UTC, coalesces, and tolerates
  a one-hour misfire so the daily reset never silently skips a day.
"""

from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx
import pytest

from app.config import Settings
from app.core.scheduler import DailyContextScheduler, macro_bias
from app.models.schemas import CalendarEvent

T0 = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)
DAY = date(2026, 8, 20)
DAY_KEY = "2026-08-20"


def event(
    title: str, at: datetime = T0, currency: str = "USD"
) -> CalendarEvent:
    """A high-impact calendar event at `at` UTC."""
    return CalendarEvent(title=title, time=at, currency=currency, impact="High")


class FakeStore:
    def __init__(self) -> None:
        self.contexts: dict[str, dict] = {}
        self.states: dict[str, dict] = {}

    async def upsert_daily_context(self, day: str, document: dict) -> None:
        self.contexts[day] = document

    async def upsert_account_state(self, account_id: str, document: dict) -> None:
        self.states[account_id] = document


class FakeCalendar:
    def __init__(self, events: list[CalendarEvent] | None = None) -> None:
        self.events = events or []
        self.min_impact: str | None = None

    async def get_calendar(self, min_impact: str = "High") -> list[CalendarEvent]:
        self.min_impact = min_impact
        return self.events


class FakeOanda:
    def __init__(self, error: Exception | None = None) -> None:
        self.summary = {"balance": "10000.0000"}
        self.error = error

    async def get_account_summary(self) -> dict:
        if self.error is not None:
            raise self.error
        return self.summary


class FakeScheduler:
    """Records add_job/start/shutdown instead of running apscheduler."""

    def __init__(self) -> None:
        self.jobs: list[dict[str, Any]] = []
        self.started = False
        self.running = False
        self.start_calls = 0
        self.shutdown_calls = 0

    def add_job(self, func, trigger=None, **kwargs) -> str:
        self.jobs.append({"func": func, "trigger": trigger, **kwargs})
        return "daily_context"

    def start(self) -> None:
        self.start_calls += 1
        self.started = True
        self.running = True

    def shutdown(self, wait: bool = True) -> None:
        self.shutdown_calls += 1
        self.running = False


@pytest.fixture
def store() -> FakeStore:
    return FakeStore()


@pytest.fixture
def calendar() -> FakeCalendar:
    return FakeCalendar()


@pytest.fixture
def oanda() -> FakeOanda:
    return FakeOanda()


@pytest.fixture
def scheduler(
    store: FakeStore,
    calendar: FakeCalendar,
    oanda: FakeOanda,
    make_settings: Callable[..., Settings],
) -> DailyContextScheduler:
    return DailyContextScheduler(store, calendar, oanda, make_settings())


# --- context build (Stage 0) -----------------------------------------------


@pytest.mark.asyncio
async def test_build_daily_context_writes_full_document(
    store: FakeStore,
    calendar: FakeCalendar,
    oanda: FakeOanda,
    scheduler: DailyContextScheduler,
) -> None:
    calendar.events = [
        event("FOMC", T0),
        event("CPI", T0 + timedelta(hours=1)),
        event("ECB", T0 + timedelta(hours=2), currency="EUR"),
    ]
    document = await scheduler.build_daily_context(DAY)

    assert document["day"] == DAY_KEY
    # Events are stored as plain dicts; datetimes stay timezone-aware (BSON).
    assert [entry["title"] for entry in document["events"]] == ["FOMC", "CPI", "ECB"]
    assert all(entry["time"].tzinfo is not None for entry in document["events"])

    assert [window["title"] for window in document["blackouts"]] == ["FOMC", "CPI", "ECB"]
    assert document["blackouts"][0]["start"] == T0 - timedelta(minutes=30)
    assert document["blackouts"][0]["end"] == T0 + timedelta(minutes=30)

    assert document["macro_bias"] == [
        {"currency": "EUR", "events": ["ECB"]},
        {"currency": "USD", "events": ["FOMC", "CPI"]},
    ]
    assert document["created_at"].tzinfo is not None

    # The same document the engine will read back is what got stored.
    assert store.contexts[DAY_KEY] == document


@pytest.mark.asyncio
async def test_build_daily_context_requests_high_impact_only(
    calendar: FakeCalendar, scheduler: DailyContextScheduler
) -> None:
    await scheduler.build_daily_context(DAY)
    assert calendar.min_impact == "High"


@pytest.mark.asyncio
async def test_build_daily_context_defaults_to_today(
    store: FakeStore, scheduler: DailyContextScheduler
) -> None:
    document = await scheduler.build_daily_context()
    assert document["day"] == datetime.now(timezone.utc).date().isoformat()
    assert document["day"] in store.contexts


@pytest.mark.asyncio
async def test_empty_calendar_yields_empty_windows_and_bias(
    scheduler: DailyContextScheduler,
) -> None:
    document = await scheduler.build_daily_context(DAY)
    assert document["events"] == []
    assert document["blackouts"] == []
    assert document["macro_bias"] == []


# --- account_state reset ---------------------------------------------------


@pytest.mark.asyncio
async def test_reset_account_state_from_oanda_balance(
    scheduler: DailyContextScheduler, store: FakeStore
) -> None:
    await scheduler.build_daily_context(DAY)

    assert store.states["001-001-1234567-001"] == {
        "account_id": "001-001-1234567-001",
        "day_start_balance": 10000.0,
        "realized_pnl": 0.0,
        "trading_halted": False,
    }


@pytest.mark.asyncio
async def test_reset_skipped_when_balance_unavailable(
    store: FakeStore,
    oanda: FakeOanda,
    scheduler: DailyContextScheduler,
) -> None:
    oanda.error = httpx.ConnectError("network down")
    document = await scheduler.build_daily_context(DAY)

    # The context is still written; only the reset is skipped.
    assert store.contexts[DAY_KEY] == document
    assert store.states == {}


# --- scheduling ------------------------------------------------------------


def test_default_scheduler_is_utc(
    store: FakeStore,
    calendar: FakeCalendar,
    oanda: FakeOanda,
    make_settings: Callable[..., Settings],
) -> None:
    scheduler = DailyContextScheduler(store, calendar, oanda, make_settings())
    assert scheduler._scheduler.timezone == timezone.utc


def test_start_schedules_daily_0000_utc_cron(
    store: FakeStore,
    calendar: FakeCalendar,
    oanda: FakeOanda,
    make_settings: Callable[..., Settings],
) -> None:
    fake = FakeScheduler()
    scheduler = DailyContextScheduler(
        store, calendar, oanda, make_settings(), scheduler=fake
    )
    scheduler.start()

    assert fake.started is True
    job = fake.jobs[0]
    assert job["id"] == "daily_context"
    assert job["trigger"] == "cron"
    assert job["hour"] == 0
    assert job["minute"] == 0
    assert job["func"] == scheduler.build_daily_context
    # Late runs coalesce into one; a missed slot still fires within the grace
    # window instead of silently skipping the day's reset.
    assert job["coalesce"] is True
    assert job["misfire_grace_time"] == 3600


def test_start_is_idempotent(
    store: FakeStore,
    calendar: FakeCalendar,
    oanda: FakeOanda,
    make_settings: Callable[..., Settings],
) -> None:
    fake = FakeScheduler()
    scheduler = DailyContextScheduler(
        store, calendar, oanda, make_settings(), scheduler=fake
    )
    scheduler.start()
    scheduler.start()
    assert fake.start_calls == 1
    assert len(fake.jobs) == 1


def test_stop_is_noop_before_start(
    store: FakeStore,
    calendar: FakeCalendar,
    oanda: FakeOanda,
    make_settings: Callable[..., Settings],
) -> None:
    fake = FakeScheduler()
    scheduler = DailyContextScheduler(
        store, calendar, oanda, make_settings(), scheduler=fake
    )
    scheduler.stop()
    assert fake.shutdown_calls == 0


def test_stop_after_start_shuts_down_and_is_idempotent(
    store: FakeStore,
    calendar: FakeCalendar,
    oanda: FakeOanda,
    make_settings: Callable[..., Settings],
) -> None:
    fake = FakeScheduler()
    scheduler = DailyContextScheduler(
        store, calendar, oanda, make_settings(), scheduler=fake
    )
    scheduler.start()
    scheduler.stop()
    scheduler.stop()  # second stop is a no-op, not a second shutdown
    assert fake.shutdown_calls == 1


# --- macro_bias ------------------------------------------------------------


def test_macro_bias_groups_high_impact_titles_by_currency() -> None:
    events = [
        event("FOMC", T0),
        event("CPI", T0 + timedelta(hours=1)),
        event("ECB", T0 + timedelta(hours=2), currency="EUR"),
    ]
    assert macro_bias(events) == [
        {"currency": "EUR", "events": ["ECB"]},
        {"currency": "USD", "events": ["FOMC", "CPI"]},
    ]


def test_macro_bias_empty_for_no_events() -> None:
    assert macro_bias([]) == []
