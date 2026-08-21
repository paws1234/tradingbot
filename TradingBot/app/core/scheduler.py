"""Daily 00:00 UTC scheduler job — Stage 0 context build + account reset.

Runs once per UTC day (Plan.md Stage 0; task 14):

1. Fetch the ForexFactory calendar (high-impact events only; degrades to []).
2. Expand events into ``±blackout_minutes`` windows
   (:func:`app.strategy.filters.build_blackout_windows`).
3. Summarize the day's high-impact currencies as a coarse macro bias.
4. Upsert the whole picture into ``daily_context`` keyed by the UTC day.
5. Reset ``account_state``: fresh OANDA balance as ``day_start_balance``,
   ``realized_pnl = 0`` and ``trading_halted = False`` — the circuit
   breaker's daily flip (app/strategy/filters.py).

The job body (:meth:`DailyContextScheduler.build_daily_context`) is a plain
async method so tests call it directly for any day; ``start``/``stop`` only
wire the apscheduler cron trigger. Everything degrades deliberately: an
unavailable calendar means no blackout windows (accepted risk,
app/data/forexfactory.py) and an OANDA balance fetch failure skips the
account reset — both logged, never raised, so the daily job always completes.
"""

import logging
from datetime import date, datetime, timezone
from typing import Sequence

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.config import Settings
from app.data.forexfactory import ForexFactoryClient
from app.data.mongo import MongoStore
from app.data.oanda import OandaClient
from app.models.schemas import CalendarEvent
from app.strategy.filters import build_blackout_windows

logger = logging.getLogger(__name__)

# A daily context build that misses its slot (service restart at 00:00, a
# busy event loop) should still run late rather than be skipped — the circuit
# breaker depends on it. One hour of grace is plenty for a once-a-day job.
JOB_GRACE_SECONDS = 3600


def macro_bias(events: Sequence[CalendarEvent]) -> list[dict]:
    """Coarse macro summary: currencies facing high-impact events today.

    Stored in ``daily_context["macro_bias"]`` because the plan names it;
    nothing consumes it in v1 (YAGNI). Grouping the day's high-impact titles
    by currency is the minimal honest summary computable here.
    """
    grouped: dict[str, list[str]] = {}
    for event in events:
        grouped.setdefault(event.currency, []).append(event.title)
    return [
        {"currency": currency, "events": titles}
        for currency, titles in sorted(grouped.items())
    ]


class DailyContextScheduler:
    """AsyncIOScheduler wrapper for the daily 00:00 UTC context job.

    ``store``, ``calendar``, ``oanda`` and ``scheduler`` are injection seams
    for tests (fake clients / a recording scheduler); production passes
    nothing and gets real clients wired to the settings.
    """

    def __init__(
        self,
        store: MongoStore,
        calendar: ForexFactoryClient,
        oanda: OandaClient,
        settings: Settings,
        scheduler: AsyncIOScheduler | None = None,
    ) -> None:
        self._store = store
        self._calendar = calendar
        self._oanda = oanda
        self._settings = settings
        self._scheduler = scheduler or AsyncIOScheduler(timezone=timezone.utc)
        # `scheduler.running` can't drive idempotency: apscheduler 3.11's
        # AsyncIOScheduler.shutdown is fire-and-forget (call_soon_threadsafe),
        # so the flag flips only on the next loop tick. Track start ourselves.
        self._started = False

    def start(self) -> None:
        """Schedule the daily 00:00 UTC job and start the scheduler."""
        if self._started:
            return
        self._scheduler.add_job(
            self.build_daily_context,
            trigger="cron",
            hour=0,
            minute=0,
            id="daily_context",
            replace_existing=True,
            coalesce=True,
            misfire_grace_time=JOB_GRACE_SECONDS,
        )
        self._scheduler.start()
        self._started = True

    def stop(self) -> None:
        """Shut the scheduler down (engine lifespan teardown)."""
        if not self._started:
            return
        self._started = False
        self._scheduler.shutdown(wait=False)

    async def build_daily_context(self, day: date | None = None) -> dict:
        """Stage 0 job body: build today's context and reset the breaker.

        ``day`` is the UTC date the context belongs to (today in production,
        fixed in tests). Returns the written ``daily_context`` document.
        """
        day = day or datetime.now(timezone.utc).date()
        day_key = day.isoformat()

        events = await self._calendar.get_calendar(min_impact="High")
        document = {
            "day": day_key,
            "created_at": datetime.now(timezone.utc),
            "events": [event.model_dump() for event in events],
            "blackouts": build_blackout_windows(
                events, self._settings.blackout_minutes
            ),
            "macro_bias": macro_bias(events),
        }
        await self._store.upsert_daily_context(day_key, document)

        await self._reset_account_state()
        return document

    async def _reset_account_state(self) -> None:
        """Reset the circuit breaker's day-start baseline from OANDA.

        A balance fetch failure skips the reset (logged) rather than crash
        the daily job; the old state doc then stands until the next reset.
        """
        try:
            summary = await self._oanda.get_account_summary()
            balance = float(summary["balance"])
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            logger.error("account_state reset skipped: balance unavailable (%s)", exc)
            return
        await self._store.upsert_account_state(
            self._settings.oanda_account_id,
            {
                "account_id": self._settings.oanda_account_id,
                "day_start_balance": balance,
                "realized_pnl": 0.0,
                "trading_halted": False,
            },
        )
