"""Async ForexFactory calendar scraper (httpx + BeautifulSoup).

ForexFactory exposes no API — scraping the calendar page is unofficial and
brittle (Plan.md risks), so this module degrades deliberately:

- `parse_calendar_html` — pure parser: one malformed row is skipped and
  logged, never fatal. Rows with "Tentative"/"All Day" times, without an
  impact level, or without a title are dropped.
- `fetch_calendar` — GET the weekly calendar, retrying transport failures
  *and* pages that parse to zero rows (a bot-check page, not a calendar)
  with backoff.
- `load_override` — manual override: `FOREXFACTORY_OVERRIDE` env var holds a
  JSON array of events the operator can force when the site changes layout.
- `get_calendar` — the engine's entry point: the override wins when set;
  scraping is the default; both failing yields [] (an empty calendar means
  the blackout filter has no windows — accepted risk, surfaced by logs).

The calendar page shows Eastern time; `America/New_York` handles the
EST/EDT switch so event times are stored as UTC without manual DST
bookkeeping. Date cells carry only month/day, so the year is resolved
against the day the scrape runs (`anchor`).

Only events at or above `min_impact` (default High) are returned — Stage 0
stores high-impact events + blackout windows in `daily_context` (Task 14).

`client` is an injection seam for tests (respx mocks the REST transport);
production code passes nothing and gets a real client.
"""

import asyncio
import json
import logging
from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

import httpx
from bs4 import BeautifulSoup

from app.config import Settings
from app.models.schemas import CalendarEvent, Impact

logger = logging.getLogger(__name__)

FOREXFACTORY_URL = "https://www.forexfactory.com/calendar"

# FF runs behind Cloudflare and rejects default request UAs outright.
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    )
}

# The calendar's home timezone — EST in winter, EDT in summer.
EASTERN = ZoneInfo("America/New_York")

_IMPACT_RANK: dict[Impact, int] = {"Low": 1, "Medium": 2, "High": 3}


class CalendarFetchError(Exception):
    """All fetch attempts failed (transport error or unparseable page)."""


def parse_calendar_html(html: str, anchor: date) -> list[CalendarEvent]:
    """Parse the calendar table into events; malformed rows are skipped.

    `anchor` is the day the scrape runs (Stage 0 runs daily at 00:00 UTC).
    Date cells carry only month/day, so the year is resolved against it —
    a week spanning New Year's stays in the right year.
    """
    soup = BeautifulSoup(html, "lxml")
    events: list[CalendarEvent] = []
    current_date: date | None = None
    for row in soup.select("tr.calendar__row"):
        month_day = _month_day(row.select_one("td.calendar__date"))
        if month_day is not None:
            current_date = _resolve_date(*month_day, anchor)
        event = _parse_row(row, current_date)
        if event is not None:
            events.append(event)
    return events


def _parse_row(row, current_date: date | None) -> CalendarEvent | None:
    """One calendar row → CalendarEvent; unusable rows return None."""
    if current_date is None:
        return None
    event_time = _parse_time(row.select_one("td.calendar__time"))
    if event_time is None:
        return None  # "Tentative", "All Day", day-breaker rows…
    impact = _parse_impact(row.select_one("td.calendar__impact"))
    if impact is None:
        return None  # Non-Economic (grey) rows have no tradable impact
    currency_cell = row.select_one("td.calendar__currency")
    title_cell = row.select_one("td.calendar__event")
    if currency_cell is None or title_cell is None:
        return None
    title = title_cell.get_text(" ", strip=True)
    if not title:
        return None
    return CalendarEvent(
        title=title,
        time=_to_utc(current_date, event_time),
        currency=currency_cell.get_text(strip=True).upper(),
        impact=impact,
        forecast=_cell_text(row.select_one("td.calendar__forecast")),
        previous=_cell_text(row.select_one("td.calendar__previous")),
    )


def _month_day(cell) -> tuple[int, int] | None:
    """Month/day from a date cell like "Tue<br>Aug 18" → (8, 18)."""
    if cell is None:
        return None
    parts = cell.get_text(" ", strip=True).split()
    if len(parts) < 2:
        return None
    try:
        month = datetime.strptime(parts[-2], "%b").month
        day = int(parts[-1])
    except ValueError:
        return None
    return month, day


def _parse_time(cell) -> time | None:
    """Time from a cell like "8:30am"; unparseable values → None."""
    if cell is None:
        return None
    try:
        return datetime.strptime(cell.get_text(strip=True), "%I:%M%p").time()
    except ValueError:
        return None


def _parse_impact(cell) -> Impact | None:
    """Impact from the `impact--high/medium/low` CSS class.

    The class can sit on the `<td>` itself or on the `<span>` inside it
    (FF's markup moved it over the years) — check both.
    """
    if cell is None:
        return None
    for node in [cell, *cell.find_all()]:
        for name in node.get("class", []):
            if name in ("impact--high", "impact--medium", "impact--low"):
                return name.removeprefix("impact--").capitalize()
    return None


def _cell_text(cell) -> str | None:
    """Stripped cell text, or None when the cell is missing or empty."""
    if cell is None:
        return None
    text = cell.get_text(strip=True)
    return text or None


def _resolve_date(month: int, day: int, anchor: date) -> date:
    """Attach the anchor's year to a month/day cell, fixing year rollover.

    The calendar page shows no year, so a Dec/Jan week would otherwise land
    ~365 days off. If the naive candidate is more than half a year from the
    anchor, it belongs to the neighbouring year.
    """
    candidate = date(anchor.year, month, day)
    delta = (candidate - anchor).days
    if delta > 182:
        return candidate.replace(year=anchor.year - 1)
    if delta < -182:
        return candidate.replace(year=anchor.year + 1)
    return candidate


def _to_utc(day: date, event_time: time) -> datetime:
    """Attach the calendar's Eastern time to its date, converted to UTC.

    `America/New_York` knows when EDT applies, so no manual DST bookkeeping
    is needed.
    """
    eastern = datetime.combine(day, event_time, tzinfo=EASTERN)
    return eastern.astimezone(timezone.utc)


class ForexFactoryClient:
    """Async scraper for the ForexFactory economic calendar."""

    def __init__(
        self,
        settings: Settings,
        client: httpx.AsyncClient | None = None,
        initial_backoff: float = 1.0,
        max_backoff: float = 30.0,
        max_attempts: int = 3,
    ) -> None:
        self._settings = settings
        if client is None:
            timeout = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)
            client = httpx.AsyncClient(timeout=timeout)
        self._client = client
        self._initial_backoff = initial_backoff
        self._max_backoff = max_backoff
        self._max_attempts = max_attempts

    async def fetch_calendar(
        self,
        week: str = "this",
        min_impact: Impact = "High",
        anchor: date | None = None,
    ) -> list[CalendarEvent]:
        """Fetch and parse the weekly calendar, retrying on failure.

        Retries transport errors and pages that parse to zero rows.
        Raises `CalendarFetchError` after `max_attempts` — `get_calendar`
        converts that into an empty calendar.
        """
        anchor = anchor or datetime.now(timezone.utc).date()
        backoff = self._initial_backoff
        for attempt in range(self._max_attempts):
            try:
                response = await self._client.get(
                    FOREXFACTORY_URL,
                    params={"week": week},
                    headers=BROWSER_HEADERS,
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                logger.warning(
                    "ForexFactory fetch attempt %d failed (%s)", attempt + 1, exc
                )
            else:
                events = parse_calendar_html(response.text, anchor)
                if events:
                    return self._filter_by_impact(events, min_impact)
                logger.warning(
                    "ForexFactory attempt %d parsed zero events — page likely "
                    "a bot-check or a layout change",
                    attempt + 1,
                )
            if attempt + 1 < self._max_attempts:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self._max_backoff)
        raise CalendarFetchError(
            f"calendar unavailable after {self._max_attempts} attempts"
        )

    async def get_calendar(self, min_impact: Impact = "High") -> list[CalendarEvent]:
        """The engine's entry point: the manual override wins, else scrape.

        Returns [] when both are unavailable — the blackout filter then has
        no windows to enforce (accepted risk, Plan.md "ForexFactory has no
        API; scraping is unofficial/brittle").
        """
        override = self.load_override()
        if override:
            return self._filter_by_impact(override, min_impact)
        try:
            return await self.fetch_calendar(min_impact=min_impact)
        except CalendarFetchError as exc:
            logger.error(
                "ForexFactory calendar unavailable (%s); proceeding without "
                "blackout windows",
                exc,
            )
            return []

    def load_override(self) -> list[CalendarEvent]:
        """Parse the manual override: `FOREXFACTORY_OVERRIDE` JSON event list.

        Invalid JSON or unusable entries are skipped with a log line, never
        raised — the scrape still runs as the fallback.
        """
        raw = self._settings.forexfactory_override
        if not raw:
            return []
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error("FOREXFACTORY_OVERRIDE is not valid JSON (%s); ignoring", exc)
            return []
        if not isinstance(payload, list):
            logger.error("FOREXFACTORY_OVERRIDE must be a JSON array of events; ignoring")
            return []
        events: list[CalendarEvent] = []
        for entry in payload:
            event = self._parse_override_event(entry)
            if event is not None:
                events.append(event)
        return events

    @staticmethod
    def _parse_override_event(raw: object) -> CalendarEvent | None:
        """One override dict → CalendarEvent; unusable entries return None."""
        if not isinstance(raw, dict):
            return None
        try:
            event_time = raw["time"]
            if isinstance(event_time, str):
                event_time = datetime.fromisoformat(event_time.replace("Z", "+00:00"))
            return CalendarEvent(
                title=str(raw["title"]),
                time=event_time,
                currency=str(raw["currency"]).upper(),
                impact=raw["impact"],
                forecast=raw.get("forecast"),
                previous=raw.get("previous"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("ForexFactory override: unusable entry (%s)", exc)
            return None

    @staticmethod
    def _filter_by_impact(
        events: list[CalendarEvent], min_impact: Impact
    ) -> list[CalendarEvent]:
        """Keep events at or above `min_impact` (High > Medium > Low)."""
        minimum = _IMPACT_RANK[min_impact]
        return [event for event in events if _IMPACT_RANK[event.impact] >= minimum]

    async def close(self) -> None:
        """Shut the HTTP client down (engine lifespan teardown)."""
        await self._client.aclose()
