"""Tests for app.data.forexfactory — calendar scraping, override, retries.

HTTP calls are mocked with respx; the pure parser is exercised directly
with realistic page fixtures. Anchors are fixed dates so the EST/EDT → UTC
math is deterministic (EDT = UTC−4 in August, EST = UTC−5 in January).
"""

import json
from datetime import date, datetime, timezone

import httpx
import pytest
import respx

from app.data.forexfactory import (
    FOREXFACTORY_URL,
    CalendarFetchError,
    ForexFactoryClient,
    parse_calendar_html,
)

ANCHOR_SUMMER = date(2026, 8, 18)  # EDT applies
ANCHOR_WINTER = date(2026, 1, 21)  # EST applies

CALENDAR_PARAMS = {"week": "this"}

# A page spanning a few days with every row shape the parser must handle.
CALENDAR_HTML = """
<html><body>
<table>
<tr class="calendar__row calendar__row--day-breaker">
  <td class="calendar__cell calendar__date">Sun<br>Aug 16</td>
  <td class="calendar__cell calendar__day">Day 1</td>
</tr>
<tr class="calendar__row">
  <td class="calendar__cell calendar__date">Mon<br>Aug 17</td>
  <td class="calendar__cell calendar__time">8:30am</td>
  <td class="calendar__cell calendar__currency">USD</td>
  <td class="calendar__cell calendar__impact">
    <span class="impact impact--high">High</span></td>
  <td class="calendar__cell calendar__event">
    <span class="calendar__event-title">Empire State Manufacturing Index</span></td>
  <td class="calendar__cell calendar__forecast">1.5</td>
  <td class="calendar__cell calendar__previous">0.9</td>
</tr>
<tr class="calendar__row">
  <td class="calendar__cell calendar__date">Mon<br>Aug 17</td>
  <td class="calendar__cell calendar__time">Tentative</td>
  <td class="calendar__cell calendar__currency">NZD</td>
  <td class="calendar__cell calendar__impact">
    <span class="impact impact--high">High</span></td>
  <td class="calendar__cell calendar__event">RBNZ Press Conference</td>
</tr>
<tr class="calendar__row">
  <td class="calendar__cell calendar__date">Mon<br>Aug 17</td>
  <td class="calendar__cell calendar__time">All Day</td>
  <td class="calendar__cell calendar__currency">EUR</td>
  <td class="calendar__cell calendar__impact">
    <span class="impact impact--medium">Medium</span></td>
  <td class="calendar__cell calendar__event">German Bank Holiday</td>
</tr>
<tr class="calendar__row">
  <td class="calendar__cell calendar__date">Mon<br>Aug 17</td>
  <td class="calendar__cell calendar__time">12:00pm</td>
  <td class="calendar__cell calendar__currency">ALL</td>
  <td class="calendar__cell calendar__impact">
    <span class="impact impact--none">None</span></td>
  <td class="calendar__cell calendar__event">Non-Economic Event</td>
</tr>
<tr class="calendar__row">
  <td class="calendar__cell calendar__date">Tue<br>Aug 18</td>
  <td class="calendar__cell calendar__time">9:30pm</td>
  <td class="calendar__cell calendar__currency">CAD</td>
  <td class="calendar__cell calendar__impact">
    <span class="impact impact--medium">Medium</span></td>
  <td class="calendar__cell calendar__event">CPI m/m</td>
  <td class="calendar__cell calendar__forecast">0.3%</td>
  <td class="calendar__cell calendar__previous">0.1%</td>
</tr>
<tr class="calendar__row">
  <td class="calendar__cell calendar__date">Tue<br>Aug 18</td>
  <td class="calendar__cell calendar__time">10:00am</td>
  <td class="calendar__cell calendar__currency">USD</td>
  <td class="calendar__cell calendar__impact">
    <span class="impact impact--low">Low</span></td>
  <td class="calendar__cell calendar__event">NAHB Housing Market Index</td>
</tr>
<tr class="calendar__row">
  <td class="calendar__cell calendar__date">Tue<br>Aug 18</td>
  <td class="calendar__cell calendar__time">2:00pm</td>
  <td class="calendar__cell calendar__currency">USD</td>
  <td class="calendar__cell calendar__impact">
    <span class="impact impact--high">High</span></td>
  <td class="calendar__cell calendar__event">FOMC Meeting Minutes</td>
  <td class="calendar__cell calendar__forecast"></td>
  <td class="calendar__cell calendar__previous"></td>
</tr>
<tr class="calendar__row">
  <td class="calendar__cell calendar__time">3:15pm</td>
  <td class="calendar__cell calendar__currency">USD</td>
  <td class="calendar__cell calendar__impact">
    <span class="impact impact--high">High</span></td>
  <td class="calendar__cell calendar__event">Jackson Hole Symposium</td>
</tr>
</table>
</body></html>
"""

EMPTY_CALENDAR_HTML = "<html><body>Access denied — please verify you are human</body></html>"


def make_client(make_settings, settings_overrides=None, **kwargs) -> ForexFactoryClient:
    overrides = settings_overrides or {}
    return ForexFactoryClient(
        make_settings(**overrides),
        client=httpx.AsyncClient(),
        initial_backoff=0.0,
        **kwargs,
    )


@respx.mock
@pytest.mark.asyncio
async def test_fetch_calendar_parses_high_impact_edt_to_utc(make_settings) -> None:
    route = respx.get(FOREXFACTORY_URL, params=CALENDAR_PARAMS).mock(
        return_value=httpx.Response(200, html=CALENDAR_HTML)
    )
    scraper = make_client(make_settings)

    events = await scraper.fetch_calendar(anchor=ANCHOR_SUMMER)
    await scraper.close()

    # High only; Tentative / All Day / Non-Economic / Low rows are dropped.
    assert [event.title for event in events] == [
        "Empire State Manufacturing Index",
        "FOMC Meeting Minutes",
        "Jackson Hole Symposium",
    ]
    first, second, third = events
    # 8:30am EDT → 12:30 UTC; 2:00pm EDT → 18:00 UTC; 3:15pm EDT → 19:15 UTC.
    assert first.time == datetime(2026, 8, 17, 12, 30, tzinfo=timezone.utc)
    assert second.time == datetime(2026, 8, 18, 18, 0, tzinfo=timezone.utc)
    assert third.time == datetime(2026, 8, 18, 19, 15, tzinfo=timezone.utc)
    assert first.currency == "USD"
    assert first.impact == "High"
    assert first.forecast == "1.5"
    assert first.previous == "0.9"
    assert second.forecast is None  # empty cell → None
    assert second.previous is None
    # The request carries the week param and a browser User-Agent (Cloudflare
    # rejects default clients).
    assert route.calls[0].request.url.params["week"] == "this"
    assert "Mozilla" in route.calls[0].request.headers["user-agent"]


@respx.mock
@pytest.mark.asyncio
async def test_fetch_min_impact_medium_includes_medium(make_settings) -> None:
    respx.get(FOREXFACTORY_URL, params=CALENDAR_PARAMS).mock(
        return_value=httpx.Response(200, html=CALENDAR_HTML)
    )
    scraper = make_client(make_settings)

    events = await scraper.fetch_calendar(
        anchor=ANCHOR_SUMMER, min_impact="Medium"
    )
    await scraper.close()

    titles = [event.title for event in events]
    assert "CPI m/m" in titles  # Medium passes at the Medium floor
    assert "NAHB Housing Market Index" not in titles  # Low stays out
    assert len(events) == 4  # 3 High + 1 Medium


def test_parse_converts_est_to_utc_in_winter() -> None:
    html = """
    <table>
    <tr class="calendar__row">
      <td class="calendar__cell calendar__date">Tue<br>Jan 20</td>
      <td class="calendar__cell calendar__time">8:30am</td>
      <td class="calendar__cell calendar__currency">CAD</td>
      <td class="calendar__cell calendar__impact">
        <span class="impact impact--high">High</span></td>
      <td class="calendar__cell calendar__event">CPI m/m</td>
    </tr>
    </table>
    """

    events = parse_calendar_html(html, ANCHOR_WINTER)

    assert len(events) == 1
    # 8:30am EST → 13:30 UTC (EST is UTC−5; no DST bookkeeping needed).
    assert events[0].time == datetime(2026, 1, 20, 13, 30, tzinfo=timezone.utc)


def test_parse_resolves_year_rollover_across_new_year() -> None:
    html = """
    <table>
    <tr class="calendar__row">
      <td class="calendar__cell calendar__date">Wed<br>Dec 30</td>
      <td class="calendar__cell calendar__time">8:30am</td>
      <td class="calendar__cell calendar__currency">USD</td>
      <td class="calendar__cell calendar__impact">
        <span class="impact impact--high">High</span></td>
      <td class="calendar__cell calendar__event">ISM Manufacturing PMI</td>
    </tr>
    </table>
    """

    events = parse_calendar_html(html, date(2026, 1, 1))

    # The cell's "Dec 30" in the anchor's year would be ~365 days out —
    # it belongs to the previous year.
    assert events[0].time == datetime(2025, 12, 30, 13, 30, tzinfo=timezone.utc)


def test_parse_skips_rows_without_usable_parts() -> None:
    html = """
    <table>
    <tr class="calendar__row">
      <td class="calendar__cell calendar__date">garbage date</td>
      <td class="calendar__cell calendar__time">8:30am</td>
      <td class="calendar__cell calendar__currency">USD</td>
      <td class="calendar__cell calendar__impact">
        <span class="impact impact--high">High</span></td>
      <td class="calendar__cell calendar__event">Unparseable date</td>
    </tr>
    <tr class="calendar__row">
      <td class="calendar__cell calendar__time">8:30am</td>
      <td class="calendar__cell calendar__currency">USD</td>
      <td class="calendar__cell calendar__impact">
        <span class="impact impact--high">High</span></td>
      <td class="calendar__cell calendar__event">No date seen yet</td>
    </tr>
    <tr class="calendar__row">
      <td class="calendar__cell calendar__date">Tue<br>Aug 18</td>
      <td class="calendar__cell calendar__time">8:30am</td>
      <td class="calendar__cell calendar__currency">USD</td>
      <td class="calendar__cell calendar__impact">
        <span class="impact impact--high">High</span></td>
    </tr>
    </table>
    """

    events = parse_calendar_html(html, ANCHOR_SUMMER)

    # All three rows are unusable: a bad date cell, no date at all yet,
    # and a missing event title.
    assert events == []


@respx.mock
@pytest.mark.asyncio
async def test_fetch_retries_until_success(make_settings) -> None:
    route = respx.get(FOREXFACTORY_URL, params=CALENDAR_PARAMS).mock(
        side_effect=[
            httpx.Response(500, json={"error": "upstream"}),
            httpx.Response(500, json={"error": "upstream"}),
            httpx.Response(200, html=CALENDAR_HTML),
        ]
    )
    scraper = make_client(make_settings, max_attempts=3)

    events = await scraper.fetch_calendar(anchor=ANCHOR_SUMMER)
    await scraper.close()

    assert len(route.calls) == 3
    assert len(events) == 3  # the High-impact rows


@respx.mock
@pytest.mark.asyncio
async def test_fetch_retries_zero_event_pages(make_settings) -> None:
    # A bot-check page is a 200 with no calendar rows — the scraper must
    # treat it like a failed attempt, not an empty calendar.
    route = respx.get(FOREXFACTORY_URL, params=CALENDAR_PARAMS).mock(
        side_effect=[
            httpx.Response(200, html=EMPTY_CALENDAR_HTML),
            httpx.Response(200, html=CALENDAR_HTML),
        ]
    )
    scraper = make_client(make_settings, max_attempts=2)

    events = await scraper.fetch_calendar(anchor=ANCHOR_SUMMER)
    await scraper.close()

    assert len(route.calls) == 2
    assert len(events) == 3


@respx.mock
@pytest.mark.asyncio
async def test_fetch_raises_after_all_attempts(make_settings) -> None:
    respx.get(FOREXFACTORY_URL, params=CALENDAR_PARAMS).mock(
        side_effect=[
            httpx.Response(500, json={"error": "upstream"}),
            httpx.Response(500, json={"error": "upstream"}),
        ]
    )
    scraper = make_client(make_settings, max_attempts=2)

    with pytest.raises(CalendarFetchError):
        await scraper.fetch_calendar(anchor=ANCHOR_SUMMER)
    await scraper.close()


VALID_OVERRIDE = json.dumps(
    [
        {
            "title": "FOMC",
            "time": "2026-08-20T18:00:00Z",
            "currency": "usd",
            "impact": "High",
        },
        {
            "title": "CPI",
            "time": "2026-08-21T12:30:00+00:00",
            "currency": "EUR",
            "impact": "Medium",
        },
        {  # unknown impact → skipped
            "title": "Mystery",
            "time": "2026-08-22T12:00:00Z",
            "currency": "USD",
            "impact": "Massive",
        },
        {  # unparseable time → skipped
            "title": "Broken",
            "time": "not-a-time",
            "currency": "USD",
            "impact": "High",
        },
        "not a dict",  # skipped
    ]
)


def test_load_override_parses_and_skips_unusable(make_settings) -> None:
    scraper = make_client(
        make_settings, settings_overrides={"forexfactory_override": VALID_OVERRIDE}
    )

    events = scraper.load_override()

    assert [event.title for event in events] == ["FOMC", "CPI"]
    assert events[0].currency == "USD"  # "usd" is normalized to upper case
    assert events[0].time == datetime(2026, 8, 20, 18, 0, tzinfo=timezone.utc)
    assert events[1].impact == "Medium"


def test_load_override_invalid_json_is_ignored(make_settings) -> None:
    scraper = make_client(
        make_settings, settings_overrides={"forexfactory_override": "{not json"}
    )

    assert scraper.load_override() == []


def test_load_override_non_list_is_ignored(make_settings) -> None:
    scraper = make_client(
        make_settings,
        settings_overrides={"forexfactory_override": '{"title": "FOMC"}'},
    )

    assert scraper.load_override() == []


@respx.mock
@pytest.mark.asyncio
async def test_get_calendar_override_skips_http(make_settings) -> None:
    scraper = make_client(
        make_settings, settings_overrides={"forexfactory_override": VALID_OVERRIDE}
    )

    events = await scraper.get_calendar()
    await scraper.close()

    # The override wins outright — no HTTP request is made at all.
    assert [event.title for event in events] == ["FOMC"]  # High floor
    assert len(respx.calls) == 0


@respx.mock
@pytest.mark.asyncio
async def test_get_calendar_scrape_failure_returns_empty(make_settings) -> None:
    respx.get(FOREXFACTORY_URL, params=CALENDAR_PARAMS).mock(
        side_effect=[
            httpx.Response(500, json={"error": "upstream"}),
            httpx.Response(500, json={"error": "upstream"}),
        ]
    )
    scraper = make_client(make_settings, max_attempts=2)

    events = await scraper.get_calendar()
    await scraper.close()

    # No override, all attempts fail → empty calendar, not an exception.
    # The blackout filter then has no windows (accepted risk in Plan.md).
    assert events == []
