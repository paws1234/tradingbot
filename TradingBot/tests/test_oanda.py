"""Tests for app.data.oanda — REST calls and the NDJSON pricing stream.

All HTTP is mocked with respx; no network access. Stream tests bound the
reconnect loop with `max_connections` (production runs it unbounded).
"""

import json
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx
import pytest
import respx

from app.config import OANDA_STREAM_URLS
from app.data.oanda import OandaClient

OANDA_REST = "https://api-fxpractice.oanda.com"
ACCOUNT_ID = "001-001-1234567-001"
CANDLES_URL = f"{OANDA_REST}/v3/instruments/XAU_USD/candles"
SUMMARY_URL = f"{OANDA_REST}/v3/accounts/{ACCOUNT_ID}/summary"
ORDERS_URL = f"{OANDA_REST}/v3/accounts/{ACCOUNT_ID}/orders"
STREAM_URL = (
    f"{OANDA_STREAM_URLS['practice']}/v3/accounts/{ACCOUNT_ID}/pricing/stream"
)

CANDLES_PARAMS = {"granularity": "M15", "count": 500, "price": "M"}

# OANDA sends candles newest-first, with the in-progress bar marked
# complete=false.
CANDLES_PAYLOAD = {
    "instrument": "XAU_USD",
    "granularity": "M15",
    "candles": [
        {
            "complete": False,
            "volume": 123,
            "time": "2026-08-18T15:45:00.000000000Z",
            "mid": {"o": "10.0", "h": "11.0", "l": "9.0", "c": "10.5"},
        },
        {
            "complete": True,
            "volume": 456,
            "time": "2026-08-18T15:30:00.000000000Z",
            "mid": {"o": "9.5", "h": "10.2", "l": "9.4", "c": "10.0"},
        },
        {
            "complete": True,
            "volume": 100,
            "time": "2026-08-18T15:15:00.000000000Z",
            "mid": {"o": "9.0", "h": "9.6", "l": "8.9", "c": "9.5"},
        },
    ],
}

PRICE_LINE = (
    '{"type":"PRICE","instrument":"XAU_USD",'
    '"time":"2026-08-18T12:00:00.123456789Z",'
    '"bids":[{"price":"2660.123","liquidity":10000000}],'
    '"asks":[{"price":"2660.456","liquidity":10000000}]}'
)

TICK_TIME = datetime(2026, 8, 18, 12, 0, 0, 123456, tzinfo=timezone.utc)


def make_client(make_settings, **kwargs) -> OandaClient:
    return OandaClient(
        make_settings(), client=httpx.AsyncClient(), initial_backoff=0.0, **kwargs
    )


@respx.mock
@pytest.mark.asyncio
async def test_get_candles_closed_only_oldest_first(make_settings) -> None:
    route = respx.get(CANDLES_URL, params=CANDLES_PARAMS).mock(
        return_value=httpx.Response(200, json=CANDLES_PAYLOAD)
    )
    oanda = make_client(make_settings)

    candles = await oanda.get_candles("XAU_USD")
    await oanda.close()

    # In-progress 15:45 bar dropped; survivors reversed into chronological order.
    assert len(candles) == 2
    first, second = candles
    assert first.time == datetime(2026, 8, 18, 15, 15, tzinfo=timezone.utc)
    assert (first.open, first.high, first.low, first.close) == (9.0, 9.6, 8.9, 9.5)
    assert first.volume == 100
    assert second.time == datetime(2026, 8, 18, 15, 30, tzinfo=timezone.utc)
    assert second.close == 10.0

    request = route.calls[0].request
    assert request.headers["authorization"] == "Bearer oanda-key"


@respx.mock
@pytest.mark.asyncio
async def test_get_candles_custom_granularity_and_count(make_settings) -> None:
    route = respx.get(
        CANDLES_URL, params={"granularity": "H1", "count": 10, "price": "M"}
    ).mock(return_value=httpx.Response(200, json={"candles": []}))
    oanda = make_client(make_settings)

    assert await oanda.get_candles("XAU_USD", granularity="H1", count=10) == []
    await oanda.close()
    # The route matched its exact params, so one call is the full contract.
    assert len(route.calls) == 1


@respx.mock
@pytest.mark.asyncio
async def test_get_candles_http_error_raises(make_settings) -> None:
    respx.get(CANDLES_URL, params=CANDLES_PARAMS).mock(
        return_value=httpx.Response(400, json={"errorMessage": "Bad granularity"})
    )
    oanda = make_client(make_settings)

    with pytest.raises(httpx.HTTPStatusError):
        await oanda.get_candles("XAU_USD")
    await oanda.close()


@respx.mock
@pytest.mark.asyncio
async def test_get_account_summary_returns_account_dict(make_settings) -> None:
    account = {
        "id": ACCOUNT_ID,
        "balance": "10000.0000",
        "currency": "USD",
        "NAV": "10000.0000",
    }
    respx.get(SUMMARY_URL).mock(
        return_value=httpx.Response(200, json={"account": account})
    )
    oanda = make_client(make_settings)

    assert await oanda.get_account_summary() == account
    await oanda.close()


@respx.mock
@pytest.mark.asyncio
async def test_get_account_summary_http_error_raises(make_settings) -> None:
    respx.get(SUMMARY_URL).mock(
        return_value=httpx.Response(401, json={"errorMessage": "Invalid token"})
    )
    oanda = make_client(make_settings)

    with pytest.raises(httpx.HTTPStatusError):
        await oanda.get_account_summary()
    await oanda.close()


@respx.mock
@pytest.mark.asyncio
async def test_place_market_order_parses_fill(make_settings) -> None:
    spec = {
        "order": {
            "type": "MARKET",
            "instrument": "XAU_USD",
            "units": "12",
            "timeInForce": "FOK",
            "stopLossOnFill": {"price": "2655.0"},
            "takeProfitOnFill": {"price": "2670.0"},
        }
    }
    payload = {
        "orderCreateTransaction": {
            "id": "456",
            "time": "2026-08-18T15:30:00.000000000Z",
            "type": "MARKET_ORDER",
            "instrument": "XAU_USD",
            "units": "12",
            "timeInForce": "FOK",
            "positionFill": "DEFAULT",
            "reason": "CLIENT_ORDER",
        },
        "orderFillTransaction": {
            "id": "457",
            "time": "2026-08-18T15:30:00.000000000Z",
            "orderID": "123",
            "instrument": "XAU_USD",
            "units": "12",
            "price": "2660.5",
            "reason": "MARKET_ORDER",
            "pl": "0",
        },
    }
    route = respx.post(ORDERS_URL).mock(
        return_value=httpx.Response(201, json=payload)
    )
    oanda = make_client(make_settings)

    result = await oanda.place_market_order(spec)
    await oanda.close()

    assert result.order_id == "123"  # the order ID, not the transaction ID
    assert result.status == "FILLED"
    assert result.instrument == "XAU_USD"
    assert result.units == "12"
    assert result.price == 2660.5
    assert result.created_at == datetime(2026, 8, 18, 15, 30, tzinfo=timezone.utc)
    # httpx serializes compactly; assert the parsed contract, not the bytes.
    assert json.loads(route.calls[0].request.content) == spec


@respx.mock
@pytest.mark.asyncio
async def test_place_market_order_pending_without_fill(make_settings) -> None:
    payload = {
        "orderCreateTransaction": {
            "id": "456",
            "time": "2026-08-18T15:30:00.000000000Z",
            "type": "MARKET_ORDER",
            "instrument": "XAU_USD",
            "units": "12",
            "timeInForce": "FOK",
            "positionFill": "DEFAULT",
            "reason": "CLIENT_ORDER",
        }
    }
    respx.post(ORDERS_URL).mock(return_value=httpx.Response(201, json=payload))
    oanda = make_client(make_settings)

    result = await oanda.place_market_order({"order": {"type": "MARKET"}})
    await oanda.close()

    assert result.order_id == "456"
    assert result.status == "PENDING"
    assert result.price is None


@respx.mock
@pytest.mark.asyncio
async def test_place_market_order_http_error_raises(make_settings) -> None:
    respx.post(ORDERS_URL).mock(
        return_value=httpx.Response(400, json={"errorMessage": "Invalid units"})
    )
    oanda = make_client(make_settings)

    with pytest.raises(httpx.HTTPStatusError):
        await oanda.place_market_order({"order": {}})
    await oanda.close()


@respx.mock
@pytest.mark.asyncio
async def test_place_market_order_without_transaction_raises(make_settings) -> None:
    respx.post(ORDERS_URL).mock(return_value=httpx.Response(201, json={}))
    oanda = make_client(make_settings)

    with pytest.raises(ValueError, match="no order transaction"):
        await oanda.place_market_order({"order": {}})
    await oanda.close()


REST_STREAM_URL = f"{OANDA_REST}/v3/accounts/{ACCOUNT_ID}/pricing/stream"


@respx.mock
@pytest.mark.asyncio
async def test_stream_uses_documented_stream_host(make_settings) -> None:
    params = {"instruments": "XAU_USD"}
    # Regression guard: the documented streaming host serves a tick, while the
    # REST host answers 401. Both are mocked, so a regression to the REST host
    # yields no ticks and fails the assertions below.
    respx.get(STREAM_URL, params=params).mock(
        return_value=httpx.Response(200, content=PRICE_LINE)
    )
    respx.get(REST_STREAM_URL, params=params).mock(
        return_value=httpx.Response(401, json={"errorMessage": "wrong host"})
    )
    oanda = make_client(make_settings)

    ticks = [
        tick async for tick in oanda.stream_prices(["XAU_USD"], max_connections=1)
    ]
    await oanda.close()

    assert len(ticks) == 1
    assert (
        respx.calls[0].request.url.host
        == urlparse(OANDA_STREAM_URLS["practice"]).hostname
    )


@respx.mock
@pytest.mark.asyncio
async def test_stream_yields_ticks_and_skips_noise(make_settings) -> None:
    body = "\n".join(
        [
            PRICE_LINE,
            '{"type":"HEARTBEAT","time":"2026-08-18T12:00:05.000000000Z"}',
            "",
            "not json",
            '{"type":"PRICE","instrument":"XAU_USD"}',  # missing bids → skipped
            '{"type":"OTHER","time":"2026-08-18T12:00:05.000000000Z"}',
            PRICE_LINE,
        ]
    )
    route = respx.get(STREAM_URL, params={"instruments": "XAU_USD"}).mock(
        return_value=httpx.Response(200, content=body)
    )
    oanda = make_client(make_settings)

    ticks = [
        tick async for tick in oanda.stream_prices(["XAU_USD"], max_connections=1)
    ]
    await oanda.close()

    assert len(ticks) == 2
    for tick in ticks:
        assert tick.instrument == "XAU_USD"
        assert tick.bid == 2660.123
        assert tick.ask == 2660.456
        assert tick.time == TICK_TIME  # nanosecond timestamp truncated to µs
    assert route.calls[0].request.headers["authorization"] == "Bearer oanda-key"


@respx.mock
@pytest.mark.asyncio
async def test_stream_reconnects_after_clean_close(make_settings) -> None:
    respx.get(STREAM_URL, params={"instruments": "XAU_USD"}).mock(
        side_effect=[
            httpx.Response(200, content=PRICE_LINE),
            httpx.Response(200, content=PRICE_LINE),
        ]
    )
    oanda = make_client(make_settings)

    ticks = [
        tick async for tick in oanda.stream_prices(["XAU_USD"], max_connections=2)
    ]
    await oanda.close()

    # The server closing the stream is normal; the feed must come back up.
    assert len(ticks) == 2
    assert len(respx.calls) == 2


@respx.mock
@pytest.mark.asyncio
async def test_stream_reconnects_after_error(make_settings) -> None:
    respx.get(STREAM_URL, params={"instruments": "XAU_USD"}).mock(
        side_effect=[
            httpx.Response(200, content=PRICE_LINE),
            httpx.ConnectError("connection reset"),
        ]
    )
    oanda = make_client(make_settings)

    ticks = [
        tick async for tick in oanda.stream_prices(["XAU_USD"], max_connections=2)
    ]
    await oanda.close()

    # Tick from the healthy connection survives; the error is absorbed and the
    # bounded loop ends instead of retrying forever.
    assert len(ticks) == 1
    assert len(respx.calls) == 2


@respx.mock
@pytest.mark.asyncio
async def test_stream_http_error_yields_nothing(make_settings) -> None:
    respx.get(STREAM_URL, params={"instruments": "XAU_USD"}).mock(
        return_value=httpx.Response(401, json={"errorMessage": "Unauthorized"})
    )
    oanda = make_client(make_settings)

    ticks = [
        tick async for tick in oanda.stream_prices(["XAU_USD"], max_connections=1)
    ]
    await oanda.close()

    assert ticks == []
