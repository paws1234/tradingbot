"""Tests for app.data.finnhub — WS news stream + REST polling fallback.

REST calls are mocked with respx; WS connections are faked through the
`ws_connect` injection seam. Streaming tests bound the loops with
`max_connections` / `max_polls` (production runs them unbounded).
"""

import asyncio
import json
from datetime import datetime, timezone

import httpx
import pytest
import respx
import websockets.exceptions

from app.data.finnhub import FINNHUB_REST_URL, FinnhubClient

NEWS_URL = f"{FINNHUB_REST_URL}/news"
NEWS_PARAMS = {"category": "general", "token": "finnhub-key"}

# Finnhub sends timestamps as epoch seconds and `related` as a CSV string.
EPOCH = 1787054400  # 2026-08-18T12:00:00Z
NEWS_TIME = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)

NEWS_FRAME = json.dumps(
    {
        "type": "news",
        "data": [
            {
                "category": "general",
                "datetime": EPOCH,
                "headline": "Gold hits record high",
                "id": 1234,
                "related": "GLD,AU",
                "source": "Reuters",
                "summary": "Gold rallied to a record.",
                "url": "https://example.com/news/1",
            }
        ],
    }
)
NEWS_FRAME_2 = json.dumps(
    {
        "type": "news",
        "data": [
            {
                "datetime": EPOCH + 3600,
                "headline": "ECB holds rates",
                "related": "EUR",
                "source": "Bloomberg",
            }
        ],
    }
)

REST_ITEMS = [
    {
        "category": "general",
        "datetime": EPOCH,
        "headline": "Gold hits record high",
        "id": 1234,
        "related": "GLD,AU",
        "source": "Reuters",
        "summary": "Gold rallied to a record.",
        "url": "https://example.com/news/1",
    },
    {  # `related` as a list, no optional fields
        "datetime": EPOCH - 3600,
        "headline": "Fed statement due",
        "related": ["USD"],
        "source": "Bloomberg",
    },
    {"datetime": EPOCH - 7200, "source": "Reuters"},  # no headline → skipped
    {"headline": "No timestamp", "source": "Reuters"},  # no datetime → skipped
    "not a dict",  # skipped
]


class FakeWS:
    """Minimal stand-in for a websockets ClientConnection."""

    def __init__(self, messages: list[str]) -> None:
        self._messages = list(messages)
        self.sent: list[str] = []

    async def send(self, text: str) -> None:
        self.sent.append(text)

    async def recv(self) -> str:
        return self._messages.pop(0)

    def __aiter__(self) -> "FakeWS":
        return self

    async def __anext__(self) -> str:
        if not self._messages:
            raise StopAsyncIteration
        return self._messages.pop(0)


class FakeConnection:
    """Async context manager wrapper so FakeWS works with `async with`."""

    def __init__(self, ws: FakeWS) -> None:
        self._ws = ws

    async def __aenter__(self) -> FakeWS:
        return self._ws

    async def __aexit__(self, *exc_info: object) -> bool:
        return False


def ws_factory(*outcomes: FakeWS | Exception):
    """Build a `ws_connect` stand-in serving one outcome per connection attempt.

    Each outcome is a FakeWS (clean connection) or an Exception (failed
    attempt). A call beyond the provided outcomes fails the test.
    """

    queue = list(outcomes)

    def connect(url: str) -> FakeConnection:
        assert queue, "ws_connect called more times than outcomes provided"
        outcome = queue.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return FakeConnection(outcome)

    return connect


def make_client(make_settings, ws_connect=None, **kwargs) -> FinnhubClient:
    return FinnhubClient(
        make_settings(),
        client=httpx.AsyncClient(),
        ws_connect=ws_connect if ws_connect is not None else ws_factory(),
        initial_backoff=0.0,
        **kwargs,
    )


@respx.mock
@pytest.mark.asyncio
async def test_fetch_news_parses_and_skips_malformed(make_settings) -> None:
    route = respx.get(NEWS_URL, params=NEWS_PARAMS).mock(
        return_value=httpx.Response(200, json=REST_ITEMS)
    )
    finnhub = make_client(make_settings)

    items = await finnhub.fetch_news()
    await finnhub.close()

    assert len(items) == 2
    first, second = items
    assert first.headline == "Gold hits record high"
    assert first.published_at == NEWS_TIME
    assert first.source == "Reuters"
    assert first.related == ["GLD", "AU"]  # CSV string → list
    assert first.url == "https://example.com/news/1"
    assert second.headline == "Fed statement due"
    assert second.related == ["USD"]  # list passes through
    assert second.url is None
    # The token travels as a query param, not a header.
    assert route.calls[0].request.url.params["token"] == "finnhub-key"


@respx.mock
@pytest.mark.asyncio
async def test_fetch_news_http_error_raises(make_settings) -> None:
    respx.get(NEWS_URL, params=NEWS_PARAMS).mock(
        return_value=httpx.Response(401, json={"error": "Invalid token"})
    )
    finnhub = make_client(make_settings)

    with pytest.raises(httpx.HTTPStatusError):
        await finnhub.fetch_news()
    await finnhub.close()


@pytest.mark.asyncio
async def test_stream_yields_news_and_skips_noise(make_settings) -> None:
    fake = FakeWS(
        [
            '{"type":"ping"}',
            "not json",
            NEWS_FRAME,
            '{"type":"trade","data":[{"datetime":1787054400,"headline":"x","source":"y"}]}',
            NEWS_FRAME_2,
        ]
    )
    finnhub = make_client(make_settings, ws_connect=ws_factory(fake))

    items = [item async for item in finnhub.stream_news(max_connections=1)]
    await finnhub.close()

    assert [item.headline for item in items] == [
        "Gold hits record high",
        "ECB holds rates",
    ]
    assert items[0].published_at == NEWS_TIME
    assert items[1].published_at == datetime(2026, 8, 18, 13, 0, tzinfo=timezone.utc)
    # Default subscription is the general news channel; the data-level ping
    # is answered with a pong, not yielded as news.
    assert fake.sent == [
        json.dumps({"type": "subscribe", "symbol": "news"}),
        PONG_FRAME,
    ]


@pytest.mark.asyncio
async def test_stream_subscribes_to_requested_symbols(make_settings) -> None:
    fake = FakeWS([])
    finnhub = make_client(make_settings, ws_connect=ws_factory(fake))

    items = [
        item async for item in finnhub.stream_news(
            ["AAPL", "MSFT"], max_connections=1
        )
    ]
    await finnhub.close()

    assert items == []
    assert fake.sent == [
        json.dumps({"type": "subscribe", "symbol": "AAPL"}),
        json.dumps({"type": "subscribe", "symbol": "MSFT"}),
    ]


@pytest.mark.asyncio
async def test_stream_reconnects_after_clean_close(make_settings) -> None:
    finnhub = make_client(
        make_settings,
        ws_connect=ws_factory(FakeWS([NEWS_FRAME]), FakeWS([NEWS_FRAME_2])),
    )

    items = [item async for item in finnhub.stream_news(max_connections=2)]
    await finnhub.close()

    # The server closing the stream is normal; the feed must come back up.
    assert [item.headline for item in items] == [
        "Gold hits record high",
        "ECB holds rates",
    ]


@pytest.mark.asyncio
async def test_stream_reconnects_after_error(make_settings) -> None:
    finnhub = make_client(
        make_settings,
        ws_connect=ws_factory(
            FakeWS([NEWS_FRAME]),
            websockets.exceptions.ConnectionClosed(None, None),
        ),
    )

    items = [item async for item in finnhub.stream_news(max_connections=2)]
    await finnhub.close()

    # Item from the healthy connection survives; the error is absorbed and
    # the bounded loop ends instead of retrying forever.
    assert [item.headline for item in items] == ["Gold hits record high"]


@respx.mock
@pytest.mark.asyncio
async def test_poll_news_dedups_and_yields_new(make_settings) -> None:
    poll_a = REST_ITEMS[:1]  # "Gold hits record high"
    poll_b = [
        REST_ITEMS[0],
        {"datetime": EPOCH + 7200, "headline": "CPI surprise", "source": "Reuters"},
    ]
    respx.get(NEWS_URL, params=NEWS_PARAMS).mock(
        side_effect=[
            httpx.Response(200, json=poll_a),
            httpx.Response(200, json=poll_b),
        ]
    )
    finnhub = make_client(make_settings)

    items = [
        item
        async for item in finnhub.poll_news(poll_interval=0.0, max_polls=2)
    ]
    await finnhub.close()

    # The repeat of "Gold hits record high" in poll_b is dropped.
    assert [item.headline for item in items] == [
        "Gold hits record high",
        "CPI surprise",
    ]


@respx.mock
@pytest.mark.asyncio
async def test_poll_news_survives_http_error(make_settings) -> None:
    respx.get(NEWS_URL, params=NEWS_PARAMS).mock(
        side_effect=[
            httpx.Response(200, json=REST_ITEMS[:1]),
            httpx.Response(500, json={"error": "upstream"}),
        ]
    )
    finnhub = make_client(make_settings)

    items = [
        item
        async for item in finnhub.poll_news(poll_interval=0.0, max_polls=2)
    ]
    await finnhub.close()

    assert [item.headline for item in items] == ["Gold hits record high"]


@pytest.mark.asyncio
async def test_watch_news_ws_healthy_streams(make_settings) -> None:
    finnhub = make_client(
        make_settings,
        ws_connect=ws_factory(FakeWS([NEWS_FRAME]), FakeWS([NEWS_FRAME_2])),
    )

    items = [item async for item in finnhub.watch_news(max_connections=1)]
    await finnhub.close()

    # Probe forwards its first frame, then the stream continues on WS.
    assert [item.headline for item in items] == [
        "Gold hits record high",
        "ECB holds rates",
    ]


@respx.mock
@pytest.mark.asyncio
async def test_watch_news_falls_back_to_polling(make_settings) -> None:
    # WS news blocked: the connection attempt itself fails → poll instead.
    ws = ws_factory(websockets.exceptions.ConnectionClosed(None, None))
    respx.get(NEWS_URL, params=NEWS_PARAMS).mock(
        return_value=httpx.Response(200, json=REST_ITEMS[:1])
    )
    finnhub = make_client(make_settings, ws_connect=ws)

    items = [item async for item in finnhub.watch_news(max_polls=1)]
    await finnhub.close()

    assert [item.headline for item in items] == ["Gold hits record high"]


@respx.mock
@pytest.mark.asyncio
async def test_watch_news_probe_timeout_falls_back(make_settings) -> None:
    class SilentWS(FakeWS):
        """Connection opens but never answers the probe."""

        async def recv(self) -> str:
            await asyncio.sleep(1)
            return "unreachable"

    respx.get(NEWS_URL, params=NEWS_PARAMS).mock(
        return_value=httpx.Response(200, json=REST_ITEMS[:1])
    )
    finnhub = make_client(
        make_settings,
        ws_connect=ws_factory(SilentWS([])),
        ws_probe_timeout=0.01,
    )

    items = [item async for item in finnhub.watch_news(max_polls=1)]
    await finnhub.close()

    assert [item.headline for item in items] == ["Gold hits record high"]


PING_FRAME = json.dumps({"type": "ping"})
PONG_FRAME = json.dumps({"type": "pong"})


@pytest.mark.asyncio
async def test_stream_answers_data_ping(make_settings) -> None:
    fake = FakeWS([NEWS_FRAME, PING_FRAME, NEWS_FRAME_2])
    finnhub = make_client(make_settings, ws_connect=ws_factory(fake))

    items = [item async for item in finnhub.stream_news(max_connections=1)]
    await finnhub.close()

    # The ping is answered, not yielded; news on both sides survives.
    assert [item.headline for item in items] == [
        "Gold hits record high",
        "ECB holds rates",
    ]
    assert fake.sent == [
        json.dumps({"type": "subscribe", "symbol": "news"}),
        PONG_FRAME,
    ]


@respx.mock
@pytest.mark.asyncio
async def test_watch_news_probe_skips_ping_then_news(make_settings) -> None:
    # A ping is transport noise, not news — the probe answers it and keeps
    # waiting for a news-carrying frame before declaring WS healthy.
    finnhub = make_client(
        make_settings,
        ws_connect=ws_factory(
            FakeWS([PING_FRAME, NEWS_FRAME]),  # probe connection
            FakeWS([NEWS_FRAME_2]),  # stream connection
        ),
    )

    items = [item async for item in finnhub.watch_news(max_connections=1)]
    await finnhub.close()

    assert [item.headline for item in items] == [
        "Gold hits record high",
        "ECB holds rates",
    ]


@respx.mock
@pytest.mark.asyncio
async def test_watch_news_ping_only_probe_falls_back(make_settings) -> None:
    class PingOnlyWS(FakeWS):
        """Sends one data-level ping, then stays silent — news blocked."""

        def __init__(self) -> None:
            super().__init__([])
            self._pings = [PING_FRAME]

        async def recv(self) -> str:
            if self._pings:
                return self._pings.pop(0)
            await asyncio.sleep(1)
            return "unreachable"

    ws = ws_factory(PingOnlyWS())
    respx.get(NEWS_URL, params=NEWS_PARAMS).mock(
        return_value=httpx.Response(200, json=REST_ITEMS[:1])
    )
    finnhub = make_client(make_settings, ws_connect=ws, ws_probe_timeout=0.01)

    items = [item async for item in finnhub.watch_news(max_polls=1)]
    await finnhub.close()

    # Pings alone do not prove news flows → REST polling takes over.
    assert [item.headline for item in items] == ["Gold hits record high"]


@respx.mock
@pytest.mark.asyncio
async def test_poll_news_survives_non_json_response(make_settings) -> None:
    respx.get(NEWS_URL, params=NEWS_PARAMS).mock(
        side_effect=[
            httpx.Response(200, content=b"<html>gateway error</html>"),
            httpx.Response(200, json=REST_ITEMS[:1]),
        ]
    )
    finnhub = make_client(make_settings)

    items = [
        item
        async for item in finnhub.poll_news(poll_interval=0.0, max_polls=2)
    ]
    await finnhub.close()

    assert [item.headline for item in items] == ["Gold hits record high"]
