"""Async Finnhub client — news WebSocket with a REST polling fallback.

Contract (Plan.md pipeline stage 1; tasks 6, 10, 15):

- `fetch_news`  — one REST poll of `/news` (general market news)
- `stream_news` — WS feed: subscribe, parse `{"type": "news", ...}` frames,
                  answer data-level pings, auto-reconnect with backoff
- `poll_news`   — infinite REST polling loop, deduped by (headline, time)
- `watch_news`  — the engine's entry point: prefer WS, and if the probe
                  connection fails or delivers no news message (the free tier
                  blocks WS news), fall back to REST polling instead of
                  retrying forever

Finnhub sends `{"type": "ping"}` JSON frames on idle connections and drops
clients that do not answer. The websockets library's automatic pong only
covers protocol-level (RFC 6455) pings, so this layer answers data-level
pings itself.

Relevance filtering (does a headline touch a traded instrument?) is Stage 2's
job (`app/strategy/filters.py`, Task 10) — this layer parses and forwards
every item it sees.

`client` / `ws_connect` are injection seams for tests (respx mocks the REST
transport; `ws_connect` supplies fake connections for the WS side).
Production code passes nothing and gets real clients.
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable
from datetime import datetime, timezone
from typing import Any

import httpx
import websockets
import websockets.exceptions

from app.config import Settings
from app.models.schemas import NewsItem

logger = logging.getLogger(__name__)

# The WS general-news channel. The engine may pass specific tickers instead
# (`stream_news(symbols=["AAPL", ...])`) to subscribe to company news.
DEFAULT_SUBSCRIPTIONS = ("news",)

FINNHUB_REST_URL = "https://finnhub.io/api/v1"
FINNHUB_WS_URL = "wss://ws.finnhub.io"


class FinnhubClient:
    """Async Finnhub news client: WS stream + REST polling fallback."""

    def __init__(
        self,
        settings: Settings,
        client: httpx.AsyncClient | None = None,
        ws_connect: Callable[[str], Any] | None = None,
        initial_backoff: float = 1.0,
        max_backoff: float = 30.0,
        ws_probe_timeout: float = 10.0,
    ) -> None:
        self._settings = settings
        if client is None:
            timeout = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)
            client = httpx.AsyncClient(timeout=timeout)
        self._client = client
        self._ws_connect = ws_connect or websockets.connect
        self._initial_backoff = initial_backoff
        self._max_backoff = max_backoff
        self._ws_probe_timeout = ws_probe_timeout

    @property
    def _ws_url(self) -> str:
        """WS endpoint — the token travels in the query string, not a header."""
        return f"{FINNHUB_WS_URL}?token={self._settings.finnhub_api_key}"

    async def fetch_news(self, category: str = "general") -> list[NewsItem]:
        """One REST poll of Finnhub `/news`, parsed in source order.

        Raises `httpx.HTTPStatusError` on auth/HTTP failures — the polling
        loop absorbs it; one-shot callers see it.
        """
        url = f"{FINNHUB_REST_URL}/news"
        params = {"category": category, "token": self._settings.finnhub_api_key}
        response = await self._client.get(url, params=params)
        response.raise_for_status()
        items: list[NewsItem] = []
        for raw in response.json():
            item = self._parse_news_item(raw)
            if item is not None:
                items.append(item)
        return items

    async def stream_news(
        self,
        symbols: list[str] | None = None,
        max_connections: int | None = None,
    ) -> AsyncIterator[NewsItem]:
        """Yield live news from the WS feed, reconnecting with backoff.

        `max_connections=None` (production) reconnects forever; tests pass a
        bound so the stream ends after that many connection attempts.
        """
        subscriptions = self._subscriptions(symbols)
        attempts = 0
        backoff = self._initial_backoff
        while True:
            try:
                attempts += 1
                async with self._ws_connect(self._ws_url) as ws:
                    await self._subscribe(ws, subscriptions)
                    async for message in ws:
                        if await self._reply_to_ping(ws, message):
                            continue
                        for item in self._parse_ws_message(message):
                            yield item
                backoff = self._initial_backoff  # healthy close → restart backoff
            except (OSError, websockets.exceptions.WebSocketException) as exc:
                logger.warning("Finnhub news WS down (%s), reconnecting", exc)
            if max_connections is not None and attempts >= max_connections:
                return
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self._max_backoff)

    async def poll_news(
        self,
        poll_interval: float = 60.0,
        max_polls: int | None = None,
    ) -> AsyncIterator[NewsItem]:
        """Yield *new* items from repeated REST polls (the WS fallback).

        Finnhub returns the same recent headlines on every poll, so each
        yielded item is deduped by (headline, published_at). Transient HTTP
        failures are logged and skipped; the loop keeps polling.
        `max_polls=None` (production) polls forever; tests pass a bound.
        """
        seen: set[tuple[str, datetime]] = set()
        polls = 0
        while True:
            polls += 1
            try:
                items = await self.fetch_news()
            except (httpx.HTTPError, ValueError) as exc:
                # ValueError covers non-JSON 200s (gateway HTML) — an
                # infinite loop must survive any malformed response.
                logger.warning("Finnhub news poll failed (%s), retrying", exc)
                items = []
            for item in items:
                key = (item.headline, item.published_at)
                if key in seen:
                    continue
                seen.add(key)
                yield item
            if max_polls is not None and polls >= max_polls:
                return
            await asyncio.sleep(poll_interval)

    async def watch_news(
        self,
        symbols: list[str] | None = None,
        poll_interval: float = 60.0,
        max_connections: int | None = None,
        max_polls: int | None = None,
    ) -> AsyncIterator[NewsItem]:
        """One continuous news feed, preferring WS.

        Probes the WS with a single connection first: if it cannot connect
        or delivers nothing but pings inside the probe window (the free tier
        blocks WS news), fall back to REST polling instead of reconnecting
        forever. A WS that delivers a news message stays on WS.
        """
        try:
            first_items = await self._probe_ws(symbols)
        except (OSError, websockets.exceptions.WebSocketException, TimeoutError) as exc:
            logger.warning(
                "Finnhub WS news unavailable (%s); falling back to REST polling", exc
            )
            async for item in self.poll_news(poll_interval, max_polls):
                yield item
            return
        for item in first_items:
            yield item
        async for item in self.stream_news(symbols, max_connections):
            yield item

    async def _probe_ws(self, symbols: list[str] | None) -> list[NewsItem]:
        """Open one WS connection and wait for the first news-carrying message.

        Data-level pings are answered and skipped — they prove the transport
        is alive, not that news flows. A channel that sends only pings within
        the window (news silently blocked on the free tier) raises
        TimeoutError; `watch_news` treats any raise as "WS news blocked" and
        falls back to polling.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._ws_probe_timeout
        async with self._ws_connect(self._ws_url) as ws:
            await self._subscribe(ws, self._subscriptions(symbols))
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise TimeoutError("no news message within the probe window")
                message = await asyncio.wait_for(ws.recv(), timeout=remaining)
                if await self._reply_to_ping(ws, message):
                    continue
                return self._parse_ws_message(message)

    @staticmethod
    def _subscriptions(symbols: list[str] | None) -> list[str]:
        """Symbols to subscribe to; None means the general news channel."""
        return list(symbols) if symbols else list(DEFAULT_SUBSCRIPTIONS)

    @staticmethod
    async def _subscribe(ws: Any, symbols: list[str]) -> None:
        for symbol in symbols:
            await ws.send(json.dumps({"type": "subscribe", "symbol": symbol}))

    @staticmethod
    async def _reply_to_ping(ws: Any, message: str | bytes) -> bool:
        """Answer a Finnhub data-level ping; True if `message` was one."""
        if isinstance(message, bytes):
            message = message.decode(errors="replace")
        try:
            payload = json.loads(message)
        except (json.JSONDecodeError, TypeError):
            return False
        if not isinstance(payload, dict) or payload.get("type") != "ping":
            return False
        await ws.send('{"type": "pong"}')
        return True

    @staticmethod
    def _parse_ws_message(message: str | bytes) -> list[NewsItem]:
        """Parse one WS frame `{"type": "news", "data": [...]}` into items."""
        if isinstance(message, bytes):
            message = message.decode(errors="replace")
        try:
            payload = json.loads(message)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Finnhub WS: unparseable message %r", message)
            return []
        if not isinstance(payload, dict) or payload.get("type") != "news":
            return []  # pings and unknown frames carry no news
        data = payload.get("data")
        if not isinstance(data, list):
            return []
        items: list[NewsItem] = []
        for raw in data:
            item = FinnhubClient._parse_news_item(raw)
            if item is not None:
                items.append(item)
        return items

    @staticmethod
    def _parse_news_item(raw: object) -> NewsItem | None:
        """Map one Finnhub news dict onto NewsItem; unusable input → None."""
        if not isinstance(raw, dict):
            return None
        try:
            published = datetime.fromtimestamp(raw["datetime"], tz=timezone.utc)
            headline = raw.get("headline")
            if not isinstance(headline, str) or not headline.strip():
                return None
            related = raw.get("related") or []
            if isinstance(related, str):
                related = [part.strip() for part in related.split(",") if part.strip()]
            if not isinstance(related, list):
                related = []
            return NewsItem(
                headline=headline,
                published_at=published,
                source=raw.get("source") or "",
                url=raw.get("url"),
                summary=raw.get("summary"),
                related=related,
            )
        except (KeyError, TypeError, ValueError, OverflowError, OSError) as exc:
            logger.warning("Finnhub news: unusable item (%s)", exc)
            return None

    async def close(self) -> None:
        """Shut the REST client down (engine lifespan teardown)."""
        await self._client.aclose()
