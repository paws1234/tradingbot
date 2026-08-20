"""Async OANDA v20 client — REST data plus the NDJSON pricing stream.

Contract (Plan.md pipeline stage 1; tasks 5, 11, 15):

- `get_candles`         — closed bars for strategy input / engine backfill
- `get_account_summary` — balance and NAV for sizing and the circuit breaker
- `place_market_order`  — transports a pre-built MARKET order spec
                          (`app/strategy/sizing.py` builds the spec in Task 11)
- `stream_prices`       — live bid/ask feed: `aiter_lines()` over HTTP chunked
                          NDJSON, `PRICE` vs `HEARTBEAT`, auto-reconnect with
                          exponential backoff

OANDA v20 streams over HTTP chunked NDJSON, not WebSocket (Plan.md risks).

`client` is an injection seam for tests (respx mocks the transport);
production code passes nothing and gets a real `httpx.AsyncClient`.
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from datetime import datetime

import httpx

from app.config import Settings
from app.models.schemas import Candle, OrderResult, PriceTick

logger = logging.getLogger(__name__)

PRICE = "PRICE"
HEARTBEAT = "HEARTBEAT"


class OandaClient:
    """Async OANDA v20 client: REST endpoints + infinite pricing stream."""

    def __init__(
        self,
        settings: Settings,
        client: httpx.AsyncClient | None = None,
        initial_backoff: float = 1.0,
        max_backoff: float = 30.0,
    ) -> None:
        self._settings = settings
        if client is None:
            # read=60 — the stream emits a HEARTBEAT at least every ~5 s, so
            # a minute without bytes means the connection is dead: the read
            # timeout then forces a reconnect. REST responses land well under.
            timeout = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)
            client = httpx.AsyncClient(timeout=timeout)
        self._client = client
        self._initial_backoff = initial_backoff
        self._max_backoff = max_backoff

    @property
    def _headers(self) -> dict[str, str]:
        """Bearer auth required by every v20 endpoint."""
        return {"Authorization": f"Bearer {self._settings.oanda_api_key}"}

    def _account_url(self, path: str) -> str:
        base = self._settings.oanda_rest_url
        account = self._settings.oanda_account_id
        return f"{base}/v3/accounts/{account}{path}"

    async def get_candles(
        self,
        instrument: str,
        granularity: str | None = None,
        count: int = 500,
    ) -> list[Candle]:
        """Return the most recent *closed* candles, oldest first.

        OANDA returns candles newest-first and the last bar is usually still
        forming. Strategies only consume closed bars (strategy.md §1.2), so
        both are normalized here. 500 M15 bars ≈ five trading days — covers
        the EMA(200) warm-up.
        """
        url = f"{self._settings.oanda_rest_url}/v3/instruments/{instrument}/candles"
        params = {
            "granularity": granularity or self._settings.granularity,
            "count": count,
            "price": "M",  # mid candles — what the strategies trade on
        }
        response = await self._client.get(url, params=params, headers=self._headers)
        response.raise_for_status()
        candles: list[Candle] = []
        for raw in response.json().get("candles", []):
            if not raw.get("complete"):
                continue
            mid = raw["mid"]
            candles.append(
                Candle(
                    time=datetime.fromisoformat(raw["time"]),
                    open=float(mid["o"]),
                    high=float(mid["h"]),
                    low=float(mid["l"]),
                    close=float(mid["c"]),
                    volume=raw.get("volume"),
                )
            )
        candles.reverse()
        return candles

    async def get_account_summary(self) -> dict:
        """Return the raw `account` object (balance, NAV, currency, …).

        OANDA v20 sends numeric fields as strings ("10000.0000") — callers
        that do math (sizing, breaker) cast with `float()`.
        """
        response = await self._client.get(
            self._account_url("/summary"), headers=self._headers
        )
        response.raise_for_status()
        return response.json()["account"]

    async def place_market_order(self, order_spec: dict) -> OrderResult:
        """Post an OANDA order spec (`{"order": {...}}`), return the outcome."""
        response = await self._client.post(
            self._account_url("/orders"), json=order_spec, headers=self._headers
        )
        response.raise_for_status()
        return self._parse_order_result(response.json())

    @staticmethod
    def _parse_order_result(payload: dict) -> OrderResult:
        filled = payload.get("orderFillTransaction")
        transaction = filled or payload.get("orderCreateTransaction")
        if transaction is None:
            raise ValueError("order response contains no order transaction")
        price = transaction.get("price")
        return OrderResult(
            order_id=str(transaction.get("orderID", transaction["id"])),
            status="FILLED" if filled is not None else "PENDING",
            instrument=transaction["instrument"],
            units=str(transaction.get("units", "0")),
            price=float(price) if price is not None else None,
            created_at=datetime.fromisoformat(transaction["time"]),
        )

    async def stream_prices(
        self,
        instruments: list[str],
        max_connections: int | None = None,
    ) -> AsyncIterator[PriceTick]:
        """Yield live bid/ask ticks, reconnecting with exponential backoff.

        `max_connections=None` (production) reconnects forever; tests pass a
        bound so the stream ends after that many connection attempts.
        """
        # The pricing stream lives on the dedicated streaming host
        # (stream-fx{practice,trade}.oanda.com), not the REST host.
        base = self._settings.oanda_stream_url
        account = self._settings.oanda_account_id
        url = f"{base}/v3/accounts/{account}/pricing/stream"
        params = {"instruments": ",".join(instruments)}
        attempts = 0
        backoff = self._initial_backoff
        while True:
            try:
                attempts += 1
                async with self._client.stream(
                    "GET", url, params=params, headers=self._headers
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        tick = self._parse_stream_line(line)
                        if tick is not None:
                            yield tick
                backoff = self._initial_backoff  # healthy close → restart backoff
            except httpx.HTTPError as exc:
                logger.warning("pricing stream down (%s), reconnecting", exc)
            if max_connections is not None and attempts >= max_connections:
                return
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self._max_backoff)

    @staticmethod
    def _parse_stream_line(line: str) -> PriceTick | None:
        """Parse one NDJSON line; HEARTBEATs and malformed lines yield None."""
        line = line.strip()
        if not line:
            return None
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("pricing stream: unparseable line %r", line)
            return None
        if payload.get("type") == HEARTBEAT:
            return None
        if payload.get("type") != PRICE:
            logger.warning(
                "pricing stream: unexpected message type %r", payload.get("type")
            )
            return None
        try:
            return PriceTick(
                instrument=payload["instrument"],
                time=datetime.fromisoformat(payload["time"]),
                bid=float(payload["bids"][0]["price"]),
                ask=float(payload["asks"][0]["price"]),
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            logger.warning("pricing stream: malformed PRICE message: %s", exc)
            return None

    async def close(self) -> None:
        """Shut the underlying client down (engine lifespan teardown)."""
        await self._client.aclose()
