"""FastAPI application entrypoint — wires real clients and the trading engine.

Production runs ``uvicorn app.main:app``; the module-level ``app`` builds from
the environment (fail-fast on missing credentials). Tests call
``create_app(settings, ...)`` with fake clients, scheduler, and engine so the
lifespan can be exercised with no network or database access.

Lifespan ownership (matching ``app/core/engine.py``: "the engine does not own
client lifecycle — app/main.py builds and closes them"):

- startup builds whatever wasn't injected, then starts the daily 00:00 UTC
  scheduler cron and the engine (which seeds today's context and launches the
  per-instrument stream tasks).
- shutdown cancels the stream tasks, stops the scheduler, and closes every
  client it wired.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI

from app.api.routes import router
from app.config import Settings, get_settings
from app.core.engine import TradingEngine
from app.core.scheduler import DailyContextScheduler
from app.data.deepseek import DeepSeekClient
from app.data.finnhub import FinnhubClient
from app.data.forexfactory import ForexFactoryClient
from app.data.mongo import MongoStore
from app.data.oanda import OandaClient

logger = logging.getLogger(__name__)


async def _close(component: object) -> None:
    """Close a wired client if it exposes one; fakes may not (getattr guard)."""
    close = getattr(component, "close", None)
    if close is not None:
        await close()


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start the scheduler cron + engine on boot; tear down cleanly on exit."""
    settings: Settings = app.state.settings
    store = app.state.store or MongoStore(settings)
    scheduler = app.state.scheduler
    engine = app.state.engine
    pipeline: tuple[object, ...] = ()

    if engine is None:
        oanda = app.state.oanda or OandaClient(settings)
        deepseek = app.state.deepseek or DeepSeekClient(settings)
        finnhub = app.state.finnhub or FinnhubClient(settings)
        forex = app.state.forex or ForexFactoryClient(settings)
        scheduler = scheduler or DailyContextScheduler(store, forex, oanda, settings)
        engine = TradingEngine(settings, store, oanda, deepseek, finnhub, scheduler)
        pipeline = (oanda, deepseek, finnhub, forex)

    app.state.store = store
    app.state.scheduler = scheduler
    app.state.engine = engine
    app.state.started_at = datetime.now(timezone.utc)

    if scheduler is not None:
        scheduler.start()
    await engine.start()
    try:
        yield
    finally:
        await engine.stop()
        if scheduler is not None:
            scheduler.stop()
        await _close(store)
        for client in pipeline:
            await _close(client)


def create_app(
    settings: Settings | None = None,
    *,
    store: MongoStore | None = None,
    oanda: OandaClient | None = None,
    deepseek: DeepSeekClient | None = None,
    finnhub: FinnhubClient | None = None,
    forex: ForexFactoryClient | None = None,
    scheduler: DailyContextScheduler | None = None,
    engine: TradingEngine | None = None,
) -> FastAPI:
    """Build the FastAPI app; pass fakes for tests, build real clients otherwise.

    ``engine`` is the switch: when injected, the lifespan starts/stops it as-is
    (the caller owns its internal clients) and skips building the pipeline.
    Otherwise the lifespan constructs the full pipeline from ``settings``.
    """
    application = FastAPI(title="TradingBot", version="0.1.0", lifespan=_lifespan)
    application.state.settings = settings or get_settings()
    application.state.store = store
    application.state.oanda = oanda
    application.state.deepseek = deepseek
    application.state.finnhub = finnhub
    application.state.forex = forex
    application.state.scheduler = scheduler
    application.state.engine = engine
    application.include_router(router)
    return application


app = create_app()
