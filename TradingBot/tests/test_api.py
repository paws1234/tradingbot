"""Tests for app.main + app.api.routes — HTTP endpoints and lifespan wiring (task 16).

Drives the app through FastAPI's TestClient with fake engine, scheduler, and
store — no network, MongoDB, or real clients. Verification points:

- ``/health`` returns 200 for UptimeRobot, no external calls.
- ``/status`` reports engine runtime, account identity, and Mongo state;
  Mongo-backed fields degrade to ``null`` instead of 500ing.
- The lifespan starts the engine + daily scheduler on boot and stops them
  (plus closing the store) on shutdown — graceful cancel.
"""

from collections.abc import Callable

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app as production_app
from app.main import create_app

ACCOUNT_ID = "001-001-1234567-001"


class FakeStore:
    """Minimal MongoStore stand-in: fixed docs, records close()."""

    def __init__(self) -> None:
        self.account_state = {
            "account_id": ACCOUNT_ID,
            "day_start_balance": 10000.0,
            "realized_pnl": 0.0,
            "trading_halted": False,
        }
        self.daily_context = {"day": "2026-08-20", "blackouts": []}
        self.closed = False

    async def get_account_state(self, account_id: str) -> dict | None:
        return self.account_state

    async def get_daily_context(self, day: str) -> dict | None:
        return self.daily_context

    async def close(self) -> None:
        self.closed = True


class BrokenStore(FakeStore):
    """A store whose reads raise — /status must degrade, not 500."""

    async def get_account_state(self, account_id: str) -> dict | None:
        raise RuntimeError("mongo down")

    async def get_daily_context(self, day: str) -> dict | None:
        raise RuntimeError("mongo down")


class FakeScheduler:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class FakeEngine:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    def status(self) -> dict:
        return {
            "started": self.started,
            "instruments": ["XAU_USD"],
            "strategies": ["asia_sweep", "atr_breakout"],
            "pending_signals": 2,
            "history_bars": {"XAU_USD": 100},
        }


def make_app(
    make_settings: Callable[..., Settings],
    store: FakeStore | None = None,
    scheduler: FakeScheduler | None = None,
    engine: FakeEngine | None = None,
) -> TestClient:
    """An app wired with fakes; the lifespan builds nothing real."""
    return create_app(
        make_settings(),
        store=store or FakeStore(),
        scheduler=scheduler or FakeScheduler(),
        engine=engine or FakeEngine(),
    )


def test_health_returns_ok(make_settings: Callable[..., Settings]) -> None:
    with TestClient(make_app(make_settings)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_lifespan_starts_engine_and_scheduler(
    make_settings: Callable[..., Settings],
) -> None:
    engine = FakeEngine()
    scheduler = FakeScheduler()

    with TestClient(make_app(make_settings, engine=engine, scheduler=scheduler)):
        pass

    assert engine.started is True
    assert scheduler.started is True


def test_lifespan_stops_and_closes_on_exit(
    make_settings: Callable[..., Settings],
) -> None:
    engine = FakeEngine()
    scheduler = FakeScheduler()
    store = FakeStore()

    with TestClient(make_app(make_settings, store=store, engine=engine, scheduler=scheduler)):
        pass

    assert engine.stopped is True
    assert scheduler.stopped is True
    assert store.closed is True


def test_status_reports_engine_runtime_and_mongo_state(
    make_settings: Callable[..., Settings],
) -> None:
    with TestClient(make_app(make_settings)) as client:
        response = client.get("/status")

    assert response.status_code == 200
    body = response.json()
    assert body["app"] == "trading-bot"
    assert body["status"] == "running"
    assert body["account"] == {"account_id": ACCOUNT_ID, "account_type": "practice"}
    assert body["engine"]["started"] is True
    assert body["engine"]["pending_signals"] == 2
    assert body["engine"]["history_bars"] == {"XAU_USD": 100}
    assert body["account_state"]["day_start_balance"] == 10000.0
    assert body["daily_context"]["day"] == "2026-08-20"
    assert body["started_at"] is not None
    assert body["uptime_seconds"] >= 0


def test_status_degrades_to_null_when_mongo_unreachable(
    make_settings: Callable[..., Settings],
) -> None:
    with TestClient(make_app(make_settings, store=BrokenStore())) as client:
        response = client.get("/status")

    assert response.status_code == 200
    body = response.json()
    assert body["account_state"] is None
    assert body["daily_context"] is None


def test_status_works_before_lifespan(
    make_settings: Callable[..., Settings],
) -> None:
    # TestClient without the context manager never runs the lifespan, so the
    # engine hasn't started and uptime hasn't begun — /status still answers.
    client = TestClient(make_app(make_settings))
    try:
        response = client.get("/status")
    finally:
        client.close()

    assert response.status_code == 200
    body = response.json()
    assert body["engine"]["started"] is False
    assert "started_at" not in body
    assert "uptime_seconds" not in body


def test_production_app_health_route_is_importable() -> None:
    # `app.main:app` is the uvicorn target; it must construct and serve /health
    # even before the lifespan builds any real clients.
    client = TestClient(production_app)
    try:
        response = client.get("/health")
    finally:
        client.close()

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
