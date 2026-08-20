"""FastAPI routes: /health liveness probe and /status operational snapshot.

``/health`` is the UptimeRobot ping target — cheap, no external calls, 200
while the process lives. ``/status`` is the richer operational view: engine
runtime plus the latest Mongo state, with every Mongo-backed field degrading
to ``null`` rather than 500ing so the endpoint stays up to report a dead
engine or a downed database.
"""

from collections.abc import Awaitable, Callable
from datetime import datetime, timezone

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    """Liveness probe for UptimeRobot — 200 while the process is alive."""
    return {"status": "ok"}


@router.get("/status")
async def status(request: Request) -> dict:
    """Operational snapshot: engine runtime plus latest Mongo state (degraded)."""
    app = request.app
    payload: dict = {"app": "trading-bot", "status": "running"}

    started_at = getattr(app.state, "started_at", None)
    if started_at is not None:
        payload["started_at"] = started_at
        payload["uptime_seconds"] = round(
            (datetime.now(timezone.utc) - started_at).total_seconds(), 1
        )

    settings = getattr(app.state, "settings", None)
    if settings is not None:
        payload["account"] = {
            "account_id": settings.oanda_account_id,
            "account_type": settings.account_type,
        }

    engine = getattr(app.state, "engine", None)
    if engine is not None:
        payload["engine"] = engine.status()

    store = getattr(app.state, "store", None)
    if store is not None and settings is not None:
        payload["account_state"] = await _mongo_state(
            store.get_account_state, settings.oanda_account_id
        )
        payload["daily_context"] = await _mongo_state(
            store.get_daily_context, datetime.now(timezone.utc).date().isoformat()
        )
    return payload


async def _mongo_state(
    call: Callable[..., Awaitable[dict | None]], *args: object
) -> dict | None:
    """Best-effort Mongo read for /status; ``None`` on failure or absence."""
    try:
        document = await call(*args)
    except Exception:  # noqa: BLE001 — /status must survive a downed DB
        return None
    if document is None:
        return None
    # Mongo ObjectIds aren't JSON-serializable by FastAPI; drop the key.
    document.pop("_id", None)
    return document
