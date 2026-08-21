"""Async MongoDB access layer backed by Motor.

Document contracts (Plan.md pipeline):

- `daily_context`  — one doc per UTC trading day: calendar events,
  blackout windows, macro bias. Keyed by `day` (YYYY-MM-DD).
- `account_state`  — one doc per OANDA account: day-start balance and
  realized day P&L for the circuit breaker. Keyed by `account_id`.
- `trade_logs`     — append-only audit trail: DeepSeek decisions and
  order results.
- `signals`        — append-only record of every signal the strategies emit.

State collections (`daily_context`, `account_state`) are upserted — each key
holds exactly one current document. Log collections are insert-only, so the
audit trail never mutates.

Datetimes must be timezone-aware (enforced by `models.schemas.UTCModel`):
naive datetimes would be stored with an assumed UTC zone and silently
corrupt blackout-window math.
"""

from datetime import datetime, timezone

from motor.motor_asyncio import (
    AsyncIOMotorClient,
    AsyncIOMotorCollection,
    AsyncIOMotorDatabase,
)

from app.config import Settings
from app.models.schemas import OrderResult, Signal, TradeDecision

DAILY_CONTEXT = "daily_context"
ACCOUNT_STATE = "account_state"
TRADE_LOGS = "trade_logs"
SIGNALS = "signals"


def _signal_context(signal: Signal) -> dict:
    """The signal fields an audit entry carries, shared by decision and order logs."""
    return {
        "strategy": signal.strategy,
        "instrument": signal.instrument,
        "side": signal.side,
        "entry": signal.entry,
        "stop_loss": signal.stop_loss,
        "take_profit": signal.take_profit,
        "atr": signal.atr,
        "reason": signal.reason,
        "pending_ai_veto": signal.pending_ai_veto,
        "signal_time": signal.timestamp,
    }


class MongoStore:
    """Motor client for the configured database, with typed accessors.

    `client` is an injection seam for tests; production code passes nothing
    and gets a real client bound to `settings.mongodb_uri`.
    """

    def __init__(
        self, settings: Settings, client: AsyncIOMotorClient | None = None
    ) -> None:
        self._client = (
            client if client is not None else AsyncIOMotorClient(settings.mongodb_uri)
        )
        self._db: AsyncIOMotorDatabase = self._client[settings.mongodb_db]

    @property
    def daily_context(self) -> AsyncIOMotorCollection:
        """One document per UTC trading day, keyed by `day`."""
        return self._db[DAILY_CONTEXT]

    @property
    def account_state(self) -> AsyncIOMotorCollection:
        """One document per account, keyed by `account_id`."""
        return self._db[ACCOUNT_STATE]

    @property
    def trade_logs(self) -> AsyncIOMotorCollection:
        """Append-only audit trail of decisions and order results."""
        return self._db[TRADE_LOGS]

    @property
    def signals(self) -> AsyncIOMotorCollection:
        """Append-only record of emitted signals."""
        return self._db[SIGNALS]

    async def get_daily_context(self, day: str) -> dict | None:
        """Return the daily context document for one UTC trading day."""
        return await self.daily_context.find_one({"day": day})

    async def upsert_daily_context(self, day: str, document: dict) -> None:
        """Write (or replace) the daily context document for one trading day."""
        await self.daily_context.replace_one({"day": day}, document, upsert=True)

    async def get_account_state(self, account_id: str) -> dict | None:
        """Return the current account state document for one account."""
        return await self.account_state.find_one({"account_id": account_id})

    async def upsert_account_state(self, account_id: str, document: dict) -> None:
        """Write (or replace) the account state document for one account."""
        await self.account_state.replace_one(
            {"account_id": account_id}, document, upsert=True
        )

    async def insert_trade_log(self, document: dict) -> str:
        """Append one entry to the trade audit trail; returns its `_id`."""
        result = await self.trade_logs.insert_one(document)
        return str(result.inserted_id)

    async def log_decision(self, signal: Signal, decision: TradeDecision) -> str:
        """Append one DeepSeek veto verdict with its signal context (Task 13).

        Every verdict — approved or vetoed — is recorded so the audit trail
        shows what the gate was shown and what it concluded. The entry is
        stamped with the decision time, not the signal time.
        """
        document = {
            "kind": "decision",
            **_signal_context(signal),
            "execute": decision.execute,
            "confidence": decision.confidence,
            "decision_reason": decision.reason,
            "timestamp": datetime.now(timezone.utc),
        }
        return await self.insert_trade_log(document)

    async def log_order(
        self,
        order: OrderResult,
        signal: Signal,
        decision: TradeDecision,
    ) -> str:
        """Append one placed order, linked to the signal and verdict behind it."""
        document = {
            "kind": "order",
            **_signal_context(signal),
            "order_id": order.order_id,
            "status": order.status,
            "order_instrument": order.instrument,
            "units": order.units,
            "price": order.price,
            "created_at": order.created_at,
            "decision_execute": decision.execute,
            "decision_confidence": decision.confidence,
            "decision_reason": decision.reason,
            "timestamp": datetime.now(timezone.utc),
        }
        return await self.insert_trade_log(document)

    async def insert_signal(self, document: dict) -> str:
        """Append one signal to the signal record; returns its `_id`."""
        result = await self.signals.insert_one(document)
        return str(result.inserted_id)

    async def close(self) -> None:
        """Shut the underlying client down (engine lifespan teardown)."""
        await self._client.close()
