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

from motor.motor_asyncio import (
    AsyncIOMotorClient,
    AsyncIOMotorCollection,
    AsyncIOMotorDatabase,
)

from app.config import Settings

DAILY_CONTEXT = "daily_context"
ACCOUNT_STATE = "account_state"
TRADE_LOGS = "trade_logs"
SIGNALS = "signals"


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

    async def insert_signal(self, document: dict) -> str:
        """Append one signal to the signal record; returns its `_id`."""
        result = await self.signals.insert_one(document)
        return str(result.inserted_id)

    async def close(self) -> None:
        """Shut the underlying client down (engine lifespan teardown)."""
        await self._client.close()
