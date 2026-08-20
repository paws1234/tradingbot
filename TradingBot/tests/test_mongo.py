"""Tests for app.data.mongo — collection wiring and CRUD helpers.

Uses a minimal in-memory fake of the Motor client (collections → dicts), so
no live MongoDB is needed. The fake records every call, which lets tests
assert both the effect (document round-trips) and the contract (exact
filter / upsert flag) the engine relies on.
"""

from typing import Any

import pytest

from app.config import Settings
from app.data.mongo import (
    ACCOUNT_STATE,
    DAILY_CONTEXT,
    SIGNALS,
    TRADE_LOGS,
    MongoStore,
)


def make_settings(**overrides: object) -> Settings:
    values = {
        "oanda_api_key": "oanda-key",
        "oanda_account_id": "001-001-1234567-001",
        "deepseek_api_key": "deepseek-key",
        "mongodb_uri": "mongodb://localhost:27017",
        "finnhub_api_key": "finnhub-key",
        **overrides,
    }
    return Settings(**values)


class FakeResult:
    def __init__(self, inserted_id: Any = None) -> None:
        self.inserted_id = inserted_id


class FakeCollection:
    def __init__(self) -> None:
        self.replaced: list[tuple[dict, dict, bool]] = []
        self.inserted: list[dict] = []
        self.docs: dict[tuple[tuple[str, Any], ...], dict] = {}

    @staticmethod
    def _key(filter: dict) -> tuple[tuple[str, Any], ...]:
        return tuple(sorted(filter.items()))

    async def replace_one(self, filter, replacement, upsert=False) -> FakeResult:
        self.replaced.append((filter, replacement, upsert))
        if upsert:
            self.docs[self._key(filter)] = replacement
        return FakeResult()

    async def insert_one(self, document) -> FakeResult:
        self.inserted.append(document)
        return FakeResult(inserted_id=len(self.inserted))

    async def find_one(self, filter):
        return self.docs.get(self._key(filter))


class FakeDatabase:
    def __init__(self) -> None:
        self.collections: dict[str, FakeCollection] = {}

    def __getitem__(self, name: str) -> FakeCollection:
        return self.collections.setdefault(name, FakeCollection())


class FakeClient:
    def __init__(self) -> None:
        self.databases: dict[str, FakeDatabase] = {}
        self.closed = False

    def __getitem__(self, name: str) -> FakeDatabase:
        return self.databases.setdefault(name, FakeDatabase())

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_client() -> FakeClient:
    return FakeClient()


@pytest.fixture
def store(fake_client: FakeClient) -> MongoStore:
    return MongoStore(make_settings(), client=fake_client)


def collection_names(store: MongoStore) -> set[str]:
    return set(store._db.collections)


@pytest.mark.asyncio
async def test_collections_bound_to_configured_db(
    fake_client: FakeClient, store: MongoStore
) -> None:
    # Default mongodb_db is "tradingbot"; all four collections live under it.
    assert set(fake_client.databases) == {"tradingbot"}

    # Motor creates collection handles lazily, like our fake: only after the
    # accessors are used do the four names resolve to four distinct handles.
    handles = [store.daily_context, store.account_state, store.trade_logs, store.signals]
    assert collection_names(store) == {
        DAILY_CONTEXT,
        ACCOUNT_STATE,
        TRADE_LOGS,
        SIGNALS,
    }
    assert len({id(handle) for handle in handles}) == 4


@pytest.mark.asyncio
async def test_custom_db_name_used(fake_client: FakeClient) -> None:
    MongoStore(make_settings(mongodb_db="other"), client=fake_client)
    assert set(fake_client.databases) == {"other"}


@pytest.mark.asyncio
async def test_upsert_daily_context_round_trips(store: MongoStore) -> None:
    doc = {"day": "2026-08-18", "events": [], "blackouts": []}
    await store.upsert_daily_context("2026-08-18", doc)
    assert await store.get_daily_context("2026-08-18") == doc

    # Same key again replaces, never appends — one document per day.
    updated = {**doc, "events": [{"title": "CPI"}]}
    await store.upsert_daily_context("2026-08-18", updated)
    assert await store.get_daily_context("2026-08-18") == updated
    assert len(store._db.collections[DAILY_CONTEXT].docs) == 1


@pytest.mark.asyncio
async def test_daily_context_upsert_uses_exact_key(store: MongoStore) -> None:
    await store.upsert_daily_context("2026-08-18", {"day": "2026-08-18"})
    filter, _, upsert = store._db.collections[DAILY_CONTEXT].replaced[-1]
    assert filter == {"day": "2026-08-18"}
    assert upsert is True


@pytest.mark.asyncio
async def test_upsert_account_state_round_trips(store: MongoStore) -> None:
    doc = {
        "account_id": "001-001-1234567-001",
        "day_start_balance": 10000.0,
        "realized_pnl": -150.0,
    }
    await store.upsert_account_state(doc["account_id"], doc)
    assert await store.get_account_state(doc["account_id"]) == doc

    filter, _, upsert = store._db.collections[ACCOUNT_STATE].replaced[-1]
    assert filter == {"account_id": doc["account_id"]}
    assert upsert is True


@pytest.mark.asyncio
async def test_getters_return_none_for_missing_keys(store: MongoStore) -> None:
    assert await store.get_daily_context("2026-08-18") is None
    assert await store.get_account_state("001-001-9999999-001") is None


@pytest.mark.asyncio
async def test_insert_trade_log_appends_and_returns_id(store: MongoStore) -> None:
    first = await store.insert_trade_log({"decision": "execute", "confidence": 8})
    second = await store.insert_trade_log({"decision": "veto", "confidence": 3})
    assert first == "1"
    assert second == "2"

    logs = store._db.collections[TRADE_LOGS].inserted
    assert [log["decision"] for log in logs] == ["execute", "veto"]
    # Insert-only: nothing may ever rewrite the audit trail.
    assert store._db.collections[TRADE_LOGS].replaced == []


@pytest.mark.asyncio
async def test_insert_signal_appends_and_returns_id(store: MongoStore) -> None:
    signal = {"strategy": "asia_sweep", "instrument": "XAU_USD", "side": "BUY"}
    assert await store.insert_signal(signal) == "1"
    assert store._db.collections[SIGNALS].inserted == [signal]


@pytest.mark.asyncio
async def test_close_closes_client(store: MongoStore, fake_client: FakeClient) -> None:
    await store.close()
    assert fake_client.closed is True
