"""Tests for app.data.mongo — collection wiring and CRUD helpers.

Uses a minimal in-memory fake of the Motor client (collections → dicts), so
no live MongoDB is needed. The fake records every call, which lets tests
assert both the effect (document round-trips) and the contract (exact
filter / upsert flag) the engine relies on.
"""

from datetime import datetime, timezone
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
from app.models.schemas import OrderResult, Signal, TradeDecision

T0 = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)


def make_signal(side: str = "BUY") -> Signal:
    """A valid Signal with SL/TP bracketing entry (BUY or SELL)."""
    stop_loss, take_profit = (90.0, 110.0) if side == "BUY" else (110.0, 90.0)
    return Signal(
        strategy="ema_fvg",  # type: ignore[arg-type]
        side=side,  # type: ignore[arg-type]
        instrument="XAU_USD",
        entry=100.0,
        stop_loss=stop_loss,
        take_profit=take_profit,
        atr=1.0,
        reason="test",
        timestamp=T0,
    )


def make_decision(execute: bool, confidence: int, reason: str) -> TradeDecision:
    return TradeDecision(execute=execute, confidence=confidence, reason=reason)


def make_order() -> OrderResult:
    return OrderResult(
        order_id="1234",
        status="FILLED",
        instrument="XAU_USD",
        units="10",
        price=100.0,
        created_at=T0,
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


# --- audit logging (Task 13) ----------------------------------------------


@pytest.mark.asyncio
async def test_log_decision_writes_verdict_with_signal_context(
    store: MongoStore,
) -> None:
    log_id = await store.log_decision(
        make_signal(), make_decision(execute=True, confidence=8, reason="trend")
    )
    assert log_id == "1"

    doc = store._db.collections[TRADE_LOGS].inserted[0]
    assert doc["kind"] == "decision"
    assert doc["strategy"] == "ema_fvg"
    assert doc["instrument"] == "XAU_USD"
    assert doc["side"] == "BUY"
    assert doc["entry"] == 100.0
    assert doc["stop_loss"] == 90.0
    assert doc["take_profit"] == 110.0
    assert doc["signal_time"] == T0
    assert doc["execute"] is True
    assert doc["confidence"] == 8
    assert doc["reason"] == "trend"
    # Decision time is stamped at log time, not taken from the signal.
    assert doc["timestamp"].tzinfo is not None
    assert abs((datetime.now(timezone.utc) - doc["timestamp"]).total_seconds()) < 5


@pytest.mark.asyncio
async def test_log_decision_records_veto_too(store: MongoStore) -> None:
    await store.log_decision(
        make_signal(), make_decision(execute=False, confidence=2, reason="overbought")
    )
    doc = store._db.collections[TRADE_LOGS].inserted[0]
    assert doc["kind"] == "decision"
    assert doc["execute"] is False
    assert doc["confidence"] == 2


@pytest.mark.asyncio
async def test_log_order_writes_order_with_signal_and_verdict(
    store: MongoStore,
) -> None:
    signal = make_signal()
    decision = make_decision(execute=True, confidence=9, reason="go")
    log_id = await store.log_order(make_order(), signal, decision)
    assert log_id == "1"

    doc = store._db.collections[TRADE_LOGS].inserted[0]
    assert doc["kind"] == "order"
    assert doc["order_id"] == "1234"
    assert doc["status"] == "FILLED"
    assert doc["units"] == "10"
    assert doc["price"] == 100.0
    assert doc["created_at"] == T0
    assert doc["decision_confidence"] == 9
    assert doc["decision_reason"] == "go"
    assert doc["strategy"] == "ema_fvg"
    assert doc["instrument"] == "XAU_USD"
    assert doc["side"] == "BUY"
    assert doc["timestamp"].tzinfo is not None


@pytest.mark.asyncio
async def test_decision_and_order_logs_share_one_append_only_trail(
    store: MongoStore,
) -> None:
    await store.log_decision(
        make_signal(), make_decision(execute=False, confidence=1, reason="veto")
    )
    await store.log_order(
        make_order(),
        make_signal(side="SELL"),
        make_decision(execute=True, confidence=8, reason="ok"),
    )
    docs = store._db.collections[TRADE_LOGS].inserted
    assert [doc["kind"] for doc in docs] == ["decision", "order"]
    # The audit trail is insert-only — nothing may ever rewrite it.
    assert store._db.collections[TRADE_LOGS].replaced == []


@pytest.mark.asyncio
async def test_close_closes_client(store: MongoStore, fake_client: FakeClient) -> None:
    await store.close()
    assert fake_client.closed is True
