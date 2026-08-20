"""Tests for app.core.engine — Stage 1–4 pipeline orchestration (task 15).

Drives the pipeline with fake clients — no network, MongoDB, or real DeepSeek
calls — and monkeypatches the strategy registry so `process_candle` emits
deterministic signals.

Verification points:

- CandleBuilder aggregates stream ticks into closed M15 OHLC candles, rolls on
  the UTC-clock boundary, and ignores out-of-order ticks.
- `active_strategies` maps instruments to strategies per strategy.md §6.3,
  intersected with the enabled set.
- `emit_signals` dispatches to each active strategy and dedups the output.
- `_process_signal` runs the Stage 2 filter chain (circuit breaker → news
  blackout) before the DeepSeek veto, applies the confidence threshold,
  sizes, dispatches, and audit-logs decisions + orders.
- One signal per setup per UTC day (the pending map), reset on day rollover.
- `start`/`stop` seed the daily context and launch per-instrument stream tasks
  plus the Finnhub feed; a streamed candle flows end to end to an order.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import httpx
import pandas as pd
import pytest

from app.config import Settings
from app.core import engine as engine_module
from app.core.engine import (
    OUTCOME_BLOCKED_BLACKOUT,
    OUTCOME_BLOCKED_BREAKER,
    OUTCOME_DISPATCHED,
    OUTCOME_DUPLICATE,
    OUTCOME_UNSIZED,
    OUTCOME_VETOED,
    CandleBuilder,
    TradingEngine,
    active_strategies,
    emit_signals,
)
from app.models.schemas import (
    Candle,
    NewsItem,
    OrderResult,
    PriceTick,
    Signal,
    TradeDecision,
)

T0 = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)


# --- helpers / fakes --------------------------------------------------------


def make_signal(
    side: str = "BUY", instrument: str = "XAU_USD", at: datetime = T0
) -> Signal:
    """A valid Signal with a 10.0 stop distance and SL/TP bracketing entry."""
    if side == "BUY":
        return Signal(
            strategy="asia_sweep",
            side="BUY",
            instrument=instrument,
            entry=100.0,
            stop_loss=90.0,
            take_profit=110.0,
            atr=1.0,
            reason="test",
            timestamp=at,
        )
    return Signal(
        strategy="asia_sweep",
        side="SELL",
        instrument=instrument,
        entry=100.0,
        stop_loss=110.0,
        take_profit=90.0,
        atr=1.0,
        reason="test",
        timestamp=at,
    )


def tick(at: datetime, bid: float = 100.0, ask: float = 100.2) -> PriceTick:
    """A mid=(100.1) XAU_USD tick at `at` UTC."""
    return PriceTick(instrument="XAU_USD", time=at, bid=bid, ask=ask)


class FakeStore:
    def __init__(self) -> None:
        self.account_state: dict | None = None
        self.daily_context: dict | None = None
        self.signal_log: list[dict] = []
        self.decisions: list[tuple[Signal, TradeDecision]] = []
        self.orders: list[tuple[OrderResult, Signal, TradeDecision]] = []

    async def get_account_state(self, account_id: str) -> dict | None:
        return self.account_state

    async def get_daily_context(self, day: str) -> dict | None:
        return self.daily_context

    async def insert_signal(self, document: dict) -> str:
        self.signal_log.append(document)
        return "signal-1"

    async def log_decision(self, signal: Signal, decision: TradeDecision) -> str:
        self.decisions.append((signal, decision))
        return "decision-1"

    async def log_order(
        self,
        order: OrderResult,
        signal: Signal,
        decision: TradeDecision,
    ) -> str:
        self.orders.append((order, signal, decision))
        return "order-1"


class FakeOanda:
    def __init__(
        self,
        ticks: list[PriceTick] | None = None,
        candles: list[Candle] | None = None,
        summary: dict | None = None,
    ) -> None:
        self.ticks = ticks or []
        self.candles = candles or []
        self.summary = summary or {"balance": "10000.0000"}
        self.candle_requests: list[tuple[str, str | None, int]] = []
        self.placed: list[dict] = []
        self.backfill_error: Exception | None = None

    async def get_candles(
        self, instrument: str, granularity: str | None = None, count: int = 500
    ) -> list[Candle]:
        self.candle_requests.append((instrument, granularity, count))
        if self.backfill_error is not None:
            raise self.backfill_error
        return self.candles

    async def stream_prices(self, instruments: list[str]):
        for price_tick in self.ticks:
            yield price_tick

    async def get_account_summary(self) -> dict:
        return self.summary

    async def place_market_order(self, order_spec: dict) -> OrderResult:
        self.placed.append(order_spec)
        return OrderResult(
            order_id="o-1",
            status="FILLED",
            instrument=order_spec["order"]["instrument"],
            units=order_spec["order"]["units"],
            price=100.0,
            created_at=T0,
        )


class FakeDeepSeek:
    def __init__(self, decision: TradeDecision | None = None) -> None:
        self.decision = decision or TradeDecision(
            execute=True, confidence=10, reason="ok"
        )
        self.calls: list[Signal] = []

    async def evaluate(self, signal: Signal) -> TradeDecision:
        self.calls.append(signal)
        return self.decision


class FakeFinnhub:
    def __init__(self, items: list[NewsItem] | None = None) -> None:
        self.items = items or []

    async def watch_news(self, *args, **kwargs):
        for item in self.items:
            yield item


class FakeDailyScheduler:
    def __init__(self) -> None:
        self.build_calls = 0

    async def build_daily_context(self, day=None) -> dict:
        self.build_calls += 1
        return {}


@dataclass
class Harness:
    engine: TradingEngine
    settings: Settings
    store: FakeStore
    oanda: FakeOanda
    deepseek: FakeDeepSeek
    finnhub: FakeFinnhub
    scheduler: FakeDailyScheduler


@pytest.fixture
def make_engine(
    make_settings: Callable[..., Settings],
) -> Callable[..., Harness]:
    def _make(
        store: FakeStore | None = None,
        oanda: FakeOanda | None = None,
        deepseek: FakeDeepSeek | None = None,
        finnhub: FakeFinnhub | None = None,
        scheduler: FakeDailyScheduler | None = None,
        **settings_overrides: object,
    ) -> Harness:
        settings = make_settings(**settings_overrides)
        store = store or FakeStore()
        oanda = oanda or FakeOanda()
        deepseek = deepseek or FakeDeepSeek()
        finnhub = finnhub or FakeFinnhub()
        scheduler = scheduler or FakeDailyScheduler()
        engine = TradingEngine(settings, store, oanda, deepseek, finnhub, scheduler)
        return Harness(engine, settings, store, oanda, deepseek, finnhub, scheduler)

    return _make


# --- CandleBuilder ----------------------------------------------------------


def test_builder_aggregates_ticks_into_closed_m15_candle() -> None:
    builder = CandleBuilder()
    assert builder.update(tick(T0, bid=100.0, ask=100.2)) is None        # mid 100.1
    assert builder.update(tick(T0 + timedelta(seconds=50), 100.4, 100.6)) is None  # 100.5
    assert builder.update(tick(T0 + timedelta(minutes=10), 100.1, 100.3)) is None  # 100.2
    closed = builder.update(tick(T0 + timedelta(minutes=15)))

    assert closed is not None
    assert closed.time == T0
    assert closed.open == pytest.approx(100.1)
    assert closed.high == pytest.approx(100.5)
    assert closed.low == pytest.approx(100.1)
    assert closed.close == pytest.approx(100.2)
    assert closed.volume == 3


def test_builder_rolls_one_candle_per_bucket() -> None:
    builder = CandleBuilder()
    builder.update(tick(T0))
    first = builder.update(tick(T0 + timedelta(minutes=15)))
    second = builder.update(tick(T0 + timedelta(minutes=30)))

    assert first is not None and first.time == T0
    assert second is not None and second.time == T0 + timedelta(minutes=15)
    assert second.open == pytest.approx(100.1)  # the first tick of the new bucket


def test_builder_ignores_out_of_order_tick() -> None:
    builder = CandleBuilder()
    builder.update(tick(T0))
    assert builder.update(tick(T0 - timedelta(minutes=5))) is None
    builder.update(tick(T0 + timedelta(seconds=30), bid=100.6, ask=100.8))  # mid 100.7
    closed = builder.update(tick(T0 + timedelta(minutes=15)))

    # The stale tick must not have corrupted the in-progress bucket.
    assert closed is not None
    assert closed.open == pytest.approx(100.1)
    assert closed.high == pytest.approx(100.7)
    assert closed.low == pytest.approx(100.1)
    assert closed.close == pytest.approx(100.7)
    assert closed.volume == 2


def test_bucket_start_floors_to_utc_clock() -> None:
    assert CandleBuilder._bucket_start(
        datetime(2026, 8, 20, 12, 34, 56, tzinfo=timezone.utc)
    ) == datetime(2026, 8, 20, 12, 30, tzinfo=timezone.utc)
    assert CandleBuilder._bucket_start(
        datetime(2026, 8, 20, 12, 59, 59, tzinfo=timezone.utc)
    ) == datetime(2026, 8, 20, 12, 45, tzinfo=timezone.utc)
    assert CandleBuilder._bucket_start(
        datetime(2026, 8, 20, 0, 7, 0, tzinfo=timezone.utc)
    ) == datetime(2026, 8, 20, 0, 0, tzinfo=timezone.utc)
    assert CandleBuilder._bucket_start(
        datetime(2026, 8, 20, 23, 59, 59, tzinfo=timezone.utc)
    ) == datetime(2026, 8, 20, 23, 45, tzinfo=timezone.utc)


# --- active_strategies / emit_signals ---------------------------------------

ALL_STRATEGIES = ["asia_sweep", "ema_fvg", "atr_breakout", "mean_reversion"]


def test_active_strategies_follows_section_6_3_for_xau() -> None:
    assert active_strategies("XAU_USD", ALL_STRATEGIES) == ["asia_sweep", "atr_breakout"]


def test_active_strategies_follows_section_6_3_for_euro() -> None:
    assert active_strategies("EUR_USD", ALL_STRATEGIES) == [
        "asia_sweep",
        "ema_fvg",
        "mean_reversion",
    ]


def test_active_strategies_runs_all_for_unknown_instrument() -> None:
    assert active_strategies("BTC_USD", ALL_STRATEGIES) == ALL_STRATEGIES


def test_active_strategies_intersects_with_enabled() -> None:
    assert active_strategies("XAU_USD", ["ema_fvg", "mean_reversion"]) == []
    assert active_strategies("EUR_USD", ["ema_fvg", "mean_reversion"]) == [
        "ema_fvg",
        "mean_reversion",
    ]


class FakeStrategy:
    """A registry entry that records its calls and returns fixed signals."""

    def __init__(self, signals: list[Signal]) -> None:
        self.signals = signals
        self.calls: list[tuple[pd.DataFrame, pd.DataFrame, str]] = []

    def __call__(
        self, df: pd.DataFrame, h1: pd.DataFrame, instrument: str
    ) -> list[Signal]:
        self.calls.append((df, h1, instrument))
        return self.signals


def sample_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame(
        {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5], "volume": [10]},
        index=pd.to_datetime([T0]),
    )
    h1 = pd.DataFrame(
        {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5]},
        index=pd.to_datetime([T0]),
    )
    return df, h1


def test_emit_signals_dispatches_to_each_active_strategy(monkeypatch) -> None:
    asia = FakeStrategy([])
    breakout = FakeStrategy([])
    monkeypatch.setattr(
        engine_module, "INSTRUMENT_STRATEGIES", {"XAU_USD": ("asia_sweep", "atr_breakout")}
    )
    monkeypatch.setattr(
        engine_module, "STRATEGY_REGISTRY", {"asia_sweep": asia, "atr_breakout": breakout}
    )
    df, h1 = sample_frames()

    signals = emit_signals("XAU_USD", df, h1, ALL_STRATEGIES)

    assert signals == []
    assert len(asia.calls) == 1
    assert asia.calls[0] == (df, h1, "XAU_USD")
    assert len(breakout.calls) == 1


def test_emit_signals_deduplicates_across_strategies(monkeypatch) -> None:
    signal = make_signal()
    monkeypatch.setattr(
        engine_module, "INSTRUMENT_STRATEGIES", {"XAU_USD": ("asia_sweep", "atr_breakout")}
    )
    monkeypatch.setattr(
        engine_module,
        "STRATEGY_REGISTRY",
        {"asia_sweep": FakeStrategy([signal]), "atr_breakout": FakeStrategy([signal])},
    )
    df, h1 = sample_frames()

    signals = emit_signals("XAU_USD", df, h1, ALL_STRATEGIES)

    # Both strategies re-emit the same setup; dedup keeps one.
    assert len(signals) == 1
    assert signals[0] is signal


# --- _process_signal: filters → veto → dispatch ------------------------------


@pytest.mark.asyncio
async def test_process_signal_dispatches_approved_signal(
    make_engine: Callable[..., Harness],
) -> None:
    h = make_engine()

    outcome = await h.engine._process_signal(make_signal())

    assert outcome["outcome"] == OUTCOME_DISPATCHED
    assert h.oanda.placed and h.oanda.placed[0]["order"]["units"] == "10"  # 10000*1% / 10
    assert len(h.deepseek.calls) == 1
    assert len(h.store.signal_log) == 1
    assert len(h.store.decisions) == 1
    assert len(h.store.orders) == 1
    assert h.store.orders[0][0].order_id == "o-1"


@pytest.mark.asyncio
async def test_process_signal_vetoed_places_no_order(
    make_engine: Callable[..., Harness],
) -> None:
    h = make_engine(
        deepseek=FakeDeepSeek(TradeDecision(execute=False, confidence=5, reason="risky"))
    )

    outcome = await h.engine._process_signal(make_signal())

    assert outcome["outcome"] == OUTCOME_VETOED
    assert outcome["reason"] == "risky"
    assert h.oanda.placed == []
    assert h.store.orders == []
    # The vetoed signal is still recorded and the verdict audit-logged.
    assert len(h.store.signal_log) == 1
    assert len(h.store.decisions) == 1


@pytest.mark.asyncio
async def test_process_signal_low_confidence_places_no_order(
    make_engine: Callable[..., Harness],
) -> None:
    h = make_engine(
        deepseek=FakeDeepSeek(TradeDecision(execute=True, confidence=5, reason="marginal"))
    )

    outcome = await h.engine._process_signal(make_signal())

    assert outcome["outcome"] == OUTCOME_VETOED
    assert h.oanda.placed == []


@pytest.mark.asyncio
async def test_process_signal_confidence_boundary_is_inclusive(
    make_engine: Callable[..., Harness],
) -> None:
    # min_confidence defaults to 7; exactly 7 dispatches.
    h = make_engine(
        deepseek=FakeDeepSeek(TradeDecision(execute=True, confidence=7, reason="ok"))
    )

    outcome = await h.engine._process_signal(make_signal())

    assert outcome["outcome"] == OUTCOME_DISPATCHED


@pytest.mark.asyncio
async def test_circuit_breaker_blocks_before_gate(
    make_engine: Callable[..., Harness],
) -> None:
    h = make_engine()
    h.store.account_state = {
        "account_id": "a",
        "day_start_balance": 10000.0,
        "realized_pnl": -500.0,
    }

    outcome = await h.engine._process_signal(make_signal())

    assert outcome["outcome"] == OUTCOME_BLOCKED_BREAKER
    assert h.deepseek.calls == []  # the gate is never consulted
    assert h.store.signal_log == []  # blocked signals are not recorded
    assert h.oanda.placed == []


@pytest.mark.asyncio
async def test_circuit_breaker_sticky_halted_flag(
    make_engine: Callable[..., Harness],
) -> None:
    h = make_engine()
    h.store.account_state = {
        "account_id": "a",
        "day_start_balance": 10000.0,
        "realized_pnl": 0.0,
        "trading_halted": True,
    }

    outcome = await h.engine._process_signal(make_signal())

    assert outcome["outcome"] == OUTCOME_BLOCKED_BREAKER
    assert h.deepseek.calls == []


@pytest.mark.asyncio
async def test_news_blackout_blocks_signal(
    make_engine: Callable[..., Harness],
) -> None:
    h = make_engine()
    h.store.daily_context = {
        "day": "2026-08-20",
        "blackouts": [
            {
                "title": "FOMC",
                "start": T0 - timedelta(minutes=30),
                "end": T0 + timedelta(minutes=30),
            }
        ],
    }

    outcome = await h.engine._process_signal(make_signal())

    assert outcome["outcome"] == OUTCOME_BLOCKED_BLACKOUT
    assert h.deepseek.calls == []
    assert h.oanda.placed == []


@pytest.mark.asyncio
async def test_signal_outside_blackout_dispatches(
    make_engine: Callable[..., Harness],
) -> None:
    h = make_engine()
    h.store.daily_context = {
        "day": "2026-08-20",
        "blackouts": [
            {
                "title": "FOMC",
                "start": T0 - timedelta(minutes=30),
                "end": T0 + timedelta(minutes=30),
            }
        ],
    }

    outcome = await h.engine._process_signal(
        make_signal(at=T0 + timedelta(minutes=31))
    )

    assert outcome["outcome"] == OUTCOME_DISPATCHED


@pytest.mark.asyncio
async def test_duplicate_signal_processed_once(
    make_engine: Callable[..., Harness],
) -> None:
    h = make_engine()

    first = await h.engine._process_signal(make_signal())
    second = await h.engine._process_signal(make_signal())

    assert first["outcome"] == OUTCOME_DISPATCHED
    assert second["outcome"] == OUTCOME_DUPLICATE
    assert len(h.deepseek.calls) == 1
    assert len(h.store.orders) == 1


@pytest.mark.asyncio
async def test_pending_resets_on_new_day(
    make_engine: Callable[..., Harness],
) -> None:
    h = make_engine()

    day1 = await h.engine._process_signal(make_signal())
    day2 = await h.engine._process_signal(make_signal(at=T0 + timedelta(days=1)))

    assert day1["outcome"] == OUTCOME_DISPATCHED
    assert day2["outcome"] == OUTCOME_DISPATCHED
    assert len(h.deepseek.calls) == 2


@pytest.mark.asyncio
async def test_unsized_signal_skips_dispatch(
    make_engine: Callable[..., Harness],
) -> None:
    h = make_engine(oanda=FakeOanda(summary={"balance": "50.0"}))

    outcome = await h.engine._process_signal(make_signal())

    assert outcome["outcome"] == OUTCOME_UNSIZED
    assert h.oanda.placed == []
    assert len(h.store.decisions) == 1  # the veto still ran and was logged


# --- process_candle: strategy feed ------------------------------------------


def _patch_single_strategy(monkeypatch, fake) -> None:
    """Route XAU_USD through one fake strategy function."""
    monkeypatch.setattr(
        engine_module, "INSTRUMENT_STRATEGIES", {"XAU_USD": ("asia_sweep",)}
    )
    monkeypatch.setattr(engine_module, "STRATEGY_REGISTRY", {"asia_sweep": fake})


@pytest.mark.asyncio
async def test_process_candle_emits_and_dispatches(
    make_engine: Callable[..., Harness], monkeypatch,
) -> None:
    h = make_engine()
    signal = make_signal()
    _patch_single_strategy(monkeypatch, FakeStrategy([signal]))

    outcomes = await h.engine.process_candle(
        "XAU_USD", Candle(time=T0, open=100.0, high=101.0, low=99.0, close=100.5, volume=10)
    )

    assert len(outcomes) == 1
    assert outcomes[0]["outcome"] == OUTCOME_DISPATCHED
    assert len(h.store.signal_log) == 1
    assert h.oanda.placed


@pytest.mark.asyncio
async def test_process_candle_appends_to_history(
    make_engine: Callable[..., Harness], monkeypatch,
) -> None:
    h = make_engine()
    seen: list[pd.DataFrame] = []

    def fake_strategy(df: pd.DataFrame, h1: pd.DataFrame, instrument: str) -> list[Signal]:
        seen.append(df)
        return []

    _patch_single_strategy(monkeypatch, fake_strategy)

    await h.engine.process_candle(
        "XAU_USD", Candle(time=T0, open=100.0, high=101.0, low=99.0, close=100.5, volume=10)
    )
    await h.engine.process_candle(
        "XAU_USD",
        Candle(
            time=T0 + timedelta(minutes=15),
            open=100.5,
            high=101.0,
            low=100.0,
            close=100.2,
            volume=8,
        ),
    )

    assert len(seen) == 2
    assert len(seen[0]) == 1 and seen[0].index[0] == pd.Timestamp(T0)
    assert len(seen[1]) == 2


@pytest.mark.asyncio
async def test_process_candle_ignores_duplicate_bar_time(
    make_engine: Callable[..., Harness], monkeypatch,
) -> None:
    h = make_engine()
    calls = []

    def fake_strategy(df: pd.DataFrame, h1: pd.DataFrame, instrument: str) -> list[Signal]:
        calls.append(df)
        return []

    monkeypatch.setattr(
        engine_module, "INSTRUMENT_STRATEGIES", {"XAU_USD": ("asia_sweep",)}
    )
    monkeypatch.setattr(engine_module, "STRATEGY_REGISTRY", {"asia_sweep": fake_strategy})
    candle = Candle(time=T0, open=100.0, high=101.0, low=99.0, close=100.5, volume=10)

    await h.engine.process_candle("XAU_USD", candle)
    frame_before = h.engine._history["XAU_USD"]
    await h.engine.process_candle("XAU_USD", candle)

    assert h.engine._history["XAU_USD"].equals(frame_before)
    assert len(calls) == 2  # strategies re-run on the unchanged frame, dedup guards dispatch


# --- start / stop / streaming ------------------------------------------------


@pytest.mark.asyncio
async def test_start_seeds_context_and_launches_tasks(
    make_engine: Callable[..., Harness],
) -> None:
    h = make_engine()

    await h.engine.start()

    assert h.scheduler.build_calls == 1
    assert h.engine._started is True
    # one per-instrument task (XAU_USD) + the news task
    assert len(h.engine._tasks) == 2
    await asyncio_gather_tasks(h)
    await h.engine.stop()
    assert h.engine._started is False
    assert h.engine._tasks == []


@pytest.mark.asyncio
async def test_start_is_idempotent(make_engine: Callable[..., Harness]) -> None:
    h = make_engine()

    await h.engine.start()
    await h.engine.start()

    assert h.scheduler.build_calls == 1
    assert len(h.engine._tasks) == 2
    await asyncio_gather_tasks(h)
    await h.engine.stop()


@pytest.mark.asyncio
async def test_stop_is_noop_before_start(
    make_engine: Callable[..., Harness],
) -> None:
    h = make_engine()

    await h.engine.stop()

    assert h.engine._started is False


def test_status_snapshots_runtime_state(
    make_engine: Callable[..., Harness],
) -> None:
    h = make_engine()
    h.engine._pending.add(("asia_sweep", "XAU_USD", T0.date(), "BUY"))

    snapshot = h.engine.status()

    assert snapshot["started"] is False
    assert snapshot["pending_signals"] == 1
    assert snapshot["instruments"] == ["XAU_USD"]
    assert snapshot["strategies"] == ALL_STRATEGIES
    assert snapshot["history_bars"] == {}


@pytest.mark.asyncio
async def test_stream_builds_candles_and_dispatches(
    make_engine: Callable[..., Harness], monkeypatch,
) -> None:
    h = make_engine()
    # Ticks spanning two M15 buckets → one closed candle on the roll.
    h.oanda.ticks = [
        tick(T0),
        tick(T0 + timedelta(seconds=30), bid=100.6, ask=100.8),
        tick(T0 + timedelta(minutes=15)),
    ]
    _patch_single_strategy(monkeypatch, FakeStrategy([make_signal()]))

    await h.engine.start()
    await asyncio_gather_tasks(h)
    await h.engine.stop()

    # Backfill used the default count (500 closed bars) on the REST client.
    assert h.oanda.candle_requests == [("XAU_USD", None, 500)]
    assert h.oanda.placed  # the streamed candle reached the order desk


@pytest.mark.asyncio
async def test_backfill_failure_degrades_to_empty_frame(
    make_engine: Callable[..., Harness],
) -> None:
    h = make_engine()
    h.oanda.backfill_error = httpx.ConnectError("network down")

    await h.engine.start()
    await asyncio_gather_tasks(h)
    await h.engine.stop()

    assert "XAU_USD" in h.engine._history
    assert h.engine._history["XAU_USD"].empty


@pytest.mark.asyncio
async def test_news_task_survives(make_engine: Callable[..., Harness]) -> None:
    h = make_engine()
    h.finnhub.items = [NewsItem(headline="Gold surges", published_at=T0, source="Finnhub")]

    await h.engine.start()
    await asyncio_gather_tasks(h)
    await h.engine.stop()

    assert h.engine._started is False


async def asyncio_gather_tasks(h: Harness) -> None:
    """Drain the engine's tasks (they end after their bounded fake streams)."""
    await asyncio.gather(*h.engine._tasks, return_exceptions=True)
