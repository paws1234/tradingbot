"""Pipeline orchestrator — Stage 1 streams → signals → filters → veto → orders.

Task 15 wires the pipeline end to end (Plan.md Stages 1–4):

1. **Per-instrument stream tasks** — one ``_run_instrument`` task per
   ``settings.instruments`` entry. Each backfills recent closed candles
   (:meth:`OandaClient.get_candles`) to warm the indicators, then consumes
   :meth:`OandaClient.stream_prices` ticks and feeds them to a
   :class:`CandleBuilder`.
2. **Candle builder** — aggregates mid prices ``(bid + ask) / 2`` into closed
   M15 candles (:class:`CandleBuilder`), emitted when the first tick of the
   next UTC-clock-aligned bucket arrives. Only closed bars ever reach the
   strategies (strategy.md §1.2).
3. **Signals** — each closed candle is appended to the instrument's history
   frame and every active strategy runs via :func:`emit_signals`
   (instrument→strategy mapping per strategy.md §6.3, intersected with
   ``settings.strategies``).
4. **Filters** — the circuit breaker
   (:func:`app.strategy.filters.circuit_breaker`) and news blackout
   (:func:`app.strategy.filters.in_blackout`) run per signal, in Stage 2
   order, before the gate. A blocked signal is logged and dropped.
5. **DeepSeek veto** — every surviving signal goes through
   :class:`~app.data.deepseek.DeepSeekClient.evaluate`; the verdict is
   audit-logged via :meth:`MongoStore.log_decision` regardless of outcome.
   Dispatch requires ``execute and confidence >= min_confidence`` — a
   fail-safe gate can never place an order.
6. **Dispatch** — sizing (:func:`app.strategy.sizing.build_market_order`)
   then :meth:`OandaClient.place_market_order`, audit-logged via
   :meth:`MongoStore.log_order`.

The `signals` collection records every signal that passes the Stage 2 filters
and reaches the veto gate (the natural join partner for the `trade_logs`
decisions); filter-blocked signals are logged, not stored.

One signal per setup per day: the engine keeps a pending set of
``dedup_key(signal)`` — (strategy, instrument, day, side) — and never
re-evaluates a key already seen that UTC day (strategy.md §6.2). The set rolls
over at day change.

The Stage 1 Finnhub feed is kept alive by ``_run_news``; nothing in the plan
consumes headlines in v1, so relevance gating (Task 10) wires in when a
consumer exists (YAGNI).

Dependencies are injected (tests pass fakes; ``app/main.py`` wires real
clients), matching the pattern in ``app/core/scheduler.py``.
"""

import asyncio
import logging
from collections.abc import Sequence
from datetime import date, datetime

import httpx
import pandas as pd

from app.config import Settings
from app.core.scheduler import DailyContextScheduler
from app.data.deepseek import DeepSeekClient
from app.data.finnhub import FinnhubClient
from app.data.mongo import MongoStore
from app.data.oanda import OandaClient
from app.indicators.technical import resample_h1
from app.models.schemas import Candle, PriceTick, Signal
from app.strategy.filters import circuit_breaker, in_blackout
from app.strategy.signals import STRATEGY_REGISTRY, dedup_key, dedup_signals
from app.strategy.sizing import build_market_order

logger = logging.getLogger(__name__)

# strategy.md §6.3 recommended instrument→strategy defaults. An instrument
# absent here runs every enabled strategy.
INSTRUMENT_STRATEGIES: dict[str, tuple[str, ...]] = {
    "XAU_USD": ("asia_sweep", "atr_breakout"),
    "EUR_USD": ("asia_sweep", "ema_fvg", "mean_reversion"),
    "GBP_USD": ("asia_sweep", "ema_fvg", "mean_reversion"),
}

# History is capped so the process never grows without bound. 1000 M15 bars
# (~10 days) comfortably covers the H1 EMA(200) trend warm-up.
CANDLE_LIMIT = 1000

# Outcomes ``_process_signal`` reports, for tests and future /status.
OUTCOME_DISPATCHED = "dispatched"
OUTCOME_VETOED = "vetoed"
OUTCOME_BLOCKED_BREAKER = "blocked_breaker"
OUTCOME_BLOCKED_BLACKOUT = "blocked_blackout"
OUTCOME_DUPLICATE = "duplicate"
OUTCOME_UNSIZED = "unsized"


class CandleBuilder:
    """Aggregate stream ticks into closed M15 OHLC candles.

    Ticks are bucketed by UTC-clock-aligned 15-minute windows; the candle's
    ``time`` is the bucket's open time (OANDA convention). The first tick that
    belongs to a *newer* bucket closes and returns the previous candle — so a
    returned candle is always fully closed, never the in-progress bar.

    The builder needs no prior state: it starts a fresh bucket on the first
    tick, which in the engine is the in-progress bar *after* the last closed
    backfilled candle (``get_candles`` returns closed bars only), so backfill
    and live candles never overlap or gap.
    """

    def __init__(self) -> None:
        self._start: datetime | None = None
        self._open = 0.0
        self._high = 0.0
        self._low = 0.0
        self._last = 0.0
        self._volume = 0

    @staticmethod
    def _bucket_start(time: datetime) -> datetime:
        """Floor a tick time to the UTC-clock-aligned M15 bucket open time."""
        minute = (time.minute // 15) * 15
        return time.replace(minute=minute, second=0, microsecond=0)

    def update(self, tick: PriceTick) -> Candle | None:
        """Feed one tick; returns the closed candle when the bucket rolls."""
        start = self._bucket_start(tick.time)
        mid = (tick.bid + tick.ask) / 2.0
        current = self._start
        if current is None:
            self._begin(start, mid)
            return None
        if start < current:
            return None  # out-of-order tick — ignore
        if start > current:
            closed = self._closed_candle(current)
            self._begin(start, mid)
            return closed
        self._high = max(self._high, mid)
        self._low = min(self._low, mid)
        self._last = mid
        self._volume += 1
        return None

    def _begin(self, start: datetime, mid: float) -> None:
        self._start = start
        self._open = self._high = self._low = self._last = mid
        self._volume = 1

    def _closed_candle(self, start: datetime) -> Candle:
        return Candle(
            time=start,
            open=self._open,
            high=self._high,
            low=self._low,
            close=self._last,
            volume=self._volume,
        )


def active_strategies(instrument: str, enabled: Sequence[str]) -> list[str]:
    """Strategies to run for an instrument (strategy.md §6.3 ∩ STRATEGIES).

    Recommended defaults come from §6.3; an instrument absent there runs every
    enabled strategy. Order follows the recommendation table.
    """
    recommended = INSTRUMENT_STRATEGIES.get(instrument, tuple(STRATEGY_REGISTRY))
    return [strategy_id for strategy_id in recommended if strategy_id in enabled]


def emit_signals(
    instrument: str,
    df: pd.DataFrame,
    h1: pd.DataFrame,
    enabled: Sequence[str],
) -> list[Signal]:
    """Run every active strategy over one instrument's closed bars."""
    signals: list[Signal] = []
    for strategy_id in active_strategies(instrument, enabled):
        signals.extend(STRATEGY_REGISTRY[strategy_id](df, h1, instrument))
    return dedup_signals(signals)


class TradingEngine:
    """Stage 1–4 orchestrator: streams → signals → filters → veto → orders.

    All dependencies are injected (tests pass fakes); the engine does not own
    client lifecycle — ``app/main.py`` builds and closes them.
    """

    def __init__(
        self,
        settings: Settings,
        store: MongoStore,
        oanda: OandaClient,
        deepseek: DeepSeekClient,
        finnhub: FinnhubClient,
        scheduler: DailyContextScheduler,
        candle_limit: int = CANDLE_LIMIT,
    ) -> None:
        self._settings = settings
        self._store = store
        self._oanda = oanda
        self._deepseek = deepseek
        self._finnhub = finnhub
        self._scheduler = scheduler
        self._candle_limit = candle_limit
        self._history: dict[str, pd.DataFrame] = {}
        self._pending: set[tuple[str, str, date, str]] = set()
        self._tasks: list[asyncio.Task[None]] = []
        self._started = False

    async def start(self) -> None:
        """Seed the day's context, then launch the stream tasks."""
        if self._started:
            return
        self._started = True
        # Stage 0 once at startup so blackouts and the breaker baseline exist
        # before the first stream tick; the scheduler reruns it daily at 00:00.
        try:
            await self._scheduler.build_daily_context()
        except Exception as exc:  # noqa: BLE001 — a bad calendar must not kill startup
            logger.error("daily context seed failed (%s); filters will be open", exc)
        for instrument in self._settings.instruments:
            self._tasks.append(asyncio.create_task(self._run_instrument(instrument)))
        self._tasks.append(asyncio.create_task(self._run_news()))

    async def stop(self) -> None:
        """Cancel the stream tasks (client teardown is the app's job)."""
        if not self._started:
            return
        self._started = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    def status(self) -> dict:
        """Runtime snapshot for /status: started flag, active scope, work in flight."""
        return {
            "started": self._started,
            "instruments": list(self._settings.instruments),
            "strategies": list(self._settings.strategies),
            "pending_signals": len(self._pending),
            "history_bars": {
                instrument: len(frame)
                for instrument, frame in self._history.items()
            },
        }

    async def _run_instrument(self, instrument: str) -> None:
        """Backfill, then stream ticks into candles and run the pipeline."""
        candles = await self._backfill(instrument)
        self._history[instrument] = self._frame(candles).tail(self._candle_limit)
        builder = CandleBuilder()
        async for tick in self._oanda.stream_prices([instrument]):
            closed = builder.update(tick)
            if closed is None:
                continue
            try:
                await self.process_candle(instrument, closed)
            except Exception:  # noqa: BLE001 — a bad candle must not kill the stream
                logger.exception("candle %s failed for %s", closed.time, instrument)

    async def _backfill(self, instrument: str) -> list[Candle]:
        """Recent closed candles to warm the indicators; [] on failure (degrade)."""
        try:
            return await self._oanda.get_candles(instrument)
        except (httpx.HTTPError, ValueError) as exc:
            logger.error(
                "backfill failed for %s (%s); strategies warm up live", instrument, exc
            )
            return []

    async def _run_news(self) -> None:
        """Keep the Stage 1 Finnhub feed alive (v1: no consumer of headlines)."""
        try:
            async for item in self._finnhub.watch_news():
                logger.debug("news headline: %s (%s)", item.headline, item.source)
        except Exception:  # noqa: BLE001 — the news feed is off the trade path
            logger.exception("Finnhub news task died")

    async def process_candle(self, instrument: str, candle: Candle) -> list[dict]:
        """Append one closed candle and run the pipeline over its signals.

        Returns one outcome dict per signal, in emission order.
        """
        frame = self._append_candle(instrument, candle)
        h1 = resample_h1(frame)
        signals = emit_signals(instrument, frame, h1, self._settings.strategies)
        outcomes = []
        for signal in signals:
            try:
                outcomes.append(await self._process_signal(signal))
            except Exception:  # noqa: BLE001 — isolate per-signal failures
                logger.exception("signal processing failed for %s %s", signal.instrument, signal.strategy)
        return outcomes

    async def _process_signal(self, signal: Signal) -> dict:
        """One signal through Stage 2 filters → Stage 3 veto → Stage 4 dispatch."""
        key = dedup_key(signal)
        self._drop_stale_pending(signal.timestamp.date())
        if key in self._pending:
            return self._outcome(signal, OUTCOME_DUPLICATE, "already processed today")
        self._pending.add(key)

        account_state = await self._store.get_account_state(
            self._settings.oanda_account_id
        )
        halted, reason = circuit_breaker(
            account_state, self._settings.daily_loss_limit_pct
        )
        if halted:
            logger.info(
                "signal blocked by circuit-breaker for %s: %s", signal.instrument, reason
            )
            return self._outcome(signal, OUTCOME_BLOCKED_BREAKER, reason)

        context = await self._store.get_daily_context(
            signal.timestamp.date().isoformat()
        )
        blackouts = (context or {}).get("blackouts", [])
        blocked, reason = in_blackout(signal.timestamp, blackouts)
        if blocked:
            logger.info(
                "signal blocked by blackout for %s: %s", signal.instrument, reason
            )
            return self._outcome(signal, OUTCOME_BLOCKED_BLACKOUT, reason)

        await self._store.insert_signal(signal.model_dump())
        decision = await self._deepseek.evaluate(signal)
        await self._store.log_decision(signal, decision)

        if not (
            decision.execute and decision.confidence >= self._settings.min_confidence
        ):
            return self._outcome(signal, OUTCOME_VETOED, decision.reason)

        summary = await self._oanda.get_account_summary()
        balance = float(summary["balance"])
        order_spec = build_market_order(
            signal, balance, self._settings.risk_per_trade_pct
        )
        if order_spec is None:
            return self._outcome(signal, OUTCOME_UNSIZED, "balance cannot size one unit")

        order = await self._oanda.place_market_order(order_spec)
        await self._store.log_order(order, signal, decision)
        return self._outcome(signal, OUTCOME_DISPATCHED, decision.reason)

    def _append_candle(self, instrument: str, candle: Candle) -> pd.DataFrame:
        """Append a closed candle to the instrument's history; cap the frame."""
        frame = self._history.setdefault(instrument, self._frame([]))
        if not frame.empty and candle.time <= frame.index[-1]:
            return frame  # duplicate or out-of-order bar — keep history intact
        row = candle.model_dump()
        time = row.pop("time")
        new_row = pd.DataFrame([row], index=pd.to_datetime([time]))
        # Skipping the concat on an empty frame avoids pandas' empty-dtype
        # FutureWarning while keeping the column layout identical.
        frame = new_row if frame.empty else pd.concat([frame, new_row])
        frame = frame.tail(self._candle_limit)
        self._history[instrument] = frame
        return frame

    def _drop_stale_pending(self, day: date) -> None:
        """Forget pending keys from earlier UTC days (strategy.md §6.2 rollover)."""
        if any(key[2] != day for key in self._pending):
            self._pending = {key for key in self._pending if key[2] == day}

    @staticmethod
    def _frame(candles: Sequence[Candle]) -> pd.DataFrame:
        """Closed candles → OHLCV DataFrame indexed by bar open time (UTC)."""
        if not candles:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        rows = [candle.model_dump() for candle in candles]
        frame = pd.DataFrame(rows).set_index("time")
        frame.index = pd.to_datetime(frame.index)
        return frame[["open", "high", "low", "close", "volume"]]

    @staticmethod
    def _outcome(signal: Signal, outcome: str, reason: str | None) -> dict:
        """One signal's pipeline result, for tests and future /status."""
        return {
            "strategy": signal.strategy,
            "instrument": signal.instrument,
            "side": signal.side,
            "outcome": outcome,
            "reason": reason,
        }
