"""Strategy signal functions (strategy.md §2–§5), registry (§6.1), dedup (§6.2).

Contract every function shares:

- Uniform signature ``(df, h1, instrument) -> list[Signal]`` so the engine
  dispatches through :data:`STRATEGY_REGISTRY` without per-strategy branching.
  Strategies that don't read the H1 frame ignore ``h1``.
- Closed bars only: entries fire on the close of bar ``t``, and every rolling
  value that feeds an entry decision is shifted by one bar — either inside
  the indicator helpers (:func:`~app.indicators.technical.donchian` and the
  swing windows below) or here (the H1 trend join, the breakout squeeze).
- Warmups: indicator NaNs compare False, so a bar whose lookback is not yet
  complete simply produces no signal.
- Every returned list is deduped to one signal per (strategy, instrument,
  day, side) — strategy.md §6.2. Cross-candle persistence ("pending until
  invalidated or filled") is the engine's job (Task 15); :func:`dedup_key`
  is its hook.
- :class:`~app.models.schemas.Signal` validates that SL/TP bracket the entry
  on the correct side. A setup whose hard stop cannot bracket the entry is
  skipped, never emitted invalid — the sizing formula (Task 11) divides by
  ``|entry - stop_loss|`` and must never see a non-bracketing stop.

The engine feeds closed M15 candles built from the OANDA stream (Task 15);
nothing here is async.
"""

from collections.abc import Callable
from datetime import date

import numpy as np
import pandas as pd

from app.config import STRATEGY_IDS
from app.indicators.technical import (
    adx,
    atr,
    bearish_fvg,
    bollinger,
    bullish_fvg,
    donchian,
    ema,
    rsi,
    session_range,
    sma,
)
from app.models.schemas import Signal

# Parameters are frozen by strategy.md; each group cites its section. The
# config.py risk/filter settings (Stage 2, Task 10) do not override these.

ATR_PERIOD = 14
RSI_PERIOD = 14

# strategy.md §2.2 — Asian sweep
ASIA_START, ASIA_END = "00:00", "07:00"
SWEEP_START, SWEEP_END = "07:00", "15:00"
ASIA_SL_ATR_BUFFER = 0.3
ASIA_TP_RR = 2.5

# strategy.md §3.2 — EMA stack + FVG retrace
EMA_FAST, EMA_MID, EMA_SLOW = 20, 50, 200
EMA_RSI_LOW, EMA_RSI_HIGH = 40.0, 53.0  # bullish pullback window
EMA_RSI_BEAR_LOW, EMA_RSI_BEAR_HIGH = 47.0, 60.0  # bearish mirror
EMA_SL_ATR_BUFFER = 0.5
EMA_TP_RR = 2.0

# strategy.md §4.2 — ATR breakout
ATR_SMA_PERIOD = 50
SQUEEZE_MULT = 0.82
BO_SQUEEZE_LOOKBACK = 5  # breakout fires if any of the prior N bars was a squeeze
DONCHIAN_N = 20
BO_SL_ATR_MULT = 1.8
BO_TP_ATR_MULT = 2.5
BO_SESSION_START, BO_SESSION_END = "13:30", "17:00"

# strategy.md §5.2 — Mean reversion
ADX_PERIOD = 14
ADX_MAX = 20.0
BB_PERIOD = 20
BB_MULT = 2.5
MR_RSI_OVERSOLD, MR_RSI_OVERBOUGHT = 28.0, 72.0
MR_SL_ATR_MULT = 1.2
SWING_N = 10


def asia_sweep_signals(df: pd.DataFrame, h1: pd.DataFrame, instrument: str) -> list[Signal]:
    """Asian-range liquidity sweep & reversal (strategy.md §2).

    A stop sweep beyond the day's 00:00–07:00 range that closes back inside
    it, confirmed by a three-candle FVG, reverses toward the opposite side.
    """
    atr14 = atr(df["high"], df["low"], df["close"], ATR_PERIOD)
    asia_highs, asia_lows = session_range(df, ASIA_START, ASIA_END)
    fvg_up = bullish_fvg(df)
    fvg_dn = bearish_fvg(df)
    signals: list[Signal] = []

    for day, h_asia in asia_highs.items():
        l_asia = asia_lows[day]
        sweep = df[df.index.date == day].between_time(
            SWEEP_START, SWEEP_END, inclusive="left"
        )
        if sweep.empty:
            continue
        # Sweep extreme through bar t (cummax/cummin include t, whose OHLC
        # is known at close) — never the max over the whole window, which
        # would leak future sweep bars into today's stop.
        swept_highs = sweep["high"].cummax()
        swept_lows = sweep["low"].cummin()

        for t, row in sweep.iterrows():
            if pd.isna(atr14.loc[t]):
                continue
            # Bearish: swept above the range high, rejected back inside.
            if row["high"] > h_asia and row["close"] < h_asia and fvg_dn.loc[t]:
                sl = swept_highs.loc[t] + ASIA_SL_ATR_BUFFER * atr14.loc[t]
                tp = min(l_asia, row["close"] - ASIA_TP_RR * (sl - row["close"]))
                signals.append(
                    Signal(
                        strategy="asia_sweep",
                        side="SELL",
                        instrument=instrument,
                        entry=row["close"],
                        stop_loss=sl,
                        take_profit=tp,
                        atr=atr14.loc[t],
                        reason="bearish_asia_sweep_fvg",
                        timestamp=t,
                    )
                )
            # Bullish mirror: swept below the range low, rejected back inside.
            if row["low"] < l_asia and row["close"] > l_asia and fvg_up.loc[t]:
                sl = swept_lows.loc[t] - ASIA_SL_ATR_BUFFER * atr14.loc[t]
                tp = max(h_asia, row["close"] + ASIA_TP_RR * (row["close"] - sl))
                signals.append(
                    Signal(
                        strategy="asia_sweep",
                        side="BUY",
                        instrument=instrument,
                        entry=row["close"],
                        stop_loss=sl,
                        take_profit=tp,
                        atr=atr14.loc[t],
                        reason="bullish_asia_sweep_fvg",
                        timestamp=t,
                    )
                )

    return dedup_signals(signals)


def _join_h1_trend(
    m15: pd.DataFrame, trend_up: pd.Series, trend_down: pd.Series
) -> pd.Series:
    """Map H1 trend flags onto M15 bars: 1.0 up, -1.0 down, NaN no trend.

    Flags are shifted one H1 bar before the asof join, so an M15 bar can only
    ever read trend computed from H1 bars that closed before it (strategy.md
    §1.1 / §6.2). The guarantee holds even if the caller derived ``h1`` from
    the full M15 series — the in-progress hour's data is never used.
    """
    side = pd.Series(np.nan, index=trend_up.index)
    side[trend_up] = 1.0
    side[trend_down] = -1.0
    shifted = side.shift(1).dropna()
    if shifted.empty:
        return pd.Series(np.nan, index=m15.index)
    right = shifted.rename("trend").reset_index().rename(columns={"index": "time"})
    left = pd.DataFrame({"time": m15.index})
    joined = pd.merge_asof(left, right, on="time", direction="backward")
    return joined.set_index("time")["trend"].reindex(m15.index)


def ema_fvg_signals(df: pd.DataFrame, h1: pd.DataFrame, instrument: str) -> list[Signal]:
    """Multi-EMA trend + FVG imbalance retrace (strategy.md §3).

    Fades M15 fair-value gaps aligned with the H1 EMA-stack trend, entering
    at the gap boundary while the pullback momentum window holds.
    """
    trend_up = (ema(h1["close"], EMA_FAST) > ema(h1["close"], EMA_MID)) & (
        ema(h1["close"], EMA_MID) > ema(h1["close"], EMA_SLOW)
    )
    trend_down = (ema(h1["close"], EMA_FAST) < ema(h1["close"], EMA_MID)) & (
        ema(h1["close"], EMA_MID) < ema(h1["close"], EMA_SLOW)
    )
    flags = _join_h1_trend(df, trend_up, trend_down)

    r = rsi(df["close"], RSI_PERIOD)
    a = atr(df["high"], df["low"], df["close"], ATR_PERIOD)
    fvg_up = bullish_fvg(df)
    fvg_dn = bearish_fvg(df)
    signals: list[Signal] = []

    for t, row in df.iterrows():
        if pd.isna(a.loc[t]):
            continue
        # Bullish: limit at the gap floor, stop below it by an ATR buffer.
        if flags.loc[t] == 1.0 and fvg_up.loc[t] and EMA_RSI_LOW <= r.loc[t] <= EMA_RSI_HIGH:
            entry = df["high"].shift(2).loc[t]
            sl = df["low"].shift(2).loc[t] - EMA_SL_ATR_BUFFER * a.loc[t]
            tp = entry + EMA_TP_RR * (entry - sl)
            signals.append(
                Signal(
                    strategy="ema_fvg",
                    side="BUY",
                    instrument=instrument,
                    entry=entry,
                    stop_loss=sl,
                    take_profit=tp,
                    atr=a.loc[t],
                    reason="bullish_ema_stack_fvg_pullback",
                    timestamp=t,
                )
            )
        # Bearish mirror: limit at the gap ceiling, RSI in the bear window.
        elif flags.loc[t] == -1.0 and fvg_dn.loc[t] and EMA_RSI_BEAR_LOW <= r.loc[t] <= EMA_RSI_BEAR_HIGH:
            entry = df["low"].shift(2).loc[t]
            sl = df["high"].shift(2).loc[t] + EMA_SL_ATR_BUFFER * a.loc[t]
            tp = entry - EMA_TP_RR * (sl - entry)
            signals.append(
                Signal(
                    strategy="ema_fvg",
                    side="SELL",
                    instrument=instrument,
                    entry=entry,
                    stop_loss=sl,
                    take_profit=tp,
                    atr=a.loc[t],
                    reason="bearish_ema_stack_fvg_pullback",
                    timestamp=t,
                )
            )

    return dedup_signals(signals)


def atr_breakout_signals(df: pd.DataFrame, h1: pd.DataFrame, instrument: str) -> list[Signal]:
    """ATR volatility breakout & Donchian squeeze (strategy.md §4).

    A low-volatility squeeze must precede the breakout bar — within the prior
    ``BO_SQUEEZE_LOOKBACK`` bars, checked on a ``shift(1)``-safe rolling
    window so the bar immediately before the breakout need not itself be the
    squeezed one (strategy.md §4.3 rule 1). The bar itself must expand
    volatility and close through the prior 20-bar Donchian channel. NY-open
    session only.
    """
    a = atr(df["high"], df["low"], df["close"], ATR_PERIOD)
    a_sma = sma(a, ATR_SMA_PERIOD)
    hh, ll = donchian(df["high"], df["low"], DONCHIAN_N)
    squeeze = a < a_sma * SQUEEZE_MULT
    # Any squeeze in the prior N bars (bars t-1 .. t-N), shifted so bar t only
    # reads closed bars. `> 0` makes the rolling warm-up NaN read as no-squeeze.
    squeeze_recent = squeeze.shift(1).rolling(BO_SQUEEZE_LOOKBACK).max()
    expand = a > a_sma
    signals: list[Signal] = []

    for t, row in df.between_time(
        BO_SESSION_START, BO_SESSION_END, inclusive="left"
    ).iterrows():
        if not squeeze_recent.loc[t] > 0:
            continue
        if expand.loc[t] and row["close"] > hh.loc[t]:
            sl = row["close"] - BO_SL_ATR_MULT * a.loc[t]
            tp = row["close"] + BO_TP_ATR_MULT * a.loc[t]
            signals.append(
                Signal(
                    strategy="atr_breakout",
                    side="BUY",
                    instrument=instrument,
                    entry=row["close"],
                    stop_loss=sl,
                    take_profit=tp,
                    atr=a.loc[t],
                    reason="donchian_breakout_squeeze",
                    timestamp=t,
                )
            )
        elif expand.loc[t] and row["close"] < ll.loc[t]:
            sl = row["close"] + BO_SL_ATR_MULT * a.loc[t]
            tp = row["close"] - BO_TP_ATR_MULT * a.loc[t]
            signals.append(
                Signal(
                    strategy="atr_breakout",
                    side="SELL",
                    instrument=instrument,
                    entry=row["close"],
                    stop_loss=sl,
                    take_profit=tp,
                    atr=a.loc[t],
                    reason="donchian_breakdown_squeeze",
                    timestamp=t,
                )
            )

    return dedup_signals(signals)


def mean_reversion_signals(df: pd.DataFrame, h1: pd.DataFrame, instrument: str) -> list[Signal]:
    """Mean-reversion range scalper, ADX + Bollinger (strategy.md §5).

    Fades 2.5-sigma band extremes while ADX confirms a sideways regime.
    NaNs (warmup) fail ``adx < ADX_MAX``, so an unconfirmed regime blocks.
    """
    a = atr(df["high"], df["low"], df["close"], ATR_PERIOD)
    adx14 = adx(df["high"], df["low"], df["close"], ADX_PERIOD)
    mid, upper, lower = bollinger(df["close"], BB_PERIOD, BB_MULT)
    r = rsi(df["close"], RSI_PERIOD)
    swing_high = df["high"].shift(1).rolling(SWING_N).max()
    swing_low = df["low"].shift(1).rolling(SWING_N).min()
    signals: list[Signal] = []

    for t, row in df.iterrows():
        if not (adx14.loc[t] < ADX_MAX):
            continue
        if row["close"] < lower.loc[t] and r.loc[t] < MR_RSI_OVERSOLD:
            sl = swing_low.loc[t] - MR_SL_ATR_MULT * a.loc[t]
            if not sl < row["close"]:
                continue  # swing stop cannot bracket a crashing entry — skip
            signals.append(
                Signal(
                    strategy="mean_reversion",
                    side="BUY",
                    instrument=instrument,
                    entry=row["close"],
                    stop_loss=sl,
                    take_profit=mid.loc[t],
                    atr=a.loc[t],
                    reason="bb_oversold_adx_sideways",
                    timestamp=t,
                )
            )
        elif row["close"] > upper.loc[t] and r.loc[t] > MR_RSI_OVERBOUGHT:
            sl = swing_high.loc[t] + MR_SL_ATR_MULT * a.loc[t]
            if not sl > row["close"]:
                continue
            signals.append(
                Signal(
                    strategy="mean_reversion",
                    side="SELL",
                    instrument=instrument,
                    entry=row["close"],
                    stop_loss=sl,
                    take_profit=mid.loc[t],
                    atr=a.loc[t],
                    reason="bb_overbought_adx_sideways",
                    timestamp=t,
                )
            )

    return dedup_signals(signals)


STRATEGY_REGISTRY: dict[str, Callable[[pd.DataFrame, pd.DataFrame, str], list[Signal]]] = {
    "asia_sweep": asia_sweep_signals,
    "ema_fvg": ema_fvg_signals,
    "atr_breakout": atr_breakout_signals,
    "mean_reversion": mean_reversion_signals,
}

# config.STRATEGY_IDS is the canonical ID set (strategy.md §6.1): keep the
# registry in lockstep so the STRATEGIES env validation never accepts an ID
# with no implementation.
assert set(STRATEGY_REGISTRY) == STRATEGY_IDS


def dedup_key(signal: Signal) -> tuple[str, str, date, str]:
    """Dedup identity (strategy.md §6.2): (strategy, instrument, day, side)."""
    return (signal.strategy, signal.instrument, signal.timestamp.date(), signal.side)


def dedup_signals(signals: list[Signal]) -> list[Signal]:
    """Keep the first signal per (strategy, instrument, day, side).

    The strategy functions re-run over the full frame on every closed candle,
    so the same setup re-fires until the engine marks it invalidated or
    filled. This guarantees one signal per setup per batch; the engine's
    pending map (Task 15) extends the guarantee across batches via
    :func:`dedup_key`.
    """
    seen: set[tuple[str, str, date, str]] = set()
    unique: list[Signal] = []
    for signal in signals:
        key = dedup_key(signal)
        if key in seen:
            continue
        seen.add(key)
        unique.append(signal)
    return unique
