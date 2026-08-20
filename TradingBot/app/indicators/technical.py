"""Technical indicator helpers shared by the four strategies.

Reference implementations from strategy.md §1.2 — the math is frozen to the
spec so strategies stay reproducible; only docstrings and type hints were
added. Conventions the strategies (Task 9) rely on:

- Inputs are closed-bar OHLCV DataFrames indexed by UTC datetime
  (strategy.md §1.1). No partial bars — the engine feeds these.
- Lookahead safety: any rolling window that feeds an entry decision is
  shifted by 1 bar inside the helper (`donchian`), so `x.loc[t]` never
  contains information from bar `t` itself. Callers shift one-shot
  conditions (`bullish_fvg` is a three-candle comparison and is safe as is).
- Warmups: `atr` and `rsi` start yielding values only after `period` bars
  (leading NaNs) — the strategies' per-bar loops naturally skip NaN
  comparisons.
- `session_range` honours the spec's `[start, end)` contract via
  `inclusive="left"`: a bar stamped exactly at `end` opens the next window
  and must not be counted in this one.
"""

import numpy as np
import pandas as pd


def ema(s: pd.Series, period: int) -> pd.Series:
    """Exponential moving average, seeded at the first value (adjust=False)."""
    return s.ewm(span=period, adjust=False).mean()


def sma(s: pd.Series, period: int) -> pd.Series:
    """Simple moving average; leading `period - 1` values are NaN."""
    return s.rolling(period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder smoothing); first value at bar `period`.

    A series with no losses yields 100 (rs = inf), one with no gains yields
    0 — both divide through pandas without error, per the spec.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True range: max of today's range and the gaps to the previous close."""
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average true range (Wilder smoothing); leading `period - 1` NaNs."""
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def bollinger(
    close: pd.Series, period: int = 20, mult: float = 2.5
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger bands (population std, per spec) as (middle, upper, lower)."""
    mid = sma(close, period)
    std = close.rolling(period).std(ddof=0)
    return mid, mid + mult * std, mid - mult * std


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average directional index (Wilder smoothing) in [0, 100].

    A flat series has no directional movement: DX is 0/0 = NaN, which then
    propagates through the smoothing — the strategies treat NaN as "no
    trend signal" rather than a regime, which is the safe read.
    """
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    atr_s = atr(high, low, close, period)
    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(
        alpha=1 / period, adjust=False
    ).mean() / atr_s
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(
        alpha=1 / period, adjust=False
    ).mean() / atr_s
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def donchian(
    high: pd.Series, low: pd.Series, period: int = 20
) -> tuple[pd.Series, pd.Series]:
    """Donchian channel (high, low) over the *prior* N bars — lookahead-safe.

    The `shift(1)` guarantees `hh.loc[t]` never includes bar `t` itself, so
    a breakout test `close[t] > hh.loc[t]` only uses closed information.
    """
    hh = high.shift(1).rolling(period).max()
    ll = low.shift(1).rolling(period).min()
    return hh, ll


def bullish_fvg(df: pd.DataFrame) -> pd.Series:
    """True where a three-candle bullish gap exists: low[t] > high[t-2]."""
    return df["low"] > df["high"].shift(2)


def bearish_fvg(df: pd.DataFrame) -> pd.Series:
    """True where a three-candle bearish gap exists: high[t] < low[t-2]."""
    return df["high"] < df["low"].shift(2)


def session_range(df: pd.DataFrame, start: str, end: str) -> tuple[pd.Series, pd.Series]:
    """Per-day session (high, low) Series keyed by date, within [start, end) UTC.

    `inclusive="left"` matches the spec contract: M15 bars are stamped at
    their open time, so a bar stamped exactly at `end` belongs to the next
    window (e.g. the 07:00 bar opens the sweep window, not the Asian range).
    """
    window = df.between_time(start, end, inclusive="left")
    highs = window["high"].groupby(window.index.date).max()
    lows = window["low"].groupby(window.index.date).min()
    return highs, lows


def resample_h1(df: pd.DataFrame) -> pd.DataFrame:
    """Resample M15 OHLCV to H1 (strategy.md §1.3): first/max/min/last."""
    return df.resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
