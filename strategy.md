# Strategy Specification — Algorithmic Trading Signals (Python-Ready)

This document defines four quantifiable strategies with exact rules, parameters,
and Python/pandas-ready conditions. Each strategy emits a **Signal** object that
the engine feeds through Stage 2 filters (circuit breaker → news blackout) and
Stage 3 (DeepSeek veto) before order dispatch.

All strategies execute on **M15** candles. H1 is used only as a trend filter.

This document also records the strategy review resolutions from the trading-bot
plan (Tasks 24–27): the lookahead convention (§1.6), the direction-safe asia TP
formula (§2.3), the rolling squeeze lookback (`BO_SQUEEZE_LOOKBACK`, §4.2), the
setup lifecycle (§6.4), and the sizing guards (§6.5).

---

## 1. Common Conventions

### 1.1 DataFrame schema

All strategies consume an OHLCV `pandas.DataFrame` indexed by **UTC datetime**:

| column   | dtype   | meaning                          |
|----------|---------|----------------------------------|
| `open`   | float64 | bar open                         |
| `high`   | float64 | bar high                         |
| `low`    | float64 | bar low                          |
| `close`  | float64 | bar close                        |
| `volume` | float64 | optional                         |

- Candles are **closed bars only** (no partial bars) to avoid lookahead bias.
- Every rolling window that feeds an entry decision is shifted by 1 bar
  (`shift(1)`) so the signal only uses information available at bar close.

### 1.2 Indicator helpers (implement in `TradingBot/app/indicators/technical.py`)

```python
import numpy as np
import pandas as pd


def ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False).mean()


def sma(s: pd.Series, period: int) -> pd.Series:
    return s.rolling(period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def bollinger(close: pd.Series, period: int = 20, mult: float = 2.5):
    mid = sma(close, period)
    std = close.rolling(period).std(ddof=0)
    return mid, mid + mult * std, mid - mult * std  # middle, upper, lower


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    atr_s = atr(high, low, close, period)
    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(
        alpha=1 / period, adjust=False).mean() / atr_s
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(
        alpha=1 / period, adjust=False).mean() / atr_s
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def donchian(high: pd.Series, low: pd.Series, period: int = 20):
    hh = high.shift(1).rolling(period).max()  # prior N bars, lookahead-safe
    ll = low.shift(1).rolling(period).min()
    return hh, ll


def bullish_fvg(df: pd.DataFrame) -> pd.Series:
    return df["low"] > df["high"].shift(2)


def bearish_fvg(df: pd.DataFrame) -> pd.Series:
    return df["high"] < df["low"].shift(2)
```

### 1.3 Resampling M15 → H1

```python
h1 = df.resample("1h").agg(
    {"open": "first", "high": "max", "low": "min", "close": "last"}
).dropna()
```

### 1.4 Session range helper (per calendar day)

```python
def session_range(df: pd.DataFrame, start: str, end: str):
    """Return daily high/low Series (keyed by date) within [start, end) UTC."""
    window = df.between_time(start, end)
    highs = window["high"].groupby(window.index.date).max()
    lows = window["low"].groupby(window.index.date).min()
    return highs, lows
```

### 1.5 Signal output contract

All strategies return the same structure (pydantic model `Signal` in
`TradingBot/app/models/schemas.py`):

| field             | type   | values / meaning                              |
|-------------------|--------|-----------------------------------------------|
| `strategy`        | str    | `"asia_sweep"`, `"ema_fvg"`, `"atr_breakout"`, `"mean_reversion"` |
| `side`            | str    | `"BUY"` or `"SELL"`                           |
| `instrument`      | str    | e.g. `"XAU_USD"`                              |
| `entry`           | float  | entry price (market or limit)                 |
| `stop_loss`       | float  | hard SL price                                 |
| `take_profit`     | float  | TP price                                      |
| `atr`             | float  | ATR14 at signal time (for sizing/trailing)    |
| `reason`          | str    | machine-readable trigger summary              |
| `timestamp`       | dt     | bar close time (UTC)                          |
| `pending_ai_veto` | bool   | always `True` — Stage 3 decides `execute`     |

Sizing (Stage 4) uses `entry` and `stop_loss`:
`units = (balance × 0.01) / |entry − stop_loss|`.

### 1.6 Lookahead convention (review resolution)

Every strategy signal is free of lookahead bias by construction (enforced by
the lookahead-safety tests in `TradingBot/tests/test_signals.py`):

1. **Indicators are read at bar close.** Entries fire only on closed bars —
   candle `t` is fully formed before its OHLC feeds any decision. No
   in-progress bar ever enters the pipeline.
2. **Channels/swings are shifted 1 bar.** Every rolling reference window a
   decision reads — Donchian channel, swing high/low, session-range extreme,
   squeeze flag, H1 trend — is shifted one bar (`shift(1)`), so the triggering
   bar `t` never participates in its own reference window. H1 trend flags are
   additionally shifted one H1 bar and merged backward (`merge_asof`) so an M15
   bar only ever sees H1 bars that closed before it.

Warm-up NaNs are treated as "no signal": a bar whose lookback is not yet
complete simply produces nothing.

---

## 2. Strategy 1 — Asian Session Liquidity Sweep & Reversal

- **ID:** `asia_sweep`
- **Best for:** XAU/USD, EUR/USD, GBP/USD
- **Timeframe:** M15
- **Sessions (UTC, configurable):** Asian range `00:00–07:00`; sweep window `07:00–15:00`

### 2.1 Concept

Stops above/below the Asian range are swept during London/NY open; price closes
back inside the range and reverses toward the opposite side.

### 2.2 Parameters

| parameter        | default | meaning                              |
|------------------|---------|--------------------------------------|
| `ASIA_START`     | `00:00` | range start (UTC)                    |
| `ASIA_END`       | `07:00` | range end (UTC)                      |
| `SWEEP_START`    | `07:00` | trigger window start (UTC)           |
| `SWEEP_END`      | `15:00` | trigger window end (UTC)             |
| `ATR_PERIOD`     | `14`    | ATR lookback                         |
| `SL_ATR_BUFFER`  | `0.3`   | SL buffer beyond sweep extreme       |
| `TP_RR`          | `2.5`   | reward:risk (fallback TP)            |

### 2.3 Rules (bearish example; mirror for bullish)

1. Compute per-day Asian range:
   - `H_asia = max(high)` over `00:00–07:00`
   - `L_asia = min(low)` over `00:00–07:00`
2. **Sweep trigger** (candle `t` inside sweep window, closed bar):
   - `high[t] > H_asia`  AND  `close[t] < H_asia`  (rejection back inside)
3. **FVG confirmation** (three-candle gap in trade direction):
   - bearish: `high[t] < low[t-2]`
   - bullish: `low[t] > high[t-2]`
4. **Entry:** market at `close[t]` (optionally limit at FVG boundary).
5. **Stop Loss:** `max(high[sweep]) + SL_ATR_BUFFER × ATR14` (bearish).
6. **Take Profit (direction-safe):** `min(L_asia, entry − TP_RR × (SL − entry))`
   for a SELL, mirrored as `max(H_asia, entry + TP_RR × (entry − SL))` for a
   BUY. The range extreme and the RR projection are compared on the trade side,
   so the TP can never land on the wrong side of entry.

### 2.4 Python-ready pseudocode

```python
def asia_sweep_signals(df: pd.DataFrame) -> list[dict]:
    signals = []
    atr14 = atr(df["high"], df["low"], df["close"], 14)
    highs, lows = session_range(df, "00:00", "07:00")

    for day, h_asia in highs.items():
        l_asia = lows[day]
        day_df = df[df.index.date == day]
        sweep = day_df.between_time("07:00", "15:00")
        # Sweep extremes through bar t (cummax/cummin include t, known at close)
        swept_highs = sweep["high"].cummax()
        swept_lows = sweep["low"].cummin()

        for t, row in sweep.iterrows():
            # bearish sweep + rejection
            if row["high"] > h_asia and row["close"] < h_asia:
                fvg = row["high"] < day_df["low"].shift(2).loc[t]  # bearish FVG
                if fvg:
                    sl = swept_highs.loc[t] + 0.3 * atr14.loc[t]
                    tp = min(l_asia, row["close"] - 2.5 * (sl - row["close"]))
                    signals.append({
                        "strategy": "asia_sweep", "side": "SELL",
                        "entry": row["close"], "stop_loss": sl,
                        "take_profit": tp, "atr": atr14.loc[t],
                        "reason": "bearish_asia_sweep_fvg", "timestamp": t,
                    })
            # bullish mirror: swept below the range low, rejected back inside
            if row["low"] < l_asia and row["close"] > l_asia and row["low"] > day_df["high"].shift(2).loc[t]:
                sl = swept_lows.loc[t] - 0.3 * atr14.loc[t]
                tp = max(h_asia, row["close"] + 2.5 * (row["close"] - sl))
                signals.append({
                    "strategy": "asia_sweep", "side": "BUY",
                    "entry": row["close"], "stop_loss": sl,
                    "take_profit": tp, "atr": atr14.loc[t],
                    "reason": "bullish_asia_sweep_fvg", "timestamp": t,
                })
    return signals
```

> Deduplicate: only one signal per day per side; track a `pending` flag in the
> engine until the setup is invalidated or filled.

---

## 3. Strategy 2 — Multi-EMA Trend + FVG Imbalance Retrace

- **ID:** `ema_fvg`
- **Best for:** strong trending markets (H1 trend filter, M15 execution)
- **Timeframe:** H1 trend + M15 entry

### 3.1 Concept

Buy/sell dips into Fair Value Gap zones aligned with the higher-timeframe trend.

### 3.2 Parameters

| parameter      | default      | meaning                              |
|----------------|--------------|--------------------------------------|
| `EMA_FAST`     | `20`         | fast EMA                             |
| `EMA_MID`      | `50`         | mid EMA                              |
| `EMA_SLOW`     | `200`        | slow EMA (trend)                     |
| `TREND_TF`     | `1h`         | trend filter timeframe               |
| `RSI_PERIOD`   | `14`         | momentum filter                      |
| `RSI_LOW`      | `40`         | pullback lower bound (bullish)       |
| `RSI_HIGH`     | `53`         | pullback upper bound (bullish)       |
| `SL_ATR_BUFFER`| `0.5`        | SL buffer beyond FVG extreme         |
| `TP_RR`        | `2.0`        | reward:risk                          |

### 3.3 Rules (bullish example; mirror for bearish)

1. **Trend filter (H1):** `EMA20 > EMA50 > EMA200` (bullish stack).
2. **Bullish FVG (M15):** `low[t] > high[t-2]`; zone = `[high[t-2], low[t]]`.
3. **Entry:** limit at `high[t-2]` when price re-enters the zone AND
   `40 ≤ RSI14 ≤ 53`.
4. **Stop Loss:** `low[t-2] − 0.5 × ATR14`.
5. **Take Profit:** `entry + 2.0 × (entry − SL)` (or `2.5`).

### 3.4 Python-ready pseudocode

```python
def ema_fvg_signals(m15: pd.DataFrame, h1: pd.DataFrame) -> list[dict]:
    # H1 trend
    trend_up = (
        (ema(h1["close"], 20) > ema(h1["close"], 50))
        & (ema(h1["close"], 50) > ema(h1["close"], 200))
    )
    trend_down = (
        (ema(h1["close"], 20) < ema(h1["close"], 50))
        & (ema(h1["close"], 50) < ema(h1["close"], 200))
    )

    r = rsi(m15["close"], 14)
    a = atr(m15["high"], m15["low"], m15["close"], 14)
    signals = []

    for t, row in m15.iterrows():
        hour = t.replace(minute=0, second=0, microsecond=0)
        fvg_up = row["low"] > m15["high"].shift(2).loc[t]
        fvg_dn = row["high"] < m15["low"].shift(2).loc[t]

        if trend_up.iloc[h1.index.get_loc(hour)] and fvg_up:
            if 40 <= r.loc[t] <= 53:
                entry = m15["high"].shift(2).loc[t]      # zone top
                sl = m15["low"].shift(2).loc[t] - 0.5 * a.loc[t]
                tp = entry + 2.0 * (entry - sl)
                signals.append({"strategy": "ema_fvg", "side": "BUY",
                                "entry": entry, "stop_loss": sl,
                                "take_profit": tp, "atr": a.loc[t],
                                "reason": "bullish_ema_stack_fvg_pullback",
                                "timestamp": t})
        # bearish mirror: trend_down and fvg_dn, RSI in [47, 60]
    return signals
```

> Implementation note: map each M15 bar to its H1 bucket (`floor(t, '1h')`) and
> read the H1 trend flag at that bucket; avoid `get_loc` lookups in production by
> pre-joining H1 flags onto M15 via `pd.merge_asof`.

---

## 4. Strategy 3 — ATR Volatility Breakout & Donchian Squeeze

- **ID:** `atr_breakout`
- **Best for:** XAU/USD (gold) during NY open `13:30–17:00` UTC
- **Timeframe:** M15

### 4.1 Concept

Low-volatility squeeze expands into an explosive breakout; ride momentum with a
trailing stop.

### 4.2 Parameters

| parameter        | default | meaning                              |
|------------------|---------|--------------------------------------|
| `ATR_PERIOD`     | `14`    | ATR lookback                         |
| `ATR_SMA_PERIOD` | `50`    | ATR mean lookback                    |
| `SQUEEZE_MULT`   | `0.82`  | squeeze threshold vs ATR SMA         |
| `BO_SQUEEZE_LOOKBACK` | `5` | breakout fires if any of the prior N bars was a squeeze |
| `DONCHIAN_N`     | `20`    | breakout channel lookback            |
| `SL_ATR_MULT`    | `1.8`   | initial stop distance                |
| `TRAIL_EMA`      | `21`    | trailing stop EMA                    |
| `TP_ATR_MULT`    | `2.5`   | expansion exit                       |
| `SESSION_START`  | `13:30` | NY open (UTC)                        |
| `SESSION_END`    | `17:00` | NY close (UTC)                       |

### 4.3 Rules

1. **Squeeze filter (pre-breakout, rolling):**
   `ATR14 < SMA(ATR14, 50) × 0.82` in **any** of the prior
   `BO_SQUEEZE_LOOKBACK` bars (`shift(1)`-safe rolling window) — a recent
   squeeze primes the breakout; the bar immediately before the breakout is
   not required to be the squeezed one.
2. **Long breakout:**
   `close[t] > max(high[t-1 .. t-20])` AND `ATR14 > SMA(ATR14, 50)`
3. **Short breakout:**
   `close[t] < min(low[t-1 .. t-20])` AND `ATR14 > SMA(ATR14, 50)`
4. **Entry:** market on bar close confirming breakout.
5. **Stop Loss:** `entry − 1.8 × ATR14` (long).
6. **Exit (whichever first):**
   - trailing stop on M15 `EMA21` touch; or
   - `2.5 × ATR14` expansion from entry.

### 4.4 Python-ready pseudocode

```python
def atr_breakout_signals(df: pd.DataFrame) -> list[dict]:
    a = atr(df["high"], df["low"], df["close"], 14)
    a_sma = sma(a, 50)
    hh, ll = donchian(df["high"], df["low"], 20)
    squeeze = a < a_sma * 0.82
    expand = a > a_sma
    squeeze_recent = squeeze.shift(1).rolling(BO_SQUEEZE_LOOKBACK).max()
    signals = []

    for t, row in df.between_time("13:30", "17:00").iterrows():
        if not squeeze_recent.loc[t]:     # any squeeze in the prior N bars
            continue
        if expand.loc[t] and row["close"] > hh.loc[t]:
            sl = row["close"] - 1.8 * a.loc[t]
            tp = row["close"] + 2.5 * a.loc[t]
            signals.append({"strategy": "atr_breakout", "side": "BUY",
                            "entry": row["close"], "stop_loss": sl,
                            "take_profit": tp, "atr": a.loc[t],
                            "reason": "donchian_breakout_squeeze",
                            "timestamp": t})
        elif expand.loc[t] and row["close"] < ll.loc[t]:
            sl = row["close"] + 1.8 * a.loc[t]
            tp = row["close"] - 2.5 * a.loc[t]
            signals.append({"strategy": "atr_breakout", "side": "SELL",
                            "entry": row["close"], "stop_loss": sl,
                            "take_profit": tp, "atr": a.loc[t],
                            "reason": "donchian_breakdown_squeeze",
                            "timestamp": t})
    return signals
```

> The engine applies the `EMA21` trailing exit after entry; the signal's
> `take_profit` is the expansion-based fallback.

---

## 5. Strategy 4 — Mean Reversion Range Scalper (ADX + Bollinger Bands)

- **ID:** `mean_reversion`
- **Best for:** low-volatility / choppy range-bound markets (FX majors)
- **Timeframe:** M15

### 5.1 Concept

Fade extreme deviations from the mean when no directional trend exists.

### 5.2 Parameters

| parameter     | default | meaning                              |
|---------------|---------|--------------------------------------|
| `ADX_PERIOD`  | `14`    | ADX lookback                         |
| `ADX_MAX`     | `20`    | regime threshold (sideways)          |
| `BB_PERIOD`   | `20`    | Bollinger lookback                   |
| `BB_MULT`     | `2.5`   | Bollinger width                      |
| `RSI_PERIOD`  | `14`    | RSI lookback                         |
| `RSI_OVERSOLD`| `28`    | long threshold                       |
| `RSI_OVERBOUGHT`| `72`  | short threshold                      |
| `TP_MID`      | `true`  | exit at SMA20 (middle band)          |
| `TP_RR`       | `1.5`   | fallback reward:risk                 |
| `SL_ATR_MULT` | `1.2`   | SL beyond swing extreme              |

### 5.3 Rules

1. **Regime filter:** `ADX14 < 20`.
2. **Long entry:** `close[t] < lowerBB(20, 2.5)` AND `RSI14 < 28`.
3. **Short entry:** `close[t] > upperBB(20, 2.5)` AND `RSI14 > 72`.
4. **Take Profit:** SMA20 (middle band) touch, or `1.5` R:R.
5. **Stop Loss:** `1.2 × ATR14` beyond the recent swing high/low.

### 5.4 Python-ready pseudocode

```python
def mean_reversion_signals(df: pd.DataFrame) -> list[dict]:
    a = atr(df["high"], df["low"], df["close"], 14)
    adx14 = adx(df["high"], df["low"], df["close"], 14)
    mid, upper, lower = bollinger(df["close"], 20, 2.5)
    r = rsi(df["close"], 14)
    swing_high = df["high"].shift(1).rolling(10).max()
    swing_low = df["low"].shift(1).rolling(10).min()
    signals = []

    for t, row in df.iterrows():
        if adx14.loc[t] >= 20:
            continue
        if row["close"] < lower.loc[t] and r.loc[t] < 28:
            sl = swing_low.loc[t] - 1.2 * a.loc[t]
            tp = mid.loc[t]  # middle band
            signals.append({"strategy": "mean_reversion", "side": "BUY",
                            "entry": row["close"], "stop_loss": sl,
                            "take_profit": tp, "atr": a.loc[t],
                            "reason": "bb_oversold_adx_sideways",
                            "timestamp": t})
        elif row["close"] > upper.loc[t] and r.loc[t] > 72:
            sl = swing_high.loc[t] + 1.2 * a.loc[t]
            tp = mid.loc[t]
            signals.append({"strategy": "mean_reversion", "side": "SELL",
                            "entry": row["close"], "stop_loss": sl,
                            "take_profit": tp, "atr": a.loc[t],
                            "reason": "bb_overbought_adx_sideways",
                            "timestamp": t})
    return signals
```

---

## 6. Strategy Registry & Engine Integration

### 6.1 Selection

Which strategies run is controlled by the `STRATEGIES` env var
(comma-separated IDs), defaulting to all four:

```
STRATEGIES=asia_sweep,ema_fvg,atr_breakout,mean_reversion
```

The engine maps each ID to its signal function via a registry:

```python
STRATEGY_REGISTRY = {
    "asia_sweep": asia_sweep_signals,
    "ema_fvg": ema_fvg_signals,
    "atr_breakout": atr_breakout_signals,
    "mean_reversion": mean_reversion_signals,
}
```

### 6.2 Shared safety guarantees

1. **Lookahead-safe:** all channel/rolling values use `shift(1)`; entries fire on
   closed-bar close.
2. **One signal per setup:** engine deduplicates by `(strategy, instrument, day,
   side)` through the setup lifecycle (§6.4) — pending until filled,
   invalidated, or expired.
3. **Hard stops always present:** every signal carries a numeric `stop_loss`.
4. **AI veto is mandatory:** signals are never executed directly — Stage 3
   (DeepSeek) returns `{execute, confidence, reason}` and only `execute=true`
   with confidence ≥ `MIN_CONFIDENCE` proceeds.

### 6.3 Instrument-to-strategy mapping (recommended defaults)

| instrument          | active strategies                              |
|---------------------|------------------------------------------------|
| `XAU_USD`           | `asia_sweep`, `atr_breakout`                   |
| `EUR_USD`/`GBP_USD` | `asia_sweep`, `ema_fvg`, `mean_reversion`      |

### 6.4 Setup lifecycle (review resolution)

Each setup — one `(strategy, instrument, day, side)` key — moves through an
explicit state machine in the engine:

| state         | meaning                                                        |
|---------------|----------------------------------------------------------------|
| `pending`     | signal emitted; key registered; awaiting Stage 3 veto / dispatch |
| `filled`      | MARKET order dispatched successfully                           |
| `invalidated` | a per-strategy predicate says the setup no longer exists       |
| `expired`     | UTC day rolls over; stale keys are dropped                     |

- A key enters `pending` the first time its signal is seen for the day
  (dedup via §6.2).
- A successful dispatch marks the key `filled`, freeing it for a new setup.
- Each strategy exposes an `is_invalidated(df, signal)` predicate in
  `TradingBot/app/strategy/signals.py`; when it returns true the key is freed as
  `invalidated` and a fresh setup may form on the same
  `(strategy, instrument, day, side)`.
- At day rollover every still-`pending` key becomes `expired` and is dropped,
  so the next UTC day starts clean.

Implemented by Task 26 (`TradingBot/app/core/engine.py`); the predicates ship in
`TradingBot/app/strategy/signals.py`.

### 6.5 Sizing guards (review resolution)

Stage 4 sizing (`TradingBot/app/strategy/sizing.py`) refuses to build an order that
violates an instrument or margin limit. `build_market_order` returns `None` on
violation and the engine skips dispatch — fail-safe, never a sub-minimum or
over-leveraged order.

- **Instrument min/max units.** A per-instrument unit bound map for `XAU_USD`,
  `EUR_USD`, `GBP_USD` limits the computed size. Outside `[min, max]` →
  `build_market_order` returns `None`.
- **Margin check.** The order's notional (units × entry) is checked against
  `marginAvailable` from the account summary; when margin data is missing, a
  notional cap fallback applies. Violation → `None`.
- **Fail-safe contract.** `build_market_order` returns `dict | None`; `None`
  means "do not place this order" — the engine logs the outcome and moves on.

Implemented by Task 27 (`TradingBot/app/strategy/sizing.py`).
