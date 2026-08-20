"""Tests for app.strategy.signals — registry, dedup, and one crafted setup
per strategy.

Each test drives a small, hand-tuned OHLCV series through one strategy and
checks the signal contract (strategy.md §1.5): correct side, a numeric hard
stop, SL/TP bracketing the entry on the correct side, and
``pending_ai_veto=True``. The series are deterministic — no randomness — so
the emitted signal (and its absence elsewhere) is reproducible.

Lookahead safety is pinned indirectly: entries fire on a closed bar's close,
and the indicator helpers already shift their windows (see
``tests/test_indicators.py``).
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from app.config import STRATEGY_IDS
from app.models.schemas import Signal
from app.strategy.signals import (
    STRATEGY_REGISTRY,
    asia_sweep_signals,
    atr_breakout_signals,
    dedup_signals,
    ema_fvg_signals,
    mean_reversion_signals,
)

T0 = datetime(2026, 8, 18, 0, 0, tzinfo=timezone.utc)


def make_df(*rows: tuple[float, float, float, float]) -> pd.DataFrame:
    """OHLC DataFrame from (open, high, low, close) rows, UTC M15 index."""
    index = pd.date_range(T0, periods=len(rows), freq="15min", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=index)


def make_signal(
    strategy: str, instrument: str, ts: datetime, side: str = "BUY"
) -> Signal:
    """A valid Signal with SL/TP bracketing the entry on the correct side."""
    if side == "BUY":
        return Signal(
            strategy=strategy,  # type: ignore[arg-type]
            side="BUY",
            instrument=instrument,
            entry=100.0,
            stop_loss=99.0,
            take_profit=101.0,
            atr=1.0,
            reason="test",
            timestamp=ts,
        )
    return Signal(
        strategy=strategy,  # type: ignore[arg-type]
        side="SELL",
        instrument=instrument,
        entry=100.0,
        stop_loss=101.0,
        take_profit=99.0,
        atr=1.0,
        reason="test",
        timestamp=ts,
    )


def assert_bracketed(sig: Signal) -> None:
    """A hard stop must bracket the entry on the correct side (strategy.md §6.2)."""
    if sig.side == "BUY":
        assert sig.stop_loss < sig.entry < sig.take_profit
    else:
        assert sig.stop_loss > sig.entry > sig.take_profit
    assert np.isfinite(sig.stop_loss)


# --- registry & dedup ------------------------------------------------------


def test_registry_matches_the_canonical_strategy_ids() -> None:
    # config.STRATEGY_IDS is the validated STRATEGIES env set (strategy.md §6.1);
    # every accepted ID must have an implementation and vice versa.
    assert set(STRATEGY_REGISTRY) == STRATEGY_IDS
    assert len(STRATEGY_REGISTRY) == 4


def test_dedupe_keeps_first_signal_per_strategy_instrument_day_side() -> None:
    t1 = datetime(2026, 8, 18, 8, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 8, 18, 9, 0, tzinfo=timezone.utc)
    t3 = datetime(2026, 8, 19, 8, 0, tzinfo=timezone.utc)

    a1 = make_signal("asia_sweep", "XAU_USD", t1, "BUY")
    a2 = make_signal("asia_sweep", "XAU_USD", t2, "BUY")  # same key → dropped
    a3 = make_signal("asia_sweep", "XAU_USD", t1, "SELL")  # different side → kept
    a4 = make_signal("asia_sweep", "XAU_USD", t3, "BUY")  # different day → kept
    a5 = make_signal("asia_sweep", "EUR_USD", t1, "BUY")  # different instrument → kept

    result = dedup_signals([a1, a2, a3, a4, a5])
    assert result == [a1, a3, a4, a5]


# --- asia_sweep ------------------------------------------------------------


def asia_frame() -> pd.DataFrame:
    """One day of M15 bars, flat 100 ± 0.5, with a sweep+rejection at 07:30."""
    rows = [(100.0, 100.5, 99.5, 100.0)] * 64
    df = make_df(*rows)
    return df


def test_asia_sweep_bearish_rejection_emits_sell() -> None:
    df = asia_frame()
    # 07:00 bar: sweeps the Asian high but closes outside (no signal); its low
    # seeds the bearish FVG two bars later.
    df.iloc[28] = [101.3, 101.5, 101.2, 101.3]
    df.iloc[29] = [100.1, 100.4, 99.9, 100.1]
    # 07:30 bar: high > H_asia, close back inside, bearish FVG vs 07:00.
    df.iloc[30] = [100.0, 101.0, 99.8, 100.0]

    sigs = asia_sweep_signals(df, pd.DataFrame(), "XAU_USD")

    assert len(sigs) == 1
    sig = sigs[0]
    assert sig.strategy == "asia_sweep"
    assert sig.side == "SELL"
    assert sig.instrument == "XAU_USD"
    assert sig.entry == pytest.approx(100.0)
    assert sig.reason == "bearish_asia_sweep_fvg"
    assert sig.pending_ai_veto is True
    assert_bracketed(sig)
    # The stop sits above the swept high (101.5) plus an ATR buffer.
    assert sig.stop_loss > 101.5


def test_asia_sweep_bullish_rejection_emits_buy() -> None:
    df = asia_frame()
    df.iloc[28] = [98.7, 98.8, 98.5, 98.6]  # sweeps the low, closes outside
    df.iloc[29] = [99.9, 100.1, 99.6, 99.9]
    df.iloc[30] = [100.0, 100.1, 99.0, 100.0]  # low < L_asia, close inside, FVG

    sigs = asia_sweep_signals(df, pd.DataFrame(), "XAU_USD")

    assert len(sigs) == 1
    sig = sigs[0]
    assert sig.side == "BUY"
    assert sig.entry == pytest.approx(100.0)
    assert sig.reason == "bullish_asia_sweep_fvg"
    assert sig.pending_ai_veto is True
    assert_bracketed(sig)
    assert sig.stop_loss < 98.5


# --- ema_fvg ---------------------------------------------------------------


def ema_fvg_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rising H1 trend plus a choppy M15 tail ending in a bullish FVG."""
    h1 = pd.DataFrame(
        {"close": np.arange(1.0, 217.0)},
        index=pd.date_range(T0, periods=216, freq="1h", tz="UTC"),
    )
    # 49 churn bars drifting gently down (up 0.3 / down 0.5), then a 0.7 gap-up
    # whose low clears high[t-2]: a bullish FVG with RSI still in [40, 53].
    closes = [100.0]
    for i in range(48):
        closes.append(closes[-1] + (0.3 if i % 2 == 0 else -0.5))
    closes.append(closes[-1] + 0.7)
    index = pd.date_range(T0 + pd.Timedelta(hours=180), periods=50, freq="15min", tz="UTC")
    m15 = pd.DataFrame(
        {"open": closes, "high": [c + 0.05 for c in closes],
         "low": [c - 0.05 for c in closes], "close": closes},
        index=index,
    )
    return m15, h1


def test_ema_fvg_bullish_stack_pullback_emits_buy() -> None:
    m15, h1 = ema_fvg_data()
    sigs = ema_fvg_signals(m15, h1, "EUR_USD")

    assert len(sigs) == 1
    sig = sigs[0]
    assert sig.strategy == "ema_fvg"
    assert sig.side == "BUY"
    assert sig.instrument == "EUR_USD"
    assert sig.entry == pytest.approx(95.75)
    assert sig.reason == "bullish_ema_stack_fvg_pullback"
    assert sig.pending_ai_veto is True
    assert_bracketed(sig)


def test_ema_fvg_trend_direction_flips_signal_side() -> None:
    m15, h1 = ema_fvg_data()
    # The M15 tail contains gaps in both directions; the H1 EMA stack decides
    # which side may fire. A falling stack flips the setup to the bearish mirror.
    falling = pd.DataFrame(
        {"close": np.arange(216.0, 0.0, -1.0)},
        index=h1.index,
    )
    sigs = ema_fvg_signals(m15, falling, "EUR_USD")
    assert sigs
    assert all(sig.side == "SELL" for sig in sigs)


# --- atr_breakout ----------------------------------------------------------


def test_atr_breakout_squeeze_then_donchian_break_emits_buy() -> None:
    n = 70
    idx = pd.date_range(T0, periods=n, freq="15min", tz="UTC")
    df = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}, index=idx)
    # Narrow range 12:00 onward collapses ATR below its own SMA (squeeze).
    df.loc[idx[31]:, "high"] = 100.01
    df.loc[idx[31]:, "low"] = 99.99
    df.loc[idx[31]:, "close"] = 100.0
    # 16:00: volatility expands and price closes through the 20-bar high.
    df.loc[idx[64], ["open", "high", "low", "close"]] = [100.0, 120.0, 99.9, 105.0]

    sigs = atr_breakout_signals(df, pd.DataFrame(), "XAU_USD")

    assert len(sigs) == 1
    sig = sigs[0]
    assert sig.strategy == "atr_breakout"
    assert sig.side == "BUY"
    assert sig.entry == pytest.approx(105.0)
    assert sig.reason == "donchian_breakout_squeeze"
    assert sig.pending_ai_veto is True
    assert_bracketed(sig)


# --- mean_reversion --------------------------------------------------------


def mean_reversion_closes() -> list[float]:
    """Deterministic Ornstein-Uhlenbeck closes (theta=0.2, sigma=0.5, seed 7).

    Mean-reverting by construction, so ADX stays below the sideways threshold
    while the 05:15 bar dips through the lower 2.5-sigma band with RSI < 28.
    """
    return [
        100.0, 100.0006, 100.1499, 99.9828, 99.541, 99.4054, 99.0285, 99.2529,
        100.0724, 99.8118, 99.5392, 99.8763, 100.0795, 100.1163, 99.6278, 99.6876,
        100.0977, 99.4061, 99.2961, 98.4862, 98.1442, 97.5945, 97.9581, 97.7327,
        98.3218, 98.7358, 98.8952, 97.8578, 98.0169, 98.3892, 98.7681, 98.2494,
        98.3606, 98.1992, 98.155, 99.0544, 98.8398, 99.0556, 99.6866, 99.4575,
        99.5102, 99.6634, 99.7626, 99.1975, 99.3961, 100.1963, 99.3835, 99.9365,
        100.0088, 99.6863, 100.7493, 100.9806, 100.1848, 100.1851, 100.4364, 100.2547,
        100.5453, 100.4029, 100.656, 101.244,
    ]


def test_mean_reversion_oversold_band_fade_emits_buy() -> None:
    closes = mean_reversion_closes()
    index = pd.date_range(T0, periods=len(closes), freq="15min", tz="UTC")
    df = pd.DataFrame(
        {"open": closes, "high": [c + 0.3 for c in closes],
         "low": [c - 0.3 for c in closes], "close": closes},
        index=index,
    )

    sigs = mean_reversion_signals(df, pd.DataFrame(), "EUR_USD")

    assert len(sigs) == 1
    sig = sigs[0]
    assert sig.strategy == "mean_reversion"
    assert sig.side == "BUY"
    assert sig.entry == pytest.approx(97.5945)
    assert sig.reason == "bb_oversold_adx_sideways"
    assert sig.pending_ai_veto is True
    assert_bracketed(sig)
