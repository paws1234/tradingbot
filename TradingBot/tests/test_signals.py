"""Tests for app.strategy.signals — registry, dedup, and one crafted setup
per strategy.

Each test drives a small, hand-tuned OHLCV series through one strategy and
checks the signal contract (strategy.md §1.5): correct side, a numeric hard
stop, SL/TP bracketing the entry on the correct side, and
``pending_ai_veto=True``. The series are deterministic — no randomness — so
the emitted signal (and its absence elsewhere) is reproducible.

Lookahead safety is pinned two ways: the indicator helpers already shift
their windows (see ``tests/test_indicators.py``), and the two tests at the
bottom of this file assert it at the signal layer — a breakout fires only on
the closed bar, and the H1 trend join reads the *previous* hour's flag, never
the current one.
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from app.config import STRATEGY_IDS
from app.indicators.technical import atr, donchian, ema, sma
from app.models.schemas import Signal
from app.strategy.signals import (
    ATR_PERIOD,
    ATR_SMA_PERIOD,
    DONCHIAN_N,
    SQUEEZE_MULT,
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
    # Direction-safe asia TP (strategy.md §2.3 rule 6):
    #   TP = min(L_asia, entry − TP_RR × (SL − entry))
    # With L_asia=99.5, entry=100.0, SL>101.5 and TP_RR=2.5 the RR projection
    # is < 100.0 − 2.5×1.5 = 96.25, so TP must be strictly below L_asia.
    assert sig.take_profit <= 99.5
    rr_projection = sig.entry - 2.5 * (sig.stop_loss - sig.entry)
    assert sig.take_profit == pytest.approx(min(99.5, rr_projection))


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
    # Direction-safe asia TP (strategy.md §2.3 rule 6):
    #   TP = max(H_asia, entry + TP_RR × (entry − SL))
    # With H_asia=100.5, entry=100.0, SL<98.5 and TP_RR=2.5 the RR projection
    # is > 100.0 + 2.5×1.5 = 103.75, so TP must be strictly above H_asia.
    assert sig.take_profit >= 100.5
    rr_projection = sig.entry + 2.5 * (sig.entry - sig.stop_loss)
    assert sig.take_profit == pytest.approx(max(100.5, rr_projection))


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
    assert len(sigs) == 1
    sig = sigs[0]
    assert sig.side == "SELL"
    assert sig.entry == pytest.approx(98.75)
    assert sig.reason == "bearish_ema_stack_fvg_pullback"
    assert sig.pending_ai_veto is True
    assert_bracketed(sig)


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


def test_atr_breakout_squeeze_then_donchian_breakdown_emits_sell() -> None:
    n = 70
    idx = pd.date_range(T0, periods=n, freq="15min", tz="UTC")
    df = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}, index=idx)
    # Narrow range 12:00 onward collapses ATR below its own SMA (squeeze).
    df.loc[idx[31]:, "high"] = 100.01
    df.loc[idx[31]:, "low"] = 99.99
    df.loc[idx[31]:, "close"] = 100.0
    # 16:00: volatility expands and price closes through the 20-bar low.
    df.loc[idx[64], ["open", "high", "low", "close"]] = [100.0, 100.1, 80.0, 95.0]

    sigs = atr_breakout_signals(df, pd.DataFrame(), "XAU_USD")

    assert len(sigs) == 1
    sig = sigs[0]
    assert sig.strategy == "atr_breakout"
    assert sig.side == "SELL"
    assert sig.entry == pytest.approx(95.0)
    assert sig.reason == "donchian_breakdown_squeeze"
    assert sig.pending_ai_veto is True
    assert_bracketed(sig)


def test_atr_breakout_fires_on_squeeze_within_lookback_not_just_predecessor() -> None:
    """A squeeze anywhere in the prior BO_SQUEEZE_LOOKBACK bars primes the
    breakout — the bar right before it need not be the squeezed one
    (strategy.md §4.3 rule 1). Here the squeeze registers at bars 62–65 and
    bar 66 (t-1) has recovered above the threshold, so the old
    immediate-predecessor check would have blocked; the rolling lookback lets
    the 16:45 breakout fire."""
    n = 90
    idx = pd.date_range(T0, periods=n, freq="15min", tz="UTC")
    df = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}, index=idx)
    # Narrow range 07:45–14:45 collapses ATR below its own SMA (squeeze) …
    df.loc[idx[31]:idx[59], "high"] = 100.01
    df.loc[idx[31]:idx[59], "low"] = 99.99
    df.loc[idx[31]:idx[59], "close"] = 100.0
    # … then the range widens, so the squeeze ends before the breakout bar.
    # 16:45: volatility expands and price closes through the 20-bar high.
    df.loc[idx[67], ["open", "high", "low", "close"]] = [100.0, 120.0, 99.9, 105.0]

    # Precondition that makes this a rolling-lookback case rather than the old
    # immediate-predecessor one: the bar before the breakout (66) is un-squeezed
    # while an earlier bar within the prior BO_SQUEEZE_LOOKBACK (62) is.
    a = atr(df["high"], df["low"], df["close"], ATR_PERIOD)
    squeeze = a < sma(a, ATR_SMA_PERIOD) * SQUEEZE_MULT
    assert not bool(squeeze.iloc[66])
    assert bool(squeeze.iloc[62])

    sigs = atr_breakout_signals(df, pd.DataFrame(), "XAU_USD")

    assert len(sigs) == 1
    sig = sigs[0]
    assert sig.side == "BUY"
    assert sig.entry == pytest.approx(105.0)
    assert sig.reason == "donchian_breakout_squeeze"
    assert sig.pending_ai_veto is True
    assert_bracketed(sig)
    assert sig.timestamp == idx[67]


def test_atr_breakout_blocks_when_no_squeeze_in_lookback_window() -> None:
    """No squeeze in the prior BO_SQUEEZE_LOOKBACK bars blocks the breakout —
    even when the bar itself expands and closes through the channel
    (strategy.md §4.3 rule 1). The squeeze ends early (bars 31–40) and the
    range stays wide through the lookback window, so the 16:45 breakout is
    correctly skipped."""
    n = 90
    idx = pd.date_range(T0, periods=n, freq="15min", tz="UTC")
    df = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}, index=idx)
    # A brief early squeeze (07:45–10:00) that is long gone by the breakout …
    df.loc[idx[31]:idx[40], "high"] = 100.01
    df.loc[idx[31]:idx[40], "low"] = 99.99
    df.loc[idx[31]:idx[40], "close"] = 100.0
    # 16:45: volatility expands and price closes through the 20-bar high.
    df.loc[idx[67], ["open", "high", "low", "close"]] = [100.0, 120.0, 99.9, 105.0]

    # Preconditions: the breakout bar itself qualifies (volatility expands and
    # close clears the channel) — the squeeze guard is the only thing standing
    # between this bar and a signal — and none of the prior BO_SQUEEZE_LOOKBACK
    # bars were squeezed.
    a = atr(df["high"], df["low"], df["close"], ATR_PERIOD)
    assert bool((a > sma(a, ATR_SMA_PERIOD)).iloc[67])
    hh, _ = donchian(df["high"], df["low"], DONCHIAN_N)
    assert df["close"].iloc[67] > hh.iloc[67]
    squeeze = a < sma(a, ATR_SMA_PERIOD) * SQUEEZE_MULT
    assert not squeeze.iloc[62:67].any()

    assert atr_breakout_signals(df, pd.DataFrame(), "XAU_USD") == []


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


def test_mean_reversion_overbought_band_fade_emits_sell() -> None:
    # Mirror of the oversold series around 100: the dip becomes a spike, so
    # the same deterministic OU data now breaks the UPPER band with RSI > 72
    # while ADX still reads a sideways regime (it is sign-flip invariant).
    closes = [200.0 - c for c in mean_reversion_closes()]
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
    assert sig.side == "SELL"
    assert sig.entry == pytest.approx(102.4055)
    assert sig.reason == "bb_overbought_adx_sideways"
    assert sig.pending_ai_veto is True
    assert_bracketed(sig)


# --- lookahead safety ------------------------------------------------------


def test_atr_breakout_fires_only_on_the_closed_breakout_bar() -> None:
    """Entries use closed bars only: the signal appears at bar 64 once that
    bar has closed, never before it, and never moved by bars after it."""
    n = 70
    idx = pd.date_range(T0, periods=n, freq="15min", tz="UTC")
    df = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}, index=idx)
    # Narrow range 12:00 onward collapses ATR below its own SMA (squeeze).
    df.loc[idx[31]:, "high"] = 100.01
    df.loc[idx[31]:, "low"] = 99.99
    df.loc[idx[31]:, "close"] = 100.0
    # 16:00: volatility expands and price closes through the 20-bar high.
    df.loc[idx[64], ["open", "high", "low", "close"]] = [100.0, 120.0, 99.9, 105.0]

    # Bar 64 is closed -> exactly one signal, stamped on it.
    sigs = atr_breakout_signals(df, pd.DataFrame(), "XAU_USD")
    assert len(sigs) == 1
    assert sigs[0].timestamp == idx[64]

    # Bar 64 not yet closed -> no signal from the closed bars before it.
    assert atr_breakout_signals(df.iloc[:64], pd.DataFrame(), "XAU_USD") == []

    # A later bar cannot create a second signal or move the original.
    later = pd.DataFrame(
        {"open": [105.0], "high": [110.0], "low": [104.0], "close": [108.0]},
        index=pd.date_range(
            idx[-1] + pd.Timedelta(minutes=15), periods=1, freq="15min", tz="UTC"
        ),
    )
    sigs = atr_breakout_signals(pd.concat([df, later]), pd.DataFrame(), "XAU_USD")
    assert len(sigs) == 1
    assert sigs[0].timestamp == idx[64]


def test_ema_fvg_trend_join_shifts_the_h1_flag_by_one_bar() -> None:
    """The H1-trend join must not leak the current hour's flag (strategy.md §6.2).

    The trend an M15 bar reads is the *previous* H1 bar's — the join shifts
    the flag series by one H1 bar before the backward asof join, so a bar at
    hour H only ever sees the flag of H-1. Verified as a control variable: the
    SAME M15 frame (identical FVG + RSI conditions) against two H1 frames whose
    first up-trended hour differs by one. In case A the target hour's own flag
    is up yet invisible to it — a leaked join would fire, the shifted one does
    not; in case B that hour's flag is up, so the target bar fires.
    """

    def make_h1(last_decline: float, _n_decline: int) -> pd.DataFrame:
        decline = np.arange(150.0, last_decline, -1.0)  # exclusive stop
        rally = last_decline + np.arange(1, 251) * 0.9
        closes = np.concatenate([decline, rally])
        return pd.DataFrame(
            {"close": closes},
            index=pd.date_range(T0, periods=len(closes), freq="1h", tz="UTC"),
        )

    def first_up_hour(h1: pd.DataFrame) -> int:
        up = (ema(h1["close"], 20) > ema(h1["close"], 50)) & (
            ema(h1["close"], 50) > ema(h1["close"], 200)
        )
        return int(np.where(up.values)[0][0])

    h1_a = make_h1(50.0, 100)
    h1_b = make_h1(51.0, 99)  # rally one hour earlier than case A
    hour_a = first_up_hour(h1_a)
    hour_b = first_up_hour(h1_b)
    assert hour_b == hour_a - 1
    # Precondition: hour_a-1 is not up-trended in case A, so the M15 bar at
    # hour_a:15 must read a non-up flag — unless the join leaked hour_a's own
    # just-closed up flag, which is exactly the bug this test pins.
    up_a = (ema(h1_a["close"], 20) > ema(h1_a["close"], 50)) & (
        ema(h1_a["close"], 50) > ema(h1_a["close"], 200)
    )
    assert not bool(up_a.iloc[hour_a - 1])

    # One M15 frame: micro-alternation (RSI ~45, a defined sideways value)
    # with a single bullish FVG gap at hour_a:15. Both cases share it, so the
    # FVG/RSI conditions are identical — only the trend flag differs.
    n = hour_a * 4 + 5
    closes = [100.0]
    for i in range(1, n):
        closes.append(closes[-1] + (-0.025 if i % 2 == 1 else 0.015))
    closes = np.array(closes)
    target = hour_a * 4 + 1  # hour_a:15
    closes[target] = closes[target - 2] + 0.045
    m15 = pd.DataFrame(
        {"open": closes, "high": closes + 0.02, "low": closes - 0.02, "close": closes},
        index=pd.date_range(T0, periods=len(closes), freq="15min", tz="UTC"),
    )

    # Case A: hour_a's own flag is up but unreadable until hour_a+1 -> none.
    assert ema_fvg_signals(m15, h1_a, "EUR_USD") == []

    # Case B: hour_a-1 is up-trended -> the hour_a:15 bar sees it and fires.
    sigs = ema_fvg_signals(m15, h1_b, "EUR_USD")
    assert len(sigs) == 1
    assert sigs[0].side == "BUY"
    assert sigs[0].timestamp == m15.index[target]
