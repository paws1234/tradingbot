"""Tests for app.indicators.technical — known values and lookahead safety.

The strategies trust these helpers blindly (strategy.md §1.2), so the tests
pin three properties in particular:

- hand-computed values on small crafted series;
- warmup NaNs — strategies loop with `.loc[t]` and rely on NaN comparisons
  being False rather than on their own guards;
- lookahead safety — `donchian` must never include the current bar.
"""

from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from app.indicators.technical import (
    adx,
    atr,
    bearish_fvg,
    bollinger,
    bullish_fvg,
    donchian,
    ema,
    resample_h1,
    rsi,
    session_range,
    sma,
    true_range,
)

T0 = datetime(2026, 8, 18, 0, 0, tzinfo=timezone.utc)


def make_df(*rows: tuple[float, float, float, float]) -> pd.DataFrame:
    """OHLC DataFrame from (open, high, low, close) rows, UTC M15 index."""
    index = pd.date_range(T0, periods=len(rows), freq="15min", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=index)


def make_series(values: list[float]) -> pd.Series:
    """Float Series on the standard UTC M15 index."""
    return pd.Series(values, index=pd.date_range(T0, periods=len(values), freq="15min", tz="UTC"))


def rising_series(n: int = 30) -> pd.Series:
    """Close of a strictly rising market: each bar up by exactly 1."""
    return pd.Series(
        np.arange(1.0, n + 1.0),
        index=pd.date_range(T0, periods=n, freq="15min", tz="UTC"),
    )


def falling_series(n: int = 30) -> pd.Series:
    """Close of a strictly falling market: each bar down by exactly 1."""
    return pd.Series(
        np.arange(float(n), 0.0, -1.0),
        index=pd.date_range(T0, periods=n, freq="15min", tz="UTC"),
    )


# --- moving averages -------------------------------------------------------


def test_sma_known_value_and_warmup() -> None:
    s = sma(make_series([1.0, 2.0, 3.0, 4.0, 5.0]), 3)
    expected = pd.Series([np.nan, np.nan, 2.0, 3.0, 4.0], index=s.index)
    assert_series_equal(s, expected)


def test_ema_known_value() -> None:
    # span=3 -> alpha=2/(3+1)=0.5, seeded at the first value.
    s = ema(make_series([1.0, 2.0, 3.0]), 3)
    expected = pd.Series([1.0, 1.5, 2.25], index=s.index)
    assert_series_equal(s, expected)


def test_ema_of_constant_series_is_constant() -> None:
    s = ema(make_series([5.0] * 20), 10)
    assert (s == 5.0).all()


# --- rsi -------------------------------------------------------------------


@pytest.mark.parametrize(
    "values,expected_tail",
    [
        # Only gains -> rs = inf -> 100; only losses -> 0.
        (list(range(1, 31)), 100.0),
        (list(range(30, 0, -1)), 0.0),
    ],
)
def test_rsi_pure_trend_saturates(values: list[int], expected_tail: float) -> None:
    r = rsi(make_series([float(v) for v in values]), 14)
    assert r.iloc[:14].isna().all()  # warmup: min_periods + the NaN delta[0]
    assert (r.iloc[14:] == expected_tail).all()


def test_rsi_bounded_and_symmetric_on_mixed_series() -> None:
    # Alternating closes with no flat bars: both avg_gain and avg_loss > 0
    # after warmup, so rs is finite and the inverse series mirrors it.
    closes = 100.0 + np.cumsum([1, -1, 2, -2, 0.5, -0.5, 1.5, -1.5] * 4)
    r = rsi(make_series(closes.tolist()), 14)
    r_inverse = rsi(make_series((-closes).tolist()), 14)
    assert ((r.iloc[15:] >= 0) & (r.iloc[15:] <= 100)).all()
    assert np.allclose(r.iloc[15:] + r_inverse.iloc[15:], 100.0)


# --- true range / atr ------------------------------------------------------


def test_true_range_known_values() -> None:
    high = make_series([10.5, 11.5, 11.0])
    low = make_series([9.5, 10.5, 10.0])
    close = make_series([10.0, 11.0, 10.5])
    tr = true_range(high, low, close)
    # Bar 0 has no previous close; pandas max skips the NaN gap terms, so
    # its TR is just the bar's range.
    assert tr.iloc[0] == pytest.approx(1.0)  # 10.5 - 9.5
    assert tr.iloc[1] == pytest.approx(1.5)  # max(1.0, 1.5, 0.5)
    assert tr.iloc[2] == pytest.approx(1.0)  # max(1.0, 0.0, 1.0)


def test_atr_known_value_and_warmup() -> None:
    # Rising by 1 each bar with a 1-wide range: every TR is exactly 1, so
    # ATR is exactly 1 from its first valid bar (index 13).
    close = rising_series()
    a = atr(close, close - 1.0, close, 14)
    assert a.iloc[:13].isna().all()
    assert a.iloc[13] == pytest.approx(1.0)
    assert np.allclose(a.iloc[13:], 1.0)


# --- bollinger -------------------------------------------------------------


def test_bollinger_flat_series_collapses_to_the_price() -> None:
    mid, upper, lower = bollinger(make_series([10.0] * 25), 20, 2.5)
    assert (mid.iloc[19:] == 10.0).all()
    assert (upper.iloc[19:] == 10.0).all()
    assert (lower.iloc[19:] == 10.0).all()


def test_bollinger_known_value() -> None:
    # Population std of [1,2,3,4] is sqrt(1.25) ~ 1.118.
    mid, upper, lower = bollinger(make_series([1.0, 2.0, 3.0, 4.0]), 4, 1.0)
    assert mid.iloc[3] == pytest.approx(2.5)
    assert upper.iloc[3] == pytest.approx(3.618, abs=1e-3)
    assert lower.iloc[3] == pytest.approx(1.382, abs=1e-3)


# --- adx -------------------------------------------------------------------


def test_adx_uptrend_known_value_and_warmup() -> None:
    # Pure uptrend: minus_dm stays 0, so dx saturates at 100 once ATR (and
    # therefore +DI) becomes valid at index 13.
    close = rising_series()
    d = adx(close, close - 1.0, close, 14)
    assert d.iloc[:13].isna().all()
    assert d.iloc[13] == pytest.approx(100.0)
    assert np.allclose(d.iloc[13:], 100.0)


def test_adx_flat_series_is_nan_not_zero() -> None:
    # No directional movement -> dx = 0/0 = NaN, which propagates. The
    # strategies read NaN as "no trend signal", which is the safe read.
    flat = make_series([10.0] * 30)
    assert adx(flat, flat, flat, 14).isna().all()


def test_adx_downtrend_known_value_and_warmup() -> None:
    # Pure downtrend: minus_dm dominates, so dx saturates at 100 once ATR
    # (and therefore -DI) becomes valid at index 13 — the mirror of the
    # uptrend case above.
    close = falling_series()
    d = adx(close, close + 1.0, close, 14)
    assert d.iloc[:13].isna().all()
    assert d.iloc[13] == pytest.approx(100.0)
    assert np.allclose(d.iloc[13:], 100.0)


# --- donchian --------------------------------------------------------------


def test_donchian_known_values() -> None:
    high = make_series([1.0, 2.0, 3.0, 4.0, 5.0])
    low = make_series([1.0, 2.0, 3.0, 4.0, 5.0])
    hh, ll = donchian(high, low, 2)
    assert_series_equal(
        hh,
        pd.Series([np.nan, np.nan, 2.0, 3.0, 4.0], index=hh.index),
    )
    assert_series_equal(
        ll,
        pd.Series([np.nan, np.nan, 1.0, 2.0, 3.0], index=ll.index),
    )


def test_donchian_never_includes_the_current_bar() -> None:
    # The lookahead guarantee: at t, the current high/low is excluded even
    # when it is the extreme. A breakout test `close[t] > hh[t]` therefore
    # only ever compares against closed bars.
    high = make_series([1.0, 2.0, 3.0, 4.0, 5.0])
    low = make_series([5.0, 4.0, 3.0, 2.0, 1.0])
    hh, ll = donchian(high, low, 3)
    assert hh.iloc[-1] == 4.0 < high.iloc[-1]
    assert ll.iloc[-1] == 2.0 > low.iloc[-1]


# --- fvg -------------------------------------------------------------------


def test_bullish_fvg_detects_three_candle_gap() -> None:
    df = make_df(
        (10.0, 10.5, 9.5, 10.0),
        (10.0, 10.5, 9.5, 10.0),
        (10.0, 10.2, 9.8, 10.0),
        (11.0, 11.5, 11.0, 11.2),  # low 11.0 > high[t-2] 10.5 -> gap
        (11.0, 11.3, 10.0, 11.0),  # low 10.0 < high[t-2] 10.2 -> no gap
    )
    assert bullish_fvg(df).tolist() == [False, False, False, True, False]


def test_bearish_fvg_detects_three_candle_gap() -> None:
    df = make_df(
        (10.0, 10.5, 9.5, 10.0),
        (10.0, 10.5, 9.5, 10.0),
        (10.0, 10.2, 9.8, 10.0),
        (9.0, 9.4, 9.0, 9.2),  # high 9.4 < low[t-2] 9.5 -> gap
        (10.0, 10.5, 10.0, 10.2),  # high 10.5 > low[t-2] 9.8 -> no gap
    )
    assert bearish_fvg(df).tolist() == [False, False, False, True, False]


# --- session range ---------------------------------------------------------


def make_two_day_df() -> pd.DataFrame:
    """Two days of M15 bars, 00:00–02:45 each, with distinctive extremes."""
    index = pd.date_range(T0, periods=12, freq="15min", tz="UTC").union(
        pd.date_range(T0 + pd.Timedelta(days=1), periods=12, freq="15min", tz="UTC")
    )
    # Bars 0-7 (00:00-01:45) are the session window on each day; bar 8 is
    # stamped exactly 02:00 — the session end — and must be excluded.
    # Day 1 extremes inside the window: high 100.0, low 50.0.
    rows = [
        (50.0, 51.0, 51.0, 50.5), (60.0, 100.0, 55.0, 70.0),
        (70.0, 71.0, 50.0, 65.0), (65.0, 66.0, 60.0, 64.0),
        (64.0, 65.0, 61.0, 63.0), (63.0, 64.0, 60.0, 62.0),
        (62.0, 63.0, 59.0, 61.0), (61.0, 62.0, 58.0, 60.0),
        (60.0, 999.0, -999.0, 60.0),  # exactly 02:00 -> outside [00:00, 02:00)
        (60.0, 61.0, 57.0, 59.0), (59.0, 60.0, 56.0, 58.0),
        (58.0, 59.0, 55.0, 57.0),
        # Day 2 extremes inside the window: high 200.0, low 150.0.
        (150.0, 151.0, 151.0, 150.5), (160.0, 200.0, 155.0, 170.0),
        (170.0, 171.0, 150.0, 165.0), (165.0, 166.0, 160.0, 164.0),
        (164.0, 165.0, 161.0, 163.0), (163.0, 164.0, 160.0, 162.0),
        (162.0, 163.0, 159.0, 161.0), (161.0, 162.0, 158.0, 160.0),
        (160.0, 999.0, -999.0, 160.0),  # 02:00 on day 2
        (160.0, 161.0, 157.0, 159.0), (159.0, 160.0, 156.0, 158.0),
        (158.0, 159.0, 155.0, 157.0),
    ]
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=index)


def test_session_range_returns_daily_extremes_keyed_by_date() -> None:
    highs, lows = session_range(make_two_day_df(), "00:00", "02:00")
    assert highs.index.tolist() == [date(2026, 8, 18), date(2026, 8, 19)]
    assert highs.tolist() == [100.0, 200.0]
    assert lows.tolist() == [50.0, 150.0]


def test_session_range_excludes_the_bar_stamped_at_end() -> None:
    # The 02:00 bar carries 999/-999; inclusive="left" keeps it out of the
    # [00:00, 02:00) window, so the day's extremes come from inside it.
    highs, lows = session_range(make_two_day_df(), "00:00", "02:00")
    assert (highs < 999.0).all()
    assert (lows > -999.0).all()


# --- resampling ------------------------------------------------------------


def test_resample_h1_known_values() -> None:
    df = make_df(
        (1.0, 5.0, 0.5, 2.0), (2.0, 3.0, 1.0, 2.5),
        (3.0, 4.0, 2.0, 3.5), (4.0, 4.5, 3.5, 4.2),
        (5.0, 6.0, 4.0, 5.5), (6.0, 7.0, 5.0, 6.5),
        (7.0, 8.0, 6.0, 7.5), (8.0, 9.0, 7.0, 8.5),
    )
    h1 = resample_h1(df)
    assert h1.index.tolist() == [
        datetime(2026, 8, 18, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 8, 18, 1, 0, tzinfo=timezone.utc),
    ]
    assert h1.iloc[0].tolist() == [1.0, 5.0, 0.5, 4.2]  # open/high/low/close
    assert h1.iloc[1].tolist() == [5.0, 9.0, 4.0, 8.5]


def test_resample_h1_drops_empty_buckets() -> None:
    # Hour 1 has no bars: its bucket aggregates to all-NaN and dropna()
    # removes it, so the H1 index has no hole.
    index = pd.date_range(T0, periods=4, freq="15min", tz="UTC").union(
        pd.date_range(T0 + pd.Timedelta(hours=2), periods=4, freq="15min", tz="UTC")
    )
    df = pd.DataFrame(
        {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5}, index=index
    )
    h1 = resample_h1(df)
    assert len(h1) == 2
    assert h1.index[1] == T0 + pd.Timedelta(hours=2)


# --- index hygiene ---------------------------------------------------------


@pytest.mark.parametrize(
    "fn",
    [
        lambda df: sma(df["close"], 10),
        lambda df: ema(df["close"], 10),
        lambda df: rsi(df["close"], 14),
        lambda df: atr(df["high"], df["low"], df["close"], 14),
        lambda df: bollinger(df["close"], 20, 2.5)[0],
        lambda df: donchian(df["high"], df["low"], 20)[0],
        lambda df: bullish_fvg(df),
        lambda df: bearish_fvg(df),
        lambda df: adx(df["high"], df["low"], df["close"], 14),
    ],
)
def test_helpers_preserve_the_input_index(fn) -> None:
    # The strategies align indicators with candles via `.loc[t]`, so every
    # helper must return the exact input DatetimeIndex.
    df = make_df(*[(c, c + 1.0, c - 1.0, c) for c in np.linspace(100, 110, 30)])
    out = fn(df)
    assert out.index.equals(df.index)
