#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Volatility-scaled swing (peak/trough) detection.

A swing high is only *confirmed* once price has retraced ``k`` ATR units away
from it.  The bar at which a pivot occurred and the bar at which it became
knowable are therefore different, and every feature derived here references
the confirmation bar so that nothing looks into the future.
"""

from collections import namedtuple

import numpy as np
import pandas as pd

Pivot = namedtuple('Pivot', ['index', 'price', 'kind', 'confirm_index'])

HIGH, LOW = 'high', 'low'


def average_true_range(high, low, close, window=14):
    """Wilder's ATR, computed causally (bar ``t`` uses bars ``<= t``)."""
    prev_close = close.shift(1)
    true_range = pd.concat([high - low,
                            (high - prev_close).abs(),
                            (low - prev_close).abs()], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def detect_swings(high, low, atr, k=2.0):
    """Detect ATR-scaled zig-zag pivots.

    Returns pivots in confirmation order.  ``Pivot.index`` is the bar the
    extreme occurred on; ``Pivot.confirm_index`` is the first bar on which the
    reversal was large enough to declare it a pivot.
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    atr = np.asarray(atr, dtype=float)
    n = len(high)

    pivots = []
    # Start once ATR is available; direction is decided by the first move that
    # clears the threshold, so seed with both candidate extremes.
    start = int(np.argmax(np.isfinite(atr))) if np.isfinite(atr).any() else n
    if start >= n:
        return pivots

    direction = 0  # 0 = undecided, +1 = tracking a high, -1 = tracking a low
    hi_val, hi_idx = high[start], start
    lo_val, lo_idx = low[start], start

    for i in range(start + 1, n):
        threshold = atr[i] * k
        if not np.isfinite(threshold) or threshold <= 0:
            continue

        if high[i] > hi_val:
            hi_val, hi_idx = high[i], i
        if low[i] < lo_val:
            lo_val, lo_idx = low[i], i

        if direction >= 0 and hi_val - low[i] >= threshold and hi_idx < i:
            # Retraced far enough below the running max: that max was a peak.
            pivots.append(Pivot(hi_idx, hi_val, HIGH, i))
            direction = -1
            lo_val, lo_idx = low[i], i
        elif direction <= 0 and high[i] - lo_val >= threshold and lo_idx < i:
            pivots.append(Pivot(lo_idx, lo_val, LOW, i))
            direction = 1
            hi_val, hi_idx = high[i], i

    return pivots


def swing_features(df, k=2.0, atr_window=14):
    """Build causal features describing the most recent *confirmed* structure.

    Every column at bar ``t`` depends only on pivots with
    ``confirm_index <= t``, so truncating the frame after ``t`` leaves the row
    unchanged (see ``tests/test_causality.py``).
    """
    atr = average_true_range(df['high'], df['low'], df['close'], atr_window)
    pivots = detect_swings(df['high'], df['low'], atr, k=k)
    n = len(df)

    last_high_price = np.full(n, np.nan)
    last_high_bar = np.full(n, np.nan)
    last_low_price = np.full(n, np.nan)
    last_low_bar = np.full(n, np.nan)
    leg_direction = np.zeros(n)

    cursor = 0
    cur_hi_price = cur_hi_bar = cur_lo_price = cur_lo_bar = np.nan
    cur_dir = 0.0
    for t in range(n):
        while cursor < len(pivots) and pivots[cursor].confirm_index <= t:
            pivot = pivots[cursor]
            if pivot.kind == HIGH:
                cur_hi_price, cur_hi_bar = pivot.price, pivot.index
                cur_dir = -1.0  # a confirmed peak means we are in a down leg
            else:
                cur_lo_price, cur_lo_bar = pivot.price, pivot.index
                cur_dir = 1.0
            cursor += 1
        last_high_price[t], last_high_bar[t] = cur_hi_price, cur_hi_bar
        last_low_price[t], last_low_bar[t] = cur_lo_price, cur_lo_bar
        leg_direction[t] = cur_dir

    close = df['close'].to_numpy(dtype=float)
    bars = np.arange(n, dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        out = pd.DataFrame({
            'swing_dir': leg_direction,
            'swing_pct_from_high': close / last_high_price - 1,
            'swing_pct_from_low': close / last_low_price - 1,
            'swing_bars_since_high': bars - last_high_bar,
            'swing_bars_since_low': bars - last_low_bar,
            'swing_leg_pct': last_high_price / last_low_price - 1,
            'swing_leg_atr': (last_high_price - last_low_price) / atr.to_numpy(),
        }, index=df.index)
    return out.replace([np.inf, -np.inf], np.nan)


def pivot_frame(df, k=2.0, atr_window=14):
    """Confirmed pivots as a tidy frame, for inspection and plotting."""
    atr = average_true_range(df['high'], df['low'], df['close'], atr_window)
    pivots = detect_swings(df['high'], df['low'], atr, k=k)
    if not pivots:
        return pd.DataFrame(columns=['date', 'price', 'kind', 'confirmed_on', 'confirm_lag'])
    return pd.DataFrame([{'date': df.index[p.index],
                          'price': p.price,
                          'kind': p.kind,
                          'confirmed_on': df.index[p.confirm_index],
                          'confirm_lag': p.confirm_index - p.index}
                         for p in pivots])
