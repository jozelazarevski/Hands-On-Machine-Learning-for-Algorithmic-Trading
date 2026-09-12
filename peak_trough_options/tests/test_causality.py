#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The load-bearing tests: no feature may see the future.

Every feature is recomputed on a truncated copy of the history.  If a value at
bar ``t`` changes when bars after ``t`` are deleted, that feature was reading
forward and any backtest built on it is fiction.
"""

import numpy as np
import pytest

from peak_trough_options.data import synthetic_ohlcv
from peak_trough_options.features import build_features
from peak_trough_options.swings import detect_swings, average_true_range, swing_features


@pytest.fixture(scope='module')
def bars():
    return synthetic_ohlcv(n=900, seed=3)


def test_features_do_not_change_when_the_future_is_removed(bars):
    full = build_features(bars)
    truncated = build_features(bars.iloc[:-120])
    overlap = truncated.index

    assert list(full.columns) == list(truncated.columns)
    for column in full.columns:
        np.testing.assert_allclose(
            full.loc[overlap, column].to_numpy(dtype=float),
            truncated[column].to_numpy(dtype=float),
            equal_nan=True, rtol=1e-12, atol=1e-12,
            err_msg=f'feature {column!r} depends on future bars')


def test_swing_features_are_causal(bars):
    full = swing_features(bars)
    truncated = swing_features(bars.iloc[:-200])
    overlap = truncated.index
    for column in full.columns:
        np.testing.assert_allclose(
            full.loc[overlap, column].to_numpy(dtype=float),
            truncated[column].to_numpy(dtype=float),
            equal_nan=True, rtol=1e-12, atol=1e-12,
            err_msg=f'swing feature {column!r} depends on future bars')


def test_atr_is_causal(bars):
    full = average_true_range(bars['high'], bars['low'], bars['close'])
    cut = bars.iloc[:-50]
    truncated = average_true_range(cut['high'], cut['low'], cut['close'])
    np.testing.assert_allclose(full.loc[truncated.index].to_numpy(),
                               truncated.to_numpy(), equal_nan=True)


def test_pivots_are_confirmed_strictly_after_they_occur(bars):
    atr = average_true_range(bars['high'], bars['low'], bars['close'])
    pivots = detect_swings(bars['high'], bars['low'], atr, k=2.0)
    assert pivots, 'expected the generator to produce some swings'
    for pivot in pivots:
        assert pivot.confirm_index > pivot.index, (
            'a pivot cannot be known on the bar it happens')


def test_pivots_alternate_between_highs_and_lows(bars):
    atr = average_true_range(bars['high'], bars['low'], bars['close'])
    pivots = detect_swings(bars['high'], bars['low'], atr, k=2.0)
    kinds = [p.kind for p in pivots]
    assert all(a != b for a, b in zip(kinds, kinds[1:])), kinds


def test_pivot_prices_match_the_bar_they_point_at(bars):
    atr = average_true_range(bars['high'], bars['low'], bars['close'])
    for pivot in detect_swings(bars['high'], bars['low'], atr, k=2.0):
        expected = (bars['high'] if pivot.kind == 'high' else bars['low']).iloc[pivot.index]
        assert pivot.price == pytest.approx(float(expected))
