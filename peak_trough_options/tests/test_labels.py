#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Labels must describe exactly the window they claim to."""

import numpy as np
import pandas as pd
import pytest

from peak_trough_options.data import synthetic_ohlcv
from peak_trough_options.labels import forward_extremes, make_labels


@pytest.fixture(scope='module')
def bars():
    return synthetic_ohlcv(n=400, seed=11)


def _naive_extremes(df, horizon):
    """Deliberately slow reference implementation."""
    rows = []
    high, low, close = df['high'].to_numpy(), df['low'].to_numpy(), df['close'].to_numpy()
    for t in range(len(df)):
        if t + horizon >= len(df):
            rows.append((np.nan,) * 5)
            continue
        window_high = high[t + 1:t + horizon + 1]
        window_low = low[t + 1:t + horizon + 1]
        rows.append((window_high.max() / close[t] - 1,
                     window_low.min() / close[t] - 1,
                     close[t + horizon] / close[t] - 1,
                     float(window_high.argmax() + 1),
                     float(window_low.argmin() + 1)))
    return pd.DataFrame(rows, columns=['mfe', 'mae', 'ret', 't_peak', 't_trough'],
                        index=df.index)


@pytest.mark.parametrize('horizon', [1, 5, 21])
def test_matches_the_naive_implementation(bars, horizon):
    fast = forward_extremes(bars, horizon)
    slow = _naive_extremes(bars, horizon)
    for column in slow.columns:
        np.testing.assert_allclose(fast[column].to_numpy(), slow[column].to_numpy(),
                                   equal_nan=True, rtol=1e-12)


def test_the_unresolved_tail_is_missing(bars):
    horizon = 21
    labels = forward_extremes(bars, horizon)
    assert labels.iloc[-horizon:].isna().all().all()
    assert labels.iloc[:-horizon].notna().all().all()


def test_the_current_bar_is_excluded_from_its_own_label():
    # A single spike on day 0 must not show up in day 0's forward maximum.
    index = pd.bdate_range('2020-01-01', periods=6)
    df = pd.DataFrame({'open': 100.0, 'high': [200., 101., 101., 101., 101., 101.],
                       'low': 99.0, 'close': 100.0}, index=index)
    labels = forward_extremes(df, horizon=3)
    assert labels['mfe'].iloc[0] == pytest.approx(0.01)


def test_excursions_bracket_the_terminal_return(bars):
    labels = forward_extremes(bars, 21).dropna()
    assert (labels['mfe'] >= labels['ret'] - 1e-12).all()
    assert (labels['mae'] <= labels['ret'] + 1e-12).all()
    assert (labels['mfe'] >= labels['mae']).all()


def test_timing_is_inside_the_horizon(bars):
    horizon = 21
    labels = forward_extremes(bars, horizon).dropna()
    for column in ('t_peak', 't_trough'):
        assert labels[column].between(1, horizon).all()


def test_scaling_uses_only_past_returns(bars):
    horizon = 10
    full = make_labels(bars, horizon)
    truncated = make_labels(bars.iloc[:-40], horizon)
    # The volatility scale is backward looking, so it must be identical.
    np.testing.assert_allclose(full.loc[truncated.index, 'scale'].to_numpy(),
                               truncated['scale'].to_numpy(), equal_nan=True)


def test_scaled_labels_invert(bars):
    labels = make_labels(bars, 21).dropna()
    np.testing.assert_allclose(labels['mfe_z'] * labels['scale'],
                               labels['mfe'], rtol=1e-10)


def test_rejects_a_nonsense_horizon(bars):
    with pytest.raises(ValueError):
        forward_extremes(bars, 0)
