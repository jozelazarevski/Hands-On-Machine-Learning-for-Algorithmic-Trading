#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Forward-looking targets: how high and how low, and when.

For an options buyer the useful question is not "will it go up" but "how far
does it travel in each direction before my contract expires, and when".  Those
are the maximum favourable / adverse excursions (MFE / MAE) of the path.

Excursions are stored both in raw return space and scaled by the volatility
known at the decision bar.  Models are fitted on the scaled version, which is
far closer to stationary, and predictions are rescaled back to return space.
"""

import numpy as np
import pandas as pd

TARGETS = ('mfe', 'mae', 'ret')


def ewma_volatility(close, span=21, min_periods=10):
    """Causal EWMA volatility of daily log returns (per day, not annualised)."""
    log_ret = np.log(close).diff()
    return log_ret.ewm(span=span, min_periods=min_periods).std()


def forward_extremes(df, horizon):
    """Path extremes over the next ``horizon`` bars, excluding the current bar.

    Returns a frame with, for each bar ``t``:

    ``mfe``      max(high[t+1 .. t+H]) / close[t] - 1
    ``mae``      min(low[t+1 .. t+H])  / close[t] - 1
    ``ret``      close[t+H] / close[t] - 1
    ``t_peak``   bars until the high was made (1 .. H)
    ``t_trough`` bars until the low was made (1 .. H)

    The final ``horizon`` rows are NaN: their outcome has not happened yet.
    """
    if horizon < 1:
        raise ValueError('horizon must be >= 1')

    high = df['high'].to_numpy(dtype=float)
    low = df['low'].to_numpy(dtype=float)
    close = df['close'].to_numpy(dtype=float)
    n = len(df)

    out = {name: np.full(n, np.nan) for name in
           ('mfe', 'mae', 'ret', 't_peak', 't_trough')}
    if n > horizon:
        hi_win = np.lib.stride_tricks.sliding_window_view(high[1:], horizon)
        lo_win = np.lib.stride_tricks.sliding_window_view(low[1:], horizon)
        valid = slice(0, n - horizon)
        base = close[valid]

        out['mfe'][valid] = hi_win.max(axis=1) / base - 1
        out['mae'][valid] = lo_win.min(axis=1) / base - 1
        out['ret'][valid] = close[horizon:] / base - 1
        out['t_peak'][valid] = hi_win.argmax(axis=1) + 1
        out['t_trough'][valid] = lo_win.argmin(axis=1) + 1

    return pd.DataFrame(out, index=df.index)


def make_labels(df, horizon, vol_span=21, min_scale=1e-4):
    """Forward extremes plus their volatility-scaled counterparts.

    The scale ``sigma_t * sqrt(H)`` uses only returns up to bar ``t``.
    """
    labels = forward_extremes(df, horizon)
    sigma = ewma_volatility(df['close'], span=vol_span)
    scale = (sigma * np.sqrt(horizon)).clip(lower=min_scale)

    labels['scale'] = scale
    for name in TARGETS:
        labels[f'{name}_z'] = labels[name] / scale
    return labels


def label_report(labels):
    """Unconditional summary of the excursion distribution."""
    cols = [c for c in labels.columns if c in TARGETS or c.endswith('_z')]
    described = labels[cols].describe(percentiles=[.1, .25, .5, .75, .9])
    return described.T
