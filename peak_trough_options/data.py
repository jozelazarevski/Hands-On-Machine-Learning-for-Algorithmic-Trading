#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Loading OHLCV bars, and a synthetic generator for tests and demos."""

import numpy as np
import pandas as pd

OHLC = ['open', 'high', 'low', 'close']
COLUMN_ALIASES = {
    'adj close': 'close', 'adj_close': 'close', 'adjclose': 'close',
    'vol': 'volume', 'date': 'date', 'timestamp': 'date', 'datetime': 'date',
}


def normalise_ohlcv(df, date_column=None):
    """Lower-case the columns, index by date, sort, and validate the bars."""
    df = df.rename(columns=lambda c: COLUMN_ALIASES.get(str(c).strip().lower(),
                                                        str(c).strip().lower()))
    if date_column is None:
        date_column = 'date' if 'date' in df.columns else None
    if date_column is not None:
        df = df.set_index(pd.to_datetime(df[date_column])).drop(columns=[date_column])
    else:
        df.index = pd.to_datetime(df.index)

    df.index.name = 'date'
    missing = set(OHLC) - set(df.columns)
    if missing:
        raise ValueError(f'missing required columns: {sorted(missing)}')

    keep = OHLC + (['volume'] if 'volume' in df.columns else [])
    df = df[keep].astype(float).sort_index()
    df = df[~df.index.duplicated(keep='last')]
    df = df.dropna(subset=OHLC)
    df = df[(df[OHLC] > 0).all(axis=1)]

    inconsistent = (df['high'] < df[['open', 'close', 'low']].max(axis=1)) | \
                   (df['low'] > df[['open', 'close', 'high']].min(axis=1))
    if inconsistent.any():
        raise ValueError(f'{int(inconsistent.sum())} bars have high/low '
                         'inconsistent with open/close')
    return df


def load_csv(path, date_column=None):
    """Read an OHLCV CSV in whatever common column spelling it uses."""
    return normalise_ohlcv(pd.read_csv(path), date_column=date_column)


def load_stooq(ticker, start='2010-01-01', end=None):
    """Fetch daily bars from Stooq via pandas-datareader, if it is installed."""
    try:
        import pandas_datareader.data as web
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError('pandas-datareader is required for --ticker; '
                          'install it or pass --csv instead') from exc
    raw = web.DataReader(ticker, 'stooq', start=start, end=end)
    return normalise_ohlcv(raw.sort_index())


def synthetic_ohlcv(n=2500, seed=0, s0=100.0, start='2012-01-02'):
    """Regime-switching stochastic-volatility bars with a predictable component.

    Deliberately not a pure random walk: volatility clusters, and the drift
    depends on the previous week's return, so a model that works has something
    real to find while a leaky one is still caught by the causality tests.
    """
    rng = np.random.default_rng(seed)

    log_vol = np.log(0.012)
    vols, rets = np.empty(n), np.empty(n)
    prev_week = 0.0
    for i in range(n):
        # Persistent stochastic volatility with occasional regime jumps.
        log_vol = 0.985 * log_vol + 0.015 * np.log(0.012) + rng.normal(0, 0.08)
        if rng.random() < 0.004:
            log_vol += rng.normal(0.6, 0.2)
        vol = float(np.clip(np.exp(log_vol), 0.003, 0.12))
        drift = 0.0003 - 0.05 * prev_week          # mild weekly mean reversion
        r = drift + rng.normal(0, vol)
        vols[i], rets[i] = vol, r
        prev_week = prev_week * 0.8 + r

    close = s0 * np.exp(np.cumsum(rets))
    prev_close = np.concatenate([[s0], close[:-1]])
    # Open gaps a fraction of the daily move; the range scales with vol.
    open_ = prev_close * np.exp(rng.normal(0, vols * 0.3))
    span = np.abs(rng.normal(0, vols)) + np.abs(close / prev_close - 1)
    high = np.maximum(open_, close) * (1 + span * rng.uniform(0.2, 1.0, n))
    low = np.minimum(open_, close) * (1 - span * rng.uniform(0.2, 1.0, n))
    volume = np.exp(rng.normal(14, 0.35, n) + 4 * vols / vols.mean() * 0.1)

    index = pd.bdate_range(start=start, periods=n, name='date')
    return pd.DataFrame({'open': open_, 'high': high, 'low': low,
                         'close': close, 'volume': volume}, index=index)
