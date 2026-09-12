#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Loading OHLCV bars, and a synthetic generator for tests and demos."""

import warnings

import numpy as np
import pandas as pd

OHLC = ['open', 'high', 'low', 'close']
COLUMN_ALIASES = {
    'adj close': 'close', 'adj_close': 'close', 'adjclose': 'close',
    'vol': 'volume', 'date': 'date', 'timestamp': 'date', 'datetime': 'date',
}


def normalise_ohlcv(df, date_column=None, float_tolerance=1e-8):
    """Lower-case the columns, index by date, sort, and validate the bars.

    ``float_tolerance`` is the relative slack allowed before a bar counts as
    genuinely inconsistent.  Adjusted prices are a product of two floats, so a
    bar whose high equals its close can land an ULP below it -- real Yahoo
    history does this a couple of times per decade.  Violations inside the
    tolerance are clamped; anything larger is real corruption and raises.
    """
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

    # Aliasing can collide -- a frame carrying both "Close" and "Adj Close"
    # ends up with two columns called "close", which would otherwise slice
    # into a DataFrame and corrupt everything downstream in silence.
    if df.columns.duplicated().any():
        clashing = sorted(set(df.columns[df.columns.duplicated()]))
        raise ValueError(
            f'duplicate columns after normalisation: {clashing}. Drop the '
            'redundant price column before loading (for Yahoo data, prefer '
            'auto-adjusted bars, which carry no separate adjusted close).')

    keep = OHLC + (['volume'] if 'volume' in df.columns else [])
    df = df[keep].astype(float).sort_index()
    df = df[~df.index.duplicated(keep='last')]
    df = df.dropna(subset=OHLC)
    df = df[(df[OHLC] > 0).all(axis=1)]

    shortfall = df[['open', 'close', 'low']].max(axis=1) - df['high']
    overshoot = df['low'] - df[['open', 'close', 'high']].min(axis=1)
    tolerance = float_tolerance * df['close'].abs()

    serious = (shortfall > tolerance) | (overshoot > tolerance)
    if serious.any():
        first = df.index[serious][0]
        worst = float(pd.concat([shortfall, overshoot], axis=1).max(axis=1).max())
        raise ValueError(
            f'{int(serious.sum())} bars have high/low inconsistent with '
            f'open/close beyond rounding (first {first.date()}, worst '
            f'{worst:.4g}); the feed is wrong, not merely imprecise')

    df['high'] = df[['high', 'open', 'close']].max(axis=1)
    df['low'] = df[['low', 'open', 'close']].min(axis=1)
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


def load_yfinance(ticker, start='2010-01-01', end=None, auto_adjust=True,
                  session=None, retry_with_requests=True, **kwargs):
    """Daily bars from Yahoo Finance.

    ``auto_adjust`` is on by default and should stay on: it applies split and
    dividend adjustments to every OHLC field consistently.  Unadjusted prices
    put artificial gaps in the series on every split and ex-dividend date,
    which the swing detector reads as genuine reversals and the excursion
    labels record as real moves.

    ``retry_with_requests`` covers a transport quirk rather than anything about
    Yahoo: recent yfinance versions fetch through ``curl_cffi``, which fails
    against TLS-terminating corporate proxies.  A plain ``requests`` session
    usually gets through, so one is tried before giving up.  Pass your own
    ``session`` to control this.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError('yfinance is required for load_yfinance; '
                          'pip install yfinance') from exc

    if not isinstance(ticker, str):
        raise TypeError('load_yfinance handles one symbol at a time; '
                        f'got {type(ticker).__name__}')

    def fetch(active_session):
        return yf.download(ticker, start=start, end=end, interval='1d',
                           auto_adjust=auto_adjust, progress=False,
                           multi_level_index=False, session=active_session,
                           **kwargs)

    frame = fetch(session)
    if _is_empty(frame) and session is None and retry_with_requests:
        import requests
        fallback = requests.Session()
        fallback.headers.update({'User-Agent': 'Mozilla/5.0'})
        frame = fetch(fallback)

    if _is_empty(frame):
        raise ValueError(
            f'Yahoo Finance returned no rows for {ticker!r} between '
            f'{start} and {end or "today"}. Check the symbol (Yahoo uses '
            'suffixes such as VOD.L or BMW.DE for non-US listings).')

    frame = frame.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    if not auto_adjust:
        # Keep the raw OHLC set consistent rather than mixing an adjusted
        # close into unadjusted highs and lows.
        frame = frame.drop(columns=[c for c in frame.columns
                                    if str(c).strip().lower() in
                                    ('adj close', 'adj_close', 'adjclose')])
        warnings.warn('auto_adjust=False returns unadjusted prices; splits and '
                      'dividends will appear as real gaps to the swing '
                      'detector and the excursion labels', stacklevel=2)

    index = pd.DatetimeIndex(frame.index)
    if index.tz is not None:
        frame.index = index.tz_localize(None)
    return normalise_ohlcv(frame)


def _is_empty(frame):
    return frame is None or len(frame) == 0


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
