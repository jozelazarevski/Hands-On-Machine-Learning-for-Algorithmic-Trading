#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal feature engineering.

Every column is a function of bars ``<= t`` only.  ``HistGradientBoosting``
consumes NaN natively, so warm-up rows are left as NaN rather than imputed.
"""

import numpy as np
import pandas as pd

from .swings import average_true_range, swing_features

RET_WINDOWS = (1, 2, 5, 10, 21, 63)
VOL_WINDOWS = (10, 21, 63)
RANGE_WINDOWS = (21, 63, 252)


def _rolling_trend(y, window):
    """Rolling OLS of ``y`` on time: slope per bar and R^2 of the fit."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    slope = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    if n < window:
        return slope, r2

    x = np.arange(window, dtype=float)
    xc = x - x.mean()
    sxx = (xc ** 2).sum()

    windows = np.lib.stride_tricks.sliding_window_view(y, window)
    sxy = windows @ xc
    syy = ((windows - windows.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)

    with np.errstate(invalid='ignore', divide='ignore'):
        slope[window - 1:] = sxy / sxx
        r2[window - 1:] = np.where(syy > 0, sxy ** 2 / (sxx * syy), np.nan)
    return slope, r2


def _variance_ratio(log_ret, q, window):
    """Lo-MacKinlay variance ratio: >1 trending, <1 mean-reverting."""
    var_1 = log_ret.rolling(window).var()
    var_q = log_ret.rolling(q).sum().rolling(window).var()
    with np.errstate(invalid='ignore', divide='ignore'):
        return var_q / (q * var_1)


def _rsi(close, window=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _range_vol_estimators(df, window=21):
    """Intraday-range volatility estimators (per-day, not annualised)."""
    hl = np.log(df['high'] / df['low'])
    co = np.log(df['close'] / df['open'])
    ho = np.log(df['high'] / df['open'])
    lo = np.log(df['low'] / df['open'])
    hc = np.log(df['high'] / df['close'])
    lc = np.log(df['low'] / df['close'])

    parkinson = (hl ** 2 / (4 * np.log(2))).rolling(window).mean()
    garman_klass = (0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2).rolling(window).mean()
    rogers_satchell = (hc * ho + lc * lo).rolling(window).mean()
    return (np.sqrt(parkinson.clip(lower=0)),
            np.sqrt(garman_klass.clip(lower=0)),
            np.sqrt(rogers_satchell.clip(lower=0)))


def build_features(df, swing_k=2.0):
    """Assemble the feature matrix from an OHLC(V) frame.

    ``df`` must be indexed by date and carry ``open/high/low/close`` columns;
    ``volume`` is optional and adds three columns when present.
    """
    required = {'open', 'high', 'low', 'close'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'missing required columns: {sorted(missing)}')

    close = df['close']
    log_price = np.log(close)
    log_ret = log_price.diff()
    feats = {}

    # --- momentum over several horizons, and its volatility-adjusted form ---
    daily_vol = log_ret.ewm(span=21, min_periods=10).std()
    for w in RET_WINDOWS:
        r = log_price.diff(w)
        feats[f'ret_{w}'] = r
        feats[f'ret_{w}_z'] = r / (daily_vol * np.sqrt(w))

    # --- volatility level, term structure and vol-of-vol ---
    for w in VOL_WINDOWS:
        feats[f'vol_{w}'] = log_ret.ewm(span=w, min_periods=w // 2).std()
    feats['vol_ratio_10_63'] = feats['vol_10'] / feats['vol_63']
    feats['vol_of_vol'] = feats['vol_21'].pct_change().rolling(21).std()

    parkinson, garman_klass, rogers_satchell = _range_vol_estimators(df)
    feats['vol_parkinson'] = parkinson
    feats['vol_garman_klass'] = garman_klass
    feats['vol_rogers_satchell'] = rogers_satchell
    # Range vol above close-to-close vol implies intraday chop that a
    # close-only estimator misses -- relevant when sizing a strangle.
    feats['vol_range_premium'] = parkinson / feats['vol_21']

    atr = average_true_range(df['high'], df['low'], close)
    feats['atr_pct'] = atr / close
    feats['gap'] = df['open'] / close.shift(1) - 1

    # --- where price sits inside its recent range ---
    for w in RANGE_WINDOWS:
        hi = df['high'].rolling(w).max()
        lo = df['low'].rolling(w).min()
        span = (hi - lo).replace(0, np.nan)
        feats[f'range_pos_{w}'] = (close - lo) / span
        feats[f'dist_high_{w}'] = close / hi - 1
        feats[f'dist_low_{w}'] = close / lo - 1
    feats['drawdown_252'] = close / close.rolling(252, min_periods=21).max() - 1

    # --- shape of the recent return distribution ---
    feats['skew_63'] = log_ret.rolling(63).skew()
    feats['kurt_63'] = log_ret.rolling(63).kurt()
    feats['autocorr_21'] = log_ret.rolling(21).corr(log_ret.shift(1))
    feats['vr_2_63'] = _variance_ratio(log_ret, 2, 63)
    feats['vr_5_63'] = _variance_ratio(log_ret, 5, 63)

    # --- trend strength and its quality ---
    for w in (21, 63):
        slope, r2 = _rolling_trend(log_price.to_numpy(dtype=float), w)
        feats[f'trend_slope_{w}'] = pd.Series(slope, index=df.index) / daily_vol
        feats[f'trend_r2_{w}'] = pd.Series(r2, index=df.index)

    # --- classic oscillators ---
    feats['rsi_14'] = _rsi(close)
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    feats['macd_hist'] = (macd - macd.ewm(span=9, adjust=False).mean()) / close
    ma_21 = close.rolling(21).mean()
    sd_21 = close.rolling(21).std()
    feats['bollinger_z'] = (close - ma_21) / sd_21.replace(0, np.nan)

    if 'volume' in df.columns:
        volume = df['volume'].astype(float).replace(0, np.nan)
        log_vol = np.log(volume)
        feats['volume_z_21'] = (log_vol - log_vol.rolling(21).mean()) / log_vol.rolling(21).std()
        feats['volume_trend_63'] = log_vol.rolling(21).mean() / log_vol.rolling(63).mean() - 1
        signed = np.sign(log_ret) * volume
        feats['obv_slope_21'] = signed.rolling(21).sum() / volume.rolling(21).sum()

    out = pd.DataFrame(feats, index=df.index)
    out = out.join(swing_features(df, k=swing_k))

    # --- seasonality, as plain integers for the tree splits ---
    idx = pd.DatetimeIndex(df.index)
    out['day_of_week'] = idx.dayofweek
    out['month'] = idx.month

    return out.replace([np.inf, -np.inf], np.nan)
