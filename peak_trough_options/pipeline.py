#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""End-to-end wiring: bars in, forecast band and ranked structures out."""

import numpy as np
import pandas as pd

from .distribution import QuantileDistribution, gbm_excursion_quantiles
from .features import build_features
from .labels import TARGETS, make_labels
from .model import DEFAULT_LEVELS, PeakTroughForecaster
from .options import TRADING_DAYS, rank_structures, volatility_edge


def prepare(df, horizon=21, vol_span=21, swing_k=2.0):
    """Build the causal feature matrix and the forward-looking labels."""
    features = build_features(df, swing_k=swing_k)
    labels = make_labels(df, horizon=horizon, vol_span=vol_span)
    return features, labels


def fit(features, labels, levels=DEFAULT_LEVELS, targets=TARGETS, horizon=21,
        **kwargs):
    """Fit on every row whose outcome is already known."""
    complete = labels[[f'{t}_z' for t in targets]].notna().all(axis=1) & \
        labels['scale'].notna()
    if complete.sum() < 250:
        raise ValueError(f'only {int(complete.sum())} fully labelled rows; '
                         'need a longer history')
    kwargs.setdefault('horizon', horizon)
    forecaster = PeakTroughForecaster(levels=levels, targets=targets, **kwargs)
    return forecaster.fit(features[complete], labels[complete])


def latest_forecast(df, forecaster, labels, horizon=21,
                    trading_days=TRADING_DAYS):
    """Forecast distributions for the most recent bar, in return space."""
    features = build_features(df)
    last = features.iloc[[-1]]
    scale = labels['scale'].iloc[-1]
    if not np.isfinite(scale):
        raise ValueError('volatility scale is unavailable for the last bar')

    quantiles = forecaster.predict_returns(last, [scale])
    forecast = {t: QuantileDistribution(forecaster.levels, quantiles[t][0])
                for t in quantiles}
    timing = {k: float(v[0]) for k, v in forecaster.predict_timing(last).items()}

    spot = float(df['close'].iloc[-1])
    sigma_daily = scale / np.sqrt(horizon)
    return {
        'date': df.index[-1],
        'spot': spot,
        'horizon': horizon,
        'scale': float(scale),
        'sigma_daily': float(sigma_daily),
        'sigma_annual': float(sigma_daily * np.sqrt(trading_days)),
        'forecast': forecast,
        'timing': timing,
    }


def price_targets(view):
    """Convert the return quantiles into price levels."""
    spot = view['spot']
    forecast = view['forecast']
    levels = forecast['mfe'].levels

    rows = []
    for level in levels:
        rows.append({
            'level': level,
            'peak_price': spot * (1 + forecast['mfe'].ppf(level)),
            'trough_price': spot * (1 + forecast['mae'].ppf(level)),
            'expiry_price': spot * (1 + forecast['ret'].ppf(level)),
            'peak_return': forecast['mfe'].ppf(level),
            'trough_return': forecast['mae'].ppf(level),
            'expiry_return': forecast['ret'].ppf(level),
        })
    return pd.DataFrame(rows)


def random_walk_reference(view):
    """Where a driftless random walk of the same volatility would reach."""
    levels = view['forecast']['mfe'].levels
    return pd.DataFrame(
        gbm_excursion_quantiles(levels, view['sigma_daily'], view['horizon']),
        index=pd.Index(levels, name='level'))


def recommend(view, implied_sigma_annual=None, vrp_multiple=1.10, rate=0.04,
              dividend=0.0, spread_pct=0.01, increment=None):
    """Rank structures for the latest bar.

    ``implied_sigma_annual`` should come from the real option chain.  Without
    it a proxy of ``realised vol * vrp_multiple`` is used, and every number
    downstream inherits that assumption.
    """
    proxied = implied_sigma_annual is None
    if proxied:
        implied_sigma_annual = view['sigma_annual'] * vrp_multiple

    edge = volatility_edge(view['forecast']['ret'], implied_sigma_annual,
                           view['horizon'])
    edge['implied_is_proxy'] = proxied

    ranked = rank_structures(view['spot'], view['forecast'], view['horizon'],
                             implied_sigma_annual, rate=rate, dividend=dividend,
                             spread_pct=spread_pct, increment=increment)
    return {'volatility_edge': edge, 'structures': ranked}


def run(df, horizon=21, implied_sigma_annual=None, vrp_multiple=1.10,
        rate=0.04, dividend=0.0, spread_pct=0.01, levels=DEFAULT_LEVELS,
        **kwargs):
    """Convenience wrapper: prepare, fit on history, forecast the last bar."""
    features, labels = prepare(df, horizon=horizon)
    forecaster = fit(features, labels, levels=levels, horizon=horizon, **kwargs)
    view = latest_forecast(df, forecaster, labels, horizon=horizon)
    view['targets'] = price_targets(view)
    view['random_walk'] = random_walk_reference(view)
    view.update(recommend(view, implied_sigma_annual=implied_sigma_annual,
                          vrp_multiple=vrp_multiple, rate=rate,
                          dividend=dividend, spread_pct=spread_pct))
    view['forecaster'] = forecaster
    view['features'] = features
    view['labels'] = labels
    return view
