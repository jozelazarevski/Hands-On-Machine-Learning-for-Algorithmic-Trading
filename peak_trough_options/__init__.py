#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Volatility-scaled peak and trough forecasting for option selection.

Predicts the *distribution* of the maximum favourable and adverse excursion
over an option's life -- how far price travels up, how far down, and when --
then turns those bands into ranked option structures.

Educational research code accompanying the book.  Not investment advice; see
README.md for the assumptions and the ways this can mislead you.
"""

from .backtest import (interval_metrics, quantile_metrics, simulate_decisions,
                       timing_metrics, verdict, vrp_sensitivity, walk_forward)
from .classification import (binary_scores, classification_report,
                             classification_verdict, confusion_frame,
                             format_confusion, summarise_report)
from .data import (load_csv, load_stooq, load_yfinance, normalise_ohlcv,
                   synthetic_ohlcv)
from .distribution import (QuantileDistribution, coverage,
                           gbm_excursion_quantiles, monotone_rearrange,
                           pinball_loss)
from .features import build_features
from .labels import forward_extremes, make_labels
from .model import DEFAULT_LEVELS, PeakTroughForecaster
from .options import (bs_greeks, bs_price, evaluate_strategy, implied_vol,
                      rank_structures, volatility_edge)
from .pipeline import (fit, latest_forecast, prepare, price_targets,
                       random_walk_reference, recommend, run)
from .swings import detect_swings, pivot_frame, swing_features
from .validation import purged_walk_forward_splits

__version__ = '0.1.0'

__all__ = [
    'PeakTroughForecaster', 'QuantileDistribution', 'DEFAULT_LEVELS',
    'binary_scores', 'build_features', 'bs_greeks', 'bs_price',
    'classification_report', 'classification_verdict', 'confusion_frame',
    'coverage', 'detect_swings', 'format_confusion',
    'evaluate_strategy', 'fit', 'forward_extremes', 'gbm_excursion_quantiles',
    'implied_vol', 'interval_metrics', 'latest_forecast', 'load_csv',
    'load_stooq', 'load_yfinance', 'make_labels', 'monotone_rearrange', 'normalise_ohlcv',
    'pinball_loss', 'pivot_frame', 'prepare', 'price_targets',
    'purged_walk_forward_splits', 'quantile_metrics', 'random_walk_reference',
    'rank_structures', 'recommend', 'run', 'simulate_decisions',
    'summarise_report', 'swing_features', 'synthetic_ohlcv', 'timing_metrics',
    'verdict', 'volatility_edge', 'vrp_sensitivity', 'walk_forward',
]
