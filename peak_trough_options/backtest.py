#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Out-of-sample evaluation.

Nothing here scores a model on data it was fitted on, and nothing scores it
against a straw man.  The benchmark is *climatology*: the unconditional
quantiles of the volatility-scaled excursion measured on the training window
alone.  That baseline already knows the typical size of a move for this asset,
so beating it is the only evidence that the features carry information about
*this* particular setup rather than about volatility in general.
"""

import numpy as np
import pandas as pd

from .distribution import QuantileDistribution, coverage, pinball_loss
from .labels import TARGETS
from .model import DEFAULT_LEVELS, PeakTroughForecaster
from .options import (TRADING_DAYS, candidate_structures, payoff_at_expiry,
                      rank_structures)
from .validation import purged_walk_forward_splits, split_report


def _level_tag(level):
    return f'q{int(round(level * 100)):02d}'


def walk_forward(features, labels, levels=DEFAULT_LEVELS, targets=TARGETS,
                 n_splits=5, horizon=21, embargo=5, min_train=756,
                 verbose=True, **forecaster_kwargs):
    """Fit and predict fold by fold, returning stacked out-of-sample rows.

    The returned frame carries, for every test bar: the model quantiles, the
    training-window climatology quantiles, the realised outcome (scaled and
    raw) and the volatility scale used to convert between the two.
    """
    common = features.index.intersection(labels.index)
    features, labels = features.loc[common], labels.loc[common]

    usable = labels[[f'{t}_z' for t in targets]].notna().all(axis=1) & \
        labels['scale'].notna()
    features, labels = features[usable], labels[usable]
    if features.empty:
        raise ValueError('no rows with complete labels')

    splits = list(purged_walk_forward_splits(
        len(features), n_splits=n_splits, horizon=horizon,
        embargo=embargo, min_train=min_train))
    if not splits:
        raise ValueError('no usable folds; lower min_train or n_splits')

    frames = []
    for fold, (train_idx, test_idx) in enumerate(splits, 1):
        X_train, y_train = features.iloc[train_idx], labels.iloc[train_idx]
        X_test, y_test = features.iloc[test_idx], labels.iloc[test_idx]

        forecaster_kwargs.setdefault('horizon', horizon)
        forecaster = PeakTroughForecaster(levels=levels, targets=targets,
                                          **forecaster_kwargs)
        forecaster.fit(X_train, y_train)
        preds = forecaster.predict(X_test)

        block = pd.DataFrame(index=X_test.index)
        block['fold'] = fold
        block['scale'] = y_test['scale'].to_numpy()
        for target in targets:
            block[f'{target}_z_true'] = y_test[f'{target}_z'].to_numpy()
            block[f'{target}_true'] = y_test[target].to_numpy()
            climatology = np.nanquantile(y_train[f'{target}_z'].to_numpy(), levels)
            for j, level in enumerate(levels):
                block[f'{target}_{_level_tag(level)}'] = preds[target][:, j]
                block[f'{target}_{_level_tag(level)}_base'] = climatology[j]
        if forecaster.timing_models_:
            for name, value in forecaster.predict_timing(X_test).items():
                block[f'{name}_pred'] = value
                if name in y_test:
                    block[f'{name}_true'] = y_test[name].to_numpy()
        frames.append(block)

        if verbose:
            print(f'  fold {fold}/{len(splits)}: '
                  f'train {len(train_idx):>5d}  test {len(test_idx):>4d}  '
                  f'{X_test.index[0].date()} .. {X_test.index[-1].date()}')

    oos = pd.concat(frames).sort_index()
    oos.attrs['levels'] = tuple(levels)
    oos.attrs['targets'] = tuple(targets)
    oos.attrs['horizon'] = horizon
    oos.attrs['folds'] = split_report(len(features), splits)
    return oos


def quantile_metrics(oos, levels=None, targets=None):
    """Pinball loss and calibration, model versus climatology.

    ``skill`` is the fractional reduction in pinball loss against the
    baseline: positive means the conditioning information helped.
    ``coverage`` is the realised P(outcome <= predicted quantile), which
    should land on the nominal level for a calibrated forecast.
    """
    levels = levels or oos.attrs.get('levels', DEFAULT_LEVELS)
    targets = targets or oos.attrs.get('targets', TARGETS)

    rows = []
    for target in targets:
        truth = oos[f'{target}_z_true'].to_numpy()
        for level in levels:
            tag = _level_tag(level)
            pred = oos[f'{target}_{tag}'].to_numpy()
            base = oos[f'{target}_{tag}_base'].to_numpy()
            model_loss = pinball_loss(truth, pred, level)
            base_loss = pinball_loss(truth, base, level)
            rows.append({
                'target': target,
                'level': level,
                'model_pinball': model_loss,
                'baseline_pinball': base_loss,
                'skill': 1 - model_loss / base_loss if base_loss > 0 else np.nan,
                'coverage': float(np.nanmean(truth <= pred)),
                'baseline_coverage': float(np.nanmean(truth <= base)),
            })
    return pd.DataFrame(rows)


def interval_metrics(oos, lower=0.10, upper=0.90, targets=None):
    """Coverage and width of the central prediction interval."""
    targets = targets or oos.attrs.get('targets', TARGETS)
    lo_tag, hi_tag = _level_tag(lower), _level_tag(upper)
    nominal = upper - lower

    rows = []
    for target in targets:
        truth = oos[f'{target}_z_true'].to_numpy()
        lo = oos[f'{target}_{lo_tag}'].to_numpy()
        hi = oos[f'{target}_{hi_tag}'].to_numpy()
        base_lo = oos[f'{target}_{lo_tag}_base'].to_numpy()
        base_hi = oos[f'{target}_{hi_tag}_base'].to_numpy()
        rows.append({
            'target': target,
            'nominal': nominal,
            'model_coverage': coverage(truth, lo, hi),
            'baseline_coverage': coverage(truth, base_lo, base_hi),
            'model_width': float(np.nanmean(hi - lo)),
            'baseline_width': float(np.nanmean(base_hi - base_lo)),
        })
    return pd.DataFrame(rows)


def timing_metrics(oos):
    """Median absolute error of the peak/trough timing forecasts, in bars."""
    rows = []
    for name in ('t_peak', 't_trough'):
        pred_col, true_col = f'{name}_pred', f'{name}_true'
        if pred_col in oos and true_col in oos:
            err = (oos[pred_col] - oos[true_col]).abs()
            naive = (oos[true_col] - oos[true_col].median()).abs()
            rows.append({'target': name,
                         'median_abs_error': float(err.median()),
                         'baseline_median_abs_error': float(naive.median())})
    return pd.DataFrame(rows)


def verdict(oos, levels=None, targets=None):
    """Turn the validation tables into a plain-language assessment.

    The honest default answer for most assets is "no edge over climatology".
    Saying so is the point: a forecast that cannot beat the unconditional
    distribution should be replaced by the unconditional distribution.
    """
    metrics = quantile_metrics(oos, levels=levels, targets=targets)
    intervals = interval_metrics(oos, targets=targets)

    mean_skill = float(metrics['skill'].mean())
    worst_coverage_error = float((metrics['coverage'] - metrics['level']).abs().max())
    interval_gap = float((intervals['model_coverage'] - intervals['nominal']).abs().max())

    if mean_skill <= 0:
        headline = ('No edge: the model does not beat training-window '
                    'climatology. Use the climatology band, not this model.')
    elif mean_skill < 0.02:
        headline = ('Marginal: skill is within the range noise alone can '
                    'produce on overlapping labels. Treat as unproven.')
    elif mean_skill < 0.15:
        headline = 'Some edge over climatology, in the plausible range.'
    else:
        headline = ('Implausibly high skill for daily bars -- suspect a data '
                    'or labelling problem before trading it.')

    warnings = []
    if worst_coverage_error > 0.10:
        warnings.append(f'quantiles are off their nominal level by up to '
                        f'{worst_coverage_error:.0%}')
    if interval_gap > 0.10:
        warnings.append(f'the 80% interval is mis-covered by up to {interval_gap:.0%}')
    if (intervals['model_width'] < 0.7 * intervals['baseline_width']).any():
        warnings.append('intervals are much narrower than climatology, which '
                        'is overconfidence unless skill is clearly positive')

    return {'mean_skill': mean_skill,
            'worst_coverage_error': worst_coverage_error,
            'interval_coverage_gap': interval_gap,
            'headline': headline,
            'warnings': warnings}


def simulate_decisions(oos, prices, horizon=21, rate=0.04, dividend=0.0,
                       vrp_multiple=1.10, spread_pct=0.01, levels=None,
                       trading_days=TRADING_DAYS):
    """Replay the top-ranked structure at non-overlapping intervals.

    **This is a decision-quality diagnostic, not a P&L backtest.**  There is no
    option chain here: premium is priced off a synthetic implied volatility
    equal to trailing realised volatility times ``vrp_multiple``.  The result
    therefore measures whether the forecast distribution leads to sensible
    structure and strike choices given a pricing assumption -- it does *not*
    establish that the strategy is profitable against real quotes, and it is
    highly sensitive to ``vrp_multiple`` (sweep it before believing anything).
    """
    levels = levels or oos.attrs.get('levels', DEFAULT_LEVELS)
    close = prices['close']

    # One position at a time: entries are spaced a full horizon apart.
    entries = oos.index[::horizon]
    rows = []
    for date in entries:
        pos = close.index.get_indexer([date])[0]
        if pos < 0 or pos + horizon >= len(close):
            continue
        spot = float(close.iloc[pos])
        realised_terminal = float(close.iloc[pos + horizon])

        scale = float(oos.loc[date, 'scale'])
        forecast = {
            target: QuantileDistribution(
                levels, [oos.loc[date, f'{target}_{_level_tag(l)}'] * scale
                         for l in levels])
            for target in ('mfe', 'mae', 'ret')
        }
        # scale is a per-day sigma times sqrt(H); recover the annualised rate.
        sigma_annual = scale / np.sqrt(horizon) * np.sqrt(trading_days)
        implied = sigma_annual * vrp_multiple

        ranked = rank_structures(spot, forecast, horizon, implied,
                                 rate=rate, dividend=dividend,
                                 spread_pct=spread_pct)
        if ranked.empty:
            continue
        best = ranked.iloc[0]

        chosen = next(s for s in candidate_structures(spot, forecast)
                      if s.name == best['strategy'])
        payoff = float(payoff_at_expiry(chosen, realised_terminal)) * 100
        rows.append({
            'date': date,
            'strategy': best['strategy'],
            'direction': best['direction'],
            'spot': spot,
            'terminal': realised_terminal,
            'net_cost': best['net_cost'],
            'expected_profit': best['expected_profit'],
            'realised_payoff': payoff,
            'realised_profit': payoff - best['net_cost'],
        })

    trades = pd.DataFrame(rows)
    if trades.empty:
        return trades, {}

    profit = trades['realised_profit']
    summary = {
        'trades': int(len(trades)),
        'total_profit': float(profit.sum()),
        'mean_profit': float(profit.mean()),
        'hit_rate': float((profit > 0).mean()),
        'mean_expected_profit': float(trades['expected_profit'].mean()),
        'profit_std': float(profit.std()),
        'vrp_multiple': vrp_multiple,
    }
    return trades, summary


def vrp_sensitivity(oos, prices, horizon=21, multiples=(0.9, 1.0, 1.1, 1.2, 1.3),
                    **kwargs):
    """Sweep the implied-volatility assumption behind ``simulate_decisions``.

    If results flip sign across this sweep, the simulation is measuring the
    pricing assumption rather than the forecast.
    """
    rows = []
    for multiple in multiples:
        _, summary = simulate_decisions(oos, prices, horizon=horizon,
                                        vrp_multiple=multiple, **kwargs)
        if summary:
            rows.append(summary)
    return pd.DataFrame(rows)
