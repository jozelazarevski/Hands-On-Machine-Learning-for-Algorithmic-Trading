#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Does the upside-touch edge survive a fair benchmark?  (It does not.)

An exploratory run on SPY, AAPL, MSFT and TSLA suggested that one of the four
decision tasks -- "does the high reach +1 sigma before expiry" -- ranks better
out of sample than training-window climatology, on three names out of four and
with a better Brier score too.  This script was written to confirm that on
tickers the exploratory run never touched, and it refutes it instead.

The problem is the threshold.  A sigma-scaled cut-off asks whether the
excursion beat the EWMA volatility estimate the labels were *divided by*.
That estimate mean-reverts, so any feature comparing short-horizon volatility
to long-horizon volatility predicts the label without knowing anything about
the price path.  The effect is real, but it is a property of the normalisation
rather than of the asset: a single feature (``-vol_ratio_10_63``), with no
model and no fitting at all, scores AUC 0.54-0.65 on this task -- matching or
beating the full gradient-boosted model on every name tried.

So each ticker is run twice, off one walk-forward:

    sigma     threshold at 1.0 x the volatility scale   (the artifact)
    absolute  threshold at a fixed +/-5% move           (the real question)

and both are scored against two references -- training-window climatology, and
a random walk that already knows today's volatility.  The second reference is
the point: an excursion forecast that cannot beat a random walk holding the
same volatility estimate has contributed nothing.

    python -m peak_trough_options.studies.upside_touch_study
"""

import argparse
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

from .. import backtest, classification, data, pipeline

# Deliberately disjoint from the exploratory set (SPY, AAPL, MSFT, TSLA), and
# spread across sectors so the result is not one industry's story.
CONFIRMATION_TICKERS = (
    'JNJ', 'XOM', 'JPM', 'KO', 'PG', 'WMT', 'DIS', 'CSCO', 'INTC', 'PFE',
    'VZ', 'MRK', 'HD', 'MCD', 'BA', 'CAT', 'IBM', 'ORCL', 'GS', 'UNH',
)
EXPLORATORY_TICKERS = ('SPY', 'AAPL', 'MSFT', 'TSLA')

PRIMARY_TASK = 'upside_touch'


def run_ticker(ticker, horizon=21, start='2005-01-01', n_splits=4,
               min_train=1250, sigma=1.0, absolute=0.05):
    """Walk-forward one name once, then score it under both threshold rules."""
    bars = data.load_yfinance(ticker, start=start)
    features, labels = pipeline.prepare(bars, horizon=horizon)
    oos = backtest.walk_forward(features, labels, horizon=horizon,
                                n_splits=n_splits, min_train=min_train,
                                verbose=False, fit_timing=False)
    quantile_skill = backtest.quantile_metrics(oos)['skill'].mean()

    frames = []
    for mode, kwargs in (('sigma', {'sigma': sigma}),
                         ('absolute', {'absolute': absolute})):
        summary = classification.summarise_report(
            classification.classification_report(oos, **kwargs))
        summary.insert(0, 'ticker', ticker)
        summary.insert(1, 'mode', mode)
        summary['horizon'] = horizon
        summary['bars'] = len(bars)
        summary['quantile_skill'] = quantile_skill
        summary['auc_vs_climatology'] = summary['roc_auc'] - summary['baseline_roc_auc']
        summary['auc_vs_randomwalk'] = summary['roc_auc'] - summary['randomwalk_roc_auc']
        # The honest score: beat whichever reference is harder.
        summary['auc_advantage'] = summary['roc_auc'] - summary[
            ['baseline_roc_auc', 'randomwalk_roc_auc']].max(axis=1)
        summary['brier_advantage'] = summary[
            ['baseline_brier', 'randomwalk_brier']].min(axis=1) - summary['brier']
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def run_study(tickers, out=None, **kwargs):
    """Walk-forward every ticker, writing partial results out as they land.

    A full sweep takes the better part of an hour, so results are flushed
    after each name: a timeout or a dropped connection then costs one ticker
    rather than the whole run.
    """
    frames, failed = [], []
    for ticker in tickers:
        try:
            frames.append(run_ticker(ticker, **kwargs))
            print(f'  {ticker:5s} done', flush=True)
            if out:
                pd.concat(frames, ignore_index=True).to_csv(out, index=False)
        except Exception as exc:  # noqa: BLE001 - a dead ticker must not stop the study
            failed.append((ticker, f'{type(exc).__name__}: {exc}'))
            print(f'  {ticker:5s} SKIPPED ({type(exc).__name__})', flush=True)
    if not frames:
        raise RuntimeError('every ticker failed; check network access')
    return pd.concat(frames, ignore_index=True), failed


def test_hypothesis(results, task=PRIMARY_TASK, mode='absolute'):
    """Sign test and Wilcoxon on the per-ticker AUC advantage.

    Under the null the model is no better than climatology, so each ticker is
    a coin flip on the sign of the advantage.  Both tests assume the tickers
    are independent, which US large caps emphatically are not -- they share a
    market factor, so the true p-value is larger than the nominal one.  Read
    these as an upper bound on the evidence, not a measurement of it.
    """
    subset = results[(results['task'] == task) & (results['mode'] == mode)]
    subset = subset.dropna(subset=['auc_advantage'])
    advantage = subset['auc_advantage'].to_numpy()
    n = len(advantage)
    wins = int((advantage > 0).sum())

    out = {
        'task': task,
        'mode': mode,
        'tickers': n,
        'wins': wins,
        'win_rate': wins / n if n else float('nan'),
        'mean_auc_advantage': float(np.mean(advantage)) if n else float('nan'),
        'median_auc_advantage': float(np.median(advantage)) if n else float('nan'),
        'mean_auc': float(subset['roc_auc'].mean()) if n else float('nan'),
        'mean_baseline_auc': float(subset['baseline_roc_auc'].mean()) if n else float('nan'),
        'mean_randomwalk_auc': float(subset['randomwalk_roc_auc'].mean()) if n else float('nan'),
        'mean_brier_advantage': float(subset['brier_advantage'].mean()) if n else float('nan'),
        'brier_wins': int((subset['brier_advantage'] > 0).sum()),
    }
    if n:
        out['sign_test_p'] = float(binomtest(wins, n, 0.5, alternative='greater').pvalue)
        if np.any(advantage != 0):
            out['wilcoxon_p'] = float(wilcoxon(advantage, alternative='greater').pvalue)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--horizon', type=int, default=21)
    parser.add_argument('--start', default='2005-01-01')
    parser.add_argument('--splits', type=int, default=4)
    parser.add_argument('--min-train', type=int, default=1250)
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--absolute', type=float, default=0.05)
    parser.add_argument('--tickers', nargs='*', default=list(CONFIRMATION_TICKERS))
    parser.add_argument('--out', default=None, help='write the per-ticker table here')
    args = parser.parse_args(argv)

    warnings.filterwarnings('ignore')
    pd.set_option('display.width', 220)

    print(f'{len(args.tickers)} tickers, horizon {args.horizon}, from {args.start}')
    print(f'sigma-scaled threshold: {args.sigma} sigma   |   '
          f'absolute threshold: {args.absolute:.1%}\n')

    results, failed = run_study(args.tickers, out=args.out,
                                horizon=args.horizon, start=args.start,
                                n_splits=args.splits, min_train=args.min_train,
                                sigma=args.sigma, absolute=args.absolute)

    for mode, caption in (('sigma', 'sigma-scaled threshold -- the artifact'),
                          ('absolute', 'absolute threshold -- the real question')):
        primary = results[(results['task'] == PRIMARY_TASK) &
                          (results['mode'] == mode)].sort_values(
                              'auc_advantage', ascending=False)
        print(f'\n--- {PRIMARY_TASK}, {caption} ---')
        print(primary[['ticker', 'roc_auc', 'baseline_roc_auc',
                       'randomwalk_roc_auc', 'auc_vs_climatology',
                       'auc_vs_randomwalk', 'auc_advantage']].to_string(
            index=False, float_format=lambda v: f'{v:.4f}'))

    print('\n--- every task, both modes, versus the harder reference ---')
    rows = [test_hypothesis(results, task, mode)
            for mode in ('sigma', 'absolute')
            for task in classification.TASKS]
    table = pd.DataFrame(rows)[
        ['task', 'mode', 'tickers', 'wins', 'win_rate', 'mean_auc',
         'mean_baseline_auc', 'mean_randomwalk_auc', 'mean_auc_advantage',
         'sign_test_p']]
    print(table.to_string(index=False, float_format=lambda v: f'{v:.4f}'))

    verdict = test_hypothesis(results, PRIMARY_TASK, 'absolute')
    print('\n--- verdict ---')
    beat = verdict['mean_auc_advantage'] > 0 and verdict.get('sign_test_p', 1) < 0.05
    print(f'  {PRIMARY_TASK} at an absolute {args.absolute:.0%} threshold beats the '
          f'harder reference on {verdict["wins"]}/{verdict["tickers"]} names, '
          f'mean AUC advantage {verdict["mean_auc_advantage"]:+.4f}, '
          f'sign-test p {verdict.get("sign_test_p", float("nan")):.3f}.')
    print(f'  -> {"SUPPORTED" if beat else "NOT SUPPORTED"}: the forecast '
          f'{"adds" if beat else "does not add"} information beyond knowing '
          'the current volatility.')
    print('  Tickers share a market factor, so they are not independent tests '
          'and the true p-value is larger than the nominal one.')

    if failed:
        print('\nskipped:')
        for ticker, reason in failed:
            print(f'  {ticker}: {reason}')

    if args.out:
        results.to_csv(args.out, index=False)
        print(f'\nper-ticker table written to {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
