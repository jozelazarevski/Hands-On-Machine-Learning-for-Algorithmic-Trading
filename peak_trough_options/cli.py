#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Command line front end.

    python -m peak_trough_options.cli --csv AAPL.csv --horizon 21
    python -m peak_trough_options.cli --ticker AAPL.US --iv 0.32 --validate
    python -m peak_trough_options.cli --demo --validate
"""

import argparse
import sys

import pandas as pd

from . import backtest, data, pipeline

DISCLAIMER = (
    'Research and educational output only. These are model estimates from '
    'historical prices, not investment advice, and options can lose their '
    'entire value. Read the assumptions in README.md before acting on any of it.'
)


def _print_header(text):
    print(f'\n{text}\n' + '-' * len(text))


def build_parser():
    parser = argparse.ArgumentParser(
        prog='peak_trough_options',
        description='Forecast the peak/trough band over an option horizon '
                    'and rank structures against it.')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--csv', help='OHLCV csv file')
    source.add_argument('--ticker', help='symbol to pull from stooq, e.g. AAPL.US')
    source.add_argument('--demo', action='store_true',
                        help='run on synthetic bars, no network needed')

    parser.add_argument('--horizon', type=int, default=21,
                        help='trading days to expiry (default: 21)')
    parser.add_argument('--iv', type=float, default=None,
                        help='annualised implied volatility from the real '
                             'option chain, e.g. 0.32; strongly recommended')
    parser.add_argument('--vrp', type=float, default=1.10,
                        help='implied/realised multiplier used when --iv is '
                             'absent (default: 1.10)')
    parser.add_argument('--rate', type=float, default=0.04, help='risk-free rate')
    parser.add_argument('--dividend', type=float, default=0.0, help='dividend yield')
    parser.add_argument('--spread', type=float, default=0.01,
                        help='option bid-ask width as a fraction of mid')
    parser.add_argument('--start', default='2010-01-01', help='history start for --ticker')
    parser.add_argument('--half-life', type=int, default=None,
                        help='sample-weight half life in bars')
    parser.add_argument('--top', type=int, default=6, help='structures to show')
    parser.add_argument('--validate', action='store_true',
                        help='run purged walk-forward validation first')
    parser.add_argument('--splits', type=int, default=5, help='validation folds')
    parser.add_argument('--min-train', type=int, default=756,
                        help='minimum training bars per fold')
    parser.add_argument('--simulate', action='store_true',
                        help='replay top-ranked structures (needs --validate)')
    return parser


def load_bars(args):
    if args.demo:
        print('Using synthetic bars (--demo): results are a smoke test, not a study.')
        return data.synthetic_ohlcv(n=2600, seed=7)
    if args.csv:
        return data.load_csv(args.csv)
    return data.load_stooq(args.ticker, start=args.start)


def run_validation(df, args):
    _print_header('Purged walk-forward validation')
    features, labels = pipeline.prepare(df, horizon=args.horizon)
    oos = backtest.walk_forward(features, labels, horizon=args.horizon,
                                n_splits=args.splits, min_train=args.min_train,
                                half_life=args.half_life)

    print('\nQuantile skill vs training-window climatology '
          '(skill > 0 means the features helped):')
    metrics = backtest.quantile_metrics(oos)
    print(metrics.to_string(index=False, float_format=lambda v: f'{v:.4f}'))

    print('\nCentral 80% interval:')
    print(backtest.interval_metrics(oos).to_string(
        index=False, float_format=lambda v: f'{v:.4f}'))

    timing = backtest.timing_metrics(oos)
    if not timing.empty:
        print('\nTiming (bars):')
        print(timing.to_string(index=False, float_format=lambda v: f'{v:.2f}'))

    assessment = backtest.verdict(oos)
    print(f"\nVerdict: {assessment['headline']}")
    print(f"  mean skill {assessment['mean_skill']:+.3f}")
    for warning in assessment['warnings']:
        print(f'  warning: {warning}')

    if args.simulate:
        _print_header('Decision replay (diagnostic only)')
        sweep = backtest.vrp_sensitivity(oos, df, horizon=args.horizon,
                                         rate=args.rate, dividend=args.dividend,
                                         spread_pct=args.spread)
        if sweep.empty:
            print('not enough out-of-sample bars to replay')
        else:
            print(sweep.to_string(index=False, float_format=lambda v: f'{v:.2f}'))
            print('\nIf the sign of total_profit flips across this sweep, the '
                  'result reflects the volatility assumption, not the forecast.')
    return oos


def report(view, args):
    _print_header(f"Forecast for {pd.Timestamp(view['date']).date()} "
                  f"over {view['horizon']} trading days")
    print(f"spot                 {view['spot']:.2f}")
    print(f"realised vol (ann.)  {view['sigma_annual']:.1%}")

    print('\nPredicted path extremes (model):')
    targets = view['targets'].copy()
    targets['level'] = targets['level'].map(lambda v: f'{v:.0%}')
    print(targets[['level', 'trough_price', 'peak_price', 'expiry_price']]
          .to_string(index=False, float_format=lambda v: f'{v:.2f}'))

    print('\nSame quantiles as returns:')
    print(targets[['level', 'trough_return', 'peak_return', 'expiry_return']]
          .to_string(index=False, float_format=lambda v: f'{v:+.2%}'))

    print('\nDriftless random walk of the same volatility, for reference:')
    reference = view['random_walk'].copy()
    reference.index = [f'{v:.0%}' for v in reference.index]
    print(reference.to_string(float_format=lambda v: f'{v:+.2%}'))

    timing = view['timing']
    if timing:
        print('\nMedian timing (trading days ahead):')
        for name, value in timing.items():
            print(f'  {name:<9} {value:.1f}')
        print('  -> a contract should have meaningfully more time than this.')

    edge = view['volatility_edge']
    print('\nVolatility comparison:')
    print(f"  model sigma (ann.)    {edge['model_sigma_annual']:.1%}")
    print(f"  implied sigma (ann.)  {edge['implied_sigma_annual']:.1%}"
          + ('   [PROXY -- supply --iv from the real chain]'
             if edge['implied_is_proxy'] else ''))
    print(f"  model / implied       {edge['model_over_implied']:.2f}  "
          f"-> {edge['stance']}")

    print('\nRanked structures (per 1 contract, expected profit under the '
          'model, premium at the implied vol above):')
    columns = ['strategy', 'direction', 'net_cost', 'expected_profit',
               'expected_return_on_risk', 'win_probability', 'max_loss',
               'delta', 'vega', 'theta']
    table = view['structures'].head(args.top)[columns]
    print(table.to_string(index=False, float_format=lambda v: f'{v:.3f}'))


def main(argv=None):
    args = build_parser().parse_args(argv)
    df = load_bars(args)
    print(f'Loaded {len(df)} bars: {df.index[0].date()} .. {df.index[-1].date()}')

    if args.validate:
        run_validation(df, args)

    view = pipeline.run(df, horizon=args.horizon,
                        implied_sigma_annual=args.iv, vrp_multiple=args.vrp,
                        rate=args.rate, dividend=args.dividend,
                        spread_pct=args.spread, half_life=args.half_life)
    report(view, args)
    print(f'\n{DISCLAIMER}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
