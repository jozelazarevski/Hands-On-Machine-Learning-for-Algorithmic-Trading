#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""End-to-end behaviour, including the leakage canary."""

import numpy as np
import pandas as pd
import pytest

from peak_trough_options import backtest, data, pipeline
from peak_trough_options.model import PeakTroughForecaster

HORIZON = 21


@pytest.fixture(scope='module')
def bars():
    return data.synthetic_ohlcv(n=1600, seed=21)


@pytest.fixture(scope='module')
def prepared(bars):
    return pipeline.prepare(bars, horizon=HORIZON)


def test_normalise_accepts_messy_column_names():
    raw = pd.DataFrame({
        'Date': ['2020-01-02', '2020-01-03'],
        'Open': [10, 11], 'High': [12, 12], 'Low': [9, 10],
        'Adj Close': [11, 11.5], 'Volume': [100, 200],
    })
    out = data.normalise_ohlcv(raw)
    assert list(out.columns) == ['open', 'high', 'low', 'close', 'volume']
    assert isinstance(out.index, pd.DatetimeIndex)


def test_normalise_rejects_impossible_bars():
    raw = pd.DataFrame({'open': [10.], 'high': [9.], 'low': [8.], 'close': [8.5]},
                       index=pd.to_datetime(['2020-01-02']))
    with pytest.raises(ValueError, match='inconsistent'):
        data.normalise_ohlcv(raw)


def test_features_are_finite_where_present(prepared):
    features, _ = prepared
    assert len(features.columns) > 30
    assert np.isfinite(features.to_numpy(dtype=float)[~np.isnan(features.to_numpy(dtype=float))]).all()
    # Warm-up aside, most of the matrix should be populated.
    assert features.iloc[300:].isna().mean().mean() < 0.02


def test_forecaster_produces_ordered_quantiles(prepared):
    features, labels = prepared
    complete = labels[['mfe_z', 'mae_z', 'ret_z']].notna().all(axis=1)
    X, y = features[complete], labels[complete]

    forecaster = PeakTroughForecaster(max_iter=40).fit(X.iloc[:800], y.iloc[:800])
    preds = forecaster.predict(X.iloc[800:900])
    for target, values in preds.items():
        assert values.shape == (100, 5)
        assert np.all(np.diff(values, axis=1) >= 0), f'{target} quantiles cross'

    frame = forecaster.forecast_frame(X.iloc[800:900], y['scale'].iloc[800:900])
    assert 't_peak' in frame and (frame['t_peak'] > 0).all()
    assert (frame['mfe_q90'] >= frame['mfe_q10']).all()


def test_column_order_does_not_change_predictions(prepared):
    features, labels = prepared
    complete = labels[['mfe_z', 'mae_z', 'ret_z']].notna().all(axis=1)
    X, y = features[complete], labels[complete]
    forecaster = PeakTroughForecaster(max_iter=30, fit_timing=False).fit(
        X.iloc[:700], y.iloc[:700])

    straight = forecaster.predict(X.iloc[700:750])['mfe']
    shuffled = forecaster.predict(X.iloc[700:750][list(reversed(list(X.columns)))])['mfe']
    np.testing.assert_allclose(straight, shuffled)


def test_predicting_before_fitting_is_an_error():
    with pytest.raises(RuntimeError):
        PeakTroughForecaster().predict(pd.DataFrame({'a': [1.0]}))


def test_walk_forward_is_out_of_sample(prepared):
    features, labels = prepared
    oos = backtest.walk_forward(features, labels, horizon=HORIZON, n_splits=3,
                                min_train=600, verbose=False)

    assert oos.index.is_monotonic_increasing
    assert not oos.index.has_duplicates
    assert oos['fold'].nunique() == 3

    metrics = backtest.quantile_metrics(oos)
    assert len(metrics) == 15
    assert metrics['model_pinball'].gt(0).all()
    assert backtest.interval_metrics(oos)['model_width'].gt(0).all()

    assessment = backtest.verdict(oos)
    assert assessment['headline']
    assert np.isfinite(assessment['mean_skill'])


def test_conformal_calibration_pulls_coverage_towards_nominal():
    """Calibration needs enough history to be worth doing, and then it works."""
    bars = data.synthetic_ohlcv(n=2800, seed=5)
    features, labels = pipeline.prepare(bars, horizon=HORIZON)
    common = dict(horizon=HORIZON, n_splits=3, min_train=1100, verbose=False)

    raw = backtest.walk_forward(features, labels, calibrate=False, **common)
    calibrated = backtest.walk_forward(features, labels, calibrate=True, **common)

    def worst(oos):
        metrics = backtest.quantile_metrics(oos)
        return (metrics['coverage'] - metrics['level']).abs().max()

    assert worst(calibrated) < worst(raw)
    assert worst(calibrated) < 0.15
    assert backtest.interval_metrics(calibrated)['model_coverage'].between(0.65, 0.92).all()


def test_calibration_is_skipped_when_there_is_too_little_history(prepared):
    features, labels = prepared
    complete = labels[['mfe_z', 'mae_z', 'ret_z']].notna().all(axis=1)
    X, y = features[complete], labels[complete]

    small = PeakTroughForecaster(horizon=HORIZON, fit_timing=False).fit(
        X.iloc[:600], y.iloc[:600])
    assert small.calibrated_ is False
    assert set(small.offsets_.values()) == {0.0}

    large = PeakTroughForecaster(horizon=HORIZON, fit_timing=False).fit(X, y)
    assert large.calibrated_ is True


def test_shuffled_labels_yield_no_skill(prepared):
    """The strict canary.

    Permuting the labels destroys any relationship with the features while
    leaving both marginal distributions intact.  Nothing is left to learn, so
    the model must not beat climatology.  Positive skill here would mean the
    evaluation harness itself is scoring the model on information it should
    not have.
    """
    features, labels = prepared
    rng = np.random.default_rng(7)
    shuffled = labels.copy()
    order = rng.permutation(len(shuffled))
    for column in ('mfe_z', 'mae_z', 'ret_z'):
        shuffled[column] = shuffled[column].to_numpy()[order]

    oos = backtest.walk_forward(features, shuffled, horizon=HORIZON, n_splits=2,
                                min_train=600, verbose=False, max_iter=60)
    skill = backtest.quantile_metrics(oos)['skill']
    assert skill.max() < 0.05, f'skill on permuted labels: {skill.max():.3f}'


def test_a_pure_random_walk_yields_almost_no_skill():
    """The softer canary, on data with no exploitable structure.

    A small positive number is legitimate here: the labels are divided by an
    EWMA volatility estimate, and the model can partly undo the noise in that
    estimator.  A *large* number would mean something is leaking.
    """
    rng = np.random.default_rng(99)
    n = 1500
    ret = rng.normal(0, 0.015, n)
    close = 100 * np.exp(np.cumsum(ret))
    span = np.abs(rng.normal(0, 0.01, n))
    df = pd.DataFrame({
        'open': close * (1 + rng.normal(0, 0.002, n)),
        'high': close * (1 + span),
        'low': close * (1 - span),
        'close': close,
    }, index=pd.bdate_range('2015-01-01', periods=n))

    features, labels = pipeline.prepare(df, horizon=HORIZON)
    oos = backtest.walk_forward(features, labels, horizon=HORIZON, n_splits=2,
                                min_train=600, verbose=False, max_iter=60)
    skill = backtest.quantile_metrics(oos)['skill']
    assert skill.max() < 0.10, f'suspicious skill on noise: {skill.max():.3f}'


def test_run_produces_a_complete_view(bars):
    view = pipeline.run(bars, horizon=HORIZON, implied_sigma_annual=0.30,
                        max_iter=60)

    assert view['date'] == bars.index[-1]
    assert view['spot'] == pytest.approx(float(bars['close'].iloc[-1]))
    assert view['sigma_annual'] > 0

    targets = view['targets']
    assert all(pd.api.types.is_float_dtype(targets[c]) for c in targets.columns), \
        f'object columns leak in: {targets.dtypes.to_dict()}'
    assert (targets['peak_price'] >= targets['trough_price']).all()
    assert targets['peak_price'].is_monotonic_increasing
    assert targets['trough_price'].is_monotonic_increasing

    edge = view['volatility_edge']
    assert edge['implied_sigma_annual'] == pytest.approx(0.30)
    assert edge['implied_is_proxy'] is False

    structures = view['structures']
    assert not structures.empty
    assert structures['expected_return_on_risk'].is_monotonic_decreasing
    assert set(['delta', 'gamma', 'vega', 'theta']).issubset(structures.columns)

    reference = view['random_walk']
    assert (reference['mfe'] >= 0).all() and (reference['mae'] <= 0).all()


def test_proxy_implied_vol_is_flagged(bars):
    view = pipeline.run(bars, horizon=HORIZON, max_iter=40)
    assert view['volatility_edge']['implied_is_proxy'] is True


def test_decision_replay_and_sensitivity_sweep(prepared, bars):
    features, labels = prepared
    oos = backtest.walk_forward(features, labels, horizon=HORIZON, n_splits=2,
                                min_train=600, verbose=False, max_iter=40)
    trades, summary = backtest.simulate_decisions(oos, bars, horizon=HORIZON)
    assert not trades.empty
    assert summary['trades'] == len(trades)
    np.testing.assert_allclose(
        trades['realised_profit'], trades['realised_payoff'] - trades['net_cost'])

    sweep = backtest.vrp_sensitivity(oos, bars, horizon=HORIZON,
                                     multiples=(1.0, 1.2))
    assert len(sweep) == 2
    assert list(sweep['vrp_multiple']) == [1.0, 1.2]
    assert sweep['trades'].gt(0).all()
    # Deliberately no monotonicity assertion: a different implied vol changes
    # which structure ranks first, so the sweep is a sensitivity check rather
    # than a monotone dial. See simulate_decisions' docstring.


def test_fit_refuses_a_short_history():
    short = data.synthetic_ohlcv(n=200, seed=1)
    features, labels = pipeline.prepare(short, horizon=HORIZON)
    with pytest.raises(ValueError, match='labelled rows'):
        pipeline.fit(features, labels)
