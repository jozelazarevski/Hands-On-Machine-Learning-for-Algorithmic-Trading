#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pricing, Greeks and structure scoring."""

import numpy as np
import pytest
from scipy.stats import norm

from peak_trough_options.distribution import QuantileDistribution
from peak_trough_options.options import (bear_put_spread, bs_greeks, bs_price,
                                         bull_call_spread, breakevens,
                                         candidate_structures,
                                         evaluate_strategy, implied_vol,
                                         iron_condor, long_call,
                                         long_straddle, payoff_at_expiry,
                                         rank_structures, round_strike,
                                         structure_cost, volatility_edge)

SPOT, STRIKE, T, RATE, SIGMA, DIV = 100.0, 105.0, 0.25, 0.04, 0.30, 0.01
LEVELS = [0.10, 0.25, 0.50, 0.75, 0.90]


def test_put_call_parity():
    call = bs_price(SPOT, STRIKE, T, RATE, SIGMA, DIV, 'call')
    put = bs_price(SPOT, STRIKE, T, RATE, SIGMA, DIV, 'put')
    forward = SPOT * np.exp(-DIV * T) - STRIKE * np.exp(-RATE * T)
    assert call - put == pytest.approx(forward, abs=1e-10)


def test_price_collapses_to_intrinsic_at_expiry():
    assert bs_price(110, 100, 0, RATE, SIGMA, kind='call') == pytest.approx(10.0)
    assert bs_price(90, 100, 0, RATE, SIGMA, kind='call') == pytest.approx(0.0)
    assert bs_price(90, 100, 0, RATE, SIGMA, kind='put') == pytest.approx(10.0)


def test_price_is_monotone_in_volatility_and_strike():
    prices = [bs_price(SPOT, STRIKE, T, RATE, s, DIV, 'call') for s in (0.1, 0.2, 0.4)]
    assert prices == sorted(prices)
    strikes = [bs_price(SPOT, k, T, RATE, SIGMA, DIV, 'call') for k in (90, 100, 110)]
    assert strikes == sorted(strikes, reverse=True)


@pytest.mark.parametrize('kind', ['call', 'put'])
def test_greeks_match_finite_differences(kind):
    greeks = bs_greeks(SPOT, STRIKE, T, RATE, SIGMA, DIV, kind)
    h = 1e-4
    price = lambda s=SPOT, t=T, v=SIGMA: bs_price(s, STRIKE, t, RATE, v, DIV, kind)

    assert greeks['delta'] == pytest.approx((price(s=SPOT + h) - price(s=SPOT - h)) / (2 * h), abs=1e-6)
    assert greeks['gamma'] == pytest.approx(
        (price(s=SPOT + h) - 2 * price() + price(s=SPOT - h)) / h ** 2, abs=1e-4)
    assert greeks['vega'] == pytest.approx((price(v=SIGMA + h) - price(v=SIGMA - h)) / (2 * h) / 100, abs=1e-6)
    assert greeks['theta'] == pytest.approx(-(price(t=T + h) - price(t=T - h)) / (2 * h) / 252, abs=1e-6)


def test_call_delta_stays_inside_its_bounds():
    for strike in (50, 100, 150):
        delta = bs_greeks(SPOT, strike, T, RATE, SIGMA, DIV, 'call')['delta']
        assert 0.0 <= delta <= 1.0


@pytest.mark.parametrize('kind', ['call', 'put'])
@pytest.mark.parametrize('strike', [80.0, 100.0, 125.0])
def test_implied_vol_round_trip(kind, strike):
    price = bs_price(SPOT, strike, T, RATE, SIGMA, DIV, kind)
    assert implied_vol(price, SPOT, strike, T, RATE, DIV, kind) == pytest.approx(SIGMA, abs=1e-6)


def test_implied_vol_gives_up_on_impossible_prices():
    # Below the discounted intrinsic of a deep in-the-money call.
    assert np.isnan(implied_vol(5.0, SPOT, 80.0, T, RATE, DIV, 'call'))
    # Above anything the model can produce, and non-positive.
    assert np.isnan(implied_vol(SPOT * 2, SPOT, STRIKE, T, RATE, DIV, 'call'))
    assert np.isnan(implied_vol(0.0, SPOT, STRIKE, T, RATE, DIV, 'call'))
    assert np.isnan(implied_vol(1.0, SPOT, STRIKE, 0.0, RATE, DIV, 'call'))


def test_round_strike_uses_a_sensible_increment():
    assert round_strike(12.4) == 12.5      # below 25 -> half point
    assert round_strike(87.4) == 87.0      # below 100 -> whole point
    assert round_strike(101.3) == 102.5    # below 250 -> 2.5
    assert round_strike(503.0) == 505.0    # above 250 -> 5
    assert round_strike(101.3, increment=5.0) == 100.0


def test_payoffs_are_what_the_structure_promises():
    assert payoff_at_expiry(long_call(100), 120) == pytest.approx(20)
    assert payoff_at_expiry(long_call(100), 80) == pytest.approx(0)
    assert payoff_at_expiry(long_straddle(100), 80) == pytest.approx(20)
    spread = bull_call_spread(100, 110)
    assert payoff_at_expiry(spread, 130) == pytest.approx(10)   # capped
    assert payoff_at_expiry(spread, 105) == pytest.approx(5)
    assert payoff_at_expiry(bear_put_spread(100, 90), 80) == pytest.approx(10)
    assert payoff_at_expiry(iron_condor(80, 90, 110, 120), 100) == pytest.approx(0)


def test_a_credit_spread_costs_less_than_nothing():
    from peak_trough_options.options import bull_put_spread
    cost = structure_cost(bull_put_spread(100, 90), SPOT, T, RATE, SIGMA, DIV, spread_pct=0.0)
    assert cost < 0


def test_a_fixed_debit_structure_gets_worse_as_premium_rises():
    """Holding the structure fixed, paying a higher implied vol costs money.

    Across a *ranked* set this need not hold, because a higher implied vol
    changes which structure wins -- which is exactly why the volatility
    assumption has to be swept rather than assumed.
    """
    costs = [structure_cost(long_call(100), SPOT, T, RATE, v, DIV) for v in (0.2, 0.3, 0.4)]
    assert costs == sorted(costs)
    terminal = 115.0
    profits = [payoff_at_expiry(long_call(100), terminal) - c for c in costs]
    assert profits == sorted(profits, reverse=True)


def test_slippage_always_works_against_you():
    for strategy in (long_call(100), long_straddle(100)):
        clean = structure_cost(strategy, SPOT, T, RATE, SIGMA, DIV, spread_pct=0.0)
        dirty = structure_cost(strategy, SPOT, T, RATE, SIGMA, DIV, spread_pct=0.05)
        assert dirty > clean


def test_breakeven_of_a_long_call():
    cost = bs_price(SPOT, 100, T, RATE, SIGMA, DIV, 'call')
    points = breakevens(long_call(100), SPOT, cost)
    assert len(points) == 1
    assert points[0] == pytest.approx(100 + cost, rel=1e-3)


def test_evaluation_reports_capped_and_uncapped_risk():
    dist = QuantileDistribution(LEVELS, norm.ppf(LEVELS, 0.01, 0.08))
    call = evaluate_strategy(long_call(100), SPOT, dist, T, RATE, SIGMA, DIV)
    assert call['max_profit'] == float('inf')
    assert np.isfinite(call['max_loss']) and call['max_loss'] < 0
    assert 0.0 <= call['win_probability'] <= 1.0
    assert call['delta'] > 0 and call['vega'] > 0 and call['theta'] < 0

    spread = evaluate_strategy(bull_call_spread(100, 110), SPOT, dist, T, RATE, SIGMA, DIV)
    assert np.isfinite(spread['max_profit']) and np.isfinite(spread['max_loss'])
    # A ten-point spread cannot make more than ten points less its debit.
    assert spread['max_profit'] == pytest.approx(1000 + spread['max_loss'], abs=1.0)


def test_a_long_put_can_only_win_down_to_a_zero_share_price():
    from peak_trough_options.options import long_put
    dist = QuantileDistribution(LEVELS, norm.ppf(LEVELS, -0.02, 0.08))
    result = evaluate_strategy(long_put(100), SPOT, dist, T, RATE, SIGMA, DIV)
    cost = -result['max_loss']                    # the debit paid, in dollars
    assert np.isfinite(result['max_profit'])
    assert result['max_profit'] == pytest.approx(100 * 100 - cost, abs=1.0)


def test_an_uncapped_short_wing_is_reported_as_unbounded_loss():
    from peak_trough_options.options import Leg, Strategy
    naked = Strategy('naked short call', [Leg('call', 100, -1)], 'bearish')
    dist = QuantileDistribution(LEVELS, norm.ppf(LEVELS, 0.0, 0.08))
    result = evaluate_strategy(naked, SPOT, dist, T, RATE, SIGMA, DIV)
    assert result['max_loss'] == float('-inf')
    assert np.isfinite(result['max_profit'])


def test_a_bullish_forecast_beats_a_bearish_structure():
    from peak_trough_options.options import long_put
    bullish = QuantileDistribution(LEVELS, norm.ppf(LEVELS, 0.10, 0.05))
    call = evaluate_strategy(long_call(100), SPOT, bullish, T, RATE, SIGMA, DIV)
    put = evaluate_strategy(long_put(100), SPOT, bullish, T, RATE, SIGMA, DIV)
    assert call['expected_profit'] > put['expected_profit']


def test_volatility_edge_calls_the_stance():
    wide = QuantileDistribution(LEVELS, norm.ppf(LEVELS, 0.0, 0.20))
    narrow = QuantileDistribution(LEVELS, norm.ppf(LEVELS, 0.0, 0.02))
    assert volatility_edge(wide, 0.30, 21)['stance'] == 'buy premium'
    assert volatility_edge(narrow, 0.30, 21)['stance'] == 'sell premium'


def test_candidates_are_unique_and_ranked():
    forecast = {
        'mfe': QuantileDistribution(LEVELS, [0.00, 0.02, 0.05, 0.09, 0.14]),
        'mae': QuantileDistribution(LEVELS, [-0.14, -0.09, -0.05, -0.02, 0.00]),
        'ret': QuantileDistribution(LEVELS, norm.ppf(LEVELS, 0.01, 0.07)),
    }
    names = [s.name for s in candidate_structures(SPOT, forecast)]
    assert len(names) == len(set(names)) and len(names) >= 6

    ranked = rank_structures(SPOT, forecast, 21, 0.30)
    assert len(ranked) == len(names)
    assert ranked['expected_return_on_risk'].is_monotonic_decreasing
    assert ranked['win_probability'].between(0, 1).all()
