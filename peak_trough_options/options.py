#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""From a forecast band to a concrete option structure.

Two distinct comparisons drive every decision here:

* **Level** -- where the predicted peak and trough sit relative to the strikes
  on offer.  That picks the direction and the strikes.
* **Width** -- the predicted spread of the terminal distribution against the
  spread the market has already priced in through implied volatility.  That
  decides whether to *buy* premium or *sell* it.

A directional view is worthless if the move is already in the premium, which
is why every structure is scored on expected profit net of its cost rather
than on probability of touching a target.
"""

from collections import namedtuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

TRADING_DAYS = 252
CONTRACT_MULTIPLIER = 100

Leg = namedtuple('Leg', ['kind', 'strike', 'quantity'])
Strategy = namedtuple('Strategy', ['name', 'legs', 'direction'])


# --------------------------------------------------------------------------
# Black-Scholes
# --------------------------------------------------------------------------

def _d1_d2(spot, strike, t, rate, sigma, dividend):
    root = sigma * np.sqrt(t)
    d1 = (np.log(spot / strike) + (rate - dividend + 0.5 * sigma ** 2) * t) / root
    return d1, d1 - root


def bs_price(spot, strike, t, rate, sigma, dividend=0.0, kind='call'):
    """European option price with a continuous dividend yield."""
    kind = kind.lower()
    if kind not in ('call', 'put'):
        raise ValueError("kind must be 'call' or 'put'")
    if t <= 0 or sigma <= 0:
        intrinsic = spot - strike if kind == 'call' else strike - spot
        return float(max(intrinsic, 0.0))

    d1, d2 = _d1_d2(spot, strike, t, rate, sigma, dividend)
    disc_s = spot * np.exp(-dividend * t)
    disc_k = strike * np.exp(-rate * t)
    if kind == 'call':
        return float(disc_s * norm.cdf(d1) - disc_k * norm.cdf(d2))
    return float(disc_k * norm.cdf(-d2) - disc_s * norm.cdf(-d1))


def bs_greeks(spot, strike, t, rate, sigma, dividend=0.0, kind='call'):
    """Delta, gamma, vega (per volatility point) and theta (per day)."""
    kind = kind.lower()
    if t <= 0 or sigma <= 0:
        itm = (spot > strike) if kind == 'call' else (spot < strike)
        delta = (1.0 if kind == 'call' else -1.0) if itm else 0.0
        return {'delta': delta, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0}

    d1, d2 = _d1_d2(spot, strike, t, rate, sigma, dividend)
    disc_q = np.exp(-dividend * t)
    disc_r = np.exp(-rate * t)
    pdf = norm.pdf(d1)

    gamma = disc_q * pdf / (spot * sigma * np.sqrt(t))
    vega = spot * disc_q * pdf * np.sqrt(t)
    common_theta = -spot * disc_q * pdf * sigma / (2 * np.sqrt(t))
    if kind == 'call':
        delta = disc_q * norm.cdf(d1)
        theta = (common_theta - rate * strike * disc_r * norm.cdf(d2)
                 + dividend * spot * disc_q * norm.cdf(d1))
    else:
        delta = -disc_q * norm.cdf(-d1)
        theta = (common_theta + rate * strike * disc_r * norm.cdf(-d2)
                 - dividend * spot * disc_q * norm.cdf(-d1))

    return {'delta': float(delta),
            'gamma': float(gamma),
            'vega': float(vega / 100.0),
            'theta': float(theta / TRADING_DAYS)}


def implied_vol(price, spot, strike, t, rate, dividend=0.0, kind='call',
                lo=1e-4, hi=5.0):
    """Recover implied volatility from a price, or NaN if it is unattainable."""
    if t <= 0 or price <= 0:
        return float('nan')
    intrinsic = (max(spot * np.exp(-dividend * t) - strike * np.exp(-rate * t), 0)
                 if kind == 'call' else
                 max(strike * np.exp(-rate * t) - spot * np.exp(-dividend * t), 0))
    if price < intrinsic - 1e-12:
        return float('nan')

    def objective(sigma):
        return bs_price(spot, strike, t, rate, sigma, dividend, kind) - price

    if objective(lo) > 0 or objective(hi) < 0:
        return float('nan')
    return float(brentq(objective, lo, hi, xtol=1e-8))


def expected_move(spot, sigma_annual, t):
    """The one-standard-deviation move the market is pricing over ``t`` years."""
    return float(spot * sigma_annual * np.sqrt(t))


# --------------------------------------------------------------------------
# Structures
# --------------------------------------------------------------------------

def round_strike(price, increment=None):
    """Snap to a plausible listed strike."""
    if increment is None:
        if price < 25:
            increment = 0.5
        elif price < 100:
            increment = 1.0
        elif price < 250:
            increment = 2.5
        else:
            increment = 5.0
    return float(np.round(price / increment) * increment)


def long_call(k):
    return Strategy('long call', [Leg('call', k, +1)], 'bullish')


def long_put(k):
    return Strategy('long put', [Leg('put', k, +1)], 'bearish')


def bull_call_spread(k_long, k_short):
    return Strategy(f'bull call spread {k_long:g}/{k_short:g}',
                    [Leg('call', k_long, +1), Leg('call', k_short, -1)], 'bullish')


def bear_put_spread(k_long, k_short):
    return Strategy(f'bear put spread {k_long:g}/{k_short:g}',
                    [Leg('put', k_long, +1), Leg('put', k_short, -1)], 'bearish')


def bull_put_spread(k_short, k_long):
    """Credit spread: sell the higher put, buy the lower one as protection."""
    return Strategy(f'bull put credit spread {k_short:g}/{k_long:g}',
                    [Leg('put', k_short, -1), Leg('put', k_long, +1)], 'bullish')


def bear_call_spread(k_short, k_long):
    return Strategy(f'bear call credit spread {k_short:g}/{k_long:g}',
                    [Leg('call', k_short, -1), Leg('call', k_long, +1)], 'bearish')


def long_straddle(k):
    return Strategy(f'long straddle {k:g}',
                    [Leg('call', k, +1), Leg('put', k, +1)], 'volatility')


def long_strangle(k_put, k_call):
    return Strategy(f'long strangle {k_put:g}/{k_call:g}',
                    [Leg('put', k_put, +1), Leg('call', k_call, +1)], 'volatility')


def iron_condor(k_put_long, k_put_short, k_call_short, k_call_long):
    return Strategy(
        f'iron condor {k_put_long:g}/{k_put_short:g}/{k_call_short:g}/{k_call_long:g}',
        [Leg('put', k_put_long, +1), Leg('put', k_put_short, -1),
         Leg('call', k_call_short, -1), Leg('call', k_call_long, +1)],
        'range')


def payoff_at_expiry(strategy, spot_at_expiry):
    """Intrinsic value of the whole structure at expiry."""
    spot_at_expiry = np.asarray(spot_at_expiry, dtype=float)
    total = np.zeros_like(spot_at_expiry)
    for leg in strategy.legs:
        if leg.kind == 'call':
            total = total + leg.quantity * np.maximum(spot_at_expiry - leg.strike, 0.0)
        else:
            total = total + leg.quantity * np.maximum(leg.strike - spot_at_expiry, 0.0)
    return total


def structure_cost(strategy, spot, t, rate, sigma, dividend=0.0, spread_pct=0.01):
    """Net debit (positive) or credit (negative), including a bid-ask haircut.

    ``spread_pct`` is the full bid-ask width as a fraction of mid; each leg
    crosses half of it, always against you.
    """
    cost = 0.0
    for leg in strategy.legs:
        mid = bs_price(spot, leg.strike, t, rate, sigma, dividend, leg.kind)
        slippage = 0.5 * spread_pct * mid
        cost += leg.quantity * mid + abs(leg.quantity) * slippage
    return float(cost)


def position_greeks(strategy, spot, t, rate, sigma, dividend=0.0):
    totals = {'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0}
    for leg in strategy.legs:
        greeks = bs_greeks(spot, leg.strike, t, rate, sigma, dividend, leg.kind)
        for key, value in greeks.items():
            totals[key] += leg.quantity * value
    return totals


def _profit_profile(strategy, spot, cost, hi_mult=3.0, points=2001):
    """Profit at expiry across the whole reachable range of the underlying.

    The grid starts at zero because equity prices are bounded below, which is
    what makes a long put's maximum profit finite and exactly ``K - cost``.
    """
    grid = np.linspace(0.0, spot * hi_mult, points)
    return grid, payoff_at_expiry(strategy, grid) - cost


def breakevens(strategy, spot, cost, **kwargs):
    """Underlying prices at which the structure breaks even at expiry."""
    grid, profit = _profit_profile(strategy, spot, cost, **kwargs)
    crossings = np.where(np.sign(profit[:-1]) * np.sign(profit[1:]) < 0)[0]
    out = []
    for i in crossings:
        x0, x1 = grid[i], grid[i + 1]
        y0, y1 = profit[i], profit[i + 1]
        out.append(float(x0 - y0 * (x1 - x0) / (y1 - y0)))
    return out


def evaluate_strategy(strategy, spot, terminal_dist, t, rate, sigma,
                      dividend=0.0, spread_pct=0.01, multiplier=CONTRACT_MULTIPLIER):
    """Score a structure against the forecast distribution.

    ``terminal_dist`` is a :class:`~.distribution.QuantileDistribution` over
    *simple returns* of the underlying over the same horizon as ``t``.
    Premium is priced at the market's ``sigma``; the payoff is integrated
    under the model's distribution.  The gap between the two is the edge.
    """
    cost = structure_cost(strategy, spot, t, rate, sigma, dividend, spread_pct)

    def profit_of_return(r):
        return payoff_at_expiry(strategy, spot * (1.0 + r)) - cost

    expected_profit = terminal_dist.expectation(profit_of_return)
    win_prob = terminal_dist.expectation(lambda r: (profit_of_return(r) > 0).astype(float))

    grid, profile = _profit_profile(strategy, spot, cost)
    max_profit, max_loss = float(profile.max()), float(profile.min())
    # Downside is capped by a zero share price, so the only open-ended wing is
    # to the upside: whichever way the payoff is still moving at the right-hand
    # edge, it keeps moving that way for ever.
    step = profile[-1] - profile[-2]
    if step > 1e-12:
        max_profit = float('inf')
    elif step < -1e-12:
        max_loss = float('-inf')

    risk = abs(max_loss) if np.isfinite(max_loss) else abs(cost)
    risk = max(risk, 1e-9)

    return {
        'strategy': strategy.name,
        'direction': strategy.direction,
        'net_cost': cost * multiplier,
        'expected_profit': expected_profit * multiplier,
        'expected_return_on_risk': expected_profit / risk,
        'win_probability': win_prob,
        'max_profit': max_profit * multiplier if np.isfinite(max_profit) else float('inf'),
        'max_loss': max_loss * multiplier if np.isfinite(max_loss) else float('-inf'),
        'breakevens': breakevens(strategy, spot, cost),
        **position_greeks(strategy, spot, t, rate, sigma, dividend),
    }


# --------------------------------------------------------------------------
# Candidate generation and ranking
# --------------------------------------------------------------------------

def volatility_edge(terminal_dist, implied_sigma_annual, horizon_days,
                    trading_days=TRADING_DAYS):
    """Compare the forecast's width with the width the market has priced.

    Returns the model's annualised sigma, the ratio to implied, and the
    resulting premium stance.
    """
    model_sigma = terminal_dist.implied_sigma() * np.sqrt(trading_days / horizon_days)
    ratio = model_sigma / implied_sigma_annual if implied_sigma_annual > 0 else np.nan
    if not np.isfinite(ratio):
        stance = 'unknown'
    elif ratio > 1.15:
        stance = 'buy premium'
    elif ratio < 0.87:
        stance = 'sell premium'
    else:
        stance = 'neutral'
    return {'model_sigma_annual': float(model_sigma),
            'implied_sigma_annual': float(implied_sigma_annual),
            'model_over_implied': float(ratio),
            'stance': stance}


def candidate_structures(spot, forecast, increment=None):
    """Build candidate structures anchored on the predicted peak and trough.

    ``forecast`` maps target names to
    :class:`~.distribution.QuantileDistribution` objects for ``mfe``, ``mae``
    and ``ret``.  Strikes are placed at the forecast quantiles, so the
    structures actually reflect the predicted travel rather than an arbitrary
    delta convention.
    """
    mfe, mae = forecast['mfe'], forecast['mae']
    atm = round_strike(spot, increment)

    peak_mid = round_strike(spot * (1 + max(mfe.ppf(0.50), 0.0)), increment)
    peak_high = round_strike(spot * (1 + max(mfe.ppf(0.75), 0.0)), increment)
    trough_mid = round_strike(spot * (1 + min(mae.ppf(0.50), 0.0)), increment)
    trough_low = round_strike(spot * (1 + min(mae.ppf(0.25), 0.0)), increment)

    out = [long_call(atm), long_put(atm), long_straddle(atm)]
    if peak_mid > atm:
        out.append(long_call(peak_mid))
    if peak_high > peak_mid > atm:
        out.append(bull_call_spread(peak_mid, peak_high))
    elif peak_high > atm:
        out.append(bull_call_spread(atm, peak_high))
    if trough_mid < atm:
        out.append(long_put(trough_mid))
    if trough_low < trough_mid < atm:
        out.append(bear_put_spread(trough_mid, trough_low))
    elif trough_low < atm:
        out.append(bear_put_spread(atm, trough_low))
    if trough_mid < atm < peak_mid:
        out.append(long_strangle(trough_mid, peak_mid))
    if trough_low < trough_mid:
        out.append(bull_put_spread(trough_mid, trough_low))
    if peak_mid < peak_high:
        out.append(bear_call_spread(peak_mid, peak_high))
    if trough_low < trough_mid < atm < peak_mid < peak_high:
        out.append(iron_condor(trough_low, trough_mid, peak_mid, peak_high))

    # De-duplicate while preserving order.
    seen, unique = set(), []
    for strategy in out:
        if strategy.name not in seen:
            seen.add(strategy.name)
            unique.append(strategy)
    return unique


def rank_structures(spot, forecast, horizon_days, implied_sigma_annual,
                    rate=0.04, dividend=0.0, spread_pct=0.01, increment=None,
                    trading_days=TRADING_DAYS, multiplier=CONTRACT_MULTIPLIER):
    """Score every candidate and return them best-first.

    Ranking is by expected profit per unit of capital at risk, computed under
    the model's distribution while paying the market's implied volatility.
    """
    t = horizon_days / trading_days
    rows = [evaluate_strategy(strategy, spot, forecast['ret'], t, rate,
                              implied_sigma_annual, dividend, spread_pct, multiplier)
            for strategy in candidate_structures(spot, forecast, increment)]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values('expected_return_on_risk', ascending=False).reset_index(drop=True)
