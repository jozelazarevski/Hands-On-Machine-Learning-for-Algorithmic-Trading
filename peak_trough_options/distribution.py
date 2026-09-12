#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Turning a handful of predicted quantiles into a usable distribution.

Independently fitted quantile models can cross (a 10% estimate above the 90%
estimate).  ``monotone_rearrange`` repairs that by sorting, which is the
Chernozhukov-Fernandez-Val-Galichon rearrangement and never increases the
quantile loss.

``QuantileDistribution`` interpolates *in normal-score space* rather than in
probability space: the quantile function is linear in ``Phi^-1(u)`` for a
Gaussian, so a handful of interior quantiles extrapolate to sane tails instead
of the flat or exploding ones you get from interpolating on ``u`` directly.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def monotone_rearrange(values):
    """Sort quantile predictions along the last axis to remove crossings."""
    return np.sort(np.asarray(values, dtype=float), axis=-1)


def pinball_loss(y_true, y_pred, level):
    """Quantile (pinball) loss -- the proper scoring rule for a quantile."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    delta = y_true - y_pred
    return float(np.nanmean(np.maximum(level * delta, (level - 1) * delta)))


def coverage(y_true, lower, upper):
    """Empirical fraction of outcomes inside ``[lower, upper]``."""
    y_true = np.asarray(y_true, dtype=float)
    inside = (y_true >= np.asarray(lower)) & (y_true <= np.asarray(upper))
    return float(np.nanmean(inside[~np.isnan(y_true)]))


class QuantileDistribution:
    """A continuous distribution pinned to a set of quantiles.

    Parameters
    ----------
    levels : sequence of float
        Probability levels in (0, 1), strictly increasing.
    values : sequence of float
        The corresponding quantiles; sorted on construction.
    """

    def __init__(self, levels, values):
        levels = np.asarray(levels, dtype=float)
        values = np.sort(np.asarray(values, dtype=float))
        if levels.ndim != 1 or levels.shape != values.shape:
            raise ValueError('levels and values must be 1-D and the same length')
        if levels.size < 2:
            raise ValueError('need at least two quantile levels')
        if np.any(levels <= 0) or np.any(levels >= 1):
            raise ValueError('levels must lie strictly inside (0, 1)')
        if np.any(np.diff(levels) <= 0):
            raise ValueError('levels must be strictly increasing')

        self.levels = levels
        self.values = values
        self._z = norm.ppf(levels)

    def ppf(self, u):
        """Inverse CDF, linear in normal-score space with linear tails.

        Returns a float for scalar input and an array otherwise.
        """
        u = np.asarray(u, dtype=float)
        scalar = u.ndim == 0
        z = norm.ppf(np.clip(u, 1e-12, 1 - 1e-12))
        out = np.interp(z, self._z, self.values)

        # np.interp clamps outside the knots; replace with the end slopes so
        # the tails keep growing instead of flattening into a point mass.
        spread = np.diff(self.values)
        gaps = np.diff(self._z)
        lo_slope = spread[0] / gaps[0] if spread[0] > 0 else 0.0
        hi_slope = spread[-1] / gaps[-1] if spread[-1] > 0 else 0.0
        out = np.where(z < self._z[0],
                       self.values[0] + lo_slope * (z - self._z[0]), out)
        out = np.where(z > self._z[-1],
                       self.values[-1] + hi_slope * (z - self._z[-1]), out)
        return float(out) if scalar else out

    def cdf(self, x):
        """CDF by inverting ``ppf`` on the normal-score grid.

        Returns a float for scalar input and an array otherwise.
        """
        x = np.asarray(x, dtype=float)
        scalar = x.ndim == 0
        z = np.interp(x, self.values, self._z)
        spread = np.diff(self.values)
        gaps = np.diff(self._z)
        lo_slope = gaps[0] / spread[0] if spread[0] > 0 else 0.0
        hi_slope = gaps[-1] / spread[-1] if spread[-1] > 0 else 0.0
        z = np.where(x < self.values[0],
                     self._z[0] + lo_slope * (x - self.values[0]), z)
        z = np.where(x > self.values[-1],
                     self._z[-1] + hi_slope * (x - self.values[-1]), z)
        out = norm.cdf(z)
        return float(out) if scalar else out

    def expectation(self, fn, n_points=4000):
        """``E[fn(X)]`` by midpoint integration of ``fn(F^-1(u))`` over u.

        Deterministic and unbiased up to truncation of the outer
        ``1/(2 n_points)`` of each tail.
        """
        u = (np.arange(n_points) + 0.5) / n_points
        return float(np.mean(fn(self.ppf(u))))

    def prob_above(self, x):
        return float(1.0 - self.cdf(x))

    def prob_below(self, x):
        return float(self.cdf(x))

    def mean(self, n_points=4000):
        return self.expectation(lambda v: v, n_points=n_points)

    def std(self, n_points=4000):
        mu = self.mean(n_points)
        var = self.expectation(lambda v: (v - mu) ** 2, n_points=n_points)
        return float(np.sqrt(max(var, 0.0)))

    def implied_sigma(self):
        """Gaussian-equivalent sigma from the widest quantile pair."""
        z_span = self._z[-1] - self._z[0]
        return float((self.values[-1] - self.values[0]) / z_span)

    def scaled(self, factor):
        """Same shape, rescaled -- used to map z-space back to returns."""
        return QuantileDistribution(self.levels, self.values * factor)

    def shifted(self, offset):
        return QuantileDistribution(self.levels, self.values + offset)

    def __repr__(self):
        pairs = ', '.join(f'{l:.2f}:{v:+.4f}' for l, v in zip(self.levels, self.values))
        return f'QuantileDistribution({pairs})'


# --------------------------------------------------------------------------
# Closed-form random-walk benchmark
# --------------------------------------------------------------------------

def _running_max_cdf(m, mu, sigma, t):
    """P(max_{s<=t} X_s <= m) for X_s = mu*s + sigma*W_s, m >= 0."""
    if m < 0:
        return 0.0
    root = sigma * np.sqrt(t)
    first = norm.cdf((m - mu * t) / root)
    second = np.exp(2 * mu * m / sigma ** 2) * norm.cdf((-m - mu * t) / root)
    return float(np.clip(first - second, 0.0, 1.0))


def gbm_running_max_quantiles(levels, mu, sigma, t):
    """Quantiles of the running maximum of an arithmetic Brownian motion."""
    upper = max(10 * sigma * np.sqrt(t) + abs(mu) * t, 1e-6)
    out = []
    for p in np.atleast_1d(levels):
        # _running_max_cdf is increasing in m, so bracket and bisect.
        hi = upper
        while _running_max_cdf(hi, mu, sigma, t) < p and hi < 1e4:
            hi *= 2
        out.append(brentq(lambda m: _running_max_cdf(m, mu, sigma, t) - p,
                          0.0, hi, xtol=1e-10))
    return np.array(out)


def gbm_excursion_quantiles(levels, sigma_daily, horizon, mu_daily=0.0):
    """Random-walk reference for MFE / MAE, in simple-return space.

    Log price follows ``mu*t + sigma*W_t``; the maximum and minimum are
    mirror images of each other once the drift is flipped.

    This assumes *continuous* monitoring of a gap-free diffusion, so the
    running maximum is never below the starting point and the reported MFE is
    never negative.  Realised labels can be negative, because price gaps over
    the open.  Treat this as a "how far would a driftless random walk of the
    same volatility travel" yardstick for sizing strikes, not as a calibrated
    forecast -- ``backtest.py`` benchmarks against training-window climatology
    instead, which matches the label definition exactly.
    """
    levels = np.asarray(levels, dtype=float)
    max_log = gbm_running_max_quantiles(levels, mu_daily, sigma_daily, horizon)
    # min of X with drift mu == -(max of X with drift -mu), with levels flipped
    min_log = -gbm_running_max_quantiles(1 - levels, -mu_daily, sigma_daily, horizon)
    return {'mfe': np.expm1(max_log), 'mae': np.expm1(min_log)}


def gbm_touch_probability(threshold, sigma_daily, horizon, mu_daily=0.0, kind='up'):
    """P(the path touches ``threshold`` (a simple return) within ``horizon``).

    Closed form from the reflection principle, for a driftless-by-default
    geometric Brownian motion of the given daily volatility.  This is the
    reference an excursion forecast has to beat to have said anything: it
    already knows how volatile the asset is right now, so beating it requires
    information beyond the current volatility level.
    """
    log_threshold = np.log1p(threshold)
    if kind == 'up':
        if log_threshold <= 0:
            return 1.0
        return 1.0 - _running_max_cdf(log_threshold, mu_daily, sigma_daily, horizon)
    if kind == 'down':
        if log_threshold >= 0:
            return 1.0
        # The minimum of X with drift mu mirrors the maximum of -X with -mu.
        return 1.0 - _running_max_cdf(-log_threshold, -mu_daily, sigma_daily, horizon)
    raise ValueError("kind must be 'up' or 'down'")


def gbm_move_probability(threshold, sigma_daily, horizon, mu_daily=0.0):
    """P(|terminal return| >= ``threshold``) under the same random walk."""
    log_threshold = abs(np.log1p(abs(threshold)))
    spread = sigma_daily * np.sqrt(horizon)
    drift = mu_daily * horizon
    return float(norm.sf((log_threshold - drift) / spread)
                 + norm.cdf((-log_threshold - drift) / spread))
