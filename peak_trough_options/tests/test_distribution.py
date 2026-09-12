#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Quantile plumbing: rearrangement, interpolation and the random-walk anchor."""

import numpy as np
import pytest
from scipy.stats import norm

from peak_trough_options.distribution import (QuantileDistribution, coverage,
                                              gbm_excursion_quantiles,
                                              gbm_running_max_quantiles,
                                              monotone_rearrange, pinball_loss)

LEVELS = np.array([0.10, 0.25, 0.50, 0.75, 0.90])


def test_rearrangement_removes_crossings():
    crossed = np.array([[0.05, -0.01, 0.02, 0.09, 0.03]])
    fixed = monotone_rearrange(crossed)
    assert np.all(np.diff(fixed, axis=-1) >= 0)
    np.testing.assert_allclose(np.sort(crossed), fixed)


def test_rearrangement_never_worsens_the_pinball_loss():
    rng = np.random.default_rng(0)
    truth = rng.normal(size=500)
    crossed = rng.normal(size=(500, len(LEVELS)))
    fixed = monotone_rearrange(crossed)
    before = sum(pinball_loss(truth, crossed[:, j], q) for j, q in enumerate(LEVELS))
    after = sum(pinball_loss(truth, fixed[:, j], q) for j, q in enumerate(LEVELS))
    assert after <= before + 1e-12


def test_a_gaussian_is_recovered_from_its_interior_quantiles():
    mu, sigma = 0.02, 0.05
    dist = QuantileDistribution(LEVELS, norm.ppf(LEVELS, mu, sigma))
    probe = np.array([0.001, 0.01, 0.05, 0.3, 0.5, 0.7, 0.95, 0.99, 0.999])
    np.testing.assert_allclose(dist.ppf(probe), norm.ppf(probe, mu, sigma), atol=1e-12)
    np.testing.assert_allclose(dist.cdf(norm.ppf(probe, mu, sigma)), probe, atol=1e-12)
    assert dist.mean() == pytest.approx(mu, abs=1e-4)
    assert dist.std() == pytest.approx(sigma, abs=1e-4)
    assert dist.implied_sigma() == pytest.approx(sigma, abs=1e-10)


def test_expectation_matches_the_closed_form_call_payoff():
    mu, sigma, strike = 0.01, 0.06, 0.04
    dist = QuantileDistribution(LEVELS, norm.ppf(LEVELS, mu, sigma))
    z = (mu - strike) / sigma
    expected = (mu - strike) * norm.cdf(z) + sigma * norm.pdf(z)
    assert dist.expectation(lambda v: np.maximum(v - strike, 0)) == \
        pytest.approx(expected, abs=1e-5)


def test_ppf_is_monotone_and_hits_its_knots():
    dist = QuantileDistribution(LEVELS, [-0.08, -0.03, 0.00, 0.04, 0.11])
    np.testing.assert_allclose(dist.ppf(LEVELS), [-0.08, -0.03, 0.00, 0.04, 0.11])
    grid = np.linspace(1e-4, 1 - 1e-4, 500)
    assert np.all(np.diff(dist.ppf(grid)) >= -1e-12)


def test_tails_keep_extending_beyond_the_outer_quantiles():
    dist = QuantileDistribution(LEVELS, [-0.08, -0.03, 0.00, 0.04, 0.11])
    assert dist.ppf(0.001) < dist.ppf(0.10)
    assert dist.ppf(0.999) > dist.ppf(0.90)


def test_scalar_input_gives_scalar_output():
    """A 0-d array here silently turns downstream columns into object dtype."""
    dist = QuantileDistribution(LEVELS, [-0.08, -0.03, 0.0, 0.04, 0.11])
    assert isinstance(dist.ppf(0.5), float)
    assert isinstance(dist.cdf(0.0), float)
    assert isinstance(dist.ppf([0.25, 0.75]), np.ndarray)
    assert dist.ppf(np.array([0.25, 0.75])).shape == (2,)


def test_probabilities_are_consistent():
    dist = QuantileDistribution(LEVELS, [-0.08, -0.03, 0.00, 0.04, 0.11])
    assert dist.prob_above(0.0) == pytest.approx(0.5, abs=1e-9)
    assert dist.prob_above(0.04) + dist.prob_below(0.04) == pytest.approx(1.0)


def test_scaling_and_shifting():
    dist = QuantileDistribution(LEVELS, [-0.08, -0.03, 0.0, 0.04, 0.11])
    np.testing.assert_allclose(dist.scaled(2.0).values, dist.values * 2)
    np.testing.assert_allclose(dist.shifted(0.5).values, dist.values + 0.5)


def test_rejects_malformed_inputs():
    with pytest.raises(ValueError):
        QuantileDistribution([0.5], [0.0])
    with pytest.raises(ValueError):
        QuantileDistribution([0.0, 0.5], [0.0, 1.0])
    with pytest.raises(ValueError):
        QuantileDistribution([0.5, 0.25], [0.0, 1.0])


def test_pinball_and_coverage():
    truth = np.array([1.0, 2.0, 3.0, 4.0])
    assert pinball_loss(truth, truth, 0.5) == 0.0
    assert coverage(truth, np.full(4, 1.5), np.full(4, 3.5)) == pytest.approx(0.5)


def test_running_max_matches_simulation():
    sigma, horizon = 0.02, 21
    rng = np.random.default_rng(5)
    substeps = 60
    paths = rng.normal(0, sigma / np.sqrt(substeps),
                       size=(60000, horizon * substeps)).cumsum(axis=1)
    simulated = np.quantile(np.maximum(paths.max(axis=1), 0.0), LEVELS)
    closed_form = gbm_running_max_quantiles(LEVELS, 0.0, sigma, horizon)
    # Discrete sub-stepping biases the simulation slightly low.
    np.testing.assert_allclose(closed_form, simulated, atol=0.004)


def test_excursions_are_mirrored_without_drift():
    result = gbm_excursion_quantiles(LEVELS, 0.02, 21, mu_daily=0.0)
    up = np.log1p(result['mfe'])
    down = -np.log1p(result['mae'])[::-1]
    np.testing.assert_allclose(up, down, rtol=1e-6)
    assert np.all(result['mfe'] >= 0) and np.all(result['mae'] <= 0)
