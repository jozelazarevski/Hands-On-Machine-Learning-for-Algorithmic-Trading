#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Conditional quantile forecasts of the peak, the trough and their timing.

One gradient-boosted quantile regressor is fitted per (target, level) pair.
Point forecasts are deliberately avoided: for choosing a strike, the shape of
the tail is the whole question, and a conditional mean tells you nothing about
how much room a contract needs.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .distribution import QuantileDistribution, monotone_rearrange
from .labels import TARGETS
from .validation import sample_weights

DEFAULT_LEVELS = (0.10, 0.25, 0.50, 0.75, 0.90)

# Deliberately heavy regularisation.  With H-day overlapping labels the
# effective sample size is roughly n/H, so a few thousand daily bars carry only
# a few dozen independent observations.  Loosening these collapses interval
# coverage long before it improves the pinball loss -- measured, not assumed.
DEFAULT_PARAMS = dict(
    loss='quantile',
    max_iter=60,
    learning_rate=0.03,
    max_leaf_nodes=8,
    min_samples_leaf=200,
    l2_regularization=1.0,
    early_stopping=False,
)


class PeakTroughForecaster:
    """Quantile forecaster for volatility-scaled path extremes.

    Targets are the ``*_z`` columns produced by :func:`labels.make_labels`
    (excursions divided by ``sigma_t * sqrt(H)``).  Predictions come back in
    the same scaled space; multiply by ``labels['scale']`` to get returns.
    """

    def __init__(self, levels=DEFAULT_LEVELS, targets=TARGETS, half_life=None,
                 fit_timing=True, calibrate=True, calibration_frac=0.25,
                 horizon=21, random_state=42, **params):
        self.levels = tuple(float(q) for q in levels)
        if any(not 0 < q < 1 for q in self.levels):
            raise ValueError('levels must lie strictly inside (0, 1)')
        if list(self.levels) != sorted(self.levels):
            raise ValueError('levels must be increasing')

        self.targets = tuple(targets)
        self.half_life = half_life
        self.fit_timing = fit_timing
        self.calibrate = calibrate
        self.calibration_frac = calibration_frac
        self.horizon = horizon
        self.random_state = random_state
        self.params = {**DEFAULT_PARAMS, **params}

        self.models_ = {}
        self.timing_models_ = {}
        self.offsets_ = {}
        self.feature_names_ = None
        self.calibrated_ = False

    # -- fitting ----------------------------------------------------------

    def _make_model(self, quantile):
        return HistGradientBoostingRegressor(
            quantile=quantile, random_state=self.random_state, **self.params)

    def _fit_one(self, X, y, quantile):
        mask = np.isfinite(y)
        if mask.sum() < 50:
            raise ValueError(f'only {int(mask.sum())} usable rows for quantile {quantile}')
        weights = sample_weights(int(mask.sum()), self.half_life)
        model = self._make_model(quantile)
        model.fit(X[mask], y[mask], sample_weight=weights)
        return model

    def _calibration_split(self, n):
        """Indices for fitting and for conformal calibration, purged apart.

        The calibration block is the most recent slice of the training window,
        separated from the fitting block by ``horizon`` bars so that no label
        used for fitting resolves inside the calibration block.
        """
        if not self.calibrate:
            return np.arange(n), None
        n_cal = int(round(n * self.calibration_frac))
        fit_end = n - n_cal - self.horizon
        # Overlapping labels mean the calibration block holds only about
        # n_cal / horizon independent observations.  Correcting a quantile
        # from a handful of them adds more noise than it removes, so below
        # this size the model is left uncalibrated rather than mis-calibrated.
        if n_cal < max(100, 8 * self.horizon) or fit_end < 200:
            return np.arange(n), None
        return np.arange(fit_end), np.arange(n - n_cal, n)

    def fit(self, X, labels):
        """Fit every (target, level) model plus the optional timing models.

        When ``calibrate`` is set, a purged tail of the training window is held
        out and each predicted quantile is shifted by the offset that makes it
        hit its nominal level on that block -- split-conformal quantile
        regression.  Gradient boosting on overlapping financial labels is
        reliably overconfident, and this corrects the coverage directly rather
        than hoping regularisation happens to fix it.
        """
        self.feature_names_ = list(X.columns)
        X_arr = X.to_numpy(dtype=float)
        fit_idx, cal_idx = self._calibration_split(len(X_arr))
        self.calibrated_ = cal_idx is not None

        for target in self.targets:
            column = f'{target}_z'
            if column not in labels:
                raise KeyError(f'labels is missing {column!r}')
            y = labels[column].to_numpy(dtype=float)
            for q in self.levels:
                model = self._fit_one(X_arr[fit_idx], y[fit_idx], q)
                self.models_[(target, q)] = model
                self.offsets_[(target, q)] = (
                    self._conformal_offset(model, X_arr[cal_idx], y[cal_idx], q)
                    if self.calibrated_ else 0.0)

        if self.fit_timing:
            for name in ('t_peak', 't_trough'):
                if name in labels:
                    y = labels[name].to_numpy(dtype=float)
                    self.timing_models_[name] = self._fit_one(
                        X_arr[fit_idx], y[fit_idx], 0.5)
        return self

    @staticmethod
    def _conformal_offset(model, X_cal, y_cal, quantile):
        """Shift that makes ``P(y <= q_hat - shift)`` equal ``quantile``.

        With scores ``E_i = q_hat(x_i) - y_i``, the requirement
        ``P(E >= c) = quantile`` gives ``c`` as the ``1 - quantile`` empirical
        quantile of the scores.  A perfectly calibrated model scores zero here
        and is left alone.
        """
        mask = np.isfinite(y_cal)
        if mask.sum() < 30:
            return 0.0
        scores = model.predict(X_cal[mask]) - y_cal[mask]
        return float(np.quantile(scores, 1.0 - quantile))

    # -- prediction -------------------------------------------------------

    def _check_fitted(self):
        if not self.models_:
            raise RuntimeError('forecaster is not fitted')

    def predict(self, X):
        """Scaled-space quantiles as ``{target: array of shape (n, n_levels)}``.

        Crossed quantiles are repaired by monotone rearrangement.
        """
        self._check_fitted()
        X_arr = self._as_array(X)
        out = {}
        for target in self.targets:
            preds = np.column_stack([
                self.models_[(target, q)].predict(X_arr) - self.offsets_[(target, q)]
                for q in self.levels])
            # Rearrange after the conformal shift, which can reintroduce crossings.
            out[target] = monotone_rearrange(preds)
        return out

    def predict_timing(self, X):
        """Median bars-ahead of the peak and the trough."""
        self._check_fitted()
        X_arr = self._as_array(X)
        return {name: model.predict(X_arr)
                for name, model in self.timing_models_.items()}

    def _as_array(self, X):
        if isinstance(X, pd.DataFrame):
            if self.feature_names_ is not None and list(X.columns) != self.feature_names_:
                X = X[self.feature_names_]
            return X.to_numpy(dtype=float)
        return np.asarray(X, dtype=float)

    def predict_returns(self, X, scale):
        """Quantiles converted from scaled space back to simple returns."""
        scale = np.asarray(scale, dtype=float).reshape(-1, 1)
        return {t: v * scale for t, v in self.predict(X).items()}

    def distributions(self, X, scale):
        """One :class:`QuantileDistribution` per row and target, in returns."""
        preds = self.predict_returns(X, scale)
        n = len(next(iter(preds.values())))
        return [{t: QuantileDistribution(self.levels, preds[t][i])
                 for t in preds} for i in range(n)]

    def forecast_frame(self, X, scale, index=None):
        """Tidy per-row forecast table in return space."""
        preds = self.predict_returns(X, scale)
        index = index if index is not None else getattr(X, 'index', None)
        data = {}
        for target, values in preds.items():
            for j, q in enumerate(self.levels):
                data[f'{target}_q{int(round(q * 100)):02d}'] = values[:, j]
        frame = pd.DataFrame(data, index=index)
        if self.timing_models_:
            for name, value in self.predict_timing(X).items():
                frame[name] = value
        return frame
