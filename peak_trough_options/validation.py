#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Purged, embargoed walk-forward validation.

Excursion labels span ``horizon`` future bars, so consecutive samples overlap.
Plain K-fold -- and even a plain expanding-window split -- lets a training
label that was resolved *inside* the test window leak the answer, which is the
usual reason a peak/trough model looks brilliant offline and dies live.

Each split therefore drops the last ``horizon`` training bars (their outcome
is not yet known at the moment the test window opens) plus an optional
``embargo`` of extra bars to blunt residual serial correlation.
"""

import numpy as np


def purged_walk_forward_splits(n_samples, n_splits=5, horizon=21,
                               embargo=0, min_train=252):
    """Yield ``(train_idx, test_idx)`` position arrays, oldest fold first.

    Training is always an expanding window strictly before the test block.
    """
    if n_splits < 1:
        raise ValueError('n_splits must be >= 1')
    if min_train <= 0:
        raise ValueError('min_train must be positive')

    purge = horizon + embargo
    first_test = min_train + purge
    if first_test >= n_samples:
        raise ValueError(
            f'not enough samples: need more than {first_test} rows for '
            f'min_train={min_train}, horizon={horizon}, embargo={embargo}, '
            f'got {n_samples}')

    blocks = np.array_split(np.arange(first_test, n_samples), n_splits)
    for block in blocks:
        if block.size == 0:
            continue
        test_start = int(block[0])
        train_end = test_start - purge
        if train_end < min_train:
            continue
        yield np.arange(train_end), block


def split_report(n_samples, splits):
    """Human-readable description of a split plan."""
    rows = []
    for i, (train_idx, test_idx) in enumerate(splits, 1):
        gap = int(test_idx[0] - train_idx[-1] - 1)
        rows.append({'fold': i,
                     'train_size': len(train_idx),
                     'test_size': len(test_idx),
                     'train_end': int(train_idx[-1]),
                     'test_start': int(test_idx[0]),
                     'test_end': int(test_idx[-1]),
                     'purged_bars': gap})
    return rows


def sample_weights(n, half_life=None):
    """Exponentially decaying weights so recent regimes dominate the fit."""
    if not half_life:
        return None
    age = np.arange(n)[::-1]
    return 0.5 ** (age / float(half_life))
