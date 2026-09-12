#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The splitter must never let an unresolved training label into the test window."""

import numpy as np
import pytest

from peak_trough_options.validation import (purged_walk_forward_splits,
                                            sample_weights, split_report)

HORIZON = 21


def _splits(n=2000, **kwargs):
    kwargs.setdefault('horizon', HORIZON)
    return list(purged_walk_forward_splits(n, **kwargs))


def test_yields_the_requested_number_of_folds():
    assert len(_splits(n_splits=5)) == 5


def test_train_and_test_never_overlap():
    for train_idx, test_idx in _splits():
        assert not set(train_idx) & set(test_idx)


def test_training_labels_resolve_before_the_test_window_opens():
    embargo = 5
    for train_idx, test_idx in _splits(embargo=embargo):
        last_train = int(train_idx[-1])
        first_test = int(test_idx[0])
        # The label at `last_train` looks HORIZON bars ahead; it must land
        # strictly before the first test bar, plus the embargo.
        assert last_train + HORIZON + embargo < first_test + 1
        assert first_test - last_train - 1 == HORIZON + embargo


def test_training_is_an_expanding_window_from_the_start():
    previous = 0
    for train_idx, _ in _splits():
        np.testing.assert_array_equal(train_idx, np.arange(len(train_idx)))
        assert len(train_idx) > previous
        previous = len(train_idx)


def test_folds_move_forward_and_tile_the_tail():
    splits = _splits()
    for (_, earlier), (_, later) in zip(splits, splits[1:]):
        assert earlier[-1] < later[0]
    covered = np.concatenate([test for _, test in splits])
    np.testing.assert_array_equal(covered, np.arange(covered[0], covered[-1] + 1))


def test_respects_the_minimum_training_size():
    for train_idx, _ in _splits(min_train=500):
        assert len(train_idx) >= 500


def test_refuses_an_impossible_configuration():
    with pytest.raises(ValueError, match='not enough samples'):
        list(purged_walk_forward_splits(300, horizon=HORIZON, min_train=756))
    with pytest.raises(ValueError):
        list(purged_walk_forward_splits(2000, n_splits=0))


def test_split_report_records_the_purged_gap():
    splits = _splits(embargo=3)
    for row in split_report(2000, splits):
        assert row['purged_bars'] == HORIZON + 3


def test_sample_weights_decay_towards_the_past():
    weights = sample_weights(100, half_life=25)
    assert weights[-1] == pytest.approx(1.0)
    assert weights[-26] == pytest.approx(0.5)
    assert np.all(np.diff(weights) > 0)
    assert sample_weights(100, None) is None
