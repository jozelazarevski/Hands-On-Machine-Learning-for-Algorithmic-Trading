#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Confusion matrices and the decisions behind them."""

import numpy as np
import pandas as pd
import pytest

from peak_trough_options import backtest, data, pipeline
from peak_trough_options import classification as cl

HORIZON = 21


@pytest.fixture(scope='module')
def oos():
    bars = data.synthetic_ohlcv(n=2000, seed=13)
    features, labels = pipeline.prepare(bars, horizon=HORIZON)
    return backtest.walk_forward(features, labels, horizon=HORIZON, n_splits=2,
                                 min_train=900, verbose=False)


# -- the counting itself ---------------------------------------------------

def test_confusion_frame_counts_each_cell():
    actual = ['up', 'up', 'down', 'down', 'down']
    predicted = ['up', 'down', 'down', 'down', 'up']
    frame = cl.confusion_frame(actual, predicted, ['down', 'up'])
    assert frame.loc['up', 'up'] == 1
    assert frame.loc['up', 'down'] == 1
    assert frame.loc['down', 'down'] == 2
    assert frame.loc['down', 'up'] == 1
    assert frame.to_numpy().sum() == len(actual)


def test_a_perfect_classifier_is_purely_diagonal():
    labels = ['down', 'up']
    truth = ['up', 'down', 'up', 'down']
    frame = cl.confusion_frame(truth, truth, labels)
    off_diagonal = frame.to_numpy().sum() - np.trace(frame.to_numpy())
    assert off_diagonal == 0


def test_totals_and_row_normalisation():
    frame = cl.confusion_frame(['a', 'a', 'b'], ['a', 'b', 'b'], ['a', 'b'])
    totals = cl.with_totals(frame)
    assert totals.loc['total', 'total'] == 3
    assert totals.loc['a', 'total'] == 2

    normalised = cl.row_normalised(frame)
    np.testing.assert_allclose(normalised.sum(axis=1).to_numpy(), [1.0, 1.0])
    assert normalised.loc['a', 'a'] == pytest.approx(0.5)


def test_format_confusion_renders_without_blowing_up():
    frame = cl.confusion_frame(['a', 'b'], ['a', 'a'], ['a', 'b'])
    text = cl.format_confusion(frame, title='task', normalise=True)
    assert 'task' in text and 'recall' in text


# -- scores ----------------------------------------------------------------

def test_scores_of_a_perfect_forecast():
    truth = np.array([True, False, True, False])
    scores = cl.binary_scores(truth, truth.astype(float))
    assert scores['accuracy'] == 1.0
    assert scores['precision'] == 1.0
    assert scores['recall'] == 1.0
    assert scores['mcc'] == pytest.approx(1.0)
    assert scores['roc_auc'] == 1.0
    assert scores['brier'] == 0.0


def test_scores_of_an_inverted_forecast():
    truth = np.array([True, False, True, False])
    scores = cl.binary_scores(truth, (~truth).astype(float))
    assert scores['accuracy'] == 0.0
    assert scores['mcc'] == pytest.approx(-1.0)
    assert scores['roc_auc'] == 0.0


def test_confusion_cells_agree_with_the_scores():
    rng = np.random.default_rng(0)
    truth = rng.random(400) < 0.4
    prob = rng.random(400)
    scores = cl.binary_scores(truth, prob, threshold=0.5)
    frame = cl.binary_confusion(truth, prob, labels=('no', 'yes'), threshold=0.5)
    assert frame.loc['yes', 'yes'] == scores['true_positive']
    assert frame.loc['no', 'yes'] == scores['false_positive']
    assert frame.loc['no', 'no'] == scores['true_negative']
    assert frame.loc['yes', 'no'] == scores['false_negative']
    assert frame.to_numpy().sum() == scores['n']


def test_accuracy_alone_flatters_a_lopsided_base_rate():
    """The reason lift_over_majority exists."""
    truth = np.array([False] * 95 + [True] * 5)
    always_no = np.zeros(100)
    scores = cl.binary_scores(truth, always_no)
    assert scores['accuracy'] == pytest.approx(0.95)
    assert scores['lift_over_majority'] == pytest.approx(0.0)
    assert scores['balanced_accuracy'] == pytest.approx(0.5)
    assert scores['recall'] == 0.0


def test_scores_survive_a_single_class():
    truth = np.zeros(50, dtype=bool)
    scores = cl.binary_scores(truth, np.full(50, 0.3))
    assert np.isnan(scores['roc_auc'])
    assert scores['accuracy'] == 1.0


def test_empty_input_returns_nothing():
    assert cl.binary_scores([], []) == {}


# -- thresholds ------------------------------------------------------------

def test_auto_threshold_is_the_mean_predicted_probability():
    prob = np.array([0.1, 0.2, 0.3, 0.4])
    assert cl.resolve_threshold(prob, 'auto') == pytest.approx(0.25)
    assert cl.resolve_threshold(prob, 0.5) == 0.5


def test_auto_threshold_rescues_a_rare_event_from_a_dead_matrix():
    """At 0.5 a rare-event forecast never says yes, and the matrix is useless."""
    rng = np.random.default_rng(4)
    prob = rng.uniform(0.05, 0.45, 500)
    truth = rng.random(500) < prob

    fixed = cl.binary_confusion(truth, prob, threshold=0.5)
    assert fixed['yes'].sum() == 0          # nothing is ever predicted yes

    auto = cl.binary_confusion(truth, prob,
                               threshold=cl.resolve_threshold(prob, 'auto'))
    assert auto['yes'].sum() > 0 and auto['no'].sum() > 0


# -- end to end on real out-of-sample output -------------------------------

def test_report_covers_every_task(oos):
    report = cl.classification_report(oos)
    assert set(report) == set(cl.TASKS)

    for task, entry in report.items():
        frame = entry['confusion']
        assert frame.shape == (2, 2)
        assert frame.to_numpy().sum() == len(oos)
        assert list(frame.index) == list(cl.TASK_LABELS[task])
        assert 0.0 <= entry['threshold'] <= 1.0
        assert entry['scores']['n'] == len(oos)
        assert 'baseline_confusion' in entry


def test_probabilities_are_probabilities_and_match_the_outcomes(oos):
    prob, outcome = cl.decision_probabilities(oos, sigma=1.0)
    for task in cl.TASKS:
        assert np.all((prob[task] >= 0) & (prob[task] <= 1)), task
        assert outcome[task].dtype == bool
        assert len(outcome[task]) == len(oos)

    # The realised outcomes must be exactly what the labels say.
    np.testing.assert_array_equal(outcome['direction'],
                                  oos['ret_z_true'].to_numpy() > 0)
    np.testing.assert_array_equal(outcome['upside_touch'],
                                  oos['mfe_z_true'].to_numpy() >= 1.0)
    np.testing.assert_array_equal(outcome['downside_touch'],
                                  oos['mae_z_true'].to_numpy() <= -1.0)


def test_a_bigger_threshold_makes_a_touch_rarer(oos):
    _, near = cl.decision_probabilities(oos, sigma=0.5)
    _, far = cl.decision_probabilities(oos, sigma=2.0)
    assert far['upside_touch'].mean() < near['upside_touch'].mean()
    assert far['downside_touch'].mean() < near['downside_touch'].mean()


def test_upside_probability_falls_as_the_bar_is_raised(oos):
    low = cl.decision_probabilities(oos, sigma=0.5)[0]['upside_touch']
    high = cl.decision_probabilities(oos, sigma=2.0)[0]['upside_touch']
    assert np.all(high <= low + 1e-12)


def test_summary_table_lines_up_with_the_report(oos):
    report = cl.classification_report(oos)
    summary = cl.summarise_report(report)
    assert list(summary['task']) == list(cl.TASKS)
    assert summary['n'].eq(len(oos)).all()
    assert summary['base_rate'].between(0, 1).all()
    assert summary['brier'].between(0, 1).all()
    for _, row in summary.iterrows():
        assert row['lift_over_majority'] == pytest.approx(
            row['accuracy'] - row['majority_accuracy'])


def test_verdict_reads_the_summary(oos):
    assessment = cl.classification_verdict(cl.classification_report(oos))
    assert assessment['headline']
    assert isinstance(assessment['notes'], list)
    assert not assessment['summary'].empty


def test_verdict_refuses_to_credit_a_model_climatology_matches():
    """A model no better than the baseline must not be called informative."""
    summary = pd.DataFrame([{'task': 'direction', 'n': 100, 'threshold': 0.5,
                             'base_rate': 0.5, 'accuracy': 0.5,
                             'majority_accuracy': 0.5, 'lift_over_majority': 0.0,
                             'balanced_accuracy': 0.5, 'mcc': 0.0,
                             'roc_auc': 0.55, 'brier': 0.25,
                             'baseline_roc_auc': 0.60, 'baseline_brier': 0.24}])
    report = {'direction': {'scores': {k: summary.iloc[0][k] for k in
                                       ('n', 'base_rate', 'accuracy',
                                        'majority_accuracy', 'lift_over_majority',
                                        'balanced_accuracy', 'mcc', 'roc_auc',
                                        'brier')},
                            'threshold': 0.5,
                            'baseline_scores': {'roc_auc': 0.60, 'brier': 0.24}}}
    assessment = cl.classification_verdict(report)
    assert 'No usable classification' in assessment['headline']
    assert any('add nothing' in note for note in assessment['notes'])
