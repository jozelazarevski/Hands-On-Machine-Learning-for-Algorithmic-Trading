#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Confusion matrices for the discrete decisions the forecast implies.

The model itself is a quantile regressor -- it produces a distribution, not a
label.  But every actual use of it collapses that distribution into a choice:
buy the call or don't, expect a big move or a small one.  Those choices are
classifications, and a confusion matrix is the right way to see where they go
wrong: whether the model is wrong in the expensive direction, and whether it is
merely repeating the base rate.

Thresholding a distribution at 0.5 throws information away, so every task is
also scored probabilistically (Brier, ROC AUC) alongside the matrix.  A
calibrated 55% is worth more to an option buyer than a hard label, and the two
views can disagree: a forecast can be genuinely informative and still lose
every argument with a majority-class guesser when the base rate is lopsided.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score

from .distribution import QuantileDistribution
from .model import DEFAULT_LEVELS


def confusion_frame(y_true, y_pred, labels):
    """Counts as a labelled frame: rows are actual, columns are predicted."""
    position = {label: i for i, label in enumerate(labels)}
    counts = np.zeros((len(labels), len(labels)), dtype=int)
    for actual, predicted in zip(y_true, y_pred):
        counts[position[actual], position[predicted]] += 1
    return pd.DataFrame(counts,
                        index=pd.Index(labels, name='actual'),
                        columns=pd.Index(labels, name='predicted'))


def with_totals(frame):
    """Append row and column totals for reading the margins off directly."""
    out = frame.copy()
    out['total'] = out.sum(axis=1)
    out.loc['total'] = out.sum(axis=0)
    return out


def row_normalised(frame):
    """Each row as a share of its actual class: the diagonal is recall."""
    totals = frame.sum(axis=1).replace(0, np.nan)
    return frame.div(totals, axis=0)


def binary_scores(y_true, y_prob, threshold=0.5):
    """Threshold and probabilistic scores for one binary decision.

    ``lift_over_majority`` is the one that matters: accuracy above what you
    would get by always guessing the more common outcome.  A model can look
    accurate and still be worthless when the base rate is lopsided.
    """
    y_true = np.asarray(y_true, dtype=bool)
    y_prob = np.asarray(y_prob, dtype=float)
    keep = np.isfinite(y_prob)
    y_true, y_prob = y_true[keep], y_prob[keep]
    n = len(y_true)
    if n == 0:
        return {}

    y_pred = y_prob > threshold
    tp = int(np.sum(y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))

    def ratio(numerator, denominator):
        return float(numerator / denominator) if denominator else float('nan')

    recall = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    precision = ratio(tp, tp + fp)
    base_rate = float(np.mean(y_true))
    accuracy = ratio(tp + tn, n)
    majority = max(base_rate, 1 - base_rate)

    both_classes = 0 < y_true.sum() < n
    return {
        'n': n,
        'base_rate': base_rate,
        'predicted_rate': float(np.mean(y_pred)),
        'accuracy': accuracy,
        'majority_accuracy': majority,
        'lift_over_majority': accuracy - majority,
        'balanced_accuracy': float(np.nanmean([recall, specificity])),
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': ratio(2 * tp, 2 * tp + fp + fn),
        'mcc': float(matthews_corrcoef(y_true, y_pred)) if both_classes and 0 < y_pred.sum() < n else float('nan'),
        'roc_auc': float(roc_auc_score(y_true, y_prob)) if both_classes else float('nan'),
        'brier': float(np.mean((y_prob - y_true) ** 2)),
        'true_positive': tp, 'false_positive': fp,
        'true_negative': tn, 'false_negative': fn,
    }


def binary_confusion(y_true, y_prob, labels=('no', 'yes'), threshold=0.5):
    """Confusion frame for a binary decision, ordered negative class first."""
    y_true = np.asarray(y_true, dtype=bool)
    y_prob = np.asarray(y_prob, dtype=float)
    keep = np.isfinite(y_prob)
    y_true, y_prob = y_true[keep], y_prob[keep]
    negative, positive = labels
    actual = np.where(y_true, positive, negative)
    predicted = np.where(y_prob > threshold, positive, negative)
    return confusion_frame(actual, predicted, [negative, positive])


def format_confusion(frame, title=None, normalise=False):
    """Render a confusion matrix as text, with totals and recall per row."""
    lines = []
    if title:
        lines.append(title)
        lines.append('-' * len(title))

    table = with_totals(frame)
    lines.append(table.to_string())

    if normalise and frame.to_numpy().sum():
        lines.append('')
        lines.append('row-normalised (diagonal = recall):')
        lines.append(row_normalised(frame).to_string(
            float_format=lambda v: f'{v:.2f}'))
    return '\n'.join(lines)


def scores_frame(results, columns=('n', 'base_rate', 'accuracy',
                                   'majority_accuracy', 'lift_over_majority',
                                   'balanced_accuracy', 'precision', 'recall',
                                   'mcc', 'roc_auc', 'brier')):
    """Stack per-task score dictionaries into one comparable table."""
    rows = []
    for name, scores in results.items():
        if scores:
            rows.append({'task': name, **{c: scores.get(c) for c in columns}})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Turning the forecast distributions into decisions
# --------------------------------------------------------------------------

TASKS = ('direction', 'upside_touch', 'downside_touch', 'big_move')

TASK_LABELS = {
    'direction': ('down', 'up'),
    'upside_touch': ('missed', 'touched'),
    'downside_touch': ('missed', 'touched'),
    'big_move': ('small', 'big'),
}

TASK_QUESTIONS = {
    'direction': 'does it close above today by expiry',
    'upside_touch': 'does the high reach +{t} sigma before expiry',
    'downside_touch': 'does the low reach -{t} sigma before expiry',
    'big_move': 'does it move more than {t} sigma either way',
}


def _distributions(oos, levels, suffix=''):
    """Per-row quantile distributions in volatility-scaled space.

    Working in scaled space keeps a sigma threshold meaningful across assets
    and regimes, and it is the space the model actually predicts in.
    """
    out = {}
    for target in ('mfe', 'mae', 'ret'):
        columns = [f'{target}_q{int(round(q * 100)):02d}{suffix}' for q in levels]
        if not all(c in oos.columns for c in columns):
            continue
        values = oos[columns].to_numpy(dtype=float)
        out[target] = [QuantileDistribution(levels, row) for row in values]
    return out


def decision_probabilities(oos, levels=None, sigma=1.0, suffix=''):
    """Model probability and realised outcome for each decision task.

    ``sigma`` sets how far the touch and big-move thresholds sit from spot, in
    units of the volatility scale used to normalise the labels -- so 1.0 is a
    one-standard-deviation move over the horizon.
    """
    levels = tuple(levels or oos.attrs.get('levels', DEFAULT_LEVELS))
    dists = _distributions(oos, levels, suffix=suffix)
    if not dists:
        raise ValueError('out-of-sample frame carries no quantile columns')

    truth = {t: oos[f'{t}_z_true'].to_numpy(dtype=float) for t in ('mfe', 'mae', 'ret')}
    n = len(oos)
    probability, outcome = {}, {}

    probability['direction'] = np.array([1.0 - dists['ret'][i].cdf(0.0) for i in range(n)])
    outcome['direction'] = truth['ret'] > 0

    probability['upside_touch'] = np.array([1.0 - dists['mfe'][i].cdf(sigma) for i in range(n)])
    outcome['upside_touch'] = truth['mfe'] >= sigma

    probability['downside_touch'] = np.array([dists['mae'][i].cdf(-sigma) for i in range(n)])
    outcome['downside_touch'] = truth['mae'] <= -sigma

    probability['big_move'] = np.array([
        (1.0 - dists['ret'][i].cdf(sigma)) + dists['ret'][i].cdf(-sigma) for i in range(n)])
    outcome['big_move'] = np.abs(truth['ret']) >= sigma

    return probability, outcome


def resolve_threshold(probabilities, threshold='auto'):
    """Pick the cut-off that turns probabilities into a yes/no call.

    A fixed 0.5 is silently wrong for a rare event: the chance of a one-sigma
    move is about 0.32 under a normal, so no honest forecast ever crosses 0.5
    and the matrix collapses into a single column.  ``'auto'`` cuts at the
    model's own mean probability instead -- "flag the setups this model rates
    above its own average" -- which is derived purely from the predictions and
    never touches the outcomes, so it cannot leak.
    """
    if threshold == 'auto':
        finite = np.asarray(probabilities, dtype=float)
        finite = finite[np.isfinite(finite)]
        return float(np.mean(finite)) if finite.size else 0.5
    return float(threshold)


def classification_report(oos, levels=None, sigma=1.0, threshold='auto',
                          include_baseline=True):
    """Confusion matrices and scores for every decision task.

    Returns ``{task: {...}}`` with the confusion frame, the scores, the
    threshold used, and the same set for the climatology baseline so the
    comparison is like for like.  Each side is cut at its own ``'auto'``
    threshold, so both are asked the same question about their own beliefs.
    """
    model_prob, outcome = decision_probabilities(oos, levels=levels, sigma=sigma)
    base_prob = (decision_probabilities(oos, levels=levels, sigma=sigma,
                                        suffix='_base')[0]
                 if include_baseline else {})

    report = {}
    for task in TASKS:
        labels = TASK_LABELS[task]
        cut = resolve_threshold(model_prob[task], threshold)
        entry = {
            'question': TASK_QUESTIONS[task].format(t=sigma),
            'threshold': cut,
            'confusion': binary_confusion(outcome[task], model_prob[task],
                                          labels=labels, threshold=cut),
            'scores': binary_scores(outcome[task], model_prob[task], cut),
        }
        if task in base_prob:
            base_cut = resolve_threshold(base_prob[task], threshold)
            entry['baseline_threshold'] = base_cut
            entry['baseline_confusion'] = binary_confusion(
                outcome[task], base_prob[task], labels=labels, threshold=base_cut)
            entry['baseline_scores'] = binary_scores(
                outcome[task], base_prob[task], base_cut)
        report[task] = entry
    return report


def summarise_report(report):
    """Model-versus-baseline scores for every task, as one table."""
    rows = []
    for task, entry in report.items():
        scores = entry['scores']
        if not scores:
            continue
        row = {'task': task,
               'n': scores['n'],
               'threshold': entry.get('threshold', float('nan')),
               'base_rate': scores['base_rate'],
               'accuracy': scores['accuracy'],
               'majority_accuracy': scores['majority_accuracy'],
               'lift_over_majority': scores['lift_over_majority'],
               'balanced_accuracy': scores['balanced_accuracy'],
               'mcc': scores['mcc'],
               'roc_auc': scores['roc_auc'],
               'brier': scores['brier']}
        baseline = entry.get('baseline_scores') or {}
        row['baseline_roc_auc'] = baseline.get('roc_auc', float('nan'))
        row['baseline_brier'] = baseline.get('brier', float('nan'))
        rows.append(row)
    return pd.DataFrame(rows)


def classification_verdict(report):
    """Plain-language reading of the confusion matrices.

    A confusion matrix flatters a model whenever one class dominates, so two
    separate questions get asked: does the ranking beat climatology (AUC), and
    does the thresholded call beat always guessing the majority class.  Those
    can disagree, and when they do the ranking is the more useful of the two.
    """
    summary = summarise_report(report)
    if summary.empty:
        return {'headline': 'no decisions could be scored', 'notes': [],
                'summary': summary}

    # Beating a coin toss is not the bar: climatology is, exactly as for the
    # quantile scores.  A constant forecast still earns an AUC away from 0.5
    # when the base rate drifts between folds, so that is what must be cleared.
    floor = np.maximum(0.52, summary['baseline_roc_auc'].fillna(0.5) + 0.01)
    informative = summary[summary['roc_auc'] > floor]

    if informative.empty:
        headline = ('No usable classification: no decision ranks better '
                    'out of sample than climatology does.')
    else:
        headline = ('Ranks better than climatology on: '
                    + ', '.join(informative['task']) + '.')

    notes = []
    for (_, row), bar in zip(summary.iterrows(), floor):
        task = row['task']
        auc, base_auc = row['roc_auc'], row['baseline_roc_auc']
        if not np.isfinite(auc):
            notes.append(f'{task}: only one class occurred, nothing to score')
            continue

        if auc <= 0.5:
            notes.append(f'{task}: the ranking is no better than a coin toss '
                         f'(AUC {auc:.2f}) -- do not trade this decision')
        elif np.isfinite(base_auc) and auc <= base_auc:
            notes.append(f'{task}: climatology ranks it at least as well '
                         f'(AUC {base_auc:.2f} vs {auc:.2f}) -- the features '
                         'add nothing here')
        elif auc <= bar:
            notes.append(f'{task}: AUC {auc:.2f} is above climatology but '
                         'inside the range noise alone produces -- unproven')
        elif row['lift_over_majority'] <= 0:
            notes.append(f'{task}: ranks better than climatology '
                         f'(AUC {auc:.2f}), but at the p>{row["threshold"]:.2f} '
                         f'cut it still loses to always guessing the majority '
                         f'class (base rate {row["base_rate"]:.0%}) -- trade the '
                         'ranking, and move the cut to suit your payoff, rather '
                         'than reading the matrix as a flat failure')

        if np.isfinite(row['brier']) and np.isfinite(row['baseline_brier']) \
                and row['brier'] > row['baseline_brier']:
            notes.append(f'{task}: climatology is better calibrated '
                         f'(Brier {row["baseline_brier"]:.3f} vs {row["brier"]:.3f})')

        if auc > 0.65:
            notes.append(f'{task}: AUC {auc:.2f} is high for daily bars -- '
                         'check for leakage before believing it')

    return {'headline': headline, 'notes': notes, 'summary': summary}
