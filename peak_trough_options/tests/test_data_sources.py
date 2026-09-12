#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Loading and validating bars.

The yfinance tests stub the download so the suite never touches the network:
what is under test is the handling of what Yahoo hands back, not Yahoo.
"""

import sys
import types

import numpy as np
import pandas as pd
import pytest

from peak_trough_options import data


def _yahoo_frame(n=6, multi_level=False, adjusted=True, tz=None):
    index = pd.bdate_range('2024-01-02', periods=n, name='Date')
    if tz:
        index = index.tz_localize(tz)
    close = np.linspace(100, 105, n)
    frame = pd.DataFrame({
        'Open': close - 0.5, 'High': close + 1.0,
        'Low': close - 1.0, 'Close': close,
        'Volume': np.full(n, 1_000_000.0),
    }, index=index)
    if not adjusted:
        frame['Adj Close'] = close * 0.9
    if multi_level:
        frame.columns = pd.MultiIndex.from_product([frame.columns, ['AAPL']])
    return frame


@pytest.fixture
def fake_yfinance(monkeypatch):
    """Install a stub yfinance whose download() is scripted per test."""
    calls = []

    module = types.ModuleType('yfinance')
    module.responses = [_yahoo_frame()]

    def download(tickers, **kwargs):
        calls.append({'tickers': tickers, **kwargs})
        return module.responses[min(len(calls) - 1, len(module.responses) - 1)]

    module.download = download
    module.calls = calls
    monkeypatch.setitem(sys.modules, 'yfinance', module)
    return module


# -- normalisation and bar validation --------------------------------------

def test_yahoo_columns_normalise(fake_yfinance):
    bars = data.load_yfinance('AAPL', start='2024-01-01')
    assert list(bars.columns) == ['open', 'high', 'low', 'close', 'volume']
    assert bars.index.name == 'date'
    assert isinstance(bars.index, pd.DatetimeIndex)


def test_auto_adjust_is_requested_by_default(fake_yfinance):
    data.load_yfinance('AAPL')
    assert fake_yfinance.calls[0]['auto_adjust'] is True
    assert fake_yfinance.calls[0]['interval'] == '1d'


def test_multi_level_columns_are_flattened(fake_yfinance):
    fake_yfinance.responses = [_yahoo_frame(multi_level=True)]
    bars = data.load_yfinance('AAPL')
    assert list(bars.columns) == ['open', 'high', 'low', 'close', 'volume']


def test_timezone_aware_index_is_made_naive(fake_yfinance):
    fake_yfinance.responses = [_yahoo_frame(tz='America/New_York')]
    bars = data.load_yfinance('AAPL')
    assert bars.index.tz is None


def test_unadjusted_data_warns_and_drops_the_adjusted_close(fake_yfinance):
    """Both columns alias to 'close'; keeping both would corrupt the frame."""
    fake_yfinance.responses = [_yahoo_frame(adjusted=False)]
    with pytest.warns(UserWarning, match='unadjusted'):
        bars = data.load_yfinance('AAPL', auto_adjust=False)
    assert list(bars.columns) == ['open', 'high', 'low', 'close', 'volume']
    assert bars['close'].iloc[0] == pytest.approx(100.0)   # not the 0.9x column


def test_colliding_price_columns_raise_rather_than_corrupt():
    raw = _yahoo_frame(adjusted=False)
    with pytest.raises(ValueError, match='duplicate columns'):
        data.normalise_ohlcv(raw)


# -- transport behaviour ---------------------------------------------------

def test_an_empty_first_response_retries_with_a_plain_session(fake_yfinance):
    """yfinance's curl_cffi transport fails behind TLS-terminating proxies."""
    fake_yfinance.responses = [_yahoo_frame(n=0), _yahoo_frame()]
    bars = data.load_yfinance('AAPL')
    assert len(bars) == 6
    assert len(fake_yfinance.calls) == 2
    assert fake_yfinance.calls[0]['session'] is None
    assert fake_yfinance.calls[1]['session'] is not None


def test_the_retry_is_skipped_when_a_session_was_supplied(fake_yfinance):
    fake_yfinance.responses = [_yahoo_frame(n=0)]
    with pytest.raises(ValueError, match='no rows'):
        data.load_yfinance('AAPL', session=object())
    assert len(fake_yfinance.calls) == 1


def test_a_persistently_empty_download_explains_itself(fake_yfinance):
    fake_yfinance.responses = [_yahoo_frame(n=0)]
    with pytest.raises(ValueError, match='no rows for'):
        data.load_yfinance('NOPE')


def test_a_list_of_symbols_is_refused(fake_yfinance):
    with pytest.raises(TypeError, match='one symbol at a time'):
        data.load_yfinance(['AAPL', 'MSFT'])


# -- the float-noise tolerance ---------------------------------------------

def test_one_ulp_of_rounding_is_repaired_not_rejected():
    """Real adjusted Yahoo history does this a couple of times per decade."""
    index = pd.bdate_range('2024-01-02', periods=3)
    close = np.array([101.813225, 100.0, 100.0])
    frame = pd.DataFrame({'open': [102.883085, 100.0, 100.0],
                          'high': [103.874998, 101.0, 101.0],
                          'low': close - np.array([1.421e-14, 1.0, 1.0]),
                          'close': close}, index=index)
    frame.loc[index[0], 'low'] = close[0] + 1.421e-14   # low a hair above close

    bars = data.normalise_ohlcv(frame)
    assert bars['low'].iloc[0] <= bars['close'].iloc[0]
    assert bars['low'].iloc[0] == pytest.approx(close[0], abs=1e-12)


def test_a_genuinely_broken_bar_still_raises():
    frame = pd.DataFrame({'open': [10.0], 'high': [9.0], 'low': [8.0],
                          'close': [8.5]}, index=pd.to_datetime(['2020-01-02']))
    with pytest.raises(ValueError, match='beyond rounding'):
        data.normalise_ohlcv(frame)


def test_the_tolerance_is_relative_to_price():
    """A one-cent error is noise on a $100k future and real on a penny stock."""
    index = pd.to_datetime(['2020-01-02'])
    penny = pd.DataFrame({'open': [0.50], 'high': [0.50], 'low': [0.50],
                          'close': [0.60]}, index=index)
    with pytest.raises(ValueError, match='beyond rounding'):
        data.normalise_ohlcv(penny)
