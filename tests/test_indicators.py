import math

import pandas as pd

from stock_screener.indicators import EMA, HHV, LLV, MA, REF, SMA


def test_ma() -> None:
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    out = MA(s, 2).tolist()
    assert math.isnan(out[0])
    assert out[1:] == [1.5, 2.5, 3.5]


def test_ema_adjust_false_alpha() -> None:
    s = pd.Series([1.0, 1.0, 2.0])
    out = EMA(s, 2).tolist()
    alpha = 2 / (2 + 1)
    expected_last = alpha * 2.0 + (1 - alpha) * 1.0
    assert out[0] == 1.0
    assert out[1] == 1.0
    assert abs(out[2] - expected_last) < 1e-12


def test_ref() -> None:
    s = pd.Series([10.0, 11.0, 12.0])
    out = REF(s, 1).tolist()
    assert math.isnan(out[0])
    assert out[1:] == [10.0, 11.0]


def test_llv_hhv() -> None:
    s = pd.Series([3.0, 1.0, 2.0, 0.0])
    llv = LLV(s, 2).tolist()
    hhv = HHV(s, 2).tolist()
    assert math.isnan(llv[0]) and math.isnan(hhv[0])
    assert llv[1:] == [1.0, 1.0, 0.0]
    assert hhv[1:] == [3.0, 2.0, 2.0]


def test_sma_tdx_recursive() -> None:
    s = pd.Series([1.0, 2.0, 3.0])
    out = SMA(s, 3, 1).tolist()
    assert out[0] == 1.0
    assert abs(out[1] - (4 / 3)) < 1e-12
    assert abs(out[2] - (17 / 9)) < 1e-12

