import pandas as pd

from stock_screener.rules.kdj_oversold import KdjOversoldRule
from stock_screener.rules.midline_ma60 import MidlineMa60Rule


def _df_one_stock(close: list[float], high: list[float], low: list[float]) -> pd.DataFrame:
    assert len(close) == len(high) == len(low)
    n = len(close)
    return pd.DataFrame(
        {
            "ts_code": ["000001.SZ"] * n,
            "trade_date": [f"202401{(i+1):02d}" for i in range(n)],
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "vol": [0.0] * n,
            "amount": [0.0] * n,
        }
    )


def test_rule1_hits_and_misses() -> None:
    rule = MidlineMa60Rule()

    close = [1.0] * 60 + [2.0]
    df = _df_one_stock(close=close, high=close, low=close)
    mask = rule.mask(df).tolist()
    assert bool(mask[-1]) is True

    df2 = _df_one_stock(close=[1.0] * 61, high=[1.0] * 61, low=[1.0] * 61)
    mask2 = rule.mask(df2).tolist()
    assert bool(mask2[-1]) is False

    ma60_last = (59 * 1.0 + 2.0) / 60
    alpha = 2 / 11
    ema1_last = alpha * 2.0 + (1 - alpha) * 1.0
    ema2_last = alpha * ema1_last + (1 - alpha) * 1.0
    assert abs(float(df["ma60"].iloc[-1]) - ma60_last) < 1e-12
    assert abs(float(df["mid_bullbear"].iloc[-1]) - ema2_last) < 1e-9


def test_rule2_kdj_oversold_hits_and_misses() -> None:
    rule = KdjOversoldRule()

    close = [1.0] * 10
    df = _df_one_stock(close=close, high=close, low=close)
    mask = rule.mask(df).tolist()
    assert bool(mask[-2]) is False
    assert bool(mask[-1]) is True
    assert abs(float(df["j"].iloc[-2]) - 0.0) < 1e-12
    assert abs(float(df["j"].iloc[-1]) - 0.0) < 1e-12

    close2 = [float(i) for i in range(1, 11)]
    high2 = close2
    low2 = [c - 1.0 for c in close2]
    df2 = _df_one_stock(close=close2, high=high2, low=low2)
    mask2 = rule.mask(df2).tolist()
    assert all(bool(m) is False for m in mask2)
    assert float(df2["j"].iloc[-1]) == 100.0
