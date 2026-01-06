from __future__ import annotations

import math
from typing import Optional

import pandas as pd


def MA(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(window=n).mean()


def EMA(series: pd.Series, n: int) -> pd.Series:
    alpha = 2 / (n + 1)
    return series.ewm(alpha=alpha, adjust=False).mean()


def REF(series: pd.Series, n: int) -> pd.Series:
    return series.shift(n)


def LLV(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(window=n).min()


def HHV(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(window=n).max()


def SMA(series: pd.Series, n: int, m: int) -> pd.Series:
    if n <= 0:
        raise ValueError("n must be positive")
    if m <= 0 or m > n:
        raise ValueError("m must be in [1, n]")

    values = series.astype("float64").to_numpy()
    out = [math.nan] * len(values)
    prev: Optional[float] = None

    for i, x in enumerate(values):
        if math.isnan(x):
            out[i] = math.nan
            continue
        if prev is None or math.isnan(prev):
            prev = x
            out[i] = prev
            continue
        prev = (m * x + (n - m) * prev) / n
        out[i] = prev

    return pd.Series(out, index=series.index, name=series.name)

