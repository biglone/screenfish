from __future__ import annotations

import math
from typing import Optional

import pandas as pd


def MA(series: pd.Series, n: int) -> pd.Series:
    """简单移动平均"""
    return series.rolling(window=n).mean()


def EMA(series: pd.Series, n: int) -> pd.Series:
    """指数移动平均"""
    alpha = 2 / (n + 1)
    return series.ewm(alpha=alpha, adjust=False).mean()


def REF(series: pd.Series, n: int) -> pd.Series:
    """引用N周期前的值"""
    return series.shift(n)


def LLV(series: pd.Series, n: int) -> pd.Series:
    """N周期最低值"""
    return series.rolling(window=n).min()


def HHV(series: pd.Series, n: int) -> pd.Series:
    """N周期最高值"""
    return series.rolling(window=n).max()


def SMA(series: pd.Series, n: int, m: int = 1) -> pd.Series:
    """加权移动平均 SMA(X,N,M) = (M*X+(N-M)*Y')/N"""
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


def CROSS(a: pd.Series, b: pd.Series) -> pd.Series:
    """A上穿B"""
    a_prev = a.shift(1)
    b_prev = b.shift(1)
    return (a_prev <= b_prev) & (a > b)


def STD(series: pd.Series, n: int) -> pd.Series:
    """标准差"""
    return series.rolling(window=n).std()


def SUM(series: pd.Series, n: int) -> pd.Series:
    """N周期求和"""
    return series.rolling(window=n).sum()


def ABS(series: pd.Series) -> pd.Series:
    """绝对值"""
    return series.abs()


def MAX(a: pd.Series, b: pd.Series) -> pd.Series:
    """取较大值"""
    return pd.concat([a, b], axis=1).max(axis=1)


def MIN(a: pd.Series, b: pd.Series) -> pd.Series:
    """取较小值"""
    return pd.concat([a, b], axis=1).min(axis=1)


def IF(cond: pd.Series, a: pd.Series, b: pd.Series) -> pd.Series:
    """条件函数"""
    return a.where(cond, b)


def COUNT(cond: pd.Series, n: int) -> pd.Series:
    """统计N周期内满足条件的次数"""
    return cond.astype(int).rolling(window=n).sum()


def EVERY(cond: pd.Series, n: int) -> pd.Series:
    """N周期内是否一直满足条件"""
    return cond.rolling(window=n).min().astype(bool)


def EXIST(cond: pd.Series, n: int) -> pd.Series:
    """N周期内是否存在满足条件"""
    return cond.rolling(window=n).max().astype(bool)


def BARSLAST(cond: pd.Series) -> pd.Series:
    """上一次条件成立到现在的周期数"""
    result = pd.Series(index=cond.index, dtype=float)
    last_true = -1
    for i, (idx, val) in enumerate(cond.items()):
        if val:
            last_true = i
            result.loc[idx] = 0
        elif last_true >= 0:
            result.loc[idx] = i - last_true
        else:
            result.loc[idx] = float('nan')
    return result


def SLOPE(series: pd.Series, n: int) -> pd.Series:
    """线性回归斜率"""
    def calc_slope(x):
        if len(x) < n:
            return float('nan')
        y = range(n)
        x_vals = x.values
        mean_x = sum(x_vals) / n
        mean_y = (n - 1) / 2
        num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x_vals, y))
        den = sum((yi - mean_y) ** 2 for yi in y)
        return num / den if den != 0 else 0
    return series.rolling(window=n).apply(calc_slope, raw=False)


