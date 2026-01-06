from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stock_screener.indicators import HHV, LLV, REF, SMA
from stock_screener.rules.base import RuleColumns


@dataclass(frozen=True)
class KdjOversoldRule:
    name: str = "kdj_oversold"
    n: int = 9
    m1: int = 3
    m2: int = 3
    jt: float = 13
    cols: RuleColumns = RuleColumns()

    def enrich(self, df: pd.DataFrame) -> None:
        if self.cols.j in df.columns:
            return

        llv_low = df.groupby("ts_code", sort=False)["low"].transform(lambda s: LLV(s, self.n))
        hhv_high = df.groupby("ts_code", sort=False)["high"].transform(lambda s: HHV(s, self.n))
        denom = hhv_high - llv_low
        rsv = (df["close"] - llv_low) / denom * 100
        rsv = rsv.mask(denom == 0, 0)

        k = rsv.groupby(df["ts_code"], sort=False).apply(lambda s: SMA(s, self.m1, 1)).reset_index(level=0, drop=True)
        d = k.groupby(df["ts_code"], sort=False).apply(lambda s: SMA(s, self.m2, 1)).reset_index(level=0, drop=True)
        j = 3 * k - 2 * d
        df[self.cols.j] = j

    def mask(self, df: pd.DataFrame) -> pd.Series:
        self.enrich(df)
        j = df[self.cols.j]
        j_prev = j.groupby(df["ts_code"], sort=False).transform(lambda s: REF(s, 1))
        return (j < self.jt) & (j_prev < self.jt)
