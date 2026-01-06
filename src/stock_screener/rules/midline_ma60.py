from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stock_screener.indicators import EMA, MA
from stock_screener.rules.base import RuleColumns


@dataclass(frozen=True)
class MidlineMa60Rule:
    name: str = "midline_ma60"
    ema_n: int = 10
    ma_n: int = 60
    cols: RuleColumns = RuleColumns()

    def enrich(self, df: pd.DataFrame) -> None:
        if self.cols.ma60 not in df.columns:
            df[self.cols.ma60] = df.groupby("ts_code", sort=False)["close"].transform(lambda s: MA(s, self.ma_n))
        if self.cols.mid_bullbear not in df.columns:
            df[self.cols.mid_bullbear] = df.groupby("ts_code", sort=False)["close"].transform(
                lambda s: EMA(EMA(s, self.ema_n), self.ema_n)
            )

    def mask(self, df: pd.DataFrame) -> pd.Series:
        self.enrich(df)
        ma = df[self.cols.ma60]
        mid = df[self.cols.mid_bullbear]
        close = df["close"]
        return (mid > ma) & (close > ma)

