from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd


class TuShareNotConfigured(RuntimeError):
    pass


def get_tushare_token() -> str | None:
    token = os.environ.get("TUSHARE_TOKEN")
    if token:
        return token.strip() or None
    return None


@dataclass(frozen=True)
class TuShareClient:
    token: str

    def _pro(self):
        try:
            import tushare as ts  # type: ignore
        except ModuleNotFoundError as e:  # pragma: no cover
            raise TuShareNotConfigured("tushare is not installed; run: pip install tushare") from e
        ts.set_token(self.token)
        return ts.pro_api()

    def _with_retry(self, fn, *, tries: int = 3, base_sleep_s: float = 1.0):
        last_err: Exception | None = None
        for i in range(tries):
            try:
                return fn()
            except Exception as e:  # pragma: no cover
                last_err = e
                time.sleep(base_sleep_s * (2**i))
        assert last_err is not None
        raise last_err

    def trade_cal_open_dates(self, start: str, end: str) -> list[str]:
        pro = self._pro()

        def _call():
            return pro.trade_cal(exchange="", start_date=start, end_date=end, is_open=1, fields="cal_date,is_open")

        df = self._with_retry(_call)
        if df is None or df.empty:
            return []
        df = df[df["is_open"].astype(int) == 1]
        return df["cal_date"].astype(str).tolist()

    def daily_by_trade_date(self, trade_date: str) -> pd.DataFrame:
        pro = self._pro()

        def _call():
            return pro.daily(
                trade_date=trade_date,
                fields="ts_code,trade_date,open,high,low,close,vol,amount",
            )

        df = self._with_retry(_call)
        if df is None:
            return pd.DataFrame()
        return df
