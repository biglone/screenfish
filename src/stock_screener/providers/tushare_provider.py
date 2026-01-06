from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stock_screener.tushare_client import TuShareClient, TuShareNotConfigured, get_tushare_token


class TuShareTokenMissing(RuntimeError):
    pass


@dataclass(frozen=True)
class TuShareProvider:
    name: str = "tushare"

    def _client(self) -> TuShareClient:
        token = get_tushare_token()
        if not token:
            raise TuShareTokenMissing("missing TUSHARE_TOKEN")
        return TuShareClient(token=token)

    def open_trade_dates(self, *, start: str, end: str) -> list[str]:
        client = self._client()
        return client.trade_cal_open_dates(start=start, end=end)

    def daily_by_trade_date(self, *, trade_date: str) -> pd.DataFrame:
        client = self._client()
        return client.daily_by_trade_date(trade_date=trade_date)

    def stock_basics(self) -> pd.DataFrame:
        client = self._client()
        pro = client._pro()

        def _fetch(status: str) -> pd.DataFrame:
            return pro.stock_basic(exchange="", list_status=status, fields="ts_code,name")

        frames = []
        for st in ("L", "D", "P"):
            df = _fetch(st)
            if df is not None and not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame(columns=["ts_code", "name"])
        out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["ts_code"])
        return out[["ts_code", "name"]]
