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

