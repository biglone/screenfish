from __future__ import annotations

from stock_screener.providers.baostock_provider import BaoStockProvider
from stock_screener.providers.tushare_provider import TuShareProvider


def get_provider(name: str, *, baostock_adjustflag: str = "3"):
    n = (name or "").strip().lower()
    if n == "tushare":
        return TuShareProvider()
    if n == "baostock":
        return BaoStockProvider(adjustflag=baostock_adjustflag)
    raise ValueError("provider must be 'baostock' or 'tushare'")
