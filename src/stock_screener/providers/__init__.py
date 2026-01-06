from __future__ import annotations

from stock_screener.providers.baostock_provider import BaoStockProvider
from stock_screener.providers.tushare_provider import TuShareProvider


def get_provider(name: str):
    n = (name or "").strip().lower()
    if n == "tushare":
        return TuShareProvider()
    if n == "baostock":
        return BaoStockProvider()
    raise ValueError("provider must be 'baostock' or 'tushare'")

