from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import time
from typing import Iterable

import pandas as pd

from stock_screener.dates import parse_yyyymmdd


class BaoStockNotConfigured(RuntimeError):
    pass


def _to_iso(yyyymmdd: str) -> str:
    d = parse_yyyymmdd(yyyymmdd)
    return d.strftime("%Y-%m-%d")


def _iso_to_yyyymmdd(iso_yyyy_mm_dd: str) -> str:
    return iso_yyyy_mm_dd.replace("-", "")


def bs_to_ts_code(bs_code: str) -> str:
    market, code = bs_code.split(".", 1)
    market = market.upper()
    if market == "SH":
        return f"{code}.SH"
    if market == "SZ":
        return f"{code}.SZ"
    return f"{code}.{market}"


@dataclass(frozen=True)
class BaoStockProvider:
    name: str = "baostock"

    def _bs(self):
        try:
            import baostock as bs  # type: ignore
        except ModuleNotFoundError as e:  # pragma: no cover
            raise BaoStockNotConfigured("baostock is not installed; run: pip install baostock") from e
        return bs

    @contextmanager
    def session(self):
        bs = self._bs()
        lg = bs.login()
        if getattr(lg, "error_code", "0") != "0":  # pragma: no cover
            raise RuntimeError(f"baostock login failed: {lg.error_msg}")
        try:
            yield bs
        finally:
            bs.logout()

    def _open_trade_dates(self, *, bs, start: str, end: str) -> list[str]:
        rs = bs.query_trade_dates(start_date=_to_iso(start), end_date=_to_iso(end))
        if rs.error_code != "0":  # pragma: no cover
            raise RuntimeError(f"baostock query_trade_dates failed: {rs.error_msg}")

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            return []
        df = pd.DataFrame(rows, columns=rs.fields)
        df = df[df["is_trading_day"].astype(str) == "1"]
        return [_iso_to_yyyymmdd(x) for x in df["calendar_date"].astype(str).tolist()]

    def open_trade_dates(self, *, start: str, end: str) -> list[str]:
        with self.session() as bs:
            return self._open_trade_dates(bs=bs, start=start, end=end)

    def _all_stock_codes(self, *, bs, day: str) -> list[str]:
        rs = bs.query_all_stock(day=_to_iso(day))
        if rs.error_code != "0":  # pragma: no cover
            raise RuntimeError(f"baostock query_all_stock failed: {rs.error_msg}")
        codes = []
        while rs.next():
            row = rs.get_row_data()
            if not row:
                continue
            codes.append(row[0])
        return codes

    def all_stock_codes(self, *, day: str) -> list[str]:
        with self.session() as bs:
            return self._all_stock_codes(bs=bs, day=day)

    def _fetch_daily_ranges(self, *, bs, bs_code: str, ranges: Iterable[tuple[str, str]]) -> pd.DataFrame:
        last_err: str | None = None
        for attempt in range(3):
            parts: list[pd.DataFrame] = []
            last_err = None
            for start, end in ranges:
                rs = bs.query_history_k_data_plus(
                    bs_code,
                    "date,code,open,high,low,close,volume,amount",
                    start_date=_to_iso(start),
                    end_date=_to_iso(end),
                    frequency="d",
                    adjustflag="3",
                )
                if rs is None:  # pragma: no cover
                    last_err = "baostock returned None resultset"
                    parts = []
                    break
                if rs.error_code != "0":
                    last_err = str(getattr(rs, "error_msg", "") or "unknown error")
                    parts = []
                    break
                rows = []
                while rs.next():
                    rows.append(rs.get_row_data())
                if not rows:
                    continue
                df = pd.DataFrame(rows, columns=rs.fields)
                parts.append(df)
            if last_err is None:
                break
            time.sleep(1.0 * (2**attempt))
        if not parts:
            if last_err:
                raise RuntimeError(f"baostock query_history_k_data_plus failed: {last_err}")
            return pd.DataFrame()
        df_all = pd.concat(parts, ignore_index=True)
        df_all = df_all.rename(columns={"date": "trade_date", "code": "ts_code", "volume": "vol"})
        df_all["trade_date"] = df_all["trade_date"].astype(str).map(_iso_to_yyyymmdd)
        df_all["ts_code"] = df_all["ts_code"].astype(str).map(bs_to_ts_code)
        for col in ["open", "high", "low", "close", "vol", "amount"]:
            df_all[col] = pd.to_numeric(df_all[col], errors="coerce")
        return df_all[["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"]]

    def fetch_daily_ranges(self, *, bs_code: str, ranges: Iterable[tuple[str, str]]) -> pd.DataFrame:
        with self.session() as bs:
            return self._fetch_daily_ranges(bs=bs, bs_code=bs_code, ranges=ranges)
