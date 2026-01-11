from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Literal
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from stock_screener.dates import parse_yyyymmdd

PriceAdjust = Literal["none", "qfq", "hfq"]


class EastmoneyError(RuntimeError):
    pass


def ts_code_to_secid(ts_code: str) -> str:
    ts = (ts_code or "").strip().upper()
    m = re.fullmatch(r"(\d{6})\.(SH|SZ|BJ)", ts)
    if not m:
        raise ValueError(f"invalid ts_code: {ts_code}")
    code, market = m.group(1), m.group(2)
    if market == "SH":
        return f"1.{code}"
    # Eastmoney uses market=0 for SZ/BJ (secid is 0.<code>)
    if market in {"SZ", "BJ"}:
        return f"0.{code}"
    raise ValueError(f"unsupported market: {market}")


def _fqt(adjust: PriceAdjust) -> int:
    if adjust == "none":
        return 0
    if adjust == "qfq":
        return 1
    if adjust == "hfq":
        return 2
    raise ValueError(f"invalid adjust: {adjust}")


@dataclass(frozen=True)
class EastmoneyProvider:
    name: str = "eastmoney"
    base_url: str = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    timeout_seconds: float = 15.0
    max_retries: int = 3

    def _get_json(self, *, params: dict[str, str]) -> dict[str, Any]:
        url = f"{self.base_url}?{urlencode(params)}"
        last_err: Exception | None = None
        for attempt in range(max(1, int(self.max_retries))):
            try:
                req = Request(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                        "Accept": "application/json,text/plain,*/*",
                    },
                )
                with urlopen(req, timeout=float(self.timeout_seconds)) as resp:
                    raw = resp.read()
                data = json.loads(raw.decode("utf-8", errors="replace"))
                if not isinstance(data, dict):
                    raise EastmoneyError("unexpected response type")
                return data
            except (URLError, TimeoutError, json.JSONDecodeError, EastmoneyError) as e:
                last_err = e
                if attempt + 1 >= max(1, int(self.max_retries)):
                    break
                time.sleep(0.5 * (2**attempt))
        raise EastmoneyError(f"eastmoney request failed: {last_err}")

    def fetch_daily(
        self,
        *,
        ts_code: str,
        start: str | None = None,
        end: str | None = None,
        adjust: PriceAdjust = "none",
    ) -> tuple[pd.DataFrame, str | None]:
        """
        Fetch daily bars from Eastmoney.

        Returns: (df, name) where df columns match sqlite daily schema.
        """

        if start is not None:
            parse_yyyymmdd(start)
        if end is not None:
            parse_yyyymmdd(end)
        beg = start or "0"
        end_eff = end or "20500101"
        secid = ts_code_to_secid(ts_code)

        js = self._get_json(
            params={
                "secid": secid,
                "klt": "101",  # 101 = daily
                "fqt": str(_fqt(adjust)),
                "beg": beg,
                "end": end_eff,
                # fields1 is required; otherwise it may return rc=102 with data=null.
                "fields1": "f1,f2,f3,f4,f5,f6",
                "fields2": "f51,f52,f53,f54,f55,f56,f57",
            }
        )
        if int(js.get("rc") or 0) != 0:
            raise EastmoneyError(f"eastmoney rc={js.get('rc')}, msg={js.get('msg') or ''}".strip())
        data = js.get("data") if isinstance(js, dict) else None
        if not isinstance(data, dict):
            return pd.DataFrame(columns=["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"]), None

        name = data.get("name")
        name_str = str(name).strip() if name is not None else None
        if name_str == "":
            name_str = None

        klines = data.get("klines")
        if not isinstance(klines, list) or not klines:
            return pd.DataFrame(columns=["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"]), name_str

        rows: list[dict[str, Any]] = []
        for line in klines:
            if not isinstance(line, str):
                continue
            parts = line.split(",")
            if len(parts) < 7:
                continue
            trade_date = str(parts[0]).replace("-", "")
            try:
                parse_yyyymmdd(trade_date)
            except ValueError:
                continue
            rows.append(
                {
                    "ts_code": ts_code.upper(),
                    "trade_date": trade_date,
                    # Eastmoney order: date, open, close, high, low, vol, amount
                    "open": parts[1],
                    "close": parts[2],
                    "high": parts[3],
                    "low": parts[4],
                    "vol": parts[5],
                    "amount": parts[6],
                }
            )

        if not rows:
            return pd.DataFrame(columns=["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"]), name_str

        df = pd.DataFrame(rows)
        for col in ["open", "high", "low", "close", "vol", "amount"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df[["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"]], name_str

