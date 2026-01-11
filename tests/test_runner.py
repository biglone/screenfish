from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from stock_screener.backends.sqlite_backend import SqliteBackend
from stock_screener.config import Settings
from stock_screener.runner import run_screen


def test_run_screen_ignores_no_trade_rows_for_indicators(tmp_path: Path) -> None:
    settings = Settings(cache_dir=tmp_path, data_backend="sqlite", price_adjust="qfq")
    backend = SqliteBackend(
        settings.sqlite_path,
        daily_table=settings.daily_table,
        update_log_table=settings.update_log_table,
        provider_stock_progress_table=settings.provider_stock_progress_table,
    )
    backend.init()

    start = date(2024, 1, 1)
    rows: list[dict[str, object]] = []
    for i in range(70):
        d = start + timedelta(days=i)
        trade_date = d.strftime("%Y%m%d")
        if i < 10:
            close = 1.0
            vol: float | None = 100.0
            amount: float | None = 1000.0
        elif i < 20:
            close = 10.0
            if i < 15:
                vol = None
                amount = None
            else:
                vol = 0.0
                amount = 0.0
        elif i < 69:
            close = 10.0
            vol = 100.0
            amount = 1000.0
        else:
            close = 9.0
            vol = 100.0
            amount = 1000.0

        rows.append(
            {
                "ts_code": "000001.SZ",
                "trade_date": trade_date,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "vol": vol,
                "amount": amount,
            }
        )

    backend.upsert_daily_df(pd.DataFrame(rows))

    last_date = (start + timedelta(days=69)).strftime("%Y%m%d")
    out = run_screen(settings=settings, date=last_date, combo="and", lookback_days=200, rules="midline_ma60")

    assert len(out) == 1
    assert out["trade_date"].iloc[0] == last_date
    assert out["ts_code"].iloc[0] == "000001.SZ"

    expected_ma60 = (10 * 1.0 + 49 * 10.0 + 9.0) / 60.0
    assert abs(float(out["ma60"].iloc[0]) - expected_ma60) < 1e-9
    assert out["rules"].iloc[0] == "midline_ma60"

