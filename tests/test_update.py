from pathlib import Path

import pandas as pd
import pytest

from stock_screener.backends.sqlite_backend import SqliteBackend
from stock_screener.config import Settings
from stock_screener.update import UpdateIncomplete, update_daily_service


class _DummyTuShareProvider:
    name = "tushare"

    def open_trade_dates(self, *, start: str, end: str) -> list[str]:
        return [end]

    def daily_by_trade_date(self, *, trade_date: str) -> pd.DataFrame:
        return pd.DataFrame()


def test_update_daily_tushare_empty_does_not_mark_updated(tmp_path: Path, monkeypatch) -> None:
    import stock_screener.update as update_mod

    monkeypatch.setattr(update_mod, "get_provider", lambda _name: _DummyTuShareProvider())

    settings = Settings(cache_dir=tmp_path, data_backend="sqlite")
    with pytest.raises(UpdateIncomplete):
        update_daily_service(settings=settings, start="20240131", end="20240131", provider="tushare", repair_days=30)

    backend = SqliteBackend(settings.sqlite_path)
    assert backend.count_daily_rows_for_trade_date("20240131") == 0
    assert backend.max_trade_date_in_update_log() is None


class _DummyBaoStockProvider:
    name = "baostock"

    def open_trade_dates(self, *, start: str, end: str) -> list[str]:
        return [end]

    def session(self):  # noqa: ANN201
        from contextlib import contextmanager

        @contextmanager
        def _cm():
            yield object()

        return _cm()

    def _all_stock_basics(self, *, bs, day: str) -> pd.DataFrame:  # noqa: ANN001
        return pd.DataFrame(columns=["ts_code", "name"])


def test_update_daily_baostock_also_updates_bj_symbols(tmp_path: Path, monkeypatch) -> None:
    import stock_screener.update as update_mod
    import stock_screener.providers.eastmoney_provider as em

    monkeypatch.setattr(update_mod, "get_provider", lambda _name, **_kw: _DummyBaoStockProvider())

    def fake_fetch_daily(self, *, ts_code: str, start=None, end=None, adjust="none"):  # noqa: ANN001
        df = pd.DataFrame(
            [
                {
                    "ts_code": ts_code.upper(),
                    "trade_date": "20240131",
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.0,
                    "close": 10.5,
                    "vol": 100.0,
                    "amount": 1000.0,
                }
            ]
        )
        return df, "安达科技"

    monkeypatch.setattr(em.EastmoneyProvider, "fetch_daily", fake_fetch_daily)

    settings = Settings(cache_dir=tmp_path, data_backend="sqlite")
    backend = SqliteBackend(settings.sqlite_path)
    backend.init()

    # Seed one updated trade date so baostock path takes the "missing == []" branch.
    seed = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "20240131",
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "vol": 100.0,
                "amount": 1000.0,
            }
        ]
    )
    backend.upsert_daily_df(seed)
    backend.mark_trade_date_updated("20240131")

    # Add a BJ symbol to watchlist_items so update service knows what to refresh.
    with backend.connect() as conn:
        now = 1700000000
        conn.execute(
            """
            INSERT INTO watchlist_items (owner_id, group_id, ts_code, name, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("u1", "g1", "920809.BJ", None, now, now),
        )

    update_daily_service(settings=settings, start="20240101", end="20240131", provider="baostock", repair_days=30)

    with backend.connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS c FROM daily WHERE ts_code = '920809.BJ'").fetchone()
        assert int(row["c"]) >= 1
        row2 = conn.execute("SELECT name FROM stock_basic WHERE ts_code = '920809.BJ'").fetchone()
        assert row2 and row2["name"] == "安达科技"
