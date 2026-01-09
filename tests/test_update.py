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

