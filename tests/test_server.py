from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from stock_screener.backends.sqlite_backend import SqliteBackend
from stock_screener.config import Settings
from stock_screener.server import create_app


def _seed_sqlite(cache_dir: Path) -> Settings:
    settings = Settings(cache_dir=cache_dir, data_backend="sqlite")
    backend = SqliteBackend(settings.sqlite_path)
    backend.init()
    df = pd.DataFrame(
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
    backend.upsert_daily_df(df)
    backend.mark_trade_date_updated("20240131")
    backend.upsert_stock_basic_df(pd.DataFrame([{"ts_code": "000001.SZ", "name": "平安银行"}]))
    return settings


def test_health_and_status(tmp_path: Path) -> None:
    settings = _seed_sqlite(tmp_path)
    app = create_app(settings=settings)
    client = TestClient(app)

    r = client.get("/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    r2 = client.get("/v1/status")
    assert r2.status_code == 200
    body = r2.json()
    assert body["max_daily_trade_date"] == "20240131"
    assert body["stocks"] == 1


def test_screen_latest(tmp_path: Path) -> None:
    settings = _seed_sqlite(tmp_path)
    app = create_app(settings=settings)
    client = TestClient(app)

    r = client.post("/v1/screen", json={"date": "latest", "combo": "and", "lookback_days": 5, "with_name": True})
    assert r.status_code == 200
    body = r.json()
    assert body["trade_date"] == "20240131"
    assert isinstance(body["hits"], list)


def test_list_stocks_supports_pinyin_initials_search(tmp_path: Path) -> None:
    settings = _seed_sqlite(tmp_path)
    app = create_app(settings=settings)
    client = TestClient(app)

    r = client.get("/v1/stocks", params={"search": "payh", "limit": 50, "offset": 0})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] >= 1
    assert any(s["ts_code"] == "000001.SZ" for s in body["stocks"])
