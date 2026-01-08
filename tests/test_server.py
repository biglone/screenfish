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


def test_watchlist_groups_and_items_crud(tmp_path: Path) -> None:
    settings = _seed_sqlite(tmp_path)
    app = create_app(settings=settings)
    client = TestClient(app)

    r = client.get("/v1/watchlist")
    assert r.status_code == 200
    body = r.json()
    assert body["version"] == 1
    assert len(body["groups"]) >= 1
    assert any(g["id"] == "default" for g in body["groups"])

    r2 = client.post("/v1/watchlist/groups", json={"name": "测试分组"})
    assert r2.status_code == 200
    group = r2.json()
    assert group["id"]
    group_id = group["id"]

    r3 = client.post(
        f"/v1/watchlist/groups/{group_id}/items",
        json={"items": [{"ts_code": "000001.SZ"}]},
    )
    assert r3.status_code == 200
    assert r3.json()["ok"] is True

    r4 = client.get("/v1/watchlist")
    assert r4.status_code == 200
    groups = {g["id"]: g for g in r4.json()["groups"]}
    assert group_id in groups
    items = groups[group_id]["items"]
    assert any(i["ts_code"] == "000001.SZ" for i in items)
    assert any(i["name"] == "平安银行" for i in items)

    r5 = client.post(
        f"/v1/watchlist/groups/{group_id}/items/remove",
        json={"ts_codes": ["000001.SZ"]},
    )
    assert r5.status_code == 200
    assert r5.json()["removed"] == 1

    r6 = client.put(f"/v1/watchlist/groups/{group_id}", json={"name": "新名称"})
    assert r6.status_code == 200
    assert r6.json()["name"] == "新名称"

    r7 = client.delete(f"/v1/watchlist/groups/{group_id}")
    assert r7.status_code == 200
    assert r7.json()["ok"] is True

    r8 = client.get("/v1/watchlist")
    assert r8.status_code == 200
    assert any(g["id"] == "default" for g in r8.json()["groups"])
