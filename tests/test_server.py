import time
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from stock_screener import __version__ as app_version
from stock_screener.backends.sqlite_backend import SqliteBackend
from stock_screener.config import Settings
from stock_screener.server import create_app
from stock_screener.pinyin import pinyin_initials


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


def _seed_sqlite_one_day(cache_dir: Path, trade_date: str) -> Settings:
    settings = Settings(cache_dir=cache_dir, data_backend="sqlite")
    backend = SqliteBackend(settings.sqlite_path)
    backend.init()
    df = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": trade_date,
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
    backend.mark_trade_date_updated(trade_date)
    return settings


def test_health_and_status(tmp_path: Path) -> None:
    settings = _seed_sqlite(tmp_path)
    app = create_app(settings=settings)
    client = TestClient(app)

    r = client.get("/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    rv = client.get("/v1/version")
    assert rv.status_code == 200
    v = rv.json()
    assert v["name"] == "stock-screener"
    assert v["version"] == app_version

    r2 = client.get("/v1/status")
    assert r2.status_code == 200
    body = r2.json()
    assert body["max_daily_trade_date"] == "20240131"
    assert body["stocks"] == 1


def test_data_integrity_ok(tmp_path: Path, monkeypatch) -> None:
    import stock_screener.server as srv

    settings = _seed_sqlite_one_day(tmp_path, "20240131")
    app = create_app(settings=settings)
    client = TestClient(app)

    class FakeProvider:
        def open_trade_dates(self, *, start: str, end: str):  # noqa: ANN001
            return ["20240131"]

    monkeypatch.setattr(srv, "get_provider", lambda _provider: FakeProvider())

    r = client.get("/v1/data/integrity", params={"provider": "baostock", "date": "20240131", "lookback_days": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["missing_update_log_count"] == 0
    assert body["missing_daily_count"] == 0


def test_data_integrity_detects_missing_update_log(tmp_path: Path, monkeypatch) -> None:
    import stock_screener.server as srv

    settings = _seed_sqlite_one_day(tmp_path, "20240131")
    backend = SqliteBackend(settings.sqlite_path)
    with backend.connect() as conn:
        conn.execute("DELETE FROM update_log")

    app = create_app(settings=settings)
    client = TestClient(app)

    class FakeProvider:
        def open_trade_dates(self, *, start: str, end: str):  # noqa: ANN001
            return ["20240131"]

    monkeypatch.setattr(srv, "get_provider", lambda _provider: FakeProvider())

    r = client.get("/v1/data/integrity", params={"provider": "baostock", "date": "20240131", "lookback_days": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["missing_update_log_count"] == 1
    assert body["missing_update_log_dates"] == ["20240131"]


def test_data_integrity_detects_missing_bj_rows(tmp_path: Path, monkeypatch) -> None:
    import stock_screener.server as srv

    settings = _seed_sqlite_one_day(tmp_path, "20240131")
    backend = SqliteBackend(settings.sqlite_path)
    backend.upsert_stock_basic_df(pd.DataFrame([{"ts_code": "920809.BJ", "name": "安达科技"}]))

    app = create_app(settings=settings)
    client = TestClient(app)

    class FakeProvider:
        def open_trade_dates(self, *, start: str, end: str):  # noqa: ANN001
            return ["20240131"]

    monkeypatch.setattr(srv, "get_provider", lambda _provider: FakeProvider())

    r = client.get("/v1/data/integrity", params={"provider": "baostock", "date": "20240131", "lookback_days": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["missing_market_daily_count"]["BJ"] == 1


def test_list_trade_dates_uses_update_log(tmp_path: Path) -> None:
    settings = _seed_sqlite_one_day(tmp_path, "20240131")
    app = create_app(settings=settings)
    client = TestClient(app)

    r = client.get("/v1/data/trade-dates", params={"limit": 10})
    assert r.status_code == 200
    body = r.json()
    assert body["price_adjust"] == "none"
    assert body["total"] == 1
    assert body["order"] == "desc"
    assert body["dates"] == ["20240131"]


def test_stock_daily_filters_no_trade_bars(tmp_path: Path) -> None:
    settings = Settings(cache_dir=tmp_path, data_backend="sqlite", price_adjust="qfq")
    backend = SqliteBackend(
        settings.sqlite_path,
        daily_table=settings.daily_table,
        update_log_table=settings.update_log_table,
        provider_stock_progress_table=settings.provider_stock_progress_table,
    )
    backend.init()
    backend.upsert_stock_basic_df(pd.DataFrame([{"ts_code": "000001.SZ", "name": "平安银行"}]))

    backend.upsert_daily_df(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240101",
                    "open": 1.0,
                    "high": 1.0,
                    "low": 1.0,
                    "close": 1.0,
                    "vol": 100.0,
                    "amount": 1000.0,
                },
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240102",
                    "open": 100.0,
                    "high": 100.0,
                    "low": 100.0,
                    "close": 100.0,
                    "vol": None,
                    "amount": None,
                },
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240103",
                    "open": 2.0,
                    "high": 2.0,
                    "low": 2.0,
                    "close": 2.0,
                    "vol": 100.0,
                    "amount": 1000.0,
                },
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240104",
                    "open": 3.0,
                    "high": 3.0,
                    "low": 3.0,
                    "close": 3.0,
                    "vol": 100.0,
                    "amount": 1000.0,
                },
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240105",
                    "open": 999.0,
                    "high": 999.0,
                    "low": 999.0,
                    "close": 999.0,
                    "vol": 0.0,
                    "amount": 0.0,
                },
            ]
        )
    )

    app = create_app(settings=settings)
    client = TestClient(app)

    r = client.get("/v1/stocks/000001.SZ/daily", params={"price_adjust": "qfq", "limit": 100})
    assert r.status_code == 200
    body = r.json()
    assert body["ts_code"] == "000001.SZ"
    assert body["name"] == "平安银行"
    assert [b["trade_date"] for b in body["bars"]] == ["20240101", "20240103", "20240104"]


def test_indicator_series_filters_no_trade_bars(tmp_path: Path) -> None:
    settings = Settings(cache_dir=tmp_path, data_backend="sqlite", price_adjust="qfq")
    backend = SqliteBackend(
        settings.sqlite_path,
        daily_table=settings.daily_table,
        update_log_table=settings.update_log_table,
        provider_stock_progress_table=settings.provider_stock_progress_table,
    )
    backend.init()

    backend.upsert_daily_df(
        pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240101",
                    "open": 1.0,
                    "high": 1.0,
                    "low": 1.0,
                    "close": 1.0,
                    "vol": 100.0,
                    "amount": 1000.0,
                },
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240102",
                    "open": 100.0,
                    "high": 100.0,
                    "low": 100.0,
                    "close": 100.0,
                    "vol": None,
                    "amount": None,
                },
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240103",
                    "open": 2.0,
                    "high": 2.0,
                    "low": 2.0,
                    "close": 2.0,
                    "vol": 100.0,
                    "amount": 1000.0,
                },
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240104",
                    "open": 3.0,
                    "high": 3.0,
                    "low": 3.0,
                    "close": 3.0,
                    "vol": 100.0,
                    "amount": 1000.0,
                },
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240105",
                    "open": 999.0,
                    "high": 999.0,
                    "low": 999.0,
                    "close": 999.0,
                    "vol": 0.0,
                    "amount": 0.0,
                },
            ]
        )
    )

    f = backend.create_formula(name="ma3_test", formula="X:MA(CLOSE,3);", kind="indicator", enabled=True)
    formula_id = int(f["id"])

    app = create_app(settings=settings)
    client = TestClient(app)

    r = client.get(f"/v1/stocks/000001.SZ/indicators/{formula_id}", params={"price_adjust": "qfq", "limit": 100})
    assert r.status_code == 200
    body = r.json()
    assert body["ts_code"] == "000001.SZ"
    assert body["formula_id"] == formula_id
    assert [p["trade_date"] for p in body["points"]] == ["20240101", "20240103", "20240104"]
    assert body["points"][-1]["value"] == 2.0


def test_auto_update_config_defaults_and_update(tmp_path: Path) -> None:
    settings = _seed_sqlite(tmp_path)
    app = create_app(settings=settings)
    client = TestClient(app)

    r = client.get("/auto-update-config")
    assert r.status_code == 200
    cfg = r.json()
    assert cfg["enabled"] is False
    assert cfg["interval_seconds"] == 600
    assert cfg["provider"] == "baostock"
    assert cfg["repair_days"] == 30

    r2 = client.put(
        "/auto-update-config",
        json={"enabled": True, "interval_seconds": 1200, "provider": "tushare", "repair_days": 15},
    )
    assert r2.status_code == 200
    cfg2 = r2.json()
    assert cfg2["enabled"] is True
    assert cfg2["interval_seconds"] == 1200
    assert cfg2["provider"] == "tushare"
    assert cfg2["repair_days"] == 15

    r3 = client.get("/auto-update-config")
    assert r3.status_code == 200
    assert r3.json() == cfg2


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
    if pinyin_initials("平安银行") is None:
        pytest.skip("pypinyin is not installed")
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


def test_watchlist_import_backfills_bj_symbols(tmp_path: Path, monkeypatch) -> None:
    import stock_screener.providers.eastmoney_provider as em

    settings = _seed_sqlite(tmp_path)
    app = create_app(settings=settings)
    client = TestClient(app)

    r2 = client.post("/v1/watchlist/groups", json={"name": "BJ"})
    assert r2.status_code == 200
    group_id = r2.json()["id"]

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

    r3 = client.post(
        f"/v1/watchlist/groups/{group_id}/items",
        json={"items": [{"ts_code": "920809.BJ"}]},
    )
    assert r3.status_code == 200
    assert r3.json()["ok"] is True
    assert r3.json().get("unknown_total") in (None, 0)

    # Ensure the symbol is searchable under qfq as well.
    r4 = client.get("/v1/stocks", params={"search": "920809.BJ", "limit": 10, "offset": 0, "price_adjust": "qfq"})
    assert r4.status_code == 200
    assert any(s["ts_code"] == "920809.BJ" for s in r4.json()["stocks"])

    r5 = client.get("/v1/stocks/920809.BJ/daily", params={"price_adjust": "qfq", "limit": 10})
    assert r5.status_code == 200
    assert r5.json()["ts_code"] == "920809.BJ"


def test_auth_register_login_and_isolation(tmp_path: Path, monkeypatch) -> None:
    settings = _seed_sqlite(tmp_path)
    monkeypatch.setenv("STOCK_SCREENER_AUTH_ENABLED", "1")
    monkeypatch.setenv("STOCK_SCREENER_AUTH_SECRET", "test-secret")
    monkeypatch.setenv("STOCK_SCREENER_AUTH_SIGNUP_MODE", "open")
    app = create_app(settings=settings)
    client = TestClient(app)

    r = client.get("/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["auth_enabled"] is True

    r0 = client.get("/v1/status")
    assert r0.status_code == 401

    r0b = client.get("/auto-update-config")
    assert r0b.status_code == 401

    r1 = client.post("/v1/auth/register", json={"username": "admin", "password": "password123"})
    assert r1.status_code == 200
    tok1 = r1.json()["token"]
    assert tok1
    assert r1.json()["user"]["role"] == "admin"

    r2 = client.post("/v1/auth/login", json={"username": "admin", "password": "password123"})
    assert r2.status_code == 200
    tok1b = r2.json()["token"]
    assert tok1b

    r3 = client.get("/v1/auth/me", headers={"Authorization": f"Bearer {tok1b}"})
    assert r3.status_code == 200
    assert r3.json()["username"] == "admin"

    r4 = client.get("/v1/status", headers={"Authorization": f"Bearer {tok1b}"})
    assert r4.status_code == 200

    r5 = client.post("/v1/auth/register", json={"username": "user1", "password": "password123"})
    assert r5.status_code == 200
    tok2 = r5.json()["token"]
    assert r5.json()["user"]["role"] == "user"

    r5b = client.get("/auto-update-config", headers={"Authorization": f"Bearer {tok2}"})
    assert r5b.status_code == 403

    r5c = client.get("/auto-update-config", headers={"Authorization": f"Bearer {tok1b}"})
    assert r5c.status_code == 200

    r6 = client.post(
        "/v1/formulas",
        json={"name": "f1", "formula": "CLOSE>OPEN", "description": None, "kind": "screen", "timeframe": None, "enabled": True},
        headers={"Authorization": f"Bearer {tok2}"},
    )
    assert r6.status_code == 403

    r7 = client.post(
        "/v1/formulas",
        json={"name": "f1", "formula": "CLOSE>OPEN", "description": None, "kind": "screen", "timeframe": None, "enabled": True},
        headers={"Authorization": f"Bearer {tok1b}"},
    )
    assert r7.status_code == 200
    assert r7.json()["name"] == "f1"

    r8 = client.post(
        "/v1/watchlist/groups",
        json={"name": "A组"},
        headers={"Authorization": f"Bearer {tok1b}"},
    )
    assert r8.status_code == 200
    group_id = r8.json()["id"]

    r9 = client.post(
        f"/v1/watchlist/groups/{group_id}/items",
        json={"items": [{"ts_code": "000001.SZ"}]},
        headers={"Authorization": f"Bearer {tok1b}"},
    )
    assert r9.status_code == 200

    r10 = client.get("/v1/watchlist", headers={"Authorization": f"Bearer {tok2}"})
    assert r10.status_code == 200
    assert group_id not in {g["id"] for g in r10.json()["groups"]}


def test_auth_email_signup_flow(tmp_path: Path, monkeypatch) -> None:
    settings = _seed_sqlite(tmp_path)
    monkeypatch.setenv("STOCK_SCREENER_AUTH_ENABLED", "1")
    monkeypatch.setenv("STOCK_SCREENER_AUTH_SECRET", "test-secret")
    monkeypatch.setenv("STOCK_SCREENER_AUTH_SIGNUP_MODE", "email")
    monkeypatch.setenv("STOCK_SCREENER_AUTH_EMAIL_DEBUG_RETURN_CODE", "1")
    app = create_app(settings=settings)
    client = TestClient(app)

    r0 = client.get("/v1/health")
    assert r0.status_code == 200
    assert r0.json()["auth_enabled"] is True
    assert r0.json()["auth_signup_mode"] == "email"

    r1 = client.post("/v1/auth/email/request", json={"email": "user@example.com"})
    assert r1.status_code == 200
    code = r1.json()["debug_code"]
    assert isinstance(code, str) and code

    r2 = client.post(
        "/v1/auth/register/email",
        json={"email": "user@example.com", "code": code, "username": "admin", "password": "password123"},
    )
    assert r2.status_code == 200
    assert r2.json()["user"]["role"] == "admin"
    token = r2.json()["token"]

    r3 = client.post("/v1/auth/register", json={"username": "u2", "password": "password123"})
    assert r3.status_code == 422  # request validation

    r4 = client.post("/v1/auth/register", json={"username": "user2", "password": "password123"})
    assert r4.status_code == 404  # open signup disabled under email mode

    r5 = client.post("/v1/auth/login", json={"username": "user@example.com", "password": "password123"})
    assert r5.status_code == 200

    r6 = client.get("/v1/status", headers={"Authorization": f"Bearer {token}"})
    assert r6.status_code == 200


def test_update_wait_skips_baostock_update_until_published(tmp_path: Path, monkeypatch) -> None:
    settings = _seed_sqlite_one_day(tmp_path, "20240130")

    import stock_screener.server as server

    calls = {"update": 0}

    def fake_update_daily_all_service(**_kwargs) -> None:
        calls["update"] += 1

    def fake_probe_baostock_daily_available(*, trade_date: str) -> tuple[bool, str]:
        return False, f"not published: {trade_date}"

    def fake_resolve_wait_target_trade_date(*, provider: str, target_date: str, lookback_days: int = 30) -> str:
        return target_date

    monkeypatch.setattr(server, "update_daily_all_service", fake_update_daily_all_service)
    monkeypatch.setattr(server, "probe_baostock_daily_available", fake_probe_baostock_daily_available)
    monkeypatch.setattr(server, "resolve_wait_target_trade_date", fake_resolve_wait_target_trade_date)

    app = create_app(settings=settings)
    client = TestClient(app)

    r = client.post(
        "/v1/update/wait",
        json={"provider": "baostock", "target_date": "20240131", "interval_seconds": 1, "timeout_seconds": 1},
    )
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    time.sleep(0.2)
    assert calls["update"] == 0

    time.sleep(1.2)
    r2 = client.get(f"/v1/update/wait/{job_id}")
    assert r2.status_code == 200
    assert r2.json()["status"] == "timeout"
    assert calls["update"] == 0


def test_update_wait_baostock_succeeds_after_update(tmp_path: Path, monkeypatch) -> None:
    settings = _seed_sqlite_one_day(tmp_path, "20240130")

    import stock_screener.server as server

    calls = {"update": 0}

    def fake_update_daily_all_service(*, settings: Settings, start, end, provider, repair_days) -> None:
        calls["update"] += 1
        df = pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": str(end),
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.0,
                    "close": 10.5,
                    "vol": 100.0,
                    "amount": 1000.0,
                }
            ]
        )
        for mode in ("none", "qfq", "hfq"):
            mode_settings = settings.model_copy(update={"price_adjust": mode})
            backend = SqliteBackend(
                mode_settings.sqlite_path,
                daily_table=mode_settings.daily_table,
                update_log_table=mode_settings.update_log_table,
                provider_stock_progress_table=mode_settings.provider_stock_progress_table,
            )
            backend.init()
            backend.upsert_daily_df(df)
            backend.mark_trade_date_updated(str(end))

    def fake_probe_baostock_daily_available(*, trade_date: str) -> tuple[bool, str]:
        return True, f"published: {trade_date}"

    def fake_resolve_wait_target_trade_date(*, provider: str, target_date: str, lookback_days: int = 30) -> str:
        return target_date

    monkeypatch.setattr(server, "update_daily_all_service", fake_update_daily_all_service)
    monkeypatch.setattr(server, "probe_baostock_daily_available", fake_probe_baostock_daily_available)
    monkeypatch.setattr(server, "resolve_wait_target_trade_date", fake_resolve_wait_target_trade_date)

    app = create_app(settings=settings)
    client = TestClient(app)

    r = client.post(
        "/v1/update/wait",
        json={"provider": "baostock", "target_date": "20240131", "interval_seconds": 1, "timeout_seconds": 5},
    )
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    deadline = time.time() + 2.0
    status = None
    while time.time() < deadline:
        status = client.get(f"/v1/update/wait/{job_id}").json()["status"]
        if status != "running":
            break
        time.sleep(0.05)

    assert status == "succeeded"
    assert calls["update"] == 1


def test_admin_user_management_and_token_revocation(tmp_path: Path, monkeypatch) -> None:
    settings = _seed_sqlite(tmp_path)
    monkeypatch.setenv("STOCK_SCREENER_AUTH_ENABLED", "1")
    monkeypatch.setenv("STOCK_SCREENER_AUTH_SECRET", "test-secret")
    monkeypatch.setenv("STOCK_SCREENER_AUTH_SIGNUP_MODE", "open")
    app = create_app(settings=settings)
    client = TestClient(app)

    r1 = client.post("/v1/auth/register", json={"username": "admin", "password": "password123"})
    assert r1.status_code == 200
    admin_tok = r1.json()["token"]
    admin_id = r1.json()["user"]["id"]
    assert r1.json()["user"]["role"] == "admin"
    admin_headers = {"Authorization": f"Bearer {admin_tok}"}

    r2 = client.post("/v1/auth/register", json={"username": "user1", "password": "password123"})
    assert r2.status_code == 200
    user_tok = r2.json()["token"]
    user_id = r2.json()["user"]["id"]
    assert r2.json()["user"]["role"] == "user"
    user_headers = {"Authorization": f"Bearer {user_tok}"}

    r3 = client.get("/v1/admin/users", headers=admin_headers)
    assert r3.status_code == 200
    body = r3.json()
    assert body["total"] >= 2
    ids = {u["id"] for u in body["users"]}
    assert admin_id in ids
    assert user_id in ids

    r4 = client.get("/v1/admin/users", headers=user_headers)
    assert r4.status_code == 403

    r5 = client.post(
        "/v1/admin/users",
        json={"username": "user2", "password": "password123", "email": "user2@example.com"},
        headers=admin_headers,
    )
    assert r5.status_code == 200
    user2 = r5.json()
    assert user2["username"] == "user2"
    assert user2["email"] == "user2@example.com"
    assert user2["role"] == "user"
    assert user2["disabled"] is False

    # Prevent lockout: you cannot disable the last enabled admin.
    r6 = client.put(f"/v1/admin/users/{admin_id}", json={"disabled": True}, headers=admin_headers)
    assert r6.status_code == 400

    # Disable user -> old token becomes unusable.
    r7 = client.put(f"/v1/admin/users/{user_id}", json={"disabled": True}, headers=admin_headers)
    assert r7.status_code == 200
    assert r7.json()["disabled"] is True

    r8 = client.get("/v1/auth/me", headers=user_headers)
    assert r8.status_code == 401

    # Re-enable user; previous token remains invalid due to token_version bump.
    r9 = client.put(f"/v1/admin/users/{user_id}", json={"disabled": False}, headers=admin_headers)
    assert r9.status_code == 200
    assert r9.json()["disabled"] is False

    r10 = client.get("/v1/auth/me", headers=user_headers)
    assert r10.status_code == 401

    r11 = client.post("/v1/auth/login", json={"username": "user1", "password": "password123"})
    assert r11.status_code == 200
    user_tok2 = r11.json()["token"]
    user_headers2 = {"Authorization": f"Bearer {user_tok2}"}

    r12 = client.get("/v1/auth/me", headers=user_headers2)
    assert r12.status_code == 200

    # Reset password -> invalidates existing tokens.
    r13 = client.post(
        f"/v1/admin/users/{user_id}/set-password",
        json={"password": "newpassword123"},
        headers=admin_headers,
    )
    assert r13.status_code == 200
    assert isinstance(r13.json()["token_version"], int)

    r14 = client.get("/v1/auth/me", headers=user_headers2)
    assert r14.status_code == 401

    r15 = client.post("/v1/auth/login", json={"username": "user1", "password": "password123"})
    assert r15.status_code == 401
    r16 = client.post("/v1/auth/login", json={"username": "user1", "password": "newpassword123"})
    assert r16.status_code == 200

    # Explicit token revocation.
    tok3 = r16.json()["token"]
    user_headers3 = {"Authorization": f"Bearer {tok3}"}
    r17 = client.get("/v1/auth/me", headers=user_headers3)
    assert r17.status_code == 200

    r18 = client.post(f"/v1/admin/users/{user_id}/revoke-tokens", headers=admin_headers)
    assert r18.status_code == 200
    assert isinstance(r18.json()["token_version"], int)

    r19 = client.get("/v1/auth/me", headers=user_headers3)
    assert r19.status_code == 401


def test_account_update_email_and_change_password(tmp_path: Path, monkeypatch) -> None:
    settings = _seed_sqlite(tmp_path)
    monkeypatch.setenv("STOCK_SCREENER_AUTH_ENABLED", "1")
    monkeypatch.setenv("STOCK_SCREENER_AUTH_SECRET", "test-secret")
    monkeypatch.setenv("STOCK_SCREENER_AUTH_SIGNUP_MODE", "open")
    app = create_app(settings=settings)
    client = TestClient(app)

    r1 = client.post("/v1/auth/register", json={"username": "user1", "password": "password123"})
    assert r1.status_code == 200
    tok1 = r1.json()["token"]
    uid1 = r1.json()["user"]["id"]
    headers1 = {"Authorization": f"Bearer {tok1}"}

    r2 = client.get("/v1/account", headers=headers1)
    assert r2.status_code == 200
    assert r2.json()["id"] == uid1
    assert r2.json()["email"] is None

    r3 = client.put(
        "/v1/account",
        json={"email": "user1@example.com", "current_password": "wrong"},
        headers=headers1,
    )
    assert r3.status_code == 401

    r4 = client.put(
        "/v1/account",
        json={"email": "user1@example.com", "current_password": "password123"},
        headers=headers1,
    )
    assert r4.status_code == 200
    assert r4.json()["email"] == "user1@example.com"

    r5 = client.post("/v1/auth/login", json={"username": "user1@example.com", "password": "password123"})
    assert r5.status_code == 200

    r6 = client.post("/v1/auth/register", json={"username": "user2", "password": "password123"})
    assert r6.status_code == 200
    tok2 = r6.json()["token"]
    headers2 = {"Authorization": f"Bearer {tok2}"}
    r7 = client.put(
        "/v1/account",
        json={"email": "user1@example.com", "current_password": "password123"},
        headers=headers2,
    )
    assert r7.status_code == 409

    r8 = client.post(
        "/v1/account/change-password",
        json={"current_password": "wrong", "new_password": "newpassword123"},
        headers=headers1,
    )
    assert r8.status_code == 401

    r9 = client.post(
        "/v1/account/change-password",
        json={"current_password": "password123", "new_password": "newpassword123"},
        headers=headers1,
    )
    assert r9.status_code == 200
    tok1_new = r9.json()["token"]
    headers1_new = {"Authorization": f"Bearer {tok1_new}"}

    r10 = client.get("/v1/auth/me", headers=headers1)
    assert r10.status_code == 401
    r11 = client.get("/v1/auth/me", headers=headers1_new)
    assert r11.status_code == 200

    r12 = client.post("/v1/auth/login", json={"username": "user1", "password": "password123"})
    assert r12.status_code == 401
    r13 = client.post("/v1/auth/login", json={"username": "user1", "password": "newpassword123"})
    assert r13.status_code == 200

    r14 = client.post("/v1/auth/login", json={"username": "user1@example.com", "password": "newpassword123"})
    assert r14.status_code == 200

    # Clearing email disables email login.
    r15 = client.put(
        "/v1/account",
        json={"email": None, "current_password": "newpassword123"},
        headers=headers1_new,
    )
    assert r15.status_code == 200
    assert r15.json()["email"] is None

    r16 = client.post("/v1/auth/login", json={"username": "user1@example.com", "password": "newpassword123"})
    assert r16.status_code == 401
