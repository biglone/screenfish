from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

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
