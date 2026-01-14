from __future__ import annotations

import math
import os
import re
import secrets
import shutil
import socket
import sqlite3
import statistics
import subprocess
import threading
import time
import uuid
from hashlib import sha256
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import date as _date
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import typer
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field

from stock_screener import __version__ as app_version
from stock_screener.auth import (
    AuthUser,
    auth_enabled,
    auth_allowed_email_domains,
    auth_email_code_cooldown_seconds,
    auth_email_code_max_attempts,
    auth_email_code_ttl_seconds,
    auth_email_debug_return_code,
    auth_secret,
    auth_signup_mode,
    create_access_token,
    decode_access_token,
    hash_password,
    normalize_username,
    send_email,
    smtp_config,
    token_ttl_seconds,
    validate_email,
    verify_password,
)
from stock_screener.backends.sqlite_backend import SqliteBackend
from stock_screener.config import Settings
from stock_screener.dates import format_yyyymmdd, parse_yyyymmdd, subtract_calendar_days
from stock_screener.formula_parser import execute_formula, execute_formula_outputs
from stock_screener.pinyin import pinyin_full, pinyin_initials
from stock_screener.providers import get_provider
from stock_screener.providers.baostock_provider import BaoStockNotConfigured
from stock_screener.providers.tushare_provider import TuShareTokenMissing
from stock_screener.rules import resolve_rules
from stock_screener.runner import run_screen
from stock_screener.tdx import TdxEbkFormat, ts_code_to_ebk_code
from stock_screener.tushare_client import TuShareNotConfigured
from stock_screener.update import UpdateBadRequest, UpdateIncomplete, UpdateNotConfigured
from stock_screener.update import update_daily_all_service, update_daily_service


def _client_ip(request: Request) -> str | None:
    forwarded_for = (request.headers.get("x-forwarded-for") or "").strip()
    if forwarded_for:
        ip = forwarded_for.split(",", 1)[0].strip()
        if ip:
            return ip
    if request.client is not None:
        return request.client.host
    return None


def _api_key_required(x_api_key: str | None = Header(default=None)) -> None:
    required = os.environ.get("STOCK_SCREENER_API_KEY")
    if not required:
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, required):
        raise HTTPException(status_code=401, detail="missing/invalid X-API-Key")


def _admin_token_required(x_admin_token: str | None = Header(default=None)) -> None:
    enabled_raw = os.environ.get("STOCK_SCREENER_ENABLE_LOGS_API", "").strip().lower()
    enabled = enabled_raw in {"1", "true", "yes", "y", "on"}
    token = os.environ.get("STOCK_SCREENER_ADMIN_TOKEN", "").strip()
    if not enabled or not token:
        raise HTTPException(status_code=404, detail="Not Found")
    if not x_admin_token or not secrets.compare_digest(x_admin_token, token):
        raise HTTPException(status_code=401, detail="missing/invalid X-Admin-Token")


def _tail_text_file(path: Path, *, max_lines: int, max_bytes: int = 256_000) -> list[str]:
    if max_lines <= 0:
        return []
    if not path.exists():
        return []
    try:
        size = path.stat().st_size
    except OSError:
        return []

    start = max(0, size - max_bytes)
    try:
        with path.open("rb") as f:
            if start:
                f.seek(start)
            data = f.read()
    except OSError:
        return []

    text = data.decode("utf-8", errors="replace")
    if start:
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1 :]
        else:
            text = ""
    lines = text.splitlines()
    return lines[-max_lines:]


def _tail_journald_user_unit(unit: str, *, max_lines: int) -> list[str]:
    if max_lines <= 0:
        return []
    if shutil.which("journalctl") is None:
        return []
    env = dict(os.environ)
    env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    cmd = [
        "journalctl",
        "--user-unit",
        unit,
        "-n",
        str(max_lines),
        "--no-pager",
        "--output",
        "short-iso",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=2)
    if res.returncode != 0:
        raise RuntimeError((res.stderr or "").strip() or "journalctl failed")
    return (res.stdout or "").splitlines()


def resolve_wait_target_trade_date(
    *,
    provider: Literal["baostock", "tushare"],
    target_date: str,
    lookback_days: int = 30,
) -> str:
    parse_yyyymmdd(target_date)
    if lookback_days < 0:
        raise ValueError("lookback_days must be >= 0")
    start = subtract_calendar_days(target_date, lookback_days)
    try:
        p = get_provider(provider)
        open_dates = p.open_trade_dates(start=start, end=target_date)
    except (TuShareNotConfigured, BaoStockNotConfigured, TuShareTokenMissing, ValueError) as e:
        raise ValueError(str(e)) from e
    except Exception as e:  # pragma: no cover
        raise ValueError(str(e)) from e
    if not open_dates:
        raise ValueError(f"no open trade dates found for {provider} in {start}..{target_date}")
    return str(open_dates[-1])


def probe_baostock_daily_available(*, trade_date: str) -> tuple[bool, str]:
    """
    Lightweight probe to decide whether BaoStock has published daily bars for a trade date.

    It avoids running a full-market update loop when the provider hasn't published yet.
    """

    parse_yyyymmdd(trade_date)
    try:
        import baostock as bs  # type: ignore
    except ModuleNotFoundError:
        return False, "baostock not installed"

    prev_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(30.0)
    try:
        try:
            lg = bs.login()
        except Exception as e:
            return False, f"login failed: {e}"
        if getattr(lg, "error_code", "0") != "0":
            return False, f"login failed: {getattr(lg, 'error_msg', '')}"

        try:
            iso = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]}"
            samples = ["sh.600000", "sz.000001"]
            for code in samples:
                try:
                    rs = bs.query_history_k_data_plus(
                        code,
                        "date,code,close",
                        start_date=iso,
                        end_date=iso,
                        frequency="d",
                        adjustflag="3",
                    )
                except Exception:
                    continue
                if rs is None or getattr(rs, "error_code", "0") != "0":
                    continue
                try:
                    if rs.next():
                        return True, f"sample ok: {code}"
                except Exception:
                    continue
            return False, "no sample rows yet"
        finally:
            try:
                bs.logout()
            except Exception:
                pass
    finally:
        socket.setdefaulttimeout(prev_timeout)


def _watchlist_owner_id(x_api_key: str | None = Header(default=None)) -> str:
    required = os.environ.get("STOCK_SCREENER_API_KEY")
    if required:
        if not x_api_key or not secrets.compare_digest(x_api_key, required):
            raise HTTPException(status_code=401, detail="missing/invalid X-API-Key")
        return sha256(x_api_key.encode("utf-8")).hexdigest()
    return "public"


class StatusResponse(BaseModel):
    today: str
    cache_dir: str
    sqlite_path: str
    provider_default: str = "baostock"
    max_daily_trade_date: str | None
    max_update_log_trade_date: str | None
    stocks: int
    rows: int


class UpdateRequest(BaseModel):
    provider: Literal["baostock", "tushare"] = "baostock"
    start: str | None = None
    end: str | None = None
    repair_days: int = 30


class UpdateWaitRequest(BaseModel):
    provider: Literal["baostock", "tushare"] = "baostock"
    target_date: str | None = None
    repair_days: int = 30
    interval_seconds: int = Field(default=300, ge=1)
    timeout_seconds: int = Field(default=7200, ge=1)


UpdateWaitJobStatus = Literal["running", "succeeded", "failed", "timeout", "canceled"]


class UpdateWaitResponse(BaseModel):
    job_id: str
    status: UpdateWaitJobStatus
    ok: bool
    provider: Literal["baostock", "tushare"]
    target_date: str
    latest_trade_date: str | None
    attempts: int
    elapsed_seconds: float
    message: str
    last_error: str | None = None

    mode: Literal["none", "qfq", "hfq"] | None = None
    mode_progress: float | None = None
    mode_completed: int | None = None
    mode_total: int | None = None
    progress: float | None = None


class AutoUpdateConfig(BaseModel):
    enabled: bool = False
    interval_seconds: int = Field(default=600, ge=1)
    provider: Literal["baostock", "tushare"] = "baostock"
    repair_days: int = Field(default=30, ge=0)

    run_status: Literal["idle", "running"] = "idle"
    run_started_at: int | None = None
    run_target_trade_date: str | None = None
    run_mode: Literal["none", "qfq", "hfq"] | None = None
    run_message: str | None = None

    last_run_at: int | None = None
    last_success_at: int | None = None
    last_success_trade_date: str | None = None
    last_error: str | None = None


class AutoScreenConfig(BaseModel):
    enabled: bool = False
    group_name: str = Field(default="自动筛选", min_length=1, max_length=30)
    group_id: str | None = None

    combo: Literal["and", "or"] = "and"
    rules: str | None = None
    lookback_days: int = Field(default=200, ge=0, le=20000)
    with_name: bool = False
    exclude_st: bool = False
    price_adjust: Literal["none", "qfq", "hfq"] = "qfq"
    replace_group: bool = True

    last_run_at: int | None = None
    last_trade_date: str | None = None
    last_count: int | None = None
    last_error: str | None = None


class AutoScreenConfigUpdate(BaseModel):
    enabled: bool = False
    group_name: str = Field(default="自动筛选", min_length=1, max_length=30)
    combo: Literal["and", "or"] = "and"
    rules: str | None = None
    lookback_days: int = Field(default=200, ge=0, le=20000)
    with_name: bool = False
    exclude_st: bool = False
    price_adjust: Literal["none", "qfq", "hfq"] = "qfq"
    replace_group: bool = True


class AutoScreenRunResponse(BaseModel):
    ok: bool
    trade_date: str
    count: int
    group_id: str
    group_name: str
    message: str
    last_error: str | None = None


class AutoScreenRunRequest(BaseModel):
    date: str | Literal["latest"] = "latest"
    force: bool = False


class ScreenRequest(BaseModel):
    date: str | Literal["latest"] = "latest"
    combo: Literal["and", "or"] = "and"
    lookback_days: int = 200
    rules: str | None = None
    with_name: bool = False
    exclude_st: bool = False
    price_adjust: Literal["none", "qfq", "hfq"] | None = None


class ScreenResponse(BaseModel):
    trade_date: str
    hits: list[dict[str, Any]]


class AvailabilityResponse(BaseModel):
    date: str
    provider: str
    available: bool
    detail: str


TradeDateOrder = Literal["asc", "desc"]


class TradeDateListResponse(BaseModel):
    price_adjust: Literal["none", "qfq", "hfq"]
    total: int
    order: TradeDateOrder
    dates: list[str] = Field(default_factory=list)


class DataIntegrityCount(BaseModel):
    trade_date: str
    rows: int


class DataIntegrityResponse(BaseModel):
    ok: bool
    provider: Literal["baostock", "tushare"]
    price_adjust: Literal["none", "qfq", "hfq"]
    requested_date: str
    target_date: str
    lookback_days: int
    range_start: str
    range_end: str
    open_trade_dates: int
    max_daily_trade_date: str | None
    max_update_log_trade_date: str | None

    missing_update_log_count: int
    missing_update_log_dates: list[str] = Field(default_factory=list)

    missing_daily_count: int
    missing_daily_dates: list[str] = Field(default_factory=list)

    daily_rows_min: int | None = None
    daily_rows_median: int | None = None
    daily_rows_max: int | None = None
    suspicious_daily_count: int
    suspicious_daily_dates: list[DataIntegrityCount] = Field(default_factory=list)

    market_stock_basic: dict[str, int] = Field(default_factory=dict)
    market_daily_rows_on_target_date: dict[str, int] = Field(default_factory=dict)
    missing_market_daily_count: dict[str, int] = Field(default_factory=dict)
    missing_market_daily_dates: dict[str, list[str]] = Field(default_factory=dict)


class StockItem(BaseModel):
    ts_code: str
    name: str | None = None


class StockListResponse(BaseModel):
    total: int
    stocks: list[StockItem]


class DailyBar(BaseModel):
    trade_date: str
    open: float
    high: float
    low: float
    close: float
    vol: float
    amount: float


class StockDailyResponse(BaseModel):
    ts_code: str
    name: str | None = None
    bars: list[DailyBar]


class WatchlistItem(BaseModel):
    ts_code: str
    name: str | None = None


class WatchlistGroupMeta(BaseModel):
    id: str
    name: str
    created_at: int
    updated_at: int


class WatchlistGroup(WatchlistGroupMeta):
    items: list[WatchlistItem] = Field(default_factory=list)


class WatchlistStateResponse(BaseModel):
    version: int = 1
    groups: list[WatchlistGroup] = Field(default_factory=list)


class WatchlistGroupCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=30)


class WatchlistGroupUpdate(BaseModel):
    name: str = Field(..., min_length=1, max_length=30)


class WatchlistItemsUpsertRequest(BaseModel):
    items: list[WatchlistItem] = Field(..., min_length=1)
    ignore_unknown: bool = False


class WatchlistItemsRemoveRequest(BaseModel):
    ts_codes: list[str] = Field(..., min_length=1)


class LogTailResponse(BaseModel):
    source: Literal["journald", "file", "none"]
    unit: str | None = None
    path: str | None = None
    lines: list[str] = Field(default_factory=list)


class AuthRegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8, max_length=128)


class AuthLoginRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=1, max_length=128)


class AuthUserResponse(BaseModel):
    id: str
    username: str
    role: Literal["admin", "user"]


class AuthTokenResponse(BaseModel):
    token: str
    expires_at: int
    user: AuthUserResponse


class AccountResponse(BaseModel):
    id: str
    username: str
    email: str | None = None
    role: Literal["admin", "user"]


class AccountUpdateRequest(BaseModel):
    email: str | None = Field(..., max_length=254)
    current_password: str = Field(..., min_length=1, max_length=128)


class AccountChangePasswordRequest(BaseModel):
    current_password: str = Field(..., min_length=1, max_length=128)
    new_password: str = Field(..., min_length=8, max_length=128)


class AdminUserItem(BaseModel):
    id: str
    username: str
    email: str | None = None
    role: Literal["admin", "user"]
    disabled: bool
    token_version: int = 0
    created_at: int
    updated_at: int
    last_login_at: int | None = None
    last_login_ip: str | None = None


class AdminUserListResponse(BaseModel):
    total: int
    users: list[AdminUserItem]


class AdminUserCreateRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8, max_length=128)
    email: str | None = Field(default=None, min_length=3, max_length=254)
    role: Literal["admin", "user"] = "user"
    disabled: bool = False


class AdminUserUpdateRequest(BaseModel):
    username: str | None = Field(default=None, min_length=3, max_length=50)
    email: str | None = Field(default=None, min_length=3, max_length=254)
    role: Literal["admin", "user"] | None = None
    disabled: bool | None = None


class AdminUserSetPasswordRequest(BaseModel):
    password: str = Field(..., min_length=8, max_length=128)


class AdminUserTokenVersionResponse(BaseModel):
    ok: bool = True
    token_version: int


class AuthEmailCodeRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=254)


class AuthEmailCodeResponse(BaseModel):
    ok: bool = True
    expires_at: int
    debug_code: str | None = None


class AuthEmailRegisterRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=254)
    code: str = Field(..., min_length=4, max_length=12)
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8, max_length=128)


class FormulaItem(BaseModel):
    id: int
    name: str
    formula: str
    description: str | None = None
    kind: Literal["screen", "indicator"] = "screen"
    timeframe: Literal["D", "W", "M"] | None = None
    enabled: bool = True
    created_at: str
    updated_at: str


class FormulaListResponse(BaseModel):
    total: int
    formulas: list[FormulaItem]


class FormulaCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    formula: str = Field(..., min_length=1)
    description: str | None = None
    kind: Literal["screen", "indicator"] = "screen"
    timeframe: Literal["D", "W", "M"] | None = None
    enabled: bool = True


class FormulaUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    formula: str | None = Field(default=None, min_length=1)
    description: str | None = None
    kind: Literal["screen", "indicator"] | None = None
    timeframe: Literal["D", "W", "M"] | None = None
    enabled: bool | None = None


class FormulaValidateRequest(BaseModel):
    formula: str


class FormulaValidateResponse(BaseModel):
    valid: bool
    message: str


class IndicatorPoint(BaseModel):
    trade_date: str
    value: float | None


class IndicatorLine(BaseModel):
    name: str
    points: list[IndicatorPoint]


class IndicatorSeriesResponse(BaseModel):
    ts_code: str
    formula_id: int
    name: str
    timeframe: Literal["D", "W", "M"]
    points: list[IndicatorPoint]
    lines: list[IndicatorLine] = Field(default_factory=list)


@dataclass(frozen=True)
class AppState:
    settings: Settings
    auth_enabled: bool
    auth_secret: bytes | None
    auth_token_ttl_seconds: int
    auth_signup_mode: Literal["open", "email", "closed"]


@dataclass
class _UpdateWaitJob:
    job_id: str
    provider: Literal["baostock", "tushare"]
    target_date: str
    repair_days: int
    interval_seconds: int
    timeout_seconds: int
    started_at: float

    status: UpdateWaitJobStatus = "running"
    attempts: int = 0
    latest_trade_date: str | None = None
    message: str = "job started"
    last_error: str | None = None

    mode: Literal["none", "qfq", "hfq"] | None = None
    mode_progress: float | None = None
    mode_completed: int | None = None
    mode_total: int | None = None
    progress: float | None = None

    finished_at: float | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)


def _backend(settings: Settings) -> SqliteBackend:
    return SqliteBackend(
        settings.sqlite_path,
        daily_table=settings.daily_table,
        update_log_table=settings.update_log_table,
        provider_stock_progress_table=settings.provider_stock_progress_table,
    )


class _StripPrefixMiddleware:
    def __init__(self, app: Any, *, prefix: str) -> None:
        self.app = app
        self.prefix = prefix.rstrip("/") or prefix

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") in {"http", "websocket"}:
            path = str(scope.get("path") or "")
            if path == self.prefix or path.startswith(self.prefix + "/"):
                new_scope = dict(scope)
                new_scope["path"] = path[len(self.prefix) :] or "/"
                root_path = str(new_scope.get("root_path") or "")
                new_scope["root_path"] = root_path + self.prefix
                await self.app(new_scope, receive, send)
                return
        await self.app(scope, receive, send)


def create_app(*, settings: Settings) -> FastAPI:
    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        backend = _backend(settings)
        backend.init()

        prewarm_pinyin_env = os.environ.get("STOCK_SCREENER_PREWARM_PINYIN", "1").strip().lower()
        prewarm_pinyin = prewarm_pinyin_env in {"1", "true", "yes", "y", "on"}
        if prewarm_pinyin:
            # Warm up pinyin cache to avoid the first pinyin search request paying the full cost.
            try:
                with backend.connect() as conn:
                    cur = conn.cursor()
                    cur.execute("SELECT name FROM stock_basic WHERE name IS NOT NULL")
                    for row in cur.fetchall():
                        name = row["name"]
                        if not name:
                            continue
                        try:
                            _ = pinyin_initials(str(name))
                            _ = pinyin_full(str(name))
                        except Exception:
                            continue
            except Exception:
                # Best-effort only; ignore warmup failures.
                pass

        auto_update_stop = threading.Event()
        auto_update_thread = threading.Thread(
            target=_auto_update_loop,
            kwargs={"settings": settings, "stop_event": auto_update_stop},
            daemon=True,
        )
        auto_update_thread.start()

        try:
            yield
        finally:
            auto_update_stop.set()
            auto_update_thread.join(timeout=5.0)

    enabled_auth = auth_enabled()
    secret: bytes | None = None
    ttl_seconds = 0
    signup_mode: Literal["open", "email", "closed"] = "open"
    if enabled_auth:
        secret = auth_secret()
        ttl_seconds = token_ttl_seconds()
        signup_mode = auth_signup_mode()

    app = FastAPI(title="stock_screener", version="0.1.0", lifespan=_lifespan)
    state = AppState(
        settings=settings,
        auth_enabled=enabled_auth,
        auth_secret=secret,
        auth_token_ttl_seconds=ttl_seconds,
        auth_signup_mode=signup_mode,
    )
    app.state.app_state = state

    # Allow the UI to call same-origin APIs under `/api/*` without requiring a reverse proxy rewrite.
    app.add_middleware(_StripPrefixMiddleware, prefix="/api")

    # Enable response compression for large payloads (e.g. indicators/longer K-line windows).
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    origins_env = os.environ.get("STOCK_SCREENER_CORS_ORIGINS", "").strip()
    if origins_env:
        origins = [o.strip() for o in origins_env.split(",") if o.strip()]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _settings_dep() -> Settings:
        return app.state.app_state.settings

    update_wait_jobs: dict[str, _UpdateWaitJob] = {}
    update_wait_jobs_lock = threading.Lock()
    update_lock = threading.Lock()

    def _normalize_watchlist_group_name(name: str) -> str:
        return re.sub(r"\s+", " ", str(name or "").strip())[:30]

    def _ensure_default_watchlist_group(conn: Any, owner_id: str) -> None:
        row = conn.execute(
            "SELECT 1 FROM watchlist_groups WHERE owner_id = ? LIMIT 1",
            (owner_id,),
        ).fetchone()
        if row:
            return
        now = int(time.time())
        conn.execute(
            """
            INSERT OR IGNORE INTO watchlist_groups (owner_id, id, name, created_at, updated_at)
            VALUES (?, 'default', '自选', ?, ?)
            """,
            (owner_id, now, now),
        )

    def _public_watchlist_owner_id() -> str:
        required = os.environ.get("STOCK_SCREENER_API_KEY")
        if required:
            return sha256(required.encode("utf-8")).hexdigest()
        return "public"

    def _resolve_auto_screen_owner_id(conn: Any, *, user: AuthUser | None) -> str:
        if user is not None:
            return user.id
        if app.state.app_state.auth_enabled:
            row = conn.execute("SELECT screen_owner_id FROM auto_update_config WHERE id = 1 LIMIT 1").fetchone()
            if row and row["screen_owner_id"]:
                return str(row["screen_owner_id"])
            row2 = conn.execute(
                """
                SELECT id
                FROM users
                WHERE role = 'admin'
                  AND disabled = 0
                ORDER BY created_at ASC
                LIMIT 1
                """
            ).fetchone()
            if not row2:
                raise ValueError("no enabled admin user found for auto screen")
            owner_id = str(row2["id"])
            now = int(time.time())
            conn.execute(
                "UPDATE auto_update_config SET screen_owner_id = ?, updated_at = ? WHERE id = 1",
                (owner_id, now),
            )
            return owner_id
        return _public_watchlist_owner_id()

    def _ensure_auto_screen_group(
        conn: Any,
        *,
        owner_id: str,
        group_name: str,
        group_id: str | None,
    ) -> str:
        name = _normalize_watchlist_group_name(group_name) or "自动筛选"
        now = int(time.time())

        if group_id:
            row = conn.execute(
                "SELECT 1 FROM watchlist_groups WHERE owner_id = ? AND id = ? LIMIT 1",
                (owner_id, group_id),
            ).fetchone()
            if row:
                conn.execute(
                    "UPDATE watchlist_groups SET name = ?, updated_at = ? WHERE owner_id = ? AND id = ?",
                    (name, now, owner_id, group_id),
                )
                return group_id

        row2 = conn.execute(
            """
            SELECT id
            FROM watchlist_groups
            WHERE owner_id = ?
              AND name = ?
            ORDER BY updated_at DESC, created_at DESC
            LIMIT 1
            """,
            (owner_id, name),
        ).fetchone()
        if row2 and row2["id"]:
            gid = str(row2["id"])
            conn.execute(
                "UPDATE watchlist_groups SET updated_at = ? WHERE owner_id = ? AND id = ?",
                (now, owner_id, gid),
            )
            return gid

        gid = uuid.uuid4().hex
        conn.execute(
            """
            INSERT INTO watchlist_groups (owner_id, id, name, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (owner_id, gid, name, now, now),
        )
        return gid

    def _auto_update_loop(*, settings: Settings, stop_event: threading.Event) -> None:
        backend = _backend(settings)
        idle_sleep = 5.0

        # If the process was restarted mid-run, the "running" marker can get stuck.
        # Clear it on startup so the UI doesn't show a phantom running task forever.
        try:
            with backend.connect() as conn:
                conn.execute("INSERT OR IGNORE INTO auto_update_config (id) VALUES (1)")
                row0 = conn.execute(
                    "SELECT run_status FROM auto_update_config WHERE id = 1 LIMIT 1",
                ).fetchone()
                if row0 and str(row0["run_status"] or "").strip().lower() == "running":
                    now0 = int(time.time())
                    conn.execute(
                        """
                        UPDATE auto_update_config
                        SET run_status = 'idle',
                            run_message = 'previous auto update interrupted (service restarted)',
                            updated_at = ?
                        WHERE id = 1
                        """,
                        (now0,),
                    )
        except sqlite3.OperationalError:
            # Older DBs will be migrated on init; ignore best-effort cleanup failures.
            pass
        except Exception:
            pass

        while not stop_event.is_set():
            try:
                with backend.connect() as conn:
                    conn.execute("INSERT OR IGNORE INTO auto_update_config (id) VALUES (1)")
                    row = conn.execute(
                        """
                        SELECT enabled, interval_seconds, provider, repair_days, last_run_at
                        FROM auto_update_config
                        WHERE id = 1
                        LIMIT 1
                        """
                    ).fetchone()

                if not row or int(row["enabled"] or 0) != 1:
                    stop_event.wait(idle_sleep)
                    continue

                provider = str(row["provider"] or "baostock")
                if provider not in {"baostock", "tushare"}:
                    provider = "baostock"
                interval_seconds = int(row["interval_seconds"] or 600)
                if interval_seconds < 1:
                    interval_seconds = 600
                repair_days = int(row["repair_days"] or 30)
                if repair_days < 0:
                    repair_days = 30

                now = int(time.time())
                last_run_at_raw = row["last_run_at"]
                if last_run_at_raw is None:
                    # Initialize schedule on first enable; wait interval before first run.
                    with backend.connect() as conn:
                        conn.execute(
                            "UPDATE auto_update_config SET last_run_at = ?, updated_at = ? WHERE id = 1",
                            (now, now),
                        )
                    stop_event.wait(idle_sleep)
                    continue

                last_run_at = int(last_run_at_raw or 0)
                due_in = interval_seconds - max(0, now - last_run_at)
                if due_in > 0:
                    stop_event.wait(min(idle_sleep, float(due_in)))
                    continue

                requested = format_yyyymmdd(_date.today())
                target_end: str | None = None
                try:
                    target_end = resolve_wait_target_trade_date(provider=provider, target_date=requested)
                except Exception as e:
                    msg = (str(e) or "failed to resolve target trade date")[:2000]
                    now2 = int(time.time())
                    with backend.connect() as conn:
                        conn.execute(
                            "UPDATE auto_update_config SET last_run_at = ?, last_error = ?, updated_at = ? WHERE id = 1",
                            (now2, msg, now2),
                        )
                    stop_event.wait(idle_sleep)
                    continue

                if provider == "baostock" and target_end:
                    ok_remote, detail = probe_baostock_daily_available(trade_date=target_end)
                    if not ok_remote:
                        msg = f"waiting for provider publish ({detail})"[:2000]
                        now2 = int(time.time())
                        with backend.connect() as conn:
                            conn.execute(
                                "UPDATE auto_update_config SET last_run_at = ?, last_error = ?, updated_at = ? WHERE id = 1",
                                (now2, msg, now2),
                            )
                        stop_event.wait(idle_sleep)
                        continue

                if not update_lock.acquire(timeout=1.0):
                    stop_event.wait(idle_sleep)
                    continue

                try:
                    run_started_at = int(time.time())
                    with backend.connect() as conn:
                        conn.execute(
                            """
                            UPDATE auto_update_config
                            SET last_run_at = ?,
                                last_error = NULL,
                                run_status = 'running',
                                run_started_at = ?,
                                run_target_trade_date = ?,
                                run_mode = NULL,
                                run_message = 'starting',
                                updated_at = ?
                            WHERE id = 1
                            """,
                            (run_started_at, run_started_at, target_end, run_started_at),
                        )

                    try:
                        errors_bad: dict[str, str] = {}
                        errors_incomplete: dict[str, str] = {}

                        def _run_progress(mode: str, message: str) -> None:
                            nowp = int(time.time())
                            msg = (str(message or "").strip() or None)
                            if msg is not None:
                                msg = msg[:2000]
                            with backend.connect() as conn:
                                conn.execute(
                                    """
                                    UPDATE auto_update_config
                                    SET run_mode = ?,
                                        run_message = ?,
                                        updated_at = ?
                                    WHERE id = 1
                                    """,
                                    (mode, msg, nowp),
                                )

                        for mode in ("none", "qfq", "hfq"):
                            _run_progress(mode, f"updating {mode}")
                            mode_settings = settings.model_copy(update={"price_adjust": mode})
                            try:
                                update_daily_service(
                                    settings=mode_settings,
                                    start=None,
                                    end=target_end,
                                    provider=provider,
                                    repair_days=repair_days,
                                    progress_cb=lambda m, _mode=mode: _run_progress(_mode, m),
                                )
                            except UpdateNotConfigured:
                                raise
                            except UpdateBadRequest as e:
                                errors_bad[mode] = str(e)
                            except UpdateIncomplete as e:
                                errors_incomplete[mode] = str(e)

                        if errors_bad:
                            msg = "; ".join([f"{k}: {v}" for k, v in sorted(errors_bad.items())])
                            raise UpdateBadRequest(msg)
                        if errors_incomplete:
                            msg = "; ".join([f"{k}: {v}" for k, v in sorted(errors_incomplete.items())])
                            raise UpdateIncomplete(msg)
                    except UpdateIncomplete as e:
                        msg = (str(e) or "update incomplete")[:2000]
                        with backend.connect() as conn:
                            now2 = int(time.time())
                            conn.execute(
                                """
                                UPDATE auto_update_config
                                SET last_error = ?,
                                    run_status = 'idle',
                                    run_started_at = NULL,
                                    run_target_trade_date = NULL,
                                    run_mode = NULL,
                                    run_message = NULL,
                                    updated_at = ?
                                WHERE id = 1
                                """,
                                (msg, now2),
                            )
                    except (UpdateBadRequest, UpdateNotConfigured) as e:
                        msg = (str(e) or "update failed")[:2000]
                        with backend.connect() as conn:
                            now2 = int(time.time())
                            conn.execute(
                                """
                                UPDATE auto_update_config
                                SET last_error = ?,
                                    run_status = 'idle',
                                    run_started_at = NULL,
                                    run_target_trade_date = NULL,
                                    run_mode = NULL,
                                    run_message = NULL,
                                    updated_at = ?
                                WHERE id = 1
                                """,
                                (msg, now2),
                            )
                    except Exception as e:
                        msg = (str(e) or "update failed")[:2000]
                        with backend.connect() as conn:
                            now2 = int(time.time())
                            conn.execute(
                                """
                                UPDATE auto_update_config
                                SET last_error = ?,
                                    run_status = 'idle',
                                    run_started_at = NULL,
                                    run_target_trade_date = NULL,
                                    run_mode = NULL,
                                    run_message = NULL,
                                    updated_at = ?
                                WHERE id = 1
                                """,
                                (msg, now2),
                            )
                    else:
                        latest: str | None = None
                        should_auto_screen = False
                        with backend.connect() as conn:
                            now2 = int(time.time())
                            latests: list[str] = []
                            for table in ("update_log", "update_log_qfq", "update_log_hfq"):
                                try:
                                    r = conn.execute(f"SELECT MAX(trade_date) AS d FROM {table}").fetchone()
                                except sqlite3.OperationalError:
                                    continue
                                if r and r["d"] is not None:
                                    latests.append(str(r["d"]))
                            latest = max(latests) if latests else None
                            conn.execute(
                                """
                                UPDATE auto_update_config
                                SET last_success_at = ?,
                                    last_success_trade_date = ?,
                                    last_error = NULL,
                                    run_status = 'idle',
                                    run_started_at = NULL,
                                    run_target_trade_date = NULL,
                                    run_mode = NULL,
                                    run_message = NULL,
                                    updated_at = ?
                                WHERE id = 1
                                """,
                                (now2, latest, now2),
                            )

                            if latest:
                                r2 = conn.execute(
                                    """
                                    SELECT screen_enabled, screen_last_trade_date
                                    FROM auto_update_config
                                    WHERE id = 1
                                    LIMIT 1
                                    """
                                ).fetchone()
                                if r2 and int(r2["screen_enabled"] or 0) == 1:
                                    last_screened = str(r2["screen_last_trade_date"]) if r2["screen_last_trade_date"] else None
                                    if last_screened != latest:
                                        should_auto_screen = True

                        if should_auto_screen and latest:
                            try:
                                _run_auto_screen_job(
                                    settings=settings,
                                    target_date=latest,
                                    force=False,
                                    require_enabled=True,
                                    user=None,
                                )
                            except Exception as e:
                                msg = (str(e) or "auto screen failed")[:2000]
                                with backend.connect() as conn:
                                    now3 = int(time.time())
                                    conn.execute("INSERT OR IGNORE INTO auto_update_config (id) VALUES (1)")
                                    conn.execute(
                                        """
                                        UPDATE auto_update_config
                                        SET screen_last_run_at = ?,
                                            screen_last_trade_date = ?,
                                            screen_last_count = ?,
                                            screen_last_error = ?,
                                            updated_at = ?
                                        WHERE id = 1
                                        """,
                                        (now3, latest, 0, msg, now3),
                                    )
                finally:
                    update_lock.release()

                stop_event.wait(1.0)
            except Exception:
                # Best-effort only; avoid crashing the background thread.
                stop_event.wait(idle_sleep)

    def _cleanup_update_wait_jobs(*, max_jobs: int = 200, max_age_seconds: int = 24 * 60 * 60) -> None:
        now = time.time()
        with update_wait_jobs_lock:
            stale = [
                job_id
                for job_id, job in update_wait_jobs.items()
                if job.finished_at is not None and now - job.finished_at > max_age_seconds
            ]
            for job_id in stale:
                update_wait_jobs.pop(job_id, None)
            if len(update_wait_jobs) <= max_jobs:
                return
            finished = [j for j in update_wait_jobs.values() if j.finished_at is not None]
            finished.sort(key=lambda j: j.finished_at or 0.0)
            for j in finished[: max(0, len(update_wait_jobs) - max_jobs)]:
                update_wait_jobs.pop(j.job_id, None)

    def _update_wait_job_response(job: _UpdateWaitJob) -> UpdateWaitResponse:
        now = time.time()
        elapsed = (job.finished_at or now) - job.started_at
        ok = job.status == "succeeded"
        return UpdateWaitResponse(
            job_id=job.job_id,
            status=job.status,
            ok=ok,
            provider=job.provider,
            target_date=job.target_date,
            latest_trade_date=job.latest_trade_date,
            attempts=job.attempts,
            elapsed_seconds=elapsed,
            message=job.message,
            last_error=job.last_error,
            mode=job.mode,
            mode_progress=job.mode_progress,
            mode_completed=job.mode_completed,
            mode_total=job.mode_total,
            progress=job.progress,
        )

    def _run_update_wait_job(*, job_id: str, settings: Settings) -> None:
        def _local_ready(trade_date: str) -> tuple[bool, str | None]:
            backend = _backend(settings)

            def _tables(mode: str) -> tuple[str, str]:
                m = (mode or "").strip().lower()
                if m == "qfq":
                    return "daily_qfq", "update_log_qfq"
                if m == "hfq":
                    return "daily_hfq", "update_log_hfq"
                return "daily", "update_log"

            latests: list[str] = []
            ready = True
            with backend.connect() as conn:
                for mode in ("none", "qfq", "hfq"):
                    daily_table, update_log_table = _tables(mode)
                    row = conn.execute(f"SELECT MAX(trade_date) AS d FROM {daily_table}").fetchone()
                    if row and row["d"] is not None:
                        latests.append(str(row["d"]))
                    has_rows = conn.execute(
                        f"SELECT 1 FROM {daily_table} WHERE trade_date = ? LIMIT 1",
                        (trade_date,),
                    ).fetchone() is not None
                    marked = conn.execute(
                        f"SELECT 1 FROM {update_log_table} WHERE trade_date = ? LIMIT 1",
                        (trade_date,),
                    ).fetchone() is not None
                    if not (has_rows and marked):
                        ready = False
                        break

            latest = max(latests) if latests else None
            return ready, latest

        mode_order: tuple[Literal["none", "qfq", "hfq"], ...] = ("none", "qfq", "hfq")
        mode_index = {m: i for i, m in enumerate(mode_order)}
        progress_re = re.compile(r"\bprogress:\s*(\d+)\s*/\s*(\d+)\b")
        resume_re = re.compile(r"\bstocks:\s*(\d+).*\bresume_done:\s*(\d+)\b")
        batch_re = re.compile(r"\bbatch:\s*(\d+)\s*/\s*(\d+)\b")

        def _set_job_state(
            *,
            mode: Literal["none", "qfq", "hfq"] | None,
            mode_progress: float | None,
            mode_completed: int | None,
            mode_total: int | None,
            progress: float | None,
            message: str | None = None,
        ) -> None:
            with update_wait_jobs_lock:
                job = update_wait_jobs.get(job_id)
                if job is None or job.status != "running":
                    return
                job.mode = mode
                job.mode_progress = mode_progress
                job.mode_completed = mode_completed
                job.mode_total = mode_total
                job.progress = progress
                if message is not None:
                    job.message = (message or "").strip()[:2000] or job.message

        def _progress_cb(mode: Literal["none", "qfq", "hfq"], msg: str) -> None:
            raw = (str(msg or "").strip() or "")[:2000]
            prev_mode_progress: float | None = None
            prev_completed: int | None = None
            prev_total: int | None = None
            with update_wait_jobs_lock:
                job = update_wait_jobs.get(job_id)
                if job is None or job.status != "running":
                    return
                prev_mode_progress = job.mode_progress
                prev_completed = job.mode_completed
                prev_total = job.mode_total

            completed: int | None = None
            total: int | None = None
            m = progress_re.search(raw)
            if m:
                completed = int(m.group(1))
                total = int(m.group(2))
            else:
                m2 = resume_re.search(raw)
                if m2:
                    total = int(m2.group(1))
                    completed = int(m2.group(2))

            if completed is None:
                completed = prev_completed
            if total is None:
                total = prev_total

            mode_prog: float | None = None
            if total is not None and total > 0 and completed is not None:
                mode_prog = min(1.0, max(0.0, float(completed) / float(total)))
            else:
                m3 = batch_re.search(raw)
                if m3:
                    cur = int(m3.group(1))
                    tot = int(m3.group(2))
                    if tot > 0:
                        mode_prog = min(1.0, max(0.0, float(cur - 1) / float(tot)))

            if mode_prog is None:
                mode_prog = prev_mode_progress

            overall: float | None = None
            if mode_prog is not None:
                idx = mode_index.get(mode, 0)
                overall = min(1.0, max(0.0, (float(idx) + mode_prog) / float(len(mode_order))))

            _set_job_state(
                mode=mode,
                mode_progress=mode_prog,
                mode_completed=completed,
                mode_total=total,
                progress=overall,
                message=f"updating {mode}: {raw}" if raw else None,
            )

        while True:
            with update_wait_jobs_lock:
                job = update_wait_jobs.get(job_id)
                if job is None:
                    return
                if job.status != "running":
                    return
                cancel_event = job.cancel_event
                target = job.target_date
                provider = job.provider
                repair_days = job.repair_days
                interval_seconds = job.interval_seconds
                timeout_seconds = job.timeout_seconds
                started_at = job.started_at

            if cancel_event.is_set():
                with update_wait_jobs_lock:
                    job = update_wait_jobs.get(job_id)
                    if job is None or job.status != "running":
                        return
                    job.status = "canceled"
                    job.message = "canceled"
                    job.finished_at = time.time()
                return

            elapsed = time.time() - started_at
            if elapsed >= timeout_seconds:
                _, latest = _local_ready(target)
                with update_wait_jobs_lock:
                    job = update_wait_jobs.get(job_id)
                    if job is None or job.status != "running":
                        return
                    job.status = "timeout"
                    job.latest_trade_date = latest
                    job.message = "timeout waiting for daily data"
                    job.finished_at = time.time()
                return

            ready, latest = _local_ready(target)
            if ready:
                with update_wait_jobs_lock:
                    job = update_wait_jobs.get(job_id)
                    if job is None:
                        return
                    job.status = "succeeded"
                    job.latest_trade_date = latest
                    job.message = "daily data available"
                    job.finished_at = time.time()
                return

            with update_wait_jobs_lock:
                job = update_wait_jobs.get(job_id)
                if job is None or job.status != "running":
                    return
                job.attempts += 1
                job.latest_trade_date = latest
                job.message = f"checking (attempt {job.attempts})"
                job.last_error = None
                job.mode = None
                job.mode_progress = None
                job.mode_completed = None
                job.mode_total = None
                job.progress = None

            if provider == "baostock":
                ok_remote, detail = probe_baostock_daily_available(trade_date=target)
                if not ok_remote:
                    with update_wait_jobs_lock:
                        job = update_wait_jobs.get(job_id)
                        if job is None or job.status != "running":
                            return
                        job.latest_trade_date = latest
                        job.message = f"waiting for provider publish ({detail})"
                        job.mode = None
                        job.mode_progress = None
                        job.mode_completed = None
                        job.mode_total = None
                        job.progress = None
                    remaining = timeout_seconds - (time.time() - started_at)
                    sleep_for = max(0.0, min(float(interval_seconds), float(remaining)))
                    if sleep_for <= 0:
                        continue
                    cancel_event.wait(sleep_for)
                    continue

            with update_wait_jobs_lock:
                job = update_wait_jobs.get(job_id)
                if job is None or job.status != "running":
                    return
                job.message = f"updating (attempt {job.attempts})"

            try:
                with update_lock:
                    errors_bad: dict[str, str] = {}
                    errors_incomplete: dict[str, str] = {}

                    for mode in mode_order:
                        _set_job_state(
                            mode=mode,
                            mode_progress=None,
                            mode_completed=None,
                            mode_total=None,
                            progress=None,
                            message=f"updating {mode}: starting",
                        )
                        mode_settings = settings.model_copy(update={"price_adjust": mode})
                        try:
                            update_daily_service(
                                settings=mode_settings,
                                start=None,
                                end=target,
                                provider=provider,
                                repair_days=repair_days,
                                progress_cb=lambda m, _mode=mode: _progress_cb(_mode, m),
                            )
                        except UpdateNotConfigured:
                            raise
                        except UpdateBadRequest as e:
                            errors_bad[mode] = str(e)
                        except UpdateIncomplete as e:
                            errors_incomplete[mode] = str(e)
                        else:
                            idx = mode_index.get(mode, 0)
                            _set_job_state(
                                mode=mode,
                                mode_progress=1.0,
                                mode_completed=None,
                                mode_total=None,
                                progress=min(1.0, max(0.0, (float(idx) + 1.0) / float(len(mode_order)))),
                                message=f"updating {mode}: done",
                            )

                    if errors_bad:
                        msg = "; ".join([f"{k}: {v}" for k, v in sorted(errors_bad.items())])
                        raise UpdateBadRequest(msg)
                    if errors_incomplete:
                        msg = "; ".join([f"{k}: {v}" for k, v in sorted(errors_incomplete.items())])
                        raise UpdateIncomplete(msg)
            except (UpdateBadRequest, UpdateNotConfigured) as e:
                _, latest = _local_ready(target)
                with update_wait_jobs_lock:
                    job = update_wait_jobs.get(job_id)
                    if job is None:
                        return
                    job.status = "failed"
                    job.latest_trade_date = latest
                    job.last_error = str(e)
                    job.message = str(e)
                    job.finished_at = time.time()
                return
            except UpdateIncomplete as e:
                with update_wait_jobs_lock:
                    job = update_wait_jobs.get(job_id)
                    if job is None or job.status != "running":
                        return
                    job.message = str(e) or "update incomplete; retrying"
            except Exception as e:
                with update_wait_jobs_lock:
                    job = update_wait_jobs.get(job_id)
                    if job is None or job.status != "running":
                        return
                    job.last_error = str(e)
                    job.message = f"error: {e}; retrying"

            ready, latest = _local_ready(target)
            if ready:
                with update_wait_jobs_lock:
                    job = update_wait_jobs.get(job_id)
                    if job is None:
                        return
                    job.status = "succeeded"
                    job.latest_trade_date = latest
                    job.message = "daily data available"
                    job.finished_at = time.time()
                return

            with update_wait_jobs_lock:
                job = update_wait_jobs.get(job_id)
                if job is None or job.status != "running":
                    return
                job.latest_trade_date = latest
                job.message = "waiting for daily data"

            remaining = timeout_seconds - (time.time() - started_at)
            sleep_for = max(0.0, min(float(interval_seconds), float(remaining)))
            if sleep_for <= 0:
                continue
            cancel_event.wait(sleep_for)

    def _auth_enabled_required() -> None:
        if not app.state.app_state.auth_enabled:
            raise HTTPException(status_code=404, detail="Not Found")

    def _api_key_required_if_auth_disabled(x_api_key: str | None = Header(default=None)) -> None:
        if app.state.app_state.auth_enabled:
            return
        _api_key_required(x_api_key)

    def _bearer_token(authorization: str | None) -> str:
        raw = (authorization or "").strip()
        if not raw:
            raise HTTPException(status_code=401, detail="missing/invalid Authorization")
        m = re.match(r"^Bearer\s+(.+)$", raw, flags=re.IGNORECASE)
        if not m:
            raise HTTPException(status_code=401, detail="missing/invalid Authorization")
        token = m.group(1).strip()
        if not token:
            raise HTTPException(status_code=401, detail="missing/invalid Authorization")
        return token

    def _user_from_bearer_token(authorization: str | None, *, settings: Settings) -> AuthUser:
        secret = app.state.app_state.auth_secret
        if secret is None:
            raise HTTPException(status_code=500, detail="auth misconfigured")
        token = _bearer_token(authorization)
        try:
            claims = decode_access_token(token, secret=secret)
        except ValueError:
            raise HTTPException(status_code=401, detail="invalid token")
        user_id = str(claims.get("uid") or "").strip()
        if not user_id:
            raise HTTPException(status_code=401, detail="invalid token")
        try:
            token_version = int(claims.get("tv") or 0)
        except (TypeError, ValueError):
            raise HTTPException(status_code=401, detail="invalid token")

        backend = _backend(settings)
        with backend.connect() as conn:
            row = conn.execute(
                "SELECT id, username, role, disabled, token_version FROM users WHERE id = ? LIMIT 1",
                (user_id,),
            ).fetchone()
        if not row or int(row["disabled"] or 0) != 0:
            raise HTTPException(status_code=401, detail="invalid token")
        db_token_version = int(row["token_version"] or 0)
        if token_version != db_token_version:
            raise HTTPException(status_code=401, detail="invalid token")
        role = str(row["role"] or "user")
        if role not in {"admin", "user"}:
            raise HTTPException(status_code=401, detail="invalid token")
        return AuthUser(id=str(row["id"]), username=str(row["username"]), role=role)  # type: ignore[arg-type]

    def _access_required(
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
        settings: Settings = Depends(_settings_dep),
    ) -> AuthUser | None:
        if app.state.app_state.auth_enabled:
            return _user_from_bearer_token(authorization, settings=settings)
        _api_key_required(x_api_key)
        return None

    def _admin_required(user: AuthUser | None = Depends(_access_required)) -> AuthUser | None:
        if app.state.app_state.auth_enabled and user is not None and user.role != "admin":
            raise HTTPException(status_code=403, detail="admin required")
        return user

    def _watchlist_owner_id(
        user: AuthUser | None = Depends(_access_required),
        x_api_key: str | None = Header(default=None),
    ) -> str:
        if user is not None:
            return user.id
        required = os.environ.get("STOCK_SCREENER_API_KEY")
        if required:
            key = (x_api_key or "").strip() or required
            return sha256(key.encode("utf-8")).hexdigest()
        return "public"

    @app.get("/v1/health", dependencies=[Depends(_api_key_required_if_auth_disabled)])
    def health(settings: Settings = Depends(_settings_dep)) -> dict[str, Any]:
        bootstrap = False
        if app.state.app_state.auth_enabled:
            backend = _backend(settings)
            with backend.connect() as conn:
                bootstrap = conn.execute("SELECT 1 FROM users LIMIT 1").fetchone() is None

        return {
            "status": "ok",
            "auth_enabled": app.state.app_state.auth_enabled,
            "auth_signup_mode": app.state.app_state.auth_signup_mode if app.state.app_state.auth_enabled else None,
            "auth_bootstrap": bootstrap if app.state.app_state.auth_enabled else None,
        }

    @app.get("/v1/version", dependencies=[Depends(_api_key_required_if_auth_disabled)])
    def version() -> dict[str, Any]:
        return {
            "name": "stock-screener",
            "version": app_version,
            "git_sha": (os.environ.get("STOCK_SCREENER_GIT_SHA") or "").strip() or None,
            "git_describe": (os.environ.get("STOCK_SCREENER_GIT_DESCRIBE") or "").strip() or None,
            "build_time": (os.environ.get("STOCK_SCREENER_BUILD_TIME") or "").strip() or None,
        }

    @app.post("/v1/auth/register", response_model=AuthTokenResponse, dependencies=[Depends(_auth_enabled_required)])
    def register(req: AuthRegisterRequest, request: Request, settings: Settings = Depends(_settings_dep)) -> AuthTokenResponse:
        username = normalize_username(req.username)
        if len(username) < 3:
            raise HTTPException(status_code=400, detail="username too short")
        if len(req.password) < 8:
            raise HTTPException(status_code=400, detail="password too short")

        backend = _backend(settings)
        now = int(time.time())
        user_id = uuid.uuid4().hex
        password_hash, password_salt = hash_password(req.password)
        with backend.connect() as conn:
            any_user = conn.execute("SELECT 1 FROM users LIMIT 1").fetchone()
            if any_user and app.state.app_state.auth_signup_mode != "open":
                raise HTTPException(status_code=404, detail="Not Found")
            exists = conn.execute("SELECT 1 FROM users WHERE username = ? LIMIT 1", (username,)).fetchone()
            if exists:
                raise HTTPException(status_code=409, detail="username already exists")
            role = "admin" if not any_user else "user"
            conn.execute(
                """
                INSERT INTO users (id, username, password_hash, password_salt, role, disabled, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (user_id, username, password_hash, password_salt, role, now, now),
            )
            ip = _client_ip(request)
            if ip:
                conn.execute(
                    "UPDATE users SET last_login_at = ?, last_login_ip = ? WHERE id = ?",
                    (now, ip, user_id),
                )
            else:
                conn.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (now, user_id))

        secret = app.state.app_state.auth_secret
        if secret is None:
            raise HTTPException(status_code=500, detail="auth misconfigured")
        expires_at = now + int(app.state.app_state.auth_token_ttl_seconds)
        token = create_access_token({"uid": user_id, "tv": 0}, secret=secret, expires_at=expires_at)
        return AuthTokenResponse(
            token=token,
            expires_at=expires_at,
            user=AuthUserResponse(id=user_id, username=username, role=role),  # type: ignore[arg-type]
        )

    @app.post(
        "/v1/auth/email/request",
        response_model=AuthEmailCodeResponse,
        dependencies=[Depends(_auth_enabled_required)],
    )
    def request_email_code(req: AuthEmailCodeRequest, settings: Settings = Depends(_settings_dep)) -> AuthEmailCodeResponse:
        if app.state.app_state.auth_signup_mode != "email":
            raise HTTPException(status_code=404, detail="Not Found")

        try:
            email = validate_email(req.email)
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid email")

        allowed_domains = auth_allowed_email_domains()
        if allowed_domains:
            domain = email.split("@", 1)[1]
            if domain not in allowed_domains:
                raise HTTPException(status_code=403, detail="email domain not allowed")

        now = int(time.time())
        ttl_seconds = auth_email_code_ttl_seconds()
        cooldown_seconds = auth_email_code_cooldown_seconds()
        expires_at = now + ttl_seconds
        code = f"{secrets.randbelow(1_000_000):06d}"
        code_hash, code_salt = hash_password(code)

        backend = _backend(settings)
        with backend.connect() as conn:
            if conn.execute("SELECT 1 FROM users WHERE email = ? LIMIT 1", (email,)).fetchone():
                raise HTTPException(status_code=409, detail="email already registered")

            row = conn.execute(
                "SELECT last_sent_at FROM email_verification_codes WHERE email = ? LIMIT 1",
                (email,),
            ).fetchone()
            if row and cooldown_seconds > 0:
                last_sent_at = int(row["last_sent_at"] or 0)
                wait = cooldown_seconds - (now - last_sent_at)
                if wait > 0:
                    raise HTTPException(status_code=429, detail=f"too many requests; retry in {wait}s")

            conn.execute(
                """
                INSERT INTO email_verification_codes
                  (email, code_hash, code_salt, expires_at, created_at, updated_at, send_count, last_sent_at, attempts)
                VALUES (?, ?, ?, ?, ?, ?, 1, ?, 0)
                ON CONFLICT(email) DO UPDATE SET
                  code_hash = excluded.code_hash,
                  code_salt = excluded.code_salt,
                  expires_at = excluded.expires_at,
                  updated_at = excluded.updated_at,
                  send_count = send_count + 1,
                  last_sent_at = excluded.last_sent_at,
                  attempts = 0
                """,
                (email, code_hash, code_salt, expires_at, now, now, now),
            )

        debug = auth_email_debug_return_code()
        if debug:
            return AuthEmailCodeResponse(expires_at=expires_at, debug_code=code)

        cfg = smtp_config()
        if cfg is None:
            raise HTTPException(status_code=500, detail="smtp not configured")
        try:
            send_email(
                config=cfg,
                to_addr=email,
                subject="ScreenFish 注册验证码",
                body=f"你的 ScreenFish 注册验证码是：{code}\n\n有效期：{ttl_seconds} 秒\n",
            )
        except Exception:
            raise HTTPException(status_code=500, detail="failed to send email")
        return AuthEmailCodeResponse(expires_at=expires_at, debug_code=None)

    @app.post(
        "/v1/auth/register/email",
        response_model=AuthTokenResponse,
        dependencies=[Depends(_auth_enabled_required)],
    )
    def register_email(
        req: AuthEmailRegisterRequest, request: Request, settings: Settings = Depends(_settings_dep)
    ) -> AuthTokenResponse:
        if app.state.app_state.auth_signup_mode != "email":
            raise HTTPException(status_code=404, detail="Not Found")

        try:
            email = validate_email(req.email)
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid email")

        allowed_domains = auth_allowed_email_domains()
        if allowed_domains:
            domain = email.split("@", 1)[1]
            if domain not in allowed_domains:
                raise HTTPException(status_code=403, detail="email domain not allowed")

        code = re.sub(r"\s+", "", (req.code or "").strip())
        if re.fullmatch(r"\d{6}", code) is None:
            raise HTTPException(status_code=400, detail="invalid code")

        username = normalize_username(req.username)
        if len(username) < 3:
            raise HTTPException(status_code=400, detail="username too short")
        if len(req.password) < 8:
            raise HTTPException(status_code=400, detail="password too short")

        backend = _backend(settings)
        now = int(time.time())
        user_id = uuid.uuid4().hex
        password_hash, password_salt = hash_password(req.password)
        max_attempts = auth_email_code_max_attempts()

        with backend.connect() as conn:
            row = conn.execute(
                """
                SELECT code_hash, code_salt, expires_at, attempts
                FROM email_verification_codes
                WHERE email = ?
                LIMIT 1
                """,
                (email,),
            ).fetchone()
            if not row:
                raise HTTPException(status_code=400, detail="invalid code")

            attempts = int(row["attempts"] or 0)
            if attempts >= max_attempts:
                raise HTTPException(status_code=429, detail="too many attempts")

            expires_at = int(row["expires_at"] or 0)
            if expires_at <= 0 or now >= expires_at:
                conn.execute("DELETE FROM email_verification_codes WHERE email = ?", (email,))
                raise HTTPException(status_code=400, detail="code expired")

            if not verify_password(code, str(row["code_hash"]), str(row["code_salt"])):
                conn.execute(
                    "UPDATE email_verification_codes SET attempts = attempts + 1, updated_at = ? WHERE email = ?",
                    (now, email),
                )
                raise HTTPException(status_code=400, detail="invalid code")

            conn.execute("DELETE FROM email_verification_codes WHERE email = ?", (email,))

            if conn.execute("SELECT 1 FROM users WHERE username = ? LIMIT 1", (username,)).fetchone():
                raise HTTPException(status_code=409, detail="username already exists")
            if conn.execute("SELECT 1 FROM users WHERE email = ? LIMIT 1", (email,)).fetchone():
                raise HTTPException(status_code=409, detail="email already registered")

            any_user = conn.execute("SELECT 1 FROM users LIMIT 1").fetchone()
            role = "admin" if not any_user else "user"
            conn.execute(
                """
                INSERT INTO users (id, username, email, password_hash, password_salt, role, disabled, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (user_id, username, email, password_hash, password_salt, role, now, now),
            )
            ip = _client_ip(request)
            if ip:
                conn.execute(
                    "UPDATE users SET last_login_at = ?, last_login_ip = ? WHERE id = ?",
                    (now, ip, user_id),
                )
            else:
                conn.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (now, user_id))

        secret = app.state.app_state.auth_secret
        if secret is None:
            raise HTTPException(status_code=500, detail="auth misconfigured")
        expires_at = now + int(app.state.app_state.auth_token_ttl_seconds)
        token = create_access_token({"uid": user_id, "tv": 0}, secret=secret, expires_at=expires_at)
        return AuthTokenResponse(
            token=token,
            expires_at=expires_at,
            user=AuthUserResponse(id=user_id, username=username, role=role),  # type: ignore[arg-type]
        )

    @app.post("/v1/auth/login", response_model=AuthTokenResponse, dependencies=[Depends(_auth_enabled_required)])
    def login(req: AuthLoginRequest, request: Request, settings: Settings = Depends(_settings_dep)) -> AuthTokenResponse:
        ident_raw = (req.username or "").strip()
        is_email = "@" in ident_raw
        if is_email:
            try:
                identifier = validate_email(ident_raw)
            except ValueError:
                raise HTTPException(status_code=401, detail="invalid username or password")
            where = "email = ?"
        else:
            identifier = normalize_username(ident_raw)
            where = "username = ?"
        backend = _backend(settings)
        now = int(time.time())
        ip = _client_ip(request)
        with backend.connect() as conn:
            row = conn.execute(
                f"""
                SELECT id, username, role, password_hash, password_salt, disabled, token_version
                FROM users
                WHERE {where}
                LIMIT 1
                """,
                (identifier,),
            ).fetchone()
            if not row or int(row["disabled"] or 0) != 0:
                raise HTTPException(status_code=401, detail="invalid username or password")
            if not verify_password(req.password, str(row["password_hash"]), str(row["password_salt"])):
                raise HTTPException(status_code=401, detail="invalid username or password")

            role = str(row["role"] or "user")
            if role not in {"admin", "user"}:
                raise HTTPException(status_code=401, detail="invalid username or password")
            user_id = str(row["id"])
            username = str(row["username"])
            token_version = int(row["token_version"] or 0)

            if ip:
                conn.execute(
                    "UPDATE users SET last_login_at = ?, last_login_ip = ? WHERE id = ?",
                    (now, ip, user_id),
                )
            else:
                conn.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (now, user_id))

        secret = app.state.app_state.auth_secret
        if secret is None:
            raise HTTPException(status_code=500, detail="auth misconfigured")
        expires_at = now + int(app.state.app_state.auth_token_ttl_seconds)
        token = create_access_token({"uid": user_id, "tv": token_version}, secret=secret, expires_at=expires_at)
        return AuthTokenResponse(
            token=token,
            expires_at=expires_at,
            user=AuthUserResponse(id=user_id, username=username, role=role),  # type: ignore[arg-type]
        )

    @app.get("/v1/auth/me", response_model=AuthUserResponse, dependencies=[Depends(_auth_enabled_required)])
    def me(user: AuthUser = Depends(_access_required)) -> AuthUserResponse:
        return AuthUserResponse(id=user.id, username=user.username, role=user.role)

    def _account_from_row(row: sqlite3.Row) -> AccountResponse:
        role = str(row["role"] or "user")
        if role not in {"admin", "user"}:
            role = "user"
        return AccountResponse(
            id=str(row["id"]),
            username=str(row["username"]),
            email=str(row["email"]) if row["email"] is not None else None,
            role=role,  # type: ignore[arg-type]
        )

    @app.get(
        "/v1/account",
        response_model=AccountResponse,
        dependencies=[Depends(_auth_enabled_required)],
    )
    def account_me(user: AuthUser = Depends(_access_required), settings: Settings = Depends(_settings_dep)) -> AccountResponse:
        backend = _backend(settings)
        with backend.connect() as conn:
            row = conn.execute(
                "SELECT id, username, email, role, disabled FROM users WHERE id = ? LIMIT 1",
                (user.id,),
            ).fetchone()
        if not row or int(row["disabled"] or 0) != 0:
            raise HTTPException(status_code=401, detail="invalid token")
        return _account_from_row(row)

    @app.put(
        "/v1/account",
        response_model=AccountResponse,
        dependencies=[Depends(_auth_enabled_required)],
    )
    def account_update(
        req: AccountUpdateRequest,
        user: AuthUser = Depends(_access_required),
        settings: Settings = Depends(_settings_dep),
    ) -> AccountResponse:
        email_raw = (req.email or "").strip()
        if not email_raw:
            email: str | None = None
        else:
            try:
                email = validate_email(email_raw)
            except ValueError:
                raise HTTPException(status_code=400, detail="invalid email")

        backend = _backend(settings)
        now = int(time.time())
        with backend.connect() as conn:
            row = conn.execute(
                "SELECT id, password_hash, password_salt, disabled FROM users WHERE id = ? LIMIT 1",
                (user.id,),
            ).fetchone()
            if not row or int(row["disabled"] or 0) != 0:
                raise HTTPException(status_code=401, detail="invalid token")
            if not verify_password(req.current_password, str(row["password_hash"]), str(row["password_salt"])):
                raise HTTPException(status_code=401, detail="invalid password")
            if email and conn.execute(
                "SELECT 1 FROM users WHERE email = ? AND id != ? LIMIT 1",
                (email, user.id),
            ).fetchone():
                raise HTTPException(status_code=409, detail="email already registered")

            conn.execute(
                "UPDATE users SET email = ?, updated_at = ? WHERE id = ?",
                (email, now, user.id),
            )
            row2 = conn.execute(
                "SELECT id, username, email, role, disabled FROM users WHERE id = ? LIMIT 1",
                (user.id,),
            ).fetchone()
        if not row2 or int(row2["disabled"] or 0) != 0:
            raise HTTPException(status_code=401, detail="invalid token")
        return _account_from_row(row2)

    @app.post(
        "/v1/account/change-password",
        response_model=AuthTokenResponse,
        dependencies=[Depends(_auth_enabled_required)],
    )
    def account_change_password(
        req: AccountChangePasswordRequest,
        user: AuthUser = Depends(_access_required),
        settings: Settings = Depends(_settings_dep),
    ) -> AuthTokenResponse:
        backend = _backend(settings)
        now = int(time.time())
        password_hash, password_salt = hash_password(req.new_password)
        with backend.connect() as conn:
            row = conn.execute(
                """
                SELECT id, username, role, password_hash, password_salt, disabled
                FROM users
                WHERE id = ?
                LIMIT 1
                """,
                (user.id,),
            ).fetchone()
            if not row or int(row["disabled"] or 0) != 0:
                raise HTTPException(status_code=401, detail="invalid token")
            if not verify_password(req.current_password, str(row["password_hash"]), str(row["password_salt"])):
                raise HTTPException(status_code=401, detail="invalid password")

            role = str(row["role"] or "user")
            if role not in {"admin", "user"}:
                role = "user"
            username = str(row["username"])
            user_id = str(row["id"])

            conn.execute(
                """
                UPDATE users
                SET password_hash = ?, password_salt = ?, token_version = token_version + 1, updated_at = ?
                WHERE id = ?
                """,
                (password_hash, password_salt, now, user_id),
            )
            row2 = conn.execute(
                "SELECT token_version FROM users WHERE id = ? LIMIT 1",
                (user_id,),
            ).fetchone()
            if not row2:
                raise HTTPException(status_code=401, detail="invalid token")
            token_version = int(row2["token_version"] or 0)

        secret = app.state.app_state.auth_secret
        if secret is None:
            raise HTTPException(status_code=500, detail="auth misconfigured")
        expires_at = now + int(app.state.app_state.auth_token_ttl_seconds)
        token = create_access_token({"uid": user_id, "tv": token_version}, secret=secret, expires_at=expires_at)
        return AuthTokenResponse(
            token=token,
            expires_at=expires_at,
            user=AuthUserResponse(id=user_id, username=username, role=role),  # type: ignore[arg-type]
        )

    def _admin_user_item_from_row(row: sqlite3.Row) -> AdminUserItem:
        role = str(row["role"] or "user")
        if role not in {"admin", "user"}:
            role = "user"
        last_login_at = row["last_login_at"]
        return AdminUserItem(
            id=str(row["id"]),
            username=str(row["username"]),
            email=str(row["email"]) if row["email"] is not None else None,
            role=role,  # type: ignore[arg-type]
            disabled=int(row["disabled"] or 0) != 0,
            token_version=int(row["token_version"] or 0),
            created_at=int(row["created_at"] or 0),
            updated_at=int(row["updated_at"] or 0),
            last_login_at=int(last_login_at) if last_login_at is not None else None,
            last_login_ip=str(row["last_login_ip"]) if row["last_login_ip"] is not None else None,
        )

    def _enabled_admin_count(conn: sqlite3.Connection) -> int:
        row = conn.execute("SELECT COUNT(*) AS n FROM users WHERE role = 'admin' AND disabled = 0").fetchone()
        return int(row["n"] or 0) if row else 0

    @app.get(
        "/v1/admin/users",
        response_model=AdminUserListResponse,
        dependencies=[Depends(_auth_enabled_required), Depends(_admin_required)],
    )
    def admin_list_users(
        search: str | None = Query(default=None, max_length=100),
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
        settings: Settings = Depends(_settings_dep),
    ) -> AdminUserListResponse:
        search_raw = (search or "").strip().lower()
        where = "1=1"
        params: list[Any] = []
        if search_raw:
            where = "(LOWER(username) LIKE ? OR LOWER(COALESCE(email, '')) LIKE ?)"
            like = f"%{search_raw}%"
            params.extend([like, like])

        backend = _backend(settings)
        with backend.connect() as conn:
            row = conn.execute(f"SELECT COUNT(*) AS n FROM users WHERE {where}", params).fetchone()
            total = int(row["n"] or 0) if row else 0
            rows = conn.execute(
                f"""
                SELECT id, username, email, role, disabled, token_version, created_at, updated_at, last_login_at, last_login_ip
                FROM users
                WHERE {where}
                ORDER BY created_at DESC
                LIMIT ?
                OFFSET ?
                """,
                [*params, limit, offset],
            ).fetchall()
        return AdminUserListResponse(total=total, users=[_admin_user_item_from_row(r) for r in rows])

    @app.post(
        "/v1/admin/users",
        response_model=AdminUserItem,
        dependencies=[Depends(_auth_enabled_required), Depends(_admin_required)],
    )
    def admin_create_user(req: AdminUserCreateRequest, settings: Settings = Depends(_settings_dep)) -> AdminUserItem:
        username = normalize_username(req.username)
        if len(username) < 3:
            raise HTTPException(status_code=400, detail="username too short")
        if len(req.password) < 8:
            raise HTTPException(status_code=400, detail="password too short")

        email: str | None
        email_raw = (req.email or "").strip()
        if not email_raw:
            email = None
        else:
            try:
                email = validate_email(email_raw)
            except ValueError:
                raise HTTPException(status_code=400, detail="invalid email")

        backend = _backend(settings)
        now = int(time.time())
        user_id = uuid.uuid4().hex
        password_hash, password_salt = hash_password(req.password)
        disabled = 1 if req.disabled else 0
        role = req.role

        with backend.connect() as conn:
            if conn.execute("SELECT 1 FROM users WHERE username = ? LIMIT 1", (username,)).fetchone():
                raise HTTPException(status_code=409, detail="username already exists")
            if email and conn.execute("SELECT 1 FROM users WHERE email = ? LIMIT 1", (email,)).fetchone():
                raise HTTPException(status_code=409, detail="email already registered")
            conn.execute(
                """
                INSERT INTO users (id, username, email, password_hash, password_salt, role, disabled, token_version, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (user_id, username, email, password_hash, password_salt, role, disabled, now, now),
            )
            row = conn.execute(
                """
                SELECT id, username, email, role, disabled, token_version, created_at, updated_at, last_login_at, last_login_ip
                FROM users
                WHERE id = ?
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()
        if not row:
            raise HTTPException(status_code=500, detail="failed to create user")
        return _admin_user_item_from_row(row)

    @app.put(
        "/v1/admin/users/{user_id}",
        response_model=AdminUserItem,
        dependencies=[Depends(_auth_enabled_required), Depends(_admin_required)],
    )
    def admin_update_user(
        user_id: str,
        req: AdminUserUpdateRequest,
        settings: Settings = Depends(_settings_dep),
    ) -> AdminUserItem:
        fields = set(req.model_fields_set)
        backend = _backend(settings)
        now = int(time.time())

        with backend.connect() as conn:
            row = conn.execute(
                """
                SELECT id, username, email, role, disabled, token_version, created_at, updated_at, last_login_at, last_login_ip
                FROM users
                WHERE id = ?
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="user not found")

            current_username = str(row["username"])
            current_email = str(row["email"]) if row["email"] is not None else None
            current_role = str(row["role"] or "user")
            current_disabled = int(row["disabled"] or 0)

            new_username = current_username
            new_email = current_email
            new_role = current_role
            new_disabled = current_disabled

            if "username" in fields:
                if req.username is None:
                    raise HTTPException(status_code=400, detail="invalid username")
                u = normalize_username(req.username)
                if len(u) < 3:
                    raise HTTPException(status_code=400, detail="username too short")
                if u != current_username and conn.execute(
                    "SELECT 1 FROM users WHERE username = ? AND id != ? LIMIT 1",
                    (u, user_id),
                ).fetchone():
                    raise HTTPException(status_code=409, detail="username already exists")
                new_username = u

            if "email" in fields:
                if req.email is None:
                    new_email = None
                else:
                    try:
                        e = validate_email(req.email)
                    except ValueError:
                        raise HTTPException(status_code=400, detail="invalid email")
                    if e != current_email and conn.execute(
                        "SELECT 1 FROM users WHERE email = ? AND id != ? LIMIT 1",
                        (e, user_id),
                    ).fetchone():
                        raise HTTPException(status_code=409, detail="email already registered")
                    new_email = e

            if "role" in fields:
                if req.role is None:
                    raise HTTPException(status_code=400, detail="invalid role")
                new_role = req.role

            if "disabled" in fields:
                if req.disabled is None:
                    raise HTTPException(status_code=400, detail="invalid disabled")
                new_disabled = 1 if req.disabled else 0

            if current_role == "admin" and current_disabled == 0 and (new_role != "admin" or new_disabled != 0):
                if _enabled_admin_count(conn) <= 1:
                    raise HTTPException(status_code=400, detail="cannot remove last admin")

            updates: list[str] = []
            args: list[Any] = []
            if new_username != current_username:
                updates.append("username = ?")
                args.append(new_username)
            if new_email != current_email:
                updates.append("email = ?")
                args.append(new_email)
            if new_role != current_role:
                updates.append("role = ?")
                args.append(new_role)
            if new_disabled != current_disabled:
                updates.append("disabled = ?")
                args.append(new_disabled)
                if current_disabled == 0 and new_disabled != 0:
                    updates.append("token_version = token_version + 1")
            if updates:
                updates.append("updated_at = ?")
                args.append(now)
                args.append(user_id)
                conn.execute(f"UPDATE users SET {', '.join(updates)} WHERE id = ?", args)

            row2 = conn.execute(
                """
                SELECT id, username, email, role, disabled, token_version, created_at, updated_at, last_login_at, last_login_ip
                FROM users
                WHERE id = ?
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()
        if not row2:
            raise HTTPException(status_code=404, detail="user not found")
        return _admin_user_item_from_row(row2)

    @app.post(
        "/v1/admin/users/{user_id}/set-password",
        response_model=AdminUserTokenVersionResponse,
        dependencies=[Depends(_auth_enabled_required), Depends(_admin_required)],
    )
    def admin_set_user_password(
        user_id: str,
        req: AdminUserSetPasswordRequest,
        settings: Settings = Depends(_settings_dep),
    ) -> AdminUserTokenVersionResponse:
        if len(req.password) < 8:
            raise HTTPException(status_code=400, detail="password too short")
        password_hash, password_salt = hash_password(req.password)
        backend = _backend(settings)
        now = int(time.time())
        with backend.connect() as conn:
            if not conn.execute("SELECT 1 FROM users WHERE id = ? LIMIT 1", (user_id,)).fetchone():
                raise HTTPException(status_code=404, detail="user not found")
            conn.execute(
                """
                UPDATE users
                SET password_hash = ?, password_salt = ?, token_version = token_version + 1, updated_at = ?
                WHERE id = ?
                """,
                (password_hash, password_salt, now, user_id),
            )
            row = conn.execute("SELECT token_version FROM users WHERE id = ? LIMIT 1", (user_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="user not found")
        return AdminUserTokenVersionResponse(token_version=int(row["token_version"] or 0))

    @app.post(
        "/v1/admin/users/{user_id}/revoke-tokens",
        response_model=AdminUserTokenVersionResponse,
        dependencies=[Depends(_auth_enabled_required), Depends(_admin_required)],
    )
    def admin_revoke_user_tokens(user_id: str, settings: Settings = Depends(_settings_dep)) -> AdminUserTokenVersionResponse:
        backend = _backend(settings)
        now = int(time.time())
        with backend.connect() as conn:
            if not conn.execute("SELECT 1 FROM users WHERE id = ? LIMIT 1", (user_id,)).fetchone():
                raise HTTPException(status_code=404, detail="user not found")
            conn.execute(
                "UPDATE users SET token_version = token_version + 1, updated_at = ? WHERE id = ?",
                (now, user_id),
            )
            row = conn.execute("SELECT token_version FROM users WHERE id = ? LIMIT 1", (user_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="user not found")
        return AdminUserTokenVersionResponse(token_version=int(row["token_version"] or 0))

    @app.get(
        "/v1/admin/logs/backend",
        response_model=LogTailResponse,
        dependencies=[Depends(_admin_token_required)],
    )
    def get_backend_logs(lines: int = Query(default=200, ge=1, le=2000)) -> LogTailResponse:
        log_file = os.environ.get("STOCK_SCREENER_LOG_FILE", "").strip()
        if log_file:
            path = Path(log_file).expanduser()
            return LogTailResponse(source="file", path=str(path), lines=_tail_text_file(path, max_lines=lines))

        unit = os.environ.get("STOCK_SCREENER_LOG_UNIT", "").strip() or "screenfish-backend.service"
        try:
            out = _tail_journald_user_unit(unit, max_lines=lines)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        if not out:
            return LogTailResponse(source="none", unit=unit, lines=[])
        return LogTailResponse(source="journald", unit=unit, lines=out[-lines:])

    @app.get("/v1/status", response_model=StatusResponse, dependencies=[Depends(_access_required)])
    def status(settings: Settings = Depends(_settings_dep)) -> StatusResponse:
        backend = _backend(settings)
        with backend.connect() as conn:
            row = conn.execute(f"SELECT MAX(trade_date) AS d FROM {backend.daily_table}").fetchone()
            max_daily = row["d"] if row else None
            row = conn.execute(f"SELECT MAX(trade_date) AS d FROM {backend.update_log_table}").fetchone()
            max_log = row["d"] if row else None
            row = conn.execute(f"SELECT COUNT(*) AS n FROM {backend.daily_table}").fetchone()
            total_rows = int(row["n"]) if row else 0
            row = conn.execute(f"SELECT COUNT(DISTINCT ts_code) AS n FROM {backend.daily_table}").fetchone()
            total_stocks = int(row["n"]) if row else 0
        return StatusResponse(
            today=format_yyyymmdd(_date.today()),
            cache_dir=str(settings.cache_dir),
            sqlite_path=str(settings.sqlite_path),
            max_daily_trade_date=max_daily,
            max_update_log_trade_date=max_log,
            stocks=total_stocks,
            rows=total_rows,
        )

    @app.get("/auto-update-config", response_model=AutoUpdateConfig, dependencies=[Depends(_admin_required)])
    def get_auto_update_config(settings: Settings = Depends(_settings_dep)) -> AutoUpdateConfig:
        backend = _backend(settings)
        with backend.connect() as conn:
            conn.execute("INSERT OR IGNORE INTO auto_update_config (id) VALUES (1)")
            row = conn.execute(
                """
                SELECT enabled, interval_seconds, provider, repair_days,
                       run_status, run_started_at, run_target_trade_date, run_mode, run_message,
                       last_run_at, last_success_at, last_success_trade_date, last_error
                FROM auto_update_config
                WHERE id = 1
                LIMIT 1
                """
            ).fetchone()
        if not row:
            raise HTTPException(status_code=500, detail="auto_update_config missing")

        provider = str(row["provider"] or "baostock")
        if provider not in {"baostock", "tushare"}:
            provider = "baostock"
        interval_seconds = int(row["interval_seconds"] or 600)
        if interval_seconds < 1:
            interval_seconds = 600
        repair_days = int(row["repair_days"] or 30)
        if repair_days < 0:
            repair_days = 30
        run_status = str(row["run_status"] or "idle").strip().lower()
        if run_status not in {"idle", "running"}:
            run_status = "idle"
        run_mode = str(row["run_mode"] or "").strip().lower() or None
        if run_mode not in {"none", "qfq", "hfq"}:
            run_mode = None
        return AutoUpdateConfig(
            enabled=bool(int(row["enabled"] or 0)),
            interval_seconds=interval_seconds,
            provider=provider,  # type: ignore[arg-type]
            repair_days=repair_days,
            run_status=run_status,  # type: ignore[arg-type]
            run_started_at=int(row["run_started_at"]) if row["run_started_at"] is not None else None,
            run_target_trade_date=str(row["run_target_trade_date"]) if row["run_target_trade_date"] else None,
            run_mode=run_mode,  # type: ignore[arg-type]
            run_message=str(row["run_message"]) if row["run_message"] else None,
            last_run_at=int(row["last_run_at"]) if row["last_run_at"] is not None else None,
            last_success_at=int(row["last_success_at"]) if row["last_success_at"] is not None else None,
            last_success_trade_date=str(row["last_success_trade_date"]) if row["last_success_trade_date"] else None,
            last_error=str(row["last_error"]) if row["last_error"] else None,
        )

    @app.put("/auto-update-config", response_model=AutoUpdateConfig, dependencies=[Depends(_admin_required)])
    def update_auto_update_config(req: AutoUpdateConfig, settings: Settings = Depends(_settings_dep)) -> AutoUpdateConfig:
        backend = _backend(settings)
        now = int(time.time())
        with backend.connect() as conn:
            conn.execute("INSERT OR IGNORE INTO auto_update_config (id) VALUES (1)")
            prev = conn.execute("SELECT last_run_at FROM auto_update_config WHERE id = 1 LIMIT 1").fetchone()
            last_run_at = prev["last_run_at"] if prev else None
            if req.enabled and last_run_at is None:
                # First enable: schedule next run after interval.
                last_run_at = now
            conn.execute(
                """
                UPDATE auto_update_config
                SET enabled = ?, interval_seconds = ?, provider = ?, repair_days = ?,
                    last_run_at = ?, last_error = NULL, updated_at = ?
                WHERE id = 1
                """,
                (1 if req.enabled else 0, req.interval_seconds, req.provider, req.repair_days, last_run_at, now),
            )
        return get_auto_update_config(settings=settings)

    def _auto_screen_config_from_row(row: Any) -> AutoScreenConfig:
        combo = str(row["screen_combo"] or "and").strip().lower()
        if combo not in {"and", "or"}:
            combo = "and"

        price_adjust = str(row["screen_price_adjust"] or "qfq").strip().lower()
        if price_adjust not in {"none", "qfq", "hfq"}:
            price_adjust = "qfq"

        lookback_days = int(row["screen_lookback_days"] or 200)
        if lookback_days < 0:
            lookback_days = 200
        if lookback_days > 20000:
            lookback_days = 20000

        group_name = _normalize_watchlist_group_name(str(row["screen_group_name"] or "自动筛选")) or "自动筛选"
        rules_raw = row["screen_rules"]
        rules = None
        if rules_raw is not None and str(rules_raw).strip():
            rules = str(rules_raw).strip()

        return AutoScreenConfig(
            enabled=bool(int(row["screen_enabled"] or 0)),
            group_name=group_name,
            group_id=str(row["screen_group_id"]) if row["screen_group_id"] else None,
            combo=combo,  # type: ignore[arg-type]
            rules=rules,
            lookback_days=lookback_days,
            with_name=bool(int(row["screen_with_name"] or 0)),
            exclude_st=bool(int(row["screen_exclude_st"] or 0)),
            price_adjust=price_adjust,  # type: ignore[arg-type]
            replace_group=bool(int(row["screen_replace_group"] or 0)),
            last_run_at=int(row["screen_last_run_at"]) if row["screen_last_run_at"] is not None else None,
            last_trade_date=str(row["screen_last_trade_date"]) if row["screen_last_trade_date"] else None,
            last_count=int(row["screen_last_count"]) if row["screen_last_count"] is not None else None,
            last_error=str(row["screen_last_error"]) if row["screen_last_error"] else None,
        )

    @app.get("/auto-screen-config", response_model=AutoScreenConfig, dependencies=[Depends(_admin_required)])
    def get_auto_screen_config(settings: Settings = Depends(_settings_dep)) -> AutoScreenConfig:
        backend = _backend(settings)
        with backend.connect() as conn:
            conn.execute("INSERT OR IGNORE INTO auto_update_config (id) VALUES (1)")
            row = conn.execute(
                """
                SELECT
                  screen_enabled, screen_combo, screen_rules, screen_lookback_days,
                  screen_with_name, screen_exclude_st, screen_price_adjust,
                  screen_group_name, screen_group_id, screen_replace_group,
                  screen_last_run_at, screen_last_trade_date, screen_last_count, screen_last_error
                FROM auto_update_config
                WHERE id = 1
                LIMIT 1
                """
            ).fetchone()
        if not row:
            raise HTTPException(status_code=500, detail="auto_update_config missing")
        return _auto_screen_config_from_row(row)

    @app.put("/auto-screen-config", response_model=AutoScreenConfig, dependencies=[Depends(_admin_required)])
    def update_auto_screen_config(
        req: AutoScreenConfigUpdate,
        settings: Settings = Depends(_settings_dep),
        user: AuthUser | None = Depends(_access_required),
    ) -> AutoScreenConfig:
        group_name = _normalize_watchlist_group_name(req.group_name)
        if not group_name:
            raise HTTPException(status_code=400, detail="group_name is required")

        backend = _backend(settings)
        try:
            resolve_rules(req.rules, backend=backend)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        now = int(time.time())
        with backend.connect() as conn:
            conn.execute("INSERT OR IGNORE INTO auto_update_config (id) VALUES (1)")

            if user is not None and app.state.app_state.auth_enabled:
                prev = conn.execute(
                    "SELECT screen_owner_id FROM auto_update_config WHERE id = 1 LIMIT 1"
                ).fetchone()
                prev_owner_id = str(prev["screen_owner_id"]) if prev and prev["screen_owner_id"] else None
                if prev_owner_id != user.id:
                    conn.execute("UPDATE auto_update_config SET screen_group_id = NULL WHERE id = 1")
                conn.execute("UPDATE auto_update_config SET screen_owner_id = ? WHERE id = 1", (user.id,))

            conn.execute(
                """
                UPDATE auto_update_config
                SET screen_enabled = ?,
                    screen_combo = ?,
                    screen_rules = ?,
                    screen_lookback_days = ?,
                    screen_with_name = ?,
                    screen_exclude_st = ?,
                    screen_price_adjust = ?,
                    screen_group_name = ?,
                    screen_replace_group = ?,
                    screen_last_error = NULL,
                    updated_at = ?
                WHERE id = 1
                """,
                (
                    1 if req.enabled else 0,
                    req.combo,
                    (req.rules.strip() if req.rules and req.rules.strip() else None),
                    req.lookback_days,
                    1 if req.with_name else 0,
                    1 if req.exclude_st else 0,
                    req.price_adjust,
                    group_name,
                    1 if req.replace_group else 0,
                    now,
                ),
            )
        return get_auto_screen_config(settings=settings)

    def _run_auto_screen_job(
        *,
        settings: Settings,
        target_date: str | None,
        force: bool,
        require_enabled: bool,
        user: AuthUser | None,
    ) -> AutoScreenRunResponse:
        backend = _backend(settings)
        with backend.connect() as conn:
            conn.execute("INSERT OR IGNORE INTO auto_update_config (id) VALUES (1)")
            row = conn.execute(
                """
                SELECT
                  screen_enabled, screen_combo, screen_rules, screen_lookback_days,
                  screen_with_name, screen_exclude_st, screen_price_adjust,
                  screen_owner_id, screen_group_name, screen_group_id, screen_replace_group,
                  screen_last_trade_date, screen_last_count
                FROM auto_update_config
                WHERE id = 1
                LIMIT 1
                """
            ).fetchone()
            if not row:
                raise HTTPException(status_code=500, detail="auto_update_config missing")

            enabled = bool(int(row["screen_enabled"] or 0))
            if require_enabled and not enabled:
                raise HTTPException(status_code=400, detail="auto screen is disabled")

            owner_id = _resolve_auto_screen_owner_id(conn, user=user)
            if user is not None and app.state.app_state.auth_enabled:
                prev_owner_id = str(row["screen_owner_id"]) if row["screen_owner_id"] else None
                if prev_owner_id != owner_id:
                    conn.execute("UPDATE auto_update_config SET screen_group_id = NULL WHERE id = 1")
                    row = dict(row)
                    row["screen_group_id"] = None
                conn.execute("UPDATE auto_update_config SET screen_owner_id = ? WHERE id = 1", (owner_id,))

            combo = str(row["screen_combo"] or "and").strip().lower()
            if combo not in {"and", "or"}:
                combo = "and"
            price_adjust = str(row["screen_price_adjust"] or "qfq").strip().lower()
            if price_adjust not in {"none", "qfq", "hfq"}:
                price_adjust = "qfq"
            lookback_days = int(row["screen_lookback_days"] or 200)
            if lookback_days < 0:
                lookback_days = 200
            rules = str(row["screen_rules"]).strip() if row["screen_rules"] and str(row["screen_rules"]).strip() else None
            with_name = bool(int(row["screen_with_name"] or 0))
            exclude_st = bool(int(row["screen_exclude_st"] or 0))
            group_name = _normalize_watchlist_group_name(str(row["screen_group_name"] or "自动筛选")) or "自动筛选"
            existing_group_id = str(row["screen_group_id"]) if row["screen_group_id"] else None
            replace_group = bool(int(row["screen_replace_group"] or 0))
            last_trade_date = str(row["screen_last_trade_date"]) if row["screen_last_trade_date"] else None
            last_count = int(row["screen_last_count"]) if row["screen_last_count"] is not None else None

        eff_settings = settings.model_copy(update={"price_adjust": price_adjust})
        eff_backend = _backend(eff_settings)
        if target_date is None or target_date == "latest":
            max_daily = eff_backend.max_trade_date_in_daily()
            if not max_daily:
                raise HTTPException(status_code=400, detail="no local data; run update first")
            trade_date = str(max_daily)
        else:
            trade_date = str(target_date)
            try:
                parse_yyyymmdd(trade_date)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e

        if not force and last_trade_date == trade_date:
            if existing_group_id is None:
                with backend.connect() as conn:
                    conn.execute("INSERT OR IGNORE INTO auto_update_config (id) VALUES (1)")
                    gid = _ensure_auto_screen_group(conn, owner_id=owner_id, group_name=group_name, group_id=None)
                    now = int(time.time())
                    conn.execute(
                        "UPDATE auto_update_config SET screen_group_id = ?, updated_at = ? WHERE id = 1",
                        (gid, now),
                    )
                existing_group_id = gid
            return AutoScreenRunResponse(
                ok=True,
                trade_date=trade_date,
                count=int(last_count or 0),
                group_id=existing_group_id,
                group_name=group_name,
                message=f"skip: already screened for {trade_date}",
            )

        try:
            df = run_screen(
                settings=eff_settings,
                date=trade_date,
                combo=combo,  # type: ignore[arg-type]
                lookback_days=lookback_days,
                rules=rules,
                with_name=with_name,
                exclude_st=exclude_st,
            )
        except typer.BadParameter as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        hits: list[str] = []
        if not df.empty and "ts_code" in df.columns:
            seen: set[str] = set()
            for v in df["ts_code"].astype(str).tolist():
                code = v.strip().upper()
                if not code or code in seen:
                    continue
                seen.add(code)
                hits.append(code)

        now = int(time.time())
        with backend.connect() as conn:
            conn.execute("INSERT OR IGNORE INTO auto_update_config (id) VALUES (1)")

            group_id = _ensure_auto_screen_group(conn, owner_id=owner_id, group_name=group_name, group_id=existing_group_id)
            if replace_group:
                conn.execute(
                    "DELETE FROM watchlist_items WHERE owner_id = ? AND group_id = ?",
                    (owner_id, group_id),
                )

            name_by_code: dict[str, str | None] = {}
            if with_name and hits:
                placeholders = ",".join(["?"] * len(hits))
                rows = conn.execute(
                    f"SELECT ts_code, name FROM stock_basic WHERE ts_code IN ({placeholders})",
                    hits,
                ).fetchall()
                for r in rows:
                    ts_code = str(r["ts_code"])
                    name_by_code[ts_code] = str(r["name"]) if r["name"] else None

            if hits:
                conn.executemany(
                    """
                    INSERT INTO watchlist_items (owner_id, group_id, ts_code, name, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(owner_id, group_id, ts_code) DO UPDATE SET
                      name = COALESCE(excluded.name, watchlist_items.name),
                      updated_at = excluded.updated_at
                    """,
                    [
                        (
                            owner_id,
                            group_id,
                            ts_code,
                            name_by_code.get(ts_code),
                            now,
                            now,
                        )
                        for ts_code in hits
                    ],
                )

            conn.execute(
                "UPDATE watchlist_groups SET updated_at = ? WHERE owner_id = ? AND id = ?",
                (now, owner_id, group_id),
            )

            conn.execute(
                """
                UPDATE auto_update_config
                SET screen_group_id = ?,
                    screen_last_run_at = ?,
                    screen_last_trade_date = ?,
                    screen_last_count = ?,
                    screen_last_error = NULL,
                    updated_at = ?
                WHERE id = 1
                """,
                (group_id, now, trade_date, len(hits), now),
            )

        return AutoScreenRunResponse(
            ok=True,
            trade_date=trade_date,
            count=len(hits),
            group_id=group_id,
            group_name=group_name,
            message=f"screened {len(hits)} stocks into '{group_name}'",
        )

    @app.post("/v1/auto-screen/run", response_model=AutoScreenRunResponse)
    def run_auto_screen(
        req: AutoScreenRunRequest,
        settings: Settings = Depends(_settings_dep),
        user: AuthUser | None = Depends(_admin_required),
    ) -> AutoScreenRunResponse:
        try:
            return _run_auto_screen_job(
                settings=settings,
                target_date=req.date,
                force=bool(req.force),
                require_enabled=False,
                user=user,
            )
        except HTTPException as e:
            # Record failure for diagnostics; then bubble up for the UI to show.
            backend = _backend(settings)
            now = int(time.time())
            msg = (str(e.detail) if isinstance(e.detail, str) else str(e.detail or "auto screen failed"))[:2000]
            with backend.connect() as conn:
                conn.execute("INSERT OR IGNORE INTO auto_update_config (id) VALUES (1)")
                conn.execute(
                    """
                    UPDATE auto_update_config
                    SET screen_last_run_at = ?,
                        screen_last_trade_date = ?,
                        screen_last_count = ?,
                        screen_last_error = ?,
                        updated_at = ?
                    WHERE id = 1
                    """,
                    (now, None, 0, msg, now),
                )
            raise

    @app.post("/v1/update", dependencies=[Depends(_admin_required)])
    def update(req: UpdateRequest, settings: Settings = Depends(_settings_dep)) -> dict[str, Any]:
        try:
            with update_lock:
                update_daily_all_service(
                    settings=settings,
                    start=req.start,
                    end=req.end,
                    provider=req.provider,
                    repair_days=req.repair_days,
                )
        except (UpdateBadRequest, UpdateNotConfigured) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except UpdateIncomplete as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        backend = _backend(settings)
        return {
            "ok": True,
            "max_daily_trade_date": backend.max_trade_date_in_daily(),
            "max_update_log_trade_date": backend.max_trade_date_in_update_log(),
        }

    @app.post("/v1/update/wait", response_model=UpdateWaitResponse, dependencies=[Depends(_admin_required)])
    def start_update_wait(req: UpdateWaitRequest, settings: Settings = Depends(_settings_dep)) -> UpdateWaitResponse:
        requested = req.target_date or format_yyyymmdd(_date.today())
        try:
            parse_yyyymmdd(requested)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        try:
            target = resolve_wait_target_trade_date(provider=req.provider, target_date=requested)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        _cleanup_update_wait_jobs()

        with update_wait_jobs_lock:
            for job in update_wait_jobs.values():
                if job.status != "running":
                    continue
                if job.provider == req.provider and job.target_date == target:
                    return _update_wait_job_response(job)

            job_id = uuid.uuid4().hex
            job = _UpdateWaitJob(
                job_id=job_id,
                provider=req.provider,
                target_date=target,
                repair_days=req.repair_days,
                interval_seconds=req.interval_seconds,
                timeout_seconds=req.timeout_seconds,
                started_at=time.time(),
            )
            if target != requested:
                job.message = f"target_date adjusted: {requested} -> {target}"
            update_wait_jobs[job_id] = job

        t = threading.Thread(target=_run_update_wait_job, kwargs={"job_id": job_id, "settings": settings}, daemon=True)
        t.start()
        return _update_wait_job_response(job)

    @app.get("/v1/update/wait/{job_id}", response_model=UpdateWaitResponse, dependencies=[Depends(_admin_required)])
    def get_update_wait(job_id: str) -> UpdateWaitResponse:
        with update_wait_jobs_lock:
            job = update_wait_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="job not found")
            return _update_wait_job_response(job)

    @app.delete("/v1/update/wait/{job_id}", response_model=UpdateWaitResponse, dependencies=[Depends(_admin_required)])
    def cancel_update_wait(job_id: str) -> UpdateWaitResponse:
        with update_wait_jobs_lock:
            job = update_wait_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="job not found")
            if job.status == "running":
                job.message = "cancel requested"
                job.cancel_event.set()
            return _update_wait_job_response(job)

    @app.post("/v1/screen", response_model=ScreenResponse, dependencies=[Depends(_access_required)])
    def screen(req: ScreenRequest, settings: Settings = Depends(_settings_dep)) -> ScreenResponse:
        eff_settings = settings
        if req.price_adjust is not None:
            eff_settings = settings.model_copy(update={"price_adjust": req.price_adjust})
        backend = _backend(eff_settings)
        date_value: str
        if req.date == "latest":
            max_daily = backend.max_trade_date_in_daily()
            if not max_daily:
                raise HTTPException(status_code=400, detail="no local data; run update first")
            date_value = str(max_daily)
        else:
            date_value = str(req.date)
            try:
                parse_yyyymmdd(date_value)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e

        try:
            df = run_screen(
                settings=eff_settings,
                date=date_value,
                combo=req.combo,
                lookback_days=req.lookback_days,
                rules=req.rules,
                with_name=req.with_name,
                exclude_st=req.exclude_st,
            )
        except typer.BadParameter as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return ScreenResponse(trade_date=date_value, hits=df.to_dict(orient="records"))

    @app.get("/v1/data/availability", response_model=AvailabilityResponse, dependencies=[Depends(_access_required)])
    def availability(
        provider: Literal["baostock", "tushare"] = Query(default="baostock"),
        date: str = Query(..., description="YYYYMMDD"),
    ) -> AvailabilityResponse:
        parse_yyyymmdd(date)
        if provider == "tushare":
            # TuShare by-trade_date is straightforward; if token configured, assume available after close.
            token = os.environ.get("TUSHARE_TOKEN")
            if not token:
                return AvailabilityResponse(date=date, provider=provider, available=False, detail="missing TUSHARE_TOKEN")
            return AvailabilityResponse(date=date, provider=provider, available=True, detail="token configured; availability depends on provider publish")

        # BaoStock: probe a couple of sample symbols for that day
        try:
            import baostock as bs  # type: ignore
        except ModuleNotFoundError:
            return AvailabilityResponse(date=date, provider=provider, available=False, detail="baostock not installed")

        lg = bs.login()
        if getattr(lg, "error_code", "0") != "0":
            return AvailabilityResponse(date=date, provider=provider, available=False, detail=f"login failed: {lg.error_msg}")
        try:
            samples = ["sh.600000", "sz.000001"]
            for code in samples:
                rs = bs.query_history_k_data_plus(
                    code,
                    "date,code,close",
                    start_date=f"{date[:4]}-{date[4:6]}-{date[6:8]}",
                    end_date=f"{date[:4]}-{date[4:6]}-{date[6:8]}",
                    frequency="d",
                    adjustflag="3",
                )
                if rs is None or rs.error_code != "0":
                    continue
                rows = []
                while rs.next():
                    rows.append(rs.get_row_data())
                if rows:
                    return AvailabilityResponse(date=date, provider=provider, available=True, detail=f"sample ok: {code}")
            return AvailabilityResponse(date=date, provider=provider, available=False, detail="no sample rows yet")
        finally:
            bs.logout()

    @app.get("/v1/data/trade-dates", response_model=TradeDateListResponse, dependencies=[Depends(_access_required)])
    def list_trade_dates(
        limit: int = Query(default=260, ge=1, le=5000),
        offset: int = Query(default=0, ge=0),
        order: TradeDateOrder = Query(default="desc"),
        price_adjust: Literal["none", "qfq", "hfq"] | None = Query(default=None, description="Price adjust mode"),
        settings: Settings = Depends(_settings_dep),
    ) -> TradeDateListResponse:
        eff_settings = settings
        if price_adjust is not None:
            eff_settings = settings.model_copy(update={"price_adjust": price_adjust})
        backend = _backend(eff_settings)

        direction = "ASC" if order == "asc" else "DESC"
        with backend.connect() as conn:
            row = conn.execute(f"SELECT COUNT(*) AS n FROM {backend.update_log_table}").fetchone()
            total = int(row["n"] or 0) if row else 0
            rows = conn.execute(
                f"""
                SELECT trade_date
                FROM {backend.update_log_table}
                ORDER BY trade_date {direction}
                LIMIT ? OFFSET ?
                """,
                (int(limit), int(offset)),
            ).fetchall()
        dates = [str(r["trade_date"]) for r in rows if r["trade_date"]]
        return TradeDateListResponse(price_adjust=eff_settings.price_adjust, total=total, order=order, dates=dates)

    @app.get("/v1/data/integrity", response_model=DataIntegrityResponse, dependencies=[Depends(_access_required)])
    def data_integrity(
        provider: Literal["baostock", "tushare"] = Query(default="baostock"),
        date: str = Query(default="latest", description="latest or YYYYMMDD"),
        lookback_days: int = Query(default=60, ge=0, le=3650),
        suspicious_ratio: float = Query(default=0.8, ge=0.0, le=1.0),
        price_adjust: Literal["none", "qfq", "hfq"] | None = Query(default=None, description="Price adjust mode"),
        settings: Settings = Depends(_settings_dep),
    ) -> DataIntegrityResponse:
        eff_settings = settings
        if price_adjust is not None:
            eff_settings = settings.model_copy(update={"price_adjust": price_adjust})
        backend = _backend(eff_settings)

        max_daily = backend.max_trade_date_in_daily()
        max_log = backend.max_trade_date_in_update_log()

        requested_date: str
        if date == "latest":
            if not max_daily:
                raise HTTPException(status_code=400, detail="no local data; run update first")
            requested_date = str(max_daily)
        else:
            requested_date = str(date)
            try:
                parse_yyyymmdd(requested_date)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e

        try:
            target_date = resolve_wait_target_trade_date(
                provider=provider,
                target_date=requested_date,
                lookback_days=lookback_days,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        range_start = subtract_calendar_days(target_date, lookback_days)
        try:
            p = get_provider(provider)
            open_dates = [str(x) for x in p.open_trade_dates(start=range_start, end=target_date)]
        except (TuShareNotConfigured, BaoStockNotConfigured, TuShareTokenMissing, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:  # pragma: no cover
            raise HTTPException(status_code=500, detail=str(e)) from e

        updated_dates = backend.get_updated_trade_dates(open_dates)
        missing_update_log_dates = [d for d in open_dates if d not in updated_dates]

        daily_counts: dict[str, int] = {}
        market_stock_basic: dict[str, int] = {}
        market_daily_rows_on_target_date: dict[str, int] = {}
        missing_market_daily_dates: dict[str, list[str]] = {}
        missing_market_daily_count: dict[str, int] = {}
        with backend.connect() as conn:
            if open_dates:
                placeholders = ",".join(["?"] * len(open_dates))
                rows = conn.execute(
                    f"""
                    SELECT trade_date, COUNT(*) AS n
                    FROM {backend.daily_table}
                    WHERE trade_date IN ({placeholders})
                    GROUP BY trade_date
                    """,
                    list(open_dates),
                ).fetchall()
                daily_counts = {str(r["trade_date"]): int(r["n"] or 0) for r in rows}

            markets = {"SH": ".SH", "SZ": ".SZ", "BJ": ".BJ"}
            for market, suffix in markets.items():
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM stock_basic WHERE ts_code LIKE ?",
                    (f"%{suffix}",),
                ).fetchone()
                stock_basic_n = int(row["n"] or 0) if row else 0
                market_stock_basic[market] = stock_basic_n

                row = conn.execute(
                    f"SELECT COUNT(*) AS n FROM {backend.daily_table} WHERE trade_date = ? AND ts_code LIKE ?",
                    (target_date, f"%{suffix}"),
                ).fetchone()
                market_daily_rows_on_target_date[market] = int(row["n"] or 0) if row else 0

                # Per-market missing trade dates: if we have that market in stock_basic, it should have rows.
                expected_market = stock_basic_n > 0
                if market == "BJ" and target_date < "20211115":
                    expected_market = False

                if not expected_market or not open_dates:
                    missing_market_daily_dates[market] = []
                    missing_market_daily_count[market] = 0
                    continue

                placeholders = ",".join(["?"] * len(open_dates))
                rows = conn.execute(
                    f"""
                    SELECT trade_date, COUNT(*) AS n
                    FROM {backend.daily_table}
                    WHERE trade_date IN ({placeholders})
                      AND ts_code LIKE ?
                    GROUP BY trade_date
                    """,
                    list(open_dates) + [f"%{suffix}"],
                ).fetchall()
                market_counts = {str(r["trade_date"]): int(r["n"] or 0) for r in rows}
                missing_dates = [d for d in open_dates if int(market_counts.get(d, 0)) == 0]
                missing_market_daily_dates[market] = missing_dates
                missing_market_daily_count[market] = len(missing_dates)

        missing_daily_dates = [d for d in open_dates if int(daily_counts.get(d, 0)) == 0]
        present_counts = [int(daily_counts[d]) for d in open_dates if int(daily_counts.get(d, 0)) > 0]
        daily_rows_min = min(present_counts) if present_counts else None
        daily_rows_max = max(present_counts) if present_counts else None
        daily_rows_median = int(statistics.median_low(present_counts)) if present_counts else None

        suspicious_dates: list[DataIntegrityCount] = []
        if daily_rows_median is not None and suspicious_ratio > 0 and open_dates:
            min_ok = max(1, int(math.floor(float(daily_rows_median) * float(suspicious_ratio))))
            for d in open_dates:
                n = int(daily_counts.get(d, 0))
                if n <= 0:
                    continue
                if n < min_ok:
                    suspicious_dates.append(DataIntegrityCount(trade_date=d, rows=n))

        suspicious_dates.sort(key=lambda x: x.trade_date)

        ok = True
        if missing_update_log_dates:
            ok = False
        if missing_daily_dates:
            ok = False
        if suspicious_dates:
            ok = False
        if any(v > 0 for v in missing_market_daily_count.values()):
            ok = False

        def _limit(xs: list[str], n: int = 60) -> list[str]:
            return xs[:n] if len(xs) > n else xs

        def _limit_counts(xs: list[DataIntegrityCount], n: int = 60) -> list[DataIntegrityCount]:
            return xs[:n] if len(xs) > n else xs

        return DataIntegrityResponse(
            ok=ok,
            provider=provider,
            price_adjust=eff_settings.price_adjust,
            requested_date=str(date),
            target_date=target_date,
            lookback_days=lookback_days,
            range_start=range_start,
            range_end=target_date,
            open_trade_dates=len(open_dates),
            max_daily_trade_date=max_daily,
            max_update_log_trade_date=max_log,
            missing_update_log_count=len(missing_update_log_dates),
            missing_update_log_dates=_limit(missing_update_log_dates),
            missing_daily_count=len(missing_daily_dates),
            missing_daily_dates=_limit(missing_daily_dates),
            daily_rows_min=daily_rows_min,
            daily_rows_median=daily_rows_median,
            daily_rows_max=daily_rows_max,
            suspicious_daily_count=len(suspicious_dates),
            suspicious_daily_dates=_limit_counts(suspicious_dates),
            market_stock_basic=market_stock_basic,
            market_daily_rows_on_target_date=market_daily_rows_on_target_date,
            missing_market_daily_count=missing_market_daily_count,
            missing_market_daily_dates={k: _limit(v) for k, v in missing_market_daily_dates.items()},
        )

    @app.post("/v1/export/ebk", dependencies=[Depends(_access_required)])
    def export_ebk(req: ScreenRequest, settings: Settings = Depends(_settings_dep)) -> dict[str, Any]:
        eff_settings = settings
        if req.price_adjust is not None:
            eff_settings = settings.model_copy(update={"price_adjust": req.price_adjust})
        backend = _backend(eff_settings)
        if req.date == "latest":
            max_daily = backend.max_trade_date_in_daily()
            if not max_daily:
                raise HTTPException(status_code=400, detail="no local data; run update first")
            date_value = str(max_daily)
        else:
            date_value = str(req.date)
            try:
                parse_yyyymmdd(date_value)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e

        try:
            df = run_screen(
                settings=eff_settings,
                date=date_value,
                combo=req.combo,
                lookback_days=req.lookback_days,
                rules=req.rules,
                with_name=False,
                exclude_st=req.exclude_st,
            )
        except typer.BadParameter as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        seen: set[str] = set()
        ebk_codes: list[str] = []
        for x in df["ts_code"].astype(str).tolist():
            code = ts_code_to_ebk_code(x)
            if code in seen:
                continue
            seen.add(code)
            ebk_codes.append(code)
        fmt = TdxEbkFormat()
        content = ("\r\n" if fmt.leading_crlf else "") + "\r\n".join(ebk_codes) + ("\r\n" if fmt.trailing_crlf else "")
        return {"trade_date": date_value, "ebk": content}

    @app.get("/v1/stocks", response_model=StockListResponse, dependencies=[Depends(_access_required)])
    def list_stocks(
        search: str | None = Query(default=None, description="Search by ts_code or name"),
        limit: int = Query(default=100, ge=1, le=1000),
        offset: int = Query(default=0, ge=0),
        price_adjust: Literal["none", "qfq", "hfq"] | None = Query(default=None, description="Price adjust mode"),
        settings: Settings = Depends(_settings_dep),
    ) -> StockListResponse:
        eff_settings = settings
        if price_adjust is not None:
            eff_settings = settings.model_copy(update={"price_adjust": price_adjust})
        backend = _backend(eff_settings)
        with backend.connect() as conn:
            cur = conn.cursor()
            daily_table = backend.daily_table
            search_raw = (search or "").strip()
            # Prefer stock_basic for listing/search; scanning DISTINCT over daily is expensive.
            cur.execute("SELECT 1 FROM stock_basic LIMIT 1")
            has_stock_basic = cur.fetchone() is not None
            if not search_raw:
                if has_stock_basic:
                    cur.execute(
                        f"""
                        SELECT sb.ts_code, sb.name
                        FROM stock_basic sb
                        WHERE EXISTS (SELECT 1 FROM {daily_table} d WHERE d.ts_code = sb.ts_code)
                        ORDER BY sb.ts_code
                        LIMIT ? OFFSET ?
                        """,
                        (limit, offset),
                    )
                    stocks = [StockItem(ts_code=row["ts_code"], name=row["name"]) for row in cur.fetchall()]
                    cur.execute(
                        f"""
                        SELECT COUNT(*) as cnt
                        FROM stock_basic sb
                        WHERE EXISTS (SELECT 1 FROM {daily_table} d WHERE d.ts_code = sb.ts_code)
                        """
                    )
                    total = cur.fetchone()["cnt"]
                else:
                    cur.execute(
                        f"""
                        SELECT DISTINCT d.ts_code, sb.name
                        FROM {daily_table} d
                        LEFT JOIN stock_basic sb ON d.ts_code = sb.ts_code
                        ORDER BY d.ts_code
                        LIMIT ? OFFSET ?
                        """,
                        (limit, offset),
                    )
                    stocks = [StockItem(ts_code=row["ts_code"], name=row["name"]) for row in cur.fetchall()]
                    cur.execute(f"SELECT COUNT(DISTINCT ts_code) as cnt FROM {daily_table}")
                    total = cur.fetchone()["cnt"]
            else:
                # Pinyin initials search (e.g. "zsyh") for Chinese stock names.
                search_compact = re.sub(r"\s+", "", search_raw)
                pinyin_mode = re.fullmatch(r"[A-Za-z]+", search_compact) is not None
                if pinyin_mode:
                    search_key = search_compact.lower()
                    if has_stock_basic:
                        ts_code_like = f"%{search_key}%"
                        pinyin_like = f"{search_key}%"
                        try:
                            cur.execute(
                                f"""
                                SELECT sb.ts_code, sb.name
                                FROM stock_basic sb
                                WHERE EXISTS (SELECT 1 FROM {daily_table} d WHERE d.ts_code = sb.ts_code)
                                  AND (
                                    LOWER(sb.ts_code) LIKE ?
                                    OR sb.pinyin_initials LIKE ?
                                    OR sb.pinyin_full LIKE ?
                                  )
                                ORDER BY sb.ts_code
                                LIMIT ? OFFSET ?
                                """,
                                (ts_code_like, pinyin_like, pinyin_like, limit, offset),
                            )
                            stocks = [StockItem(ts_code=row["ts_code"], name=row["name"]) for row in cur.fetchall()]
                            cur.execute(
                                f"""
                                SELECT COUNT(*) as cnt
                                FROM stock_basic sb
                                WHERE EXISTS (SELECT 1 FROM {daily_table} d WHERE d.ts_code = sb.ts_code)
                                  AND (
                                    LOWER(sb.ts_code) LIKE ?
                                    OR sb.pinyin_initials LIKE ?
                                    OR sb.pinyin_full LIKE ?
                                  )
                                """,
                                (ts_code_like, pinyin_like, pinyin_like),
                            )
                            total = cur.fetchone()["cnt"]
                            return StockListResponse(total=total, stocks=stocks)
                        except sqlite3.OperationalError:
                            # Fallback for legacy DB schema without pinyin cache columns.
                            cur.execute(
                                f"""
                                SELECT sb.ts_code, sb.name
                                FROM stock_basic sb
                                WHERE EXISTS (SELECT 1 FROM {daily_table} d WHERE d.ts_code = sb.ts_code)
                                ORDER BY sb.ts_code
                                """
                            )
                    else:
                        cur.execute(
                            f"""
                            SELECT DISTINCT d.ts_code, sb.name
                            FROM {daily_table} d
                            LEFT JOIN stock_basic sb ON d.ts_code = sb.ts_code
                            ORDER BY d.ts_code
                            """
                        )
                    rows = cur.fetchall()
                    matched: list[StockItem] = []
                    for row in rows:
                        ts_code = str(row["ts_code"])
                        name = row["name"]
                        if search_key in ts_code.lower():
                            matched.append(StockItem(ts_code=ts_code, name=name))
                            continue
                        if name is None:
                            continue
                        name_str = str(name)
                        initials = pinyin_initials(name_str)
                        full = pinyin_full(name_str)
                        if (initials and initials.startswith(search_key)) or (full and full.startswith(search_key)):
                            matched.append(StockItem(ts_code=ts_code, name=name_str))
                    total = len(matched)
                    stocks = matched[offset : offset + limit]
                else:
                    search_pattern = f"%{search_raw}%"
                    if has_stock_basic:
                        cur.execute(
                            f"""
                            SELECT sb.ts_code, sb.name
                            FROM stock_basic sb
                            WHERE EXISTS (SELECT 1 FROM {daily_table} d WHERE d.ts_code = sb.ts_code)
                              AND (sb.ts_code LIKE ? OR sb.name LIKE ?)
                            ORDER BY sb.ts_code
                            LIMIT ? OFFSET ?
                            """,
                            (search_pattern, search_pattern, limit, offset),
                        )
                        stocks = [StockItem(ts_code=row["ts_code"], name=row["name"]) for row in cur.fetchall()]
                        cur.execute(
                            f"""
                            SELECT COUNT(*) as cnt
                            FROM stock_basic sb
                            WHERE EXISTS (SELECT 1 FROM {daily_table} d WHERE d.ts_code = sb.ts_code)
                              AND (sb.ts_code LIKE ? OR sb.name LIKE ?)
                            """,
                            (search_pattern, search_pattern),
                        )
                        total = cur.fetchone()["cnt"]
                    else:
                        cur.execute(
                            f"""
                            SELECT DISTINCT d.ts_code, sb.name
                            FROM {daily_table} d
                            LEFT JOIN stock_basic sb ON d.ts_code = sb.ts_code
                            WHERE d.ts_code LIKE ? OR sb.name LIKE ?
                            ORDER BY d.ts_code
                            LIMIT ? OFFSET ?
                            """,
                            (search_pattern, search_pattern, limit, offset),
                        )
                        stocks = [StockItem(ts_code=row["ts_code"], name=row["name"]) for row in cur.fetchall()]
                        cur.execute(
                            f"""
                            SELECT COUNT(DISTINCT d.ts_code) as cnt
                            FROM {daily_table} d
                            LEFT JOIN stock_basic sb ON d.ts_code = sb.ts_code
                            WHERE d.ts_code LIKE ? OR sb.name LIKE ?
                            """,
                            (search_pattern, search_pattern),
                        )
                        total = cur.fetchone()["cnt"]
        return StockListResponse(total=total, stocks=stocks)

    @app.get("/v1/stocks/{ts_code}/daily", response_model=StockDailyResponse, dependencies=[Depends(_access_required)])
    def get_stock_daily(
        ts_code: str,
        start: str | None = Query(default=None, description="Start date YYYYMMDD"),
        end: str | None = Query(default=None, description="End date YYYYMMDD"),
        limit: int = Query(default=250, ge=1, le=20000),
        price_adjust: Literal["none", "qfq", "hfq"] | None = Query(default=None, description="Price adjust mode"),
        settings: Settings = Depends(_settings_dep),
    ) -> StockDailyResponse:
        start_parsed = None
        end_parsed = None
        if start is not None:
            try:
                start_parsed = parse_yyyymmdd(start)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
        if end is not None:
            try:
                end_parsed = parse_yyyymmdd(end)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
        if start_parsed is not None and end_parsed is not None and start_parsed > end_parsed:
            raise HTTPException(status_code=400, detail="start must be <= end")

        eff_settings = settings
        if price_adjust is not None:
            eff_settings = settings.model_copy(update={"price_adjust": price_adjust})
        backend = _backend(eff_settings)
        with backend.connect() as conn:
            cur = conn.cursor()
            # Get stock name
            cur.execute("SELECT name FROM stock_basic WHERE ts_code = ?", (ts_code,))
            name_row = cur.fetchone()
            name = name_row["name"] if name_row else None

            # Build query for daily bars
            query = f"""
            SELECT
                trade_date,
                open,
                high,
                low,
                close,
                vol,
                amount
            FROM {backend.daily_table}
            WHERE ts_code = ?
              AND open IS NOT NULL
              AND high IS NOT NULL
              AND low IS NOT NULL
              AND close IS NOT NULL
              AND vol IS NOT NULL
              AND amount IS NOT NULL
              AND NOT (vol = 0 AND amount = 0)
            """
            params: list[Any] = [ts_code]
            if start:
                query += " AND trade_date >= ?"
                params.append(start)
            if end:
                query += " AND trade_date <= ?"
                params.append(end)
            query += " ORDER BY trade_date DESC LIMIT ?"
            params.append(limit)

            cur.execute(query, params)
            rows = cur.fetchall()
            bars = [
                DailyBar(
                    trade_date=row["trade_date"],
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    vol=row["vol"],
                    amount=row["amount"],
                )
                for row in reversed(rows)  # Return in ascending order
            ]

        if not bars:
            raise HTTPException(status_code=404, detail=f"No data found for {ts_code}")

        return StockDailyResponse(ts_code=ts_code, name=name, bars=bars)

    @app.post("/v1/sync-stock-names", dependencies=[Depends(_admin_required)])
    def sync_stock_names(settings: Settings = Depends(_settings_dep)) -> dict[str, Any]:
        """Sync stock names from baostock."""
        from stock_screener.providers.baostock_provider import BaoStockProvider

        backend = _backend(settings)
        max_date = backend.max_trade_date_in_daily()
        if not max_date:
            raise HTTPException(status_code=400, detail="no local data; run update first")

        provider = BaoStockProvider()
        with provider.session() as bs:
            basics_df = provider._all_stock_basics(bs=bs, day=max_date)
            if basics_df.empty:
                return {"ok": False, "synced": 0, "message": "no stock basics returned"}
            backend.upsert_stock_basic_df(basics_df)
            return {"ok": True, "synced": len(basics_df), "message": f"synced {len(basics_df)} stock names"}

    # Watchlist endpoints (groups + items)

    @app.get(
        "/v1/watchlist",
        response_model=WatchlistStateResponse,
        dependencies=[Depends(_access_required)],
    )
    def get_watchlist(
        settings: Settings = Depends(_settings_dep),
        owner_id: str = Depends(_watchlist_owner_id),
    ) -> WatchlistStateResponse:
        backend = _backend(settings)
        with backend.connect() as conn:
            _ensure_default_watchlist_group(conn, owner_id)
            group_rows = conn.execute(
                """
                SELECT id, name, created_at, updated_at
                FROM watchlist_groups
                WHERE owner_id = ?
                ORDER BY updated_at DESC, created_at DESC
                """,
                (owner_id,),
            ).fetchall()

            items_by_group: dict[str, list[WatchlistItem]] = {}
            item_rows = conn.execute(
                """
                SELECT
                  i.group_id AS group_id,
                  i.ts_code AS ts_code,
                  COALESCE(i.name, sb.name) AS name
                FROM watchlist_items i
                LEFT JOIN stock_basic sb ON sb.ts_code = i.ts_code
                WHERE i.owner_id = ?
                ORDER BY i.group_id ASC, i.updated_at DESC, i.ts_code ASC
                """,
                (owner_id,),
            ).fetchall()
            for row in item_rows:
                gid = str(row["group_id"])
                items_by_group.setdefault(gid, []).append(
                    WatchlistItem(ts_code=str(row["ts_code"]), name=row["name"])
                )

            groups = [
                WatchlistGroup(
                    id=str(row["id"]),
                    name=str(row["name"]),
                    created_at=int(row["created_at"]),
                    updated_at=int(row["updated_at"]),
                    items=items_by_group.get(str(row["id"]), []),
                )
                for row in group_rows
            ]
            return WatchlistStateResponse(version=1, groups=groups)

    @app.post(
        "/v1/watchlist/groups",
        response_model=WatchlistGroupMeta,
        dependencies=[Depends(_access_required)],
    )
    def create_watchlist_group(
        req: WatchlistGroupCreate,
        settings: Settings = Depends(_settings_dep),
        owner_id: str = Depends(_watchlist_owner_id),
    ) -> WatchlistGroupMeta:
        name = _normalize_watchlist_group_name(req.name)
        if not name:
            raise HTTPException(status_code=400, detail="group name is required")
        group_id = uuid.uuid4().hex
        now = int(time.time())

        backend = _backend(settings)
        with backend.connect() as conn:
            _ensure_default_watchlist_group(conn, owner_id)
            conn.execute(
                """
                INSERT INTO watchlist_groups (owner_id, id, name, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (owner_id, group_id, name, now, now),
            )
        return WatchlistGroupMeta(id=group_id, name=name, created_at=now, updated_at=now)

    @app.put(
        "/v1/watchlist/groups/{group_id}",
        response_model=WatchlistGroupMeta,
        dependencies=[Depends(_access_required)],
    )
    def update_watchlist_group(
        group_id: str,
        req: WatchlistGroupUpdate,
        settings: Settings = Depends(_settings_dep),
        owner_id: str = Depends(_watchlist_owner_id),
    ) -> WatchlistGroupMeta:
        name = _normalize_watchlist_group_name(req.name)
        if not name:
            raise HTTPException(status_code=400, detail="group name is required")
        now = int(time.time())

        backend = _backend(settings)
        with backend.connect() as conn:
            row = conn.execute(
                "SELECT id, created_at FROM watchlist_groups WHERE owner_id = ? AND id = ?",
                (owner_id, group_id),
            ).fetchone()
            if not row:
                raise HTTPException(status_code=404, detail=f"watchlist group {group_id} not found")
            created_at = int(row["created_at"])
            conn.execute(
                "UPDATE watchlist_groups SET name = ?, updated_at = ? WHERE owner_id = ? AND id = ?",
                (name, now, owner_id, group_id),
            )
        return WatchlistGroupMeta(id=group_id, name=name, created_at=created_at, updated_at=now)

    @app.delete(
        "/v1/watchlist/groups/{group_id}",
        dependencies=[Depends(_access_required)],
    )
    def delete_watchlist_group(
        group_id: str,
        settings: Settings = Depends(_settings_dep),
        owner_id: str = Depends(_watchlist_owner_id),
    ) -> dict[str, Any]:
        backend = _backend(settings)
        with backend.connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM watchlist_groups WHERE owner_id = ? AND id = ?",
                (owner_id, group_id),
            ).fetchone()
            if not row:
                raise HTTPException(status_code=404, detail=f"watchlist group {group_id} not found")

            conn.execute(
                "DELETE FROM watchlist_items WHERE owner_id = ? AND group_id = ?",
                (owner_id, group_id),
            )
            conn.execute(
                "DELETE FROM watchlist_groups WHERE owner_id = ? AND id = ?",
                (owner_id, group_id),
            )
            _ensure_default_watchlist_group(conn, owner_id)
        return {"ok": True, "deleted": group_id}

    @app.post(
        "/v1/watchlist/groups/{group_id}/items",
        dependencies=[Depends(_access_required)],
    )
    def upsert_watchlist_items(
        group_id: str,
        req: WatchlistItemsUpsertRequest,
        settings: Settings = Depends(_settings_dep),
        owner_id: str = Depends(_watchlist_owner_id),
    ) -> dict[str, Any]:
        items = req.items or []
        raw_ts_codes = [str(i.ts_code).strip() for i in items if str(i.ts_code).strip()]
        if not raw_ts_codes:
            raise HTTPException(status_code=400, detail="items is required")

        ts_codes: list[str] = []
        seen: set[str] = set()
        for c in raw_ts_codes:
            norm = c.upper()
            if norm in seen:
                continue
            seen.add(norm)
            ts_codes.append(norm)

        invalid_format = [c for c in ts_codes if re.fullmatch(r"\d{6}\.(SZ|SH|BJ)", c) is None]
        if invalid_format:
            raise HTTPException(status_code=400, detail=f"invalid ts_code: {invalid_format[0]}")
        now = int(time.time())
        missing: list[str] = []

        backend = _backend(settings)
        with backend.connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM watchlist_groups WHERE owner_id = ? AND id = ?",
                (owner_id, group_id),
            ).fetchone()
            if not row:
                raise HTTPException(status_code=404, detail=f"watchlist group {group_id} not found")

            placeholders = ",".join(["?"] * len(ts_codes))
            union_query = (
                f"SELECT ts_code FROM daily WHERE ts_code IN ({placeholders}) "
                f"UNION SELECT ts_code FROM daily_qfq WHERE ts_code IN ({placeholders}) "
                f"UNION SELECT ts_code FROM daily_hfq WHERE ts_code IN ({placeholders})"
            )
            valid_rows = conn.execute(union_query, ts_codes + ts_codes + ts_codes).fetchall()
            valid_ts_codes = {str(r["ts_code"]) for r in valid_rows}
            missing = [c for c in ts_codes if c not in valid_ts_codes]
            # Best-effort: BaoStock doesn't cover BJ; try to backfill missing BJ symbols via Eastmoney.
            bj_missing = [c for c in missing if c.endswith(".BJ")]
            if bj_missing:
                try:
                    from stock_screener.providers.eastmoney_provider import EastmoneyProvider

                    provider = EastmoneyProvider()
                    backends: dict[str, SqliteBackend] = {
                        "none": SqliteBackend(settings.sqlite_path, daily_table="daily"),
                        "qfq": SqliteBackend(settings.sqlite_path, daily_table="daily_qfq"),
                        "hfq": SqliteBackend(settings.sqlite_path, daily_table="daily_hfq"),
                    }
                    basics: list[dict[str, str]] = []
                    for code in bj_missing[:50]:
                        # Full history is OK here; BJ coverage is typically small.
                        for mode, b in backends.items():
                            try:
                                df, name = provider.fetch_daily(ts_code=code, adjust=mode)  # type: ignore[arg-type]
                            except Exception:
                                continue
                            if not df.empty:
                                b.upsert_daily_df_in_conn(conn, df)
                            if name:
                                basics.append({"ts_code": code, "name": name})
                    if basics:
                        backend.upsert_stock_basic_df_in_conn(conn, pd.DataFrame(basics).drop_duplicates(subset=["ts_code"]))
                except Exception:
                    # Ignore; we'll fall back to treating them as unknown.
                    pass

                valid_rows = conn.execute(union_query, ts_codes + ts_codes + ts_codes).fetchall()
                valid_ts_codes = {str(r["ts_code"]) for r in valid_rows}
                missing = [c for c in ts_codes if c not in valid_ts_codes]
            if missing and not bool(getattr(req, "ignore_unknown", False)):
                sample = ", ".join(missing[:10])
                more = "" if len(missing) <= 10 else f" (+{len(missing) - 10} more)"
                raise HTTPException(status_code=400, detail=f"unknown ts_code(s): {sample}{more}")
            if missing:
                ts_codes = [c for c in ts_codes if c in valid_ts_codes]
                if not ts_codes:
                    sample = ", ".join(missing[:10])
                    more = "" if len(missing) <= 10 else f" (+{len(missing) - 10} more)"
                    raise HTTPException(status_code=400, detail=f"no valid ts_code(s); unknown: {sample}{more}")

            name_by_code: dict[str, str | None] = {}
            for item in items:
                code = str(item.ts_code).strip()
                if not code:
                    continue
                norm = code.upper()
                if norm not in valid_ts_codes:
                    continue
                if norm in name_by_code:
                    continue
                name_by_code[norm] = item.name

            conn.executemany(
                """
                INSERT INTO watchlist_items (owner_id, group_id, ts_code, name, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(owner_id, group_id, ts_code) DO UPDATE SET
                  name = COALESCE(excluded.name, watchlist_items.name),
                  updated_at = excluded.updated_at
                """,
                [
                    (
                        owner_id,
                        group_id,
                        ts_code,
                        name_by_code.get(ts_code),
                        now,
                        now,
                    )
                    for ts_code in ts_codes
                ],
            )
            conn.execute(
                "UPDATE watchlist_groups SET updated_at = ? WHERE owner_id = ? AND id = ?",
                (now, owner_id, group_id),
            )
            total = conn.execute(
                "SELECT COUNT(1) AS c FROM watchlist_items WHERE owner_id = ? AND group_id = ?",
                (owner_id, group_id),
            ).fetchone()["c"]
        resp: dict[str, Any] = {"ok": True, "group_id": group_id, "updated_at": now, "total": int(total)}
        if missing:
            resp["unknown_total"] = len(missing)
            resp["unknown"] = missing[:50]
        return resp

    @app.post(
        "/v1/watchlist/groups/{group_id}/items/remove",
        dependencies=[Depends(_access_required)],
    )
    def remove_watchlist_items(
        group_id: str,
        req: WatchlistItemsRemoveRequest,
        settings: Settings = Depends(_settings_dep),
        owner_id: str = Depends(_watchlist_owner_id),
    ) -> dict[str, Any]:
        raw_ts_codes = [str(c).strip() for c in (req.ts_codes or []) if c and str(c).strip()]
        ts_codes: list[str] = []
        seen: set[str] = set()
        for c in raw_ts_codes:
            if c in seen:
                continue
            seen.add(c)
            ts_codes.append(c)
        if not ts_codes:
            raise HTTPException(status_code=400, detail="ts_codes is required")
        now = int(time.time())

        backend = _backend(settings)
        with backend.connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM watchlist_groups WHERE owner_id = ? AND id = ?",
                (owner_id, group_id),
            ).fetchone()
            if not row:
                raise HTTPException(status_code=404, detail=f"watchlist group {group_id} not found")

            placeholders = ",".join(["?"] * len(ts_codes))
            params: list[Any] = [owner_id, group_id, *ts_codes]
            cur = conn.execute(
                f"DELETE FROM watchlist_items WHERE owner_id = ? AND group_id = ? AND ts_code IN ({placeholders})",
                params,
            )
            conn.execute(
                "UPDATE watchlist_groups SET updated_at = ? WHERE owner_id = ? AND id = ?",
                (now, owner_id, group_id),
            )
            removed = cur.rowcount
        return {"ok": True, "group_id": group_id, "updated_at": now, "removed": int(removed)}

    # Formula CRUD endpoints

    @app.get("/v1/formulas", response_model=FormulaListResponse, dependencies=[Depends(_access_required)])
    def list_formulas(
        enabled_only: bool = Query(default=False, description="Only return enabled formulas"),
        kind: Literal["screen", "indicator"] | None = Query(default=None, description="Filter by formula kind"),
        settings: Settings = Depends(_settings_dep),
    ) -> FormulaListResponse:
        """List all formulas."""
        backend = _backend(settings)
        formulas = backend.list_formulas(enabled_only=enabled_only, kind=kind)
        return FormulaListResponse(
            total=len(formulas),
            formulas=[FormulaItem(**f) for f in formulas],
        )

    @app.post("/v1/formulas", response_model=FormulaItem, dependencies=[Depends(_admin_required)])
    def create_formula(
        req: FormulaCreate,
        settings: Settings = Depends(_settings_dep),
    ) -> FormulaItem:
        """Create a new formula."""
        from stock_screener.formula_parser import validate_formula

        # Validate formula syntax first
        valid, msg = validate_formula(req.formula)
        if not valid:
            raise HTTPException(status_code=400, detail=f"Invalid formula syntax: {msg}")

        backend = _backend(settings)
        # Check if name already exists
        existing = backend.get_formula_by_name(req.name)
        if existing:
            raise HTTPException(status_code=409, detail=f"Formula with name '{req.name}' already exists")

        try:
            formula = backend.create_formula(
                name=req.name,
                formula=req.formula,
                description=req.description,
                kind=req.kind,
                timeframe=req.timeframe,
                enabled=req.enabled,
            )
            return FormulaItem(**formula)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/v1/formulas/{formula_id}", response_model=FormulaItem, dependencies=[Depends(_access_required)])
    def get_formula(
        formula_id: int,
        settings: Settings = Depends(_settings_dep),
    ) -> FormulaItem:
        """Get a formula by ID."""
        backend = _backend(settings)
        formula = backend.get_formula(formula_id)
        if not formula:
            raise HTTPException(status_code=404, detail=f"Formula {formula_id} not found")
        return FormulaItem(**formula)

    @app.put("/v1/formulas/{formula_id}", response_model=FormulaItem, dependencies=[Depends(_admin_required)])
    def update_formula(
        formula_id: int,
        req: FormulaUpdate,
        settings: Settings = Depends(_settings_dep),
    ) -> FormulaItem:
        """Update a formula."""
        from stock_screener.formula_parser import validate_formula

        backend = _backend(settings)

        # Check formula exists
        existing = backend.get_formula(formula_id)
        if not existing:
            raise HTTPException(status_code=404, detail=f"Formula {formula_id} not found")

        # Validate formula syntax if provided
        if req.formula is not None:
            valid, msg = validate_formula(req.formula)
            if not valid:
                raise HTTPException(status_code=400, detail=f"Invalid formula syntax: {msg}")

        # Check name uniqueness if changing name
        if req.name is not None and req.name != existing["name"]:
            name_conflict = backend.get_formula_by_name(req.name)
            if name_conflict:
                raise HTTPException(status_code=409, detail=f"Formula with name '{req.name}' already exists")

        try:
            formula = backend.update_formula(
                formula_id,
                name=req.name,
                formula=req.formula,
                description=req.description,
                kind=req.kind,
                timeframe=req.timeframe,
                enabled=req.enabled,
            )
            return FormulaItem(**formula)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.delete("/v1/formulas/{formula_id}", dependencies=[Depends(_admin_required)])
    def delete_formula(
        formula_id: int,
        settings: Settings = Depends(_settings_dep),
    ) -> dict[str, Any]:
        """Delete a formula."""
        backend = _backend(settings)
        if not backend.delete_formula(formula_id):
            raise HTTPException(status_code=404, detail=f"Formula {formula_id} not found")
        return {"ok": True, "deleted": formula_id}

    @app.post("/v1/formulas/validate", response_model=FormulaValidateResponse, dependencies=[Depends(_access_required)])
    def validate_formula_endpoint(
        req: FormulaValidateRequest,
    ) -> FormulaValidateResponse:
        """Validate formula syntax without saving."""
        from stock_screener.formula_parser import validate_formula

        valid, msg = validate_formula(req.formula)
        return FormulaValidateResponse(valid=valid, message=msg)

    @app.get(
        "/v1/stocks/{ts_code}/indicators/{formula_id}",
        response_model=IndicatorSeriesResponse,
        dependencies=[Depends(_access_required)],
    )
    def get_indicator_series(
        ts_code: str,
        formula_id: int,
        start: str | None = Query(default=None, description="Start date YYYYMMDD"),
        end: str | None = Query(default=None, description="End date YYYYMMDD"),
        limit: int = Query(default=250, ge=1, le=20000),
        price_adjust: Literal["none", "qfq", "hfq"] | None = Query(default=None, description="Price adjust mode"),
        settings: Settings = Depends(_settings_dep),
    ) -> IndicatorSeriesResponse:
        """Evaluate an indicator formula and return points aligned to daily bars."""
        if start is not None:
            try:
                parse_yyyymmdd(start)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
        if end is not None:
            try:
                parse_yyyymmdd(end)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e

        eff_settings = settings
        if price_adjust is not None:
            eff_settings = settings.model_copy(update={"price_adjust": price_adjust})
        backend = _backend(eff_settings)
        formula = backend.get_formula(formula_id)
        if not formula or formula.get("kind") != "indicator":
            raise HTTPException(status_code=404, detail=f"Indicator formula {formula_id} not found")

        timeframe = str(formula.get("timeframe") or "D").upper()
        if timeframe not in ("D", "W", "M"):
            raise HTTPException(status_code=400, detail=f"Invalid indicator timeframe: {timeframe}")

        with backend.connect() as conn:
            query = f"""
            SELECT
                trade_date,
                open,
                high,
                low,
                close,
                vol,
                amount
            FROM {backend.daily_table}
            WHERE ts_code = ?
              AND open IS NOT NULL
              AND high IS NOT NULL
              AND low IS NOT NULL
              AND close IS NOT NULL
              AND vol IS NOT NULL
              AND amount IS NOT NULL
              AND NOT (vol = 0 AND amount = 0)
            """
            params: list[Any] = [ts_code]
            if start:
                query += " AND trade_date >= ?"
                params.append(start)
            if end:
                query += " AND trade_date <= ?"
                params.append(end)
            query += " ORDER BY trade_date DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(query, params).fetchall()

        if not rows:
            raise HTTPException(status_code=404, detail=f"No data found for {ts_code}")

        daily_df = pd.DataFrame([dict(r) for r in reversed(rows)])
        daily_df["trade_date"] = daily_df["trade_date"].astype(str)

        def _to_float_or_none(x: Any) -> float | None:
            if x is None:
                return None
            if isinstance(x, bool):
                return 1.0 if x else 0.0
            try:
                v = float(x)
            except (TypeError, ValueError):
                return None
            if math.isnan(v) or math.isinf(v):
                return None
            return v

        def _is_hidden(attrs: list[str]) -> bool:
            if not attrs:
                return False
            if "NODRAW" in attrs:
                return True
            return "LINETHICK0" in attrs

        def _build_line_points(trade_dates: list[str], values: list[Any]) -> list[IndicatorPoint]:
            return [
                IndicatorPoint(trade_date=td, value=_to_float_or_none(val))
                for td, val in zip(trade_dates, values, strict=True)
            ]

        trade_dates = daily_df["trade_date"].tolist()

        if timeframe == "D":
            outputs = [o for o in execute_formula_outputs(formula["formula"], daily_df) if not _is_hidden(o.draw_attrs)]
            lines: list[IndicatorLine] = []
            for i, out in enumerate(outputs, start=1):
                line_name = out.name or f"output_{i}"
                values = out.series.reset_index(drop=True).tolist()
                lines.append(IndicatorLine(name=line_name, points=_build_line_points(trade_dates, values)))
        else:
            dt = pd.to_datetime(daily_df["trade_date"], format="%Y%m%d")
            period = dt.dt.to_period("W-FRI") if timeframe == "W" else dt.dt.to_period("M")
            tmp = daily_df.copy()
            tmp["period"] = period
            agg = (
                tmp.groupby("period", sort=False)
                .agg(
                    open=("open", "first"),
                    high=("high", "max"),
                    low=("low", "min"),
                    close=("close", "last"),
                    vol=("vol", "sum"),
                    amount=("amount", "sum"),
                )
                .copy()
            )
            outputs_period = [o for o in execute_formula_outputs(formula["formula"], agg) if not _is_hidden(o.draw_attrs)]
            lines = []
            for i, out in enumerate(outputs_period, start=1):
                line_name = out.name or f"output_{i}"
                values = period.map(out.series).tolist()
                lines.append(IndicatorLine(name=line_name, points=_build_line_points(trade_dates, values)))

        primary_name = str(formula.get("name") or "")
        primary = next((ln for ln in lines if ln.name == primary_name), None) or (lines[0] if lines else None)
        points = primary.points if primary is not None else []
        return IndicatorSeriesResponse(
            ts_code=ts_code,
            formula_id=formula_id,
            name=str(formula["name"]),
            timeframe=timeframe,  # type: ignore[arg-type]
            points=points,
            lines=lines,
        )

    return app


def create_app_from_env() -> FastAPI:
    """
    Uvicorn factory entrypoint.

    Env:
      - STOCK_SCREENER_CACHE_DIR: default ./data
      - STOCK_SCREENER_DATA_BACKEND: default sqlite
      - STOCK_SCREENER_PRICE_ADJUST: none|qfq|hfq (default: none)
      - STOCK_SCREENER_API_KEY: optional (require X-API-Key)
      - STOCK_SCREENER_AUTH_ENABLED: optional (require Authorization: Bearer ... for /v1/*)
      - STOCK_SCREENER_AUTH_SECRET: required when auth enabled
      - STOCK_SCREENER_AUTH_TOKEN_TTL_SECONDS: optional, default 30 days
      - STOCK_SCREENER_AUTH_SIGNUP_MODE: open|email|closed (default: open; first user can always register)
      - STOCK_SCREENER_AUTH_ALLOWED_EMAIL_DOMAINS: optional, comma-separated domains (email signup only)
      - STOCK_SCREENER_AUTH_EMAIL_CODE_TTL_SECONDS: optional, default 600
      - STOCK_SCREENER_AUTH_EMAIL_CODE_COOLDOWN_SECONDS: optional, default 60
      - STOCK_SCREENER_AUTH_EMAIL_CODE_MAX_ATTEMPTS: optional, default 5
      - STOCK_SCREENER_AUTH_EMAIL_DEBUG_RETURN_CODE: optional, return code in API response (dev only)
      - STOCK_SCREENER_SMTP_HOST/PORT/USERNAME/PASSWORD/FROM: required for email signup (unless debug)
      - STOCK_SCREENER_SMTP_TLS: optional, default 1 (STARTTLS)
      - STOCK_SCREENER_SMTP_SSL: optional, default 0
      - STOCK_SCREENER_CORS_ORIGINS: optional comma-separated origins
    """

    cache_dir = os.environ.get("STOCK_SCREENER_CACHE_DIR", "./data")
    data_backend = os.environ.get("STOCK_SCREENER_DATA_BACKEND", "sqlite")
    price_adjust = os.environ.get("STOCK_SCREENER_PRICE_ADJUST", "none")
    settings = Settings(cache_dir=cache_dir, data_backend=data_backend, price_adjust=price_adjust)  # type: ignore[arg-type]
    return create_app(settings=settings)
