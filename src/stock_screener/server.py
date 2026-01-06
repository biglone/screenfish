from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import date as _date
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field

from stock_screener.backends.sqlite_backend import SqliteBackend
from stock_screener.config import Settings
from stock_screener.dates import format_yyyymmdd, parse_yyyymmdd
from stock_screener.providers.baostock_provider import BaoStockNotConfigured
from stock_screener.providers.tushare_provider import TuShareTokenMissing
from stock_screener.runner import run_screen
from stock_screener.tdx import TdxEbkFormat, ts_code_to_ebk_code
from stock_screener.update import update_daily


def _api_key_required(x_api_key: str | None = Header(default=None)) -> None:
    required = os.environ.get("STOCK_SCREENER_API_KEY")
    if not required:
        return
    if not x_api_key or x_api_key != required:
        raise HTTPException(status_code=401, detail="missing/invalid X-API-Key")


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


class UpdateWaitResponse(BaseModel):
    ok: bool
    target_date: str
    latest_trade_date: str | None
    attempts: int
    elapsed_seconds: float
    message: str


class ScreenRequest(BaseModel):
    date: str | Literal["latest"] = "latest"
    combo: Literal["and", "or"] = "and"
    lookback_days: int = 200
    rules: str | None = None
    with_name: bool = False


class ScreenResponse(BaseModel):
    trade_date: str
    hits: list[dict[str, Any]]


class AvailabilityResponse(BaseModel):
    date: str
    provider: str
    available: bool
    detail: str


@dataclass(frozen=True)
class AppState:
    settings: Settings


def _backend(settings: Settings) -> SqliteBackend:
    backend = SqliteBackend(settings.sqlite_path)
    backend.init()
    return backend


def create_app(*, settings: Settings) -> FastAPI:
    app = FastAPI(title="stock_screener", version="0.1.0")
    state = AppState(settings=settings)
    app.state.app_state = state

    def _settings_dep() -> Settings:
        return app.state.app_state.settings

    @app.get("/v1/health", dependencies=[Depends(_api_key_required)])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/status", response_model=StatusResponse, dependencies=[Depends(_api_key_required)])
    def status(settings: Settings = Depends(_settings_dep)) -> StatusResponse:
        backend = _backend(settings)
        max_daily = backend.max_trade_date_in_daily()
        max_log = backend.max_trade_date_in_update_log()
        # Query counts directly to avoid loading data
        import sqlite3

        conn = sqlite3.connect(str(settings.sqlite_path))
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM daily")
            total_rows = int(cur.fetchone()[0])
            cur.execute("SELECT COUNT(DISTINCT ts_code) FROM daily")
            total_stocks = int(cur.fetchone()[0])
        finally:
            conn.close()
        return StatusResponse(
            today=format_yyyymmdd(_date.today()),
            cache_dir=str(settings.cache_dir),
            sqlite_path=str(settings.sqlite_path),
            max_daily_trade_date=max_daily,
            max_update_log_trade_date=max_log,
            stocks=total_stocks,
            rows=total_rows,
        )

    @app.post("/v1/update", dependencies=[Depends(_api_key_required)])
    def update(req: UpdateRequest, settings: Settings = Depends(_settings_dep)) -> dict[str, Any]:
        try:
            update_daily(settings=settings, start=req.start, end=req.end, provider=req.provider, repair_days=req.repair_days)
        except (TuShareTokenMissing, BaoStockNotConfigured) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        backend = _backend(settings)
        return {
            "ok": True,
            "max_daily_trade_date": backend.max_trade_date_in_daily(),
            "max_update_log_trade_date": backend.max_trade_date_in_update_log(),
        }

    @app.post("/v1/update/wait", response_model=UpdateWaitResponse, dependencies=[Depends(_api_key_required)])
    def update_wait(req: UpdateWaitRequest, settings: Settings = Depends(_settings_dep)) -> UpdateWaitResponse:
        target = req.target_date or format_yyyymmdd(_date.today())
        parse_yyyymmdd(target)

        backend = _backend(settings)
        started = time.time()
        attempts = 0

        while True:
            attempts += 1
            try:
                update_daily(settings=settings, start=None, end=target, provider=req.provider, repair_days=req.repair_days)
            except Exception:
                pass

            backend = _backend(settings)
            latest = backend.max_trade_date_in_daily()
            if latest == target and backend.count_daily_rows_for_trade_date(target) > 0:
                return UpdateWaitResponse(
                    ok=True,
                    target_date=target,
                    latest_trade_date=latest,
                    attempts=attempts,
                    elapsed_seconds=time.time() - started,
                    message="daily data available",
                )

            if time.time() - started >= req.timeout_seconds:
                return UpdateWaitResponse(
                    ok=False,
                    target_date=target,
                    latest_trade_date=latest,
                    attempts=attempts,
                    elapsed_seconds=time.time() - started,
                    message="timeout waiting for daily data",
                )
            time.sleep(req.interval_seconds)

    @app.post("/v1/screen", response_model=ScreenResponse, dependencies=[Depends(_api_key_required)])
    def screen(req: ScreenRequest, settings: Settings = Depends(_settings_dep)) -> ScreenResponse:
        backend = _backend(settings)
        date_value: str
        if req.date == "latest":
            max_daily = backend.max_trade_date_in_daily()
            if not max_daily:
                raise HTTPException(status_code=400, detail="no local data; run update first")
            date_value = str(max_daily)
        else:
            date_value = str(req.date)
            parse_yyyymmdd(date_value)

        df = run_screen(
            settings=settings,
            date=date_value,
            combo=req.combo,
            lookback_days=req.lookback_days,
            rules=req.rules,
            with_name=req.with_name,
        )
        return ScreenResponse(trade_date=date_value, hits=df.to_dict(orient="records"))

    @app.get("/v1/data/availability", response_model=AvailabilityResponse, dependencies=[Depends(_api_key_required)])
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

    @app.post("/v1/export/ebk", dependencies=[Depends(_api_key_required)])
    def export_ebk(req: ScreenRequest, settings: Settings = Depends(_settings_dep)) -> dict[str, Any]:
        backend = _backend(settings)
        if req.date == "latest":
            max_daily = backend.max_trade_date_in_daily()
            if not max_daily:
                raise HTTPException(status_code=400, detail="no local data; run update first")
            date_value = str(max_daily)
        else:
            date_value = str(req.date)
            parse_yyyymmdd(date_value)

        df = run_screen(
            settings=settings,
            date=date_value,
            combo=req.combo,
            lookback_days=req.lookback_days,
            rules=req.rules,
            with_name=False,
        )
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

    return app
