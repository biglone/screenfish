from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import date as _date
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
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


class FormulaItem(BaseModel):
    id: int
    name: str
    formula: str
    description: str | None = None
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
    enabled: bool = True


class FormulaUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    formula: str | None = Field(default=None, min_length=1)
    description: str | None = None
    enabled: bool | None = None


class FormulaValidateRequest(BaseModel):
    formula: str


class FormulaValidateResponse(BaseModel):
    valid: bool
    message: str


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

    @app.get("/v1/stocks", response_model=StockListResponse, dependencies=[Depends(_api_key_required)])
    def list_stocks(
        search: str | None = Query(default=None, description="Search by ts_code or name"),
        limit: int = Query(default=100, ge=1, le=1000),
        offset: int = Query(default=0, ge=0),
        settings: Settings = Depends(_settings_dep),
    ) -> StockListResponse:
        import sqlite3 as _sqlite3

        conn = _sqlite3.connect(str(settings.sqlite_path))
        conn.row_factory = _sqlite3.Row
        try:
            cur = conn.cursor()
            if search:
                search_pattern = f"%{search}%"
                cur.execute(
                    """
                    SELECT DISTINCT d.ts_code, sb.name
                    FROM daily d
                    LEFT JOIN stock_basic sb ON d.ts_code = sb.ts_code
                    WHERE d.ts_code LIKE ? OR sb.name LIKE ?
                    ORDER BY d.ts_code
                    LIMIT ? OFFSET ?
                    """,
                    (search_pattern, search_pattern, limit, offset),
                )
                stocks = [StockItem(ts_code=row["ts_code"], name=row["name"]) for row in cur.fetchall()]
                cur.execute(
                    """
                    SELECT COUNT(DISTINCT d.ts_code) as cnt
                    FROM daily d
                    LEFT JOIN stock_basic sb ON d.ts_code = sb.ts_code
                    WHERE d.ts_code LIKE ? OR sb.name LIKE ?
                    """,
                    (search_pattern, search_pattern),
                )
                total = cur.fetchone()["cnt"]
            else:
                cur.execute(
                    """
                    SELECT DISTINCT d.ts_code, sb.name
                    FROM daily d
                    LEFT JOIN stock_basic sb ON d.ts_code = sb.ts_code
                    ORDER BY d.ts_code
                    LIMIT ? OFFSET ?
                    """,
                    (limit, offset),
                )
                stocks = [StockItem(ts_code=row["ts_code"], name=row["name"]) for row in cur.fetchall()]
                cur.execute("SELECT COUNT(DISTINCT ts_code) as cnt FROM daily")
                total = cur.fetchone()["cnt"]
        finally:
            conn.close()
        return StockListResponse(total=total, stocks=stocks)

    @app.get("/v1/stocks/{ts_code}/daily", response_model=StockDailyResponse, dependencies=[Depends(_api_key_required)])
    def get_stock_daily(
        ts_code: str,
        start: str | None = Query(default=None, description="Start date YYYYMMDD"),
        end: str | None = Query(default=None, description="End date YYYYMMDD"),
        limit: int = Query(default=250, ge=1, le=1000),
        settings: Settings = Depends(_settings_dep),
    ) -> StockDailyResponse:
        import sqlite3 as _sqlite3

        conn = _sqlite3.connect(str(settings.sqlite_path))
        conn.row_factory = _sqlite3.Row
        try:
            cur = conn.cursor()
            # Get stock name
            cur.execute("SELECT name FROM stock_basic WHERE ts_code = ?", (ts_code,))
            name_row = cur.fetchone()
            name = name_row["name"] if name_row else None

            # Build query for daily bars
            query = "SELECT trade_date, open, high, low, close, vol, amount FROM daily WHERE ts_code = ?"
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
        finally:
            conn.close()

        if not bars:
            raise HTTPException(status_code=404, detail=f"No data found for {ts_code}")

        return StockDailyResponse(ts_code=ts_code, name=name, bars=bars)

    @app.post("/v1/sync-stock-names", dependencies=[Depends(_api_key_required)])
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

    # Formula CRUD endpoints

    @app.get("/v1/formulas", response_model=FormulaListResponse, dependencies=[Depends(_api_key_required)])
    def list_formulas(
        enabled_only: bool = Query(default=False, description="Only return enabled formulas"),
        settings: Settings = Depends(_settings_dep),
    ) -> FormulaListResponse:
        """List all formulas."""
        backend = _backend(settings)
        formulas = backend.list_formulas(enabled_only=enabled_only)
        return FormulaListResponse(
            total=len(formulas),
            formulas=[FormulaItem(**f) for f in formulas],
        )

    @app.post("/v1/formulas", response_model=FormulaItem, dependencies=[Depends(_api_key_required)])
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
                enabled=req.enabled,
            )
            return FormulaItem(**formula)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/v1/formulas/{formula_id}", response_model=FormulaItem, dependencies=[Depends(_api_key_required)])
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

    @app.put("/v1/formulas/{formula_id}", response_model=FormulaItem, dependencies=[Depends(_api_key_required)])
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
                enabled=req.enabled,
            )
            return FormulaItem(**formula)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.delete("/v1/formulas/{formula_id}", dependencies=[Depends(_api_key_required)])
    def delete_formula(
        formula_id: int,
        settings: Settings = Depends(_settings_dep),
    ) -> dict[str, Any]:
        """Delete a formula."""
        backend = _backend(settings)
        if not backend.delete_formula(formula_id):
            raise HTTPException(status_code=404, detail=f"Formula {formula_id} not found")
        return {"ok": True, "deleted": formula_id}

    @app.post("/v1/formulas/validate", response_model=FormulaValidateResponse, dependencies=[Depends(_api_key_required)])
    def validate_formula_endpoint(
        req: FormulaValidateRequest,
    ) -> FormulaValidateResponse:
        """Validate formula syntax without saving."""
        from stock_screener.formula_parser import validate_formula

        valid, msg = validate_formula(req.formula)
        return FormulaValidateResponse(valid=valid, message=msg)

    return app


def create_app_from_env() -> FastAPI:
    """
    Uvicorn factory entrypoint.

    Env:
      - STOCK_SCREENER_CACHE_DIR: default ./data
      - STOCK_SCREENER_DATA_BACKEND: default sqlite
      - STOCK_SCREENER_API_KEY: optional (require X-API-Key)
      - STOCK_SCREENER_CORS_ORIGINS: optional comma-separated origins
    """

    cache_dir = os.environ.get("STOCK_SCREENER_CACHE_DIR", "./data")
    data_backend = os.environ.get("STOCK_SCREENER_DATA_BACKEND", "sqlite")
    settings = Settings(cache_dir=cache_dir, data_backend=data_backend)  # type: ignore[arg-type]
    return create_app(settings=settings)
