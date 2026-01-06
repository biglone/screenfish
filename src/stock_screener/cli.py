from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from stock_screener.config import Settings
from stock_screener.names import sync_names
from stock_screener.runner import run_screen
from stock_screener.tdx import write_ebk
from stock_screener.update import update_daily
from stock_screener.server import create_app

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def update(
    start: Optional[str] = typer.Option(None, help="Start date YYYYMMDD (optional; auto if omitted)"),
    end: Optional[str] = typer.Option(None, help="End date YYYYMMDD (optional; defaults to today)"),
    provider: str = typer.Option("baostock", help="baostock|tushare"),
    repair_days: int = typer.Option(30, help="Auto mode lookback calendar days to repair gaps"),
    data_backend: str = typer.Option("sqlite", help="sqlite|parquet (only sqlite implemented)"),
    cache: Path = typer.Option(Path("./data"), help="Cache directory"),
) -> None:
    settings = Settings(cache_dir=cache, data_backend=data_backend)
    update_daily(settings=settings, start=start, end=end, provider=provider, repair_days=repair_days)


@app.command("sync-names")
def sync_names_cmd(
    date: str = typer.Option(..., help="Reference date YYYYMMDD (used by some providers)"),
    provider: str = typer.Option("baostock", help="baostock|tushare"),
    data_backend: str = typer.Option("sqlite", help="sqlite|parquet (only sqlite implemented)"),
    cache: Path = typer.Option(Path("./data"), help="Cache directory"),
) -> None:
    settings = Settings(cache_dir=cache, data_backend=data_backend)
    sync_names(settings=settings, provider=provider, date=date)


@app.command()
def run(
    date: str = typer.Option(..., help="Trade date YYYYMMDD"),
    combo: str = typer.Option("and", help="and|or"),
    out: Path = typer.Option(..., help="Output file path (.csv or .json)"),
    data_backend: str = typer.Option("sqlite", help="sqlite|parquet (only sqlite implemented)"),
    cache: Path = typer.Option(Path("./data"), help="Cache directory"),
    lookback_days: int = typer.Option(200, help="Calendar days lookback to compute indicators"),
    rules: Optional[str] = typer.Option(None, help="Comma-separated rule names (default: built-in)"),
    with_name: bool = typer.Option(False, help="Include stock name (requires local stock_basic cache)"),
) -> None:
    settings = Settings(cache_dir=cache, data_backend=data_backend)
    results = run_screen(
        settings=settings,
        date=date,
        combo=combo,
        lookback_days=lookback_days,
        rules=rules,
        with_name=with_name,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".csv":
        results.to_csv(out, index=False)
        return
    if out.suffix.lower() == ".json":
        out.write_text(json.dumps(results.to_dict(orient="records"), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return
    raise typer.BadParameter("out must end with .csv or .json")


@app.command("export-ebk")
def export_ebk_cmd(
    date: str = typer.Option(..., help="Trade date YYYYMMDD"),
    combo: str = typer.Option("and", help="and|or"),
    out: Path = typer.Option(..., help="Output file path (.EBK)"),
    data_backend: str = typer.Option("sqlite", help="sqlite|parquet (only sqlite implemented)"),
    cache: Path = typer.Option(Path("./data"), help="Cache directory"),
    lookback_days: int = typer.Option(200, help="Calendar days lookback to compute indicators"),
    rules: Optional[str] = typer.Option(None, help="Comma-separated rule names (default: built-in)"),
) -> None:
    if out.suffix.lower() != ".ebk":
        raise typer.BadParameter("out must end with .EBK")
    settings = Settings(cache_dir=cache, data_backend=data_backend)
    results = run_screen(settings=settings, date=date, combo=combo, lookback_days=lookback_days, rules=rules)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_ebk(results["ts_code"].astype(str).tolist(), out)
@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind host"),
    port: int = typer.Option(8000, help="Bind port"),
    data_backend: str = typer.Option("sqlite", help="sqlite|parquet (only sqlite implemented)"),
    cache: Path = typer.Option(Path("./data"), help="Cache directory"),
) -> None:
    settings = Settings(cache_dir=cache, data_backend=data_backend)
    import uvicorn

    uvicorn.run(create_app(settings=settings), host=host, port=port)


if __name__ == "__main__":
    app()
