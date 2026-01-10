from __future__ import annotations

import typer

from stock_screener.backends.sqlite_backend import SqliteBackend
from stock_screener.config import Settings
from stock_screener.dates import parse_yyyymmdd
from stock_screener.providers import get_provider
from stock_screener.providers.baostock_provider import BaoStockNotConfigured
from stock_screener.providers.tushare_provider import TuShareTokenMissing
from stock_screener.tushare_client import TuShareNotConfigured


def sync_names(*, settings: Settings, provider: str, date: str) -> None:
    if settings.data_backend != "sqlite":
        raise typer.BadParameter("only sqlite backend is implemented")
    parse_yyyymmdd(date)

    backend = SqliteBackend(
        settings.sqlite_path,
        daily_table=settings.daily_table,
        update_log_table=settings.update_log_table,
        provider_stock_progress_table=settings.provider_stock_progress_table,
    )
    backend.init()

    try:
        p = get_provider(provider)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e

    if getattr(p, "name", "") == "baostock":
        try:
            with p.session() as bs:
                df = p._all_stock_basics(bs=bs, day=date)
        except BaoStockNotConfigured as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(code=2) from e
        backend.upsert_stock_basic_df(df)
        typer.echo(f"synced names: {len(df)}")
        return

    if getattr(p, "name", "") == "tushare":
        try:
            df = p.stock_basics()
        except (TuShareNotConfigured, TuShareTokenMissing) as e:
            raise typer.BadParameter(str(e)) from e
        backend.upsert_stock_basic_df(df)
        typer.echo(f"synced names: {len(df)}")
        return

    raise typer.BadParameter("unknown provider")
