from __future__ import annotations

import typer

from stock_screener.backends.sqlite_backend import SqliteBackend
from stock_screener.config import Settings
from stock_screener.dates import parse_yyyymmdd
from stock_screener.tushare_client import TuShareClient, TuShareNotConfigured, get_tushare_token


def update_daily(*, settings: Settings, start: str, end: str) -> None:
    if settings.data_backend != "sqlite":
        raise typer.BadParameter("only sqlite backend is implemented")
    if parse_yyyymmdd(start) > parse_yyyymmdd(end):
        raise typer.BadParameter("start must be <= end")

    token = get_tushare_token()
    if not token:
        raise typer.BadParameter("missing TUSHARE_TOKEN; offline mode supports only `run` with local cache")

    backend = SqliteBackend(settings.sqlite_path)
    backend.init()

    try:
        client = TuShareClient(token=token)
        open_dates = client.trade_cal_open_dates(start=start, end=end)
    except TuShareNotConfigured as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=2)

    updated = backend.get_updated_trade_dates(open_dates)
    missing = [d for d in open_dates if d not in updated]
    typer.echo(f"open trade dates: {len(open_dates)}, missing: {len(missing)}")

    for d in missing:
        typer.echo(f"updating {d} ...")
        df = client.daily_by_trade_date(trade_date=d)
        if df.empty:
            typer.echo(f"warning: empty daily for {d}", err=True)
            backend.mark_trade_date_updated(d)
            continue
        backend.upsert_daily_df(df)
        backend.mark_trade_date_updated(d)

    typer.echo("done")
