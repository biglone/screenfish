from __future__ import annotations

import typer

from stock_screener.backends.sqlite_backend import SqliteBackend
from stock_screener.config import Settings
from stock_screener.dates import parse_yyyymmdd
from stock_screener.providers import get_provider
from stock_screener.providers.baostock_provider import BaoStockNotConfigured
from stock_screener.providers.tushare_provider import TuShareTokenMissing
from stock_screener.tushare_client import TuShareNotConfigured


def update_daily(*, settings: Settings, start: str, end: str, provider: str) -> None:
    if settings.data_backend != "sqlite":
        raise typer.BadParameter("only sqlite backend is implemented")
    if parse_yyyymmdd(start) > parse_yyyymmdd(end):
        raise typer.BadParameter("start must be <= end")

    backend = SqliteBackend(settings.sqlite_path)
    backend.init()

    try:
        p = get_provider(provider)
        open_dates = p.open_trade_dates(start=start, end=end)
    except (TuShareNotConfigured, BaoStockNotConfigured) as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=2) from e
    except (TuShareTokenMissing, ValueError) as e:
        raise typer.BadParameter(str(e)) from e

    updated = backend.get_updated_trade_dates(open_dates)
    missing = [d for d in open_dates if d not in updated]
    typer.echo(f"open trade dates: {len(open_dates)}, missing: {len(missing)}")

    if getattr(p, "name", "") == "tushare":
        for d in missing:
            typer.echo(f"updating {d} ...")
            df = p.daily_by_trade_date(trade_date=d)
            if df.empty:
                typer.echo(f"warning: empty daily for {d}", err=True)
                backend.mark_trade_date_updated(d)
                continue
            backend.upsert_daily_df(df)
            backend.mark_trade_date_updated(d)
        typer.echo("done")
        return

    if getattr(p, "name", "") == "baostock":
        if not missing:
            typer.echo("done")
            return

        missing_sorted = sorted(missing)
        ranges = [(missing_sorted[0], missing_sorted[-1])]

        with p.session() as bs:
            codes = p._all_stock_codes(bs=bs, day=end)
            typer.echo(f"stocks: {len(codes)}, range: {ranges[0][0]}..{ranges[0][1]}")
            for i, code in enumerate(codes, start=1):
                if i % 200 == 0:
                    typer.echo(f"progress: {i}/{len(codes)}")
                df = p._fetch_daily_ranges(bs=bs, bs_code=code, ranges=ranges)
                if df.empty:
                    continue
                backend.upsert_daily_df(df)

        for d in missing_sorted:
            backend.mark_trade_date_updated(d)
        typer.echo("done")
        return

    raise typer.BadParameter("unknown provider")
