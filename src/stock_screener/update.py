from __future__ import annotations

import typer

from stock_screener.backends.sqlite_backend import SqliteBackend
from stock_screener.config import Settings
from stock_screener.dates import parse_yyyymmdd
from stock_screener.providers import get_provider
from stock_screener.providers.baostock_provider import BaoStockNotConfigured
from stock_screener.providers.baostock_provider import bs_to_ts_code
from stock_screener.providers.tushare_provider import TuShareTokenMissing
from stock_screener.tushare_client import TuShareNotConfigured


def _batched(values: list[str], batch_size: int) -> list[list[str]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [values[i : i + batch_size] for i in range(0, len(values), batch_size)]


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
        with backend.connect() as conn:
            for d in missing:
                typer.echo(f"updating {d} ...")
                df = p.daily_by_trade_date(trade_date=d)
                if df.empty:
                    typer.echo(f"warning: empty daily for {d}", err=True)
                    backend.mark_trade_date_updated_in_conn(conn, d)
                    continue
                backend.upsert_daily_df_in_conn(conn, df)
                backend.mark_trade_date_updated_in_conn(conn, d)
        typer.echo("done")
        return

    if getattr(p, "name", "") == "baostock":
        if not missing:
            typer.echo("done")
            return

        missing_sorted = sorted(missing)
        range_start, range_end = missing_sorted[0], missing_sorted[-1]
        ranges = [(range_start, range_end)]

        with p.session() as bs:
            codes = p._all_stock_codes(bs=bs, day=end)
            provider_name = "baostock"
            done = backend.get_progress_ts_codes(provider=provider_name, range_start=range_start, range_end=range_end)
            if not done:
                done = backend.distinct_ts_codes_in_range(start=range_start, end=range_end)
            typer.echo(
                f"stocks: {len(codes)}, range: {range_start}..{range_end}, resume_done: {len(done)}"
            )

            batches = _batched(codes, batch_size=200)
            for bi, batch in enumerate(batches, start=1):
                typer.echo(f"batch: {bi}/{len(batches)}")
                with backend.connect() as conn:
                    for code in batch:
                        ts_code = bs_to_ts_code(code)
                        if ts_code in done:
                            continue
                        df = p._fetch_daily_ranges(bs=bs, bs_code=code, ranges=ranges)
                        if df.empty:
                            backend.mark_progress_ts_code_in_conn(
                                conn,
                                provider=provider_name,
                                range_start=range_start,
                                range_end=range_end,
                                ts_code=ts_code,
                            )
                            continue
                        backend.upsert_daily_df_in_conn(conn, df)
                        backend.mark_progress_ts_code_in_conn(
                            conn,
                            provider=provider_name,
                            range_start=range_start,
                            range_end=range_end,
                            ts_code=ts_code,
                        )
                        done.add(ts_code)

            with backend.connect() as conn:
                for d in missing_sorted:
                    backend.mark_trade_date_updated_in_conn(conn, d)
        typer.echo("done")
        return

    raise typer.BadParameter("unknown provider")
