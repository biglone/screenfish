from __future__ import annotations

from datetime import date as _date

import typer

from stock_screener.backends.sqlite_backend import SqliteBackend
from stock_screener.config import Settings
from stock_screener.dates import format_yyyymmdd, parse_yyyymmdd, subtract_calendar_days
from stock_screener.providers import get_provider
from stock_screener.providers.baostock_provider import BaoStockNotConfigured
from stock_screener.providers.baostock_provider import bs_to_ts_code
from stock_screener.providers.tushare_provider import TuShareTokenMissing
from stock_screener.tushare_client import TuShareNotConfigured


def _batched(values: list[str], batch_size: int) -> list[list[str]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [values[i : i + batch_size] for i in range(0, len(values), batch_size)]


def _resolve_start_end(
    *,
    backend: SqliteBackend,
    start: str | None,
    end: str | None,
    repair_days: int,
) -> tuple[str, str]:
    end_eff = end or format_yyyymmdd(_date.today())
    parse_yyyymmdd(end_eff)

    if start is not None:
        parse_yyyymmdd(start)
        if parse_yyyymmdd(start) > parse_yyyymmdd(end_eff):
            raise typer.BadParameter("start must be <= end")
        return start, end_eff

    # Auto mode: choose a recent lookback window to both fetch "latest" and repair gaps.
    last = backend.max_trade_date_in_update_log() or backend.max_trade_date_in_daily()
    if not last:
        raise typer.BadParameter("no local cache found; please specify --start for the first update")
    parse_yyyymmdd(last)
    if repair_days < 0:
        raise typer.BadParameter("repair-days must be >= 0")
    start_eff = subtract_calendar_days(last, repair_days)
    if parse_yyyymmdd(start_eff) > parse_yyyymmdd(end_eff):
        start_eff = end_eff
    return start_eff, end_eff


def update_daily(*, settings: Settings, start: str | None, end: str | None, provider: str, repair_days: int) -> None:
    if settings.data_backend != "sqlite":
        raise typer.BadParameter("only sqlite backend is implemented")

    backend = SqliteBackend(settings.sqlite_path)
    backend.init()
    start_eff, end_eff = _resolve_start_end(backend=backend, start=start, end=end, repair_days=repair_days)

    try:
        p = get_provider(provider)
        open_dates = p.open_trade_dates(start=start_eff, end=end_eff)
    except (TuShareNotConfigured, BaoStockNotConfigured) as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=2) from e
    except (TuShareTokenMissing, ValueError) as e:
        raise typer.BadParameter(str(e)) from e

    updated = backend.get_updated_trade_dates(open_dates)
    missing = [d for d in open_dates if d not in updated]
    typer.echo(f"range: {start_eff}..{end_eff}, open trade dates: {len(open_dates)}, missing: {len(missing)}")

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
            target_ts_codes = {bs_to_ts_code(c) for c in codes}

            done = backend.get_progress_ts_codes(provider=provider_name, range_start=range_start, range_end=range_end)
            if not done:
                seed = backend.distinct_ts_codes_in_range(start=range_start, end=range_end)
                if seed:
                    with backend.connect() as conn:
                        for ts_code in seed:
                            backend.mark_progress_ts_code_in_conn(
                                conn,
                                provider=provider_name,
                                range_start=range_start,
                                range_end=range_end,
                                ts_code=ts_code,
                            )
                    done = set(seed)
            typer.echo(
                f"stocks: {len(codes)}, range: {range_start}..{range_end}, resume_done: {len(done & target_ts_codes)}"
            )

            batches = _batched(codes, batch_size=200)
            for bi, batch in enumerate(batches, start=1):
                typer.echo(f"batch: {bi}/{len(batches)}")
                with backend.connect() as conn:
                    for code in batch:
                        ts_code = bs_to_ts_code(code)
                        if ts_code in done:
                            continue
                        try:
                            df = p._fetch_daily_ranges(bs=bs, bs_code=code, ranges=ranges)
                        except RuntimeError as e:
                            typer.echo(f"warning: {code} failed: {e}", err=True)
                            continue
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

            progress = backend.get_progress_ts_codes(provider=provider_name, range_start=range_start, range_end=range_end)
            completed = len(progress & target_ts_codes)
            if completed < len(target_ts_codes):
                typer.echo(f"incomplete: completed {completed}/{len(target_ts_codes)} stocks; rerun to resume", err=True)
                raise typer.Exit(code=1)

            with backend.connect() as conn:
                for d in missing_sorted:
                    backend.mark_trade_date_updated_in_conn(conn, d)
        typer.echo("done")
        return

    raise typer.BadParameter("unknown provider")
