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


class UpdateError(Exception):
    pass


class UpdateBadRequest(UpdateError):
    pass


class UpdateNotConfigured(UpdateError):
    pass


class UpdateIncomplete(UpdateError):
    pass


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
    try:
        parse_yyyymmdd(end_eff)
    except ValueError as e:
        raise UpdateBadRequest(str(e)) from e

    if start is not None:
        try:
            start_parsed = parse_yyyymmdd(start)
        except ValueError as e:
            raise UpdateBadRequest(str(e)) from e
        if start_parsed > parse_yyyymmdd(end_eff):
            raise UpdateBadRequest("start must be <= end")
        return start, end_eff

    # Auto mode: choose a recent lookback window to both fetch "latest" and repair gaps.
    last = backend.max_trade_date_in_update_log() or backend.max_trade_date_in_daily()
    if not last:
        raise UpdateBadRequest("no local cache found; please specify --start for the first update")
    try:
        parse_yyyymmdd(last)
    except ValueError as e:
        raise UpdateBadRequest(str(e)) from e
    if repair_days < 0:
        raise UpdateBadRequest("repair-days must be >= 0")
    start_eff = subtract_calendar_days(last, repair_days)
    if parse_yyyymmdd(start_eff) > parse_yyyymmdd(end_eff):
        start_eff = end_eff
    return start_eff, end_eff


def update_daily_service(*, settings: Settings, start: str | None, end: str | None, provider: str, repair_days: int) -> None:
    if settings.data_backend != "sqlite":
        raise UpdateBadRequest("only sqlite backend is implemented")

    backend = SqliteBackend(settings.sqlite_path)
    backend.init()
    start_eff, end_eff = _resolve_start_end(backend=backend, start=start, end=end, repair_days=repair_days)

    try:
        p = get_provider(provider)
        open_dates = p.open_trade_dates(start=start_eff, end=end_eff)
    except (TuShareNotConfigured, BaoStockNotConfigured) as e:
        raise UpdateNotConfigured(str(e)) from e
    except (TuShareTokenMissing, ValueError) as e:
        raise UpdateBadRequest(str(e)) from e

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
        # Always sync stock names when using baostock
        def _sync_stock_names(bs, day: str) -> None:
            try:
                basics_df = p._all_stock_basics(bs=bs, day=day)
                if not basics_df.empty:
                    backend.upsert_stock_basic_df(basics_df)
                    typer.echo(f"synced {len(basics_df)} stock names")
            except Exception as e:
                typer.echo(f"warning: failed to sync stock names: {e}", err=True)

        if not missing:
            # No daily data to update, but still sync stock names
            max_date = backend.max_trade_date_in_daily()
            if max_date:
                with p.session() as bs:
                    _sync_stock_names(bs, max_date)
            typer.echo("done")
            return

        missing_sorted = sorted(missing)
        range_start, range_end = missing_sorted[0], missing_sorted[-1]
        ranges = [(range_start, range_end)]

        with p.session() as bs:
            codes = p._all_stock_codes(bs=bs, day=range_end)
            if not codes:
                fallback_day = backend.max_trade_date_in_daily()
                if fallback_day and fallback_day != range_end:
                    typer.echo(f"warning: empty stock list for {range_end}, fallback to {fallback_day}", err=True)
                    codes = p._all_stock_codes(bs=bs, day=fallback_day)
            if not codes:
                typer.echo(f"error: empty stock list; cannot update {range_start}..{range_end}", err=True)
                raise UpdateNotConfigured(f"empty stock list; cannot update {range_start}..{range_end}")

            # Sync stock names
            _sync_stock_names(bs, range_end)

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
        typer.echo(f"stocks: {len(codes)}, range: {range_start}..{range_end}, resume_done: {len(done & target_ts_codes)}")

        batches = _batched(codes, batch_size=200)
        for bi, batch in enumerate(batches, start=1):
            typer.echo(f"batch: {bi}/{len(batches)}")
            with p.session() as bs:
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
                            conn.commit()
                            continue
                        backend.upsert_daily_df_in_conn(conn, df)
                        backend.mark_progress_ts_code_in_conn(
                            conn,
                            provider=provider_name,
                            range_start=range_start,
                            range_end=range_end,
                            ts_code=ts_code,
                        )
                        conn.commit()
                        done.add(ts_code)

        progress = backend.get_progress_ts_codes(provider=provider_name, range_start=range_start, range_end=range_end)
        completed = len(progress & target_ts_codes)
        if completed < len(target_ts_codes):
            raise UpdateIncomplete(f"incomplete: completed {completed}/{len(target_ts_codes)} stocks; rerun to resume")

        dates_with_rows = [d for d in missing_sorted if backend.count_daily_rows_for_trade_date(d) > 0]
        if not dates_with_rows:
            backend.clear_progress(provider=provider_name, range_start=range_start, range_end=range_end)
            typer.echo(
                f"no daily rows for {range_start}..{range_end}; provider may not have published data yet, please rerun later",
                err=True,
            )
            raise UpdateIncomplete(
                f"no daily rows for {range_start}..{range_end}; provider may not have published data yet, please rerun later"
            )

        with backend.connect() as conn:
            for d in dates_with_rows:
                backend.mark_trade_date_updated_in_conn(conn, d)

        if len(dates_with_rows) != len(missing_sorted):
            backend.clear_progress(provider=provider_name, range_start=range_start, range_end=range_end)
            remaining = [d for d in missing_sorted if d not in dates_with_rows]
            raise UpdateIncomplete(f"partial update; remaining dates: {remaining}; rerun to resume")
        typer.echo("done")
        return

    raise UpdateBadRequest("unknown provider")


def update_daily(*, settings: Settings, start: str | None, end: str | None, provider: str, repair_days: int) -> None:
    """
    CLI-friendly wrapper.

    It preserves Click/Typer exit codes and error types, while the core implementation
    (`update_daily_service`) is safe to be used by the REST API without raising Typer-specific
    exceptions that would otherwise map poorly to HTTP responses.
    """

    try:
        update_daily_service(settings=settings, start=start, end=end, provider=provider, repair_days=repair_days)
    except UpdateBadRequest as e:
        raise typer.BadParameter(str(e)) from e
    except UpdateNotConfigured as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=2) from e
    except UpdateIncomplete as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1) from e
