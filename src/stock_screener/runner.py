from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
import typer

from stock_screener.backends.sqlite_backend import SqliteBackend
from stock_screener.config import Settings
from stock_screener.dates import parse_yyyymmdd, subtract_calendar_days
from stock_screener.rules import resolve_rules


def _is_st_name(name: str | None) -> bool:
    if not name:
        return False
    n = str(name).strip().upper().replace(" ", "")
    return n.startswith("ST") or n.startswith("*ST") or n.startswith("S*ST") or n.startswith("SST")


@dataclass(frozen=True)
class ScreenResultColumns:
    trade_date: str = "trade_date"
    ts_code: str = "ts_code"
    name: str = "name"
    close: str = "close"
    amount: str = "amount"
    ma60: str = "ma60"
    mid_bullbear: str = "mid_bullbear"
    j: str = "j"
    rules: str = "rules"


def run_screen(
    *,
    settings: Settings,
    date: str,
    combo: Literal["and", "or"],
    lookback_days: int = 200,
    rules: str | None = None,
    with_name: bool = False,
    exclude_st: bool = False,
) -> pd.DataFrame:
    if settings.data_backend != "sqlite":
        raise typer.BadParameter("only sqlite backend is implemented")
    if combo not in ("and", "or"):
        raise typer.BadParameter("combo must be 'and' or 'or'")
    parse_yyyymmdd(date)

    backend = SqliteBackend(
        settings.sqlite_path,
        daily_table=settings.daily_table,
        update_log_table=settings.update_log_table,
        provider_stock_progress_table=settings.provider_stock_progress_table,
    )
    backend.init()

    if lookback_days <= 0:
        start = date
    else:
        with backend.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT trade_date
                FROM {backend.update_log_table}
                WHERE trade_date <= ?
                ORDER BY trade_date DESC
                LIMIT ?
                """,
                (date, int(lookback_days)),
            ).fetchall()
            trade_dates = [str(r["trade_date"]) for r in rows if r["trade_date"]]
            if not trade_dates:
                rows = conn.execute(
                    f"""
                    SELECT trade_date
                    FROM {backend.daily_table}
                    WHERE trade_date <= ?
                    GROUP BY trade_date
                    ORDER BY trade_date DESC
                    LIMIT ?
                    """,
                    (date, int(lookback_days)),
                ).fetchall()
                trade_dates = [str(r["trade_date"]) for r in rows if r["trade_date"]]
        if trade_dates:
            start = trade_dates[-1]
        else:
            start = subtract_calendar_days(date, lookback_days)
    df = backend.load_daily_lookback(end=date, start=start)
    if df.empty:
        raise typer.BadParameter(f"no local data in cache for [{start}, {date}]")

    df = df.copy()
    df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(drop=True)
    df = df.loc[df["vol"].notna() & df["amount"].notna() & ~((df["vol"] == 0) & (df["amount"] == 0))].reset_index(
        drop=True
    )

    try:
        rule_objs = resolve_rules(rules, backend=backend)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    masks: dict[str, pd.Series] = {}
    for rule in rule_objs:
        masks[rule.name] = rule.mask(df).fillna(False)

    if combo == "and":
        combined = pd.Series(True, index=df.index)
        for m in masks.values():
            combined &= m
    else:
        combined = pd.Series(False, index=df.index)
        for m in masks.values():
            combined |= m

    selected = df["trade_date"] == date
    hits = df.loc[selected & combined].copy()

    cols = ScreenResultColumns()
    base_cols = [
        cols.trade_date,
        cols.ts_code,
        cols.close,
        cols.amount,
        cols.ma60,
        cols.mid_bullbear,
        cols.j,
        cols.rules,
    ]

    def _ensure_col(frame: pd.DataFrame, col: str) -> None:
        if col in frame.columns:
            return
        frame[col] = pd.Series([None] * len(frame), dtype="object")

    for col in (cols.ma60, cols.mid_bullbear, cols.j):
        _ensure_col(hits, col)

    name_map: dict[str, str | None] | None = None
    if (with_name or exclude_st) and not hits.empty:
        name_map = backend.load_stock_names(hits["ts_code"].astype(str).unique().tolist())
        if exclude_st and name_map:
            names = hits["ts_code"].astype(str).map(name_map)
            is_st = names.astype(object).where(names.notna(), None).map(_is_st_name).fillna(False)
            hits = hits.loc[~is_st].copy()

    if with_name:
        base_cols.insert(2, cols.name)
        _ensure_col(hits, cols.name)
        if not hits.empty:
            if name_map is None:
                name_map = backend.load_stock_names(hits["ts_code"].astype(str).unique().tolist())
            names = hits["ts_code"].astype(str).map(name_map)
            hits[cols.name] = names.astype(object).where(names.notna(), None)

    if hits.empty:
        hits[cols.rules] = pd.Series(dtype="string")
        return hits[base_cols]

    def _rules_for_row(idx: int) -> str:
        matched = [name for name, m in masks.items() if bool(m.loc[idx])]
        return ",".join(matched)

    hits["rules"] = [_rules_for_row(i) for i in hits.index]
    out = hits[base_cols].sort_values([cols.trade_date, cols.ts_code], kind="mergesort")
    # Avoid NaN/Infinity that may break JSON responses.
    out = out.where(pd.notna(out), None)
    return out.reset_index(drop=True)
