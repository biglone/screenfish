from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
import typer

from stock_screener.backends.sqlite_backend import SqliteBackend
from stock_screener.config import Settings
from stock_screener.dates import parse_yyyymmdd, subtract_calendar_days
from stock_screener.rules import resolve_rules


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
) -> pd.DataFrame:
    if settings.data_backend != "sqlite":
        raise typer.BadParameter("only sqlite backend is implemented")
    if combo not in ("and", "or"):
        raise typer.BadParameter("combo must be 'and' or 'or'")
    parse_yyyymmdd(date)

    backend = SqliteBackend(settings.sqlite_path)
    backend.init()

    start = subtract_calendar_days(date, lookback_days)
    df = backend.load_daily_lookback(end=date, start=start)
    if df.empty:
        raise typer.BadParameter(f"no local data in cache for [{start}, {date}]")

    df = df.copy()
    df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(drop=True)

    try:
        rule_objs = resolve_rules(rules)
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
    if with_name:
        base_cols.insert(2, cols.name)
        if not hits.empty:
            name_map = backend.load_stock_names(hits["ts_code"].astype(str).unique().tolist())
            hits[cols.name] = hits["ts_code"].astype(str).map(name_map)

    if hits.empty:
        return hits.assign(rules=pd.Series(dtype="string"))[base_cols]

    def _rules_for_row(idx: int) -> str:
        matched = [name for name, m in masks.items() if bool(m.loc[idx])]
        return ",".join(matched)

    hits["rules"] = [_rules_for_row(i) for i in hits.index]
    out = hits[base_cols].sort_values([cols.trade_date, cols.ts_code], kind="mergesort")
    return out.reset_index(drop=True)
