from __future__ import annotations

from datetime import date, datetime, timedelta


def parse_yyyymmdd(value: str) -> date:
    if len(value) != 8 or not value.isdigit():
        raise ValueError(f"expected YYYYMMDD, got: {value!r}")
    return datetime.strptime(value, "%Y%m%d").date()


def format_yyyymmdd(value: date) -> str:
    return value.strftime("%Y%m%d")


def subtract_calendar_days(yyyymmdd: str, days: int) -> str:
    d = parse_yyyymmdd(yyyymmdd)
    return format_yyyymmdd(d - timedelta(days=days))

