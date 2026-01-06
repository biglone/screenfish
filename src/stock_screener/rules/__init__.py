from __future__ import annotations

from stock_screener.rules.base import Rule
from stock_screener.rules.kdj_oversold import KdjOversoldRule
from stock_screener.rules.midline_ma60 import MidlineMa60Rule


def builtin_rules() -> list[Rule]:
    return [MidlineMa60Rule(), KdjOversoldRule()]


def resolve_rules(names: str | None) -> list[Rule]:
    rules = builtin_rules()
    if not names:
        return rules
    wanted = {n.strip() for n in names.split(",") if n.strip()}
    by_name = {r.name: r for r in rules}
    unknown = sorted(wanted - set(by_name.keys()))
    if unknown:
        raise ValueError(f"unknown rules: {unknown}; available: {sorted(by_name.keys())}")
    return [by_name[n] for n in sorted(wanted)]

