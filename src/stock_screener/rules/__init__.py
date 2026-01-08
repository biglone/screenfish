from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from stock_screener.rules.base import Rule
from stock_screener.rules.formula_rule import FormulaRule
from stock_screener.rules.kdj_oversold import KdjOversoldRule
from stock_screener.rules.midline_ma60 import MidlineMa60Rule

if TYPE_CHECKING:
    from stock_screener.backends.sqlite_backend import SqliteBackend


def builtin_rules() -> list[Rule]:
    return [MidlineMa60Rule(), KdjOversoldRule()]


def resolve_rules(names: str | None, backend: "SqliteBackend | None" = None) -> list[Rule]:
    """解析规则名称，支持内置规则和数据库中的公式

    Args:
        names: 规则名称，逗号分隔。可以是:
               - 内置规则名称 (midline_ma60, kdj_oversold)
               - 公式名称 (formula:公式名称)
               - 公式ID (formula_id:123)
               - 留空使用所有启用的公式
        backend: SQLite 后端，用于加载公式

    Returns:
        规则列表
    """
    builtin = builtin_rules()
    builtin_by_name = {r.name: r for r in builtin}

    # 如果没有指定规则名称，使用所有启用的公式
    if not names:
        if backend is None:
            return builtin
        formulas = backend.list_formulas(enabled_only=True, kind="screen")
        if not formulas:
            return builtin
        return [FormulaRule(name=f["name"], formula=f["formula"]) for f in formulas]

    # 解析指定的规则
    wanted = [n.strip() for n in names.split(",") if n.strip()]
    result: list[Rule] = []

    for name in wanted:
        # 检查是否是内置规则
        if name in builtin_by_name:
            result.append(builtin_by_name[name])
            continue

        # 检查是否是公式引用 (formula:名称)
        if name.startswith("formula:"):
            formula_name = name[8:]
            if backend is None:
                raise ValueError(f"无法加载公式 '{formula_name}'：未提供数据库后端")
            formula = backend.get_formula_by_name(formula_name)
            if not formula:
                raise ValueError(f"未知公式: {formula_name}")
            if formula.get("kind") != "screen":
                raise ValueError(f"公式 '{formula_name}' 不是筛选类型")
            result.append(FormulaRule(name=formula["name"], formula=formula["formula"]))
            continue

        # 检查是否是公式ID引用 (formula_id:123)
        if name.startswith("formula_id:"):
            formula_id = int(name[11:])
            if backend is None:
                raise ValueError(f"无法加载公式 ID {formula_id}：未提供数据库后端")
            formula = backend.get_formula(formula_id)
            if not formula:
                raise ValueError(f"未知公式 ID: {formula_id}")
            if formula.get("kind") != "screen":
                raise ValueError(f"公式 ID {formula_id} 不是筛选类型")
            result.append(FormulaRule(name=formula["name"], formula=formula["formula"]))
            continue

        # 尝试作为公式名称查找
        if backend is not None:
            formula = backend.get_formula_by_name(name)
            if formula:
                if formula.get("kind") != "screen":
                    raise ValueError(f"公式 '{name}' 不是筛选类型")
                result.append(FormulaRule(name=formula["name"], formula=formula["formula"]))
                continue

        # 未找到匹配的规则
        available = sorted(builtin_by_name.keys())
        raise ValueError(f"未知规则: {name}; 可用的内置规则: {available}")

    return result
