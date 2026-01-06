from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stock_screener.formula_parser import execute_formula


@dataclass
class FormulaRule:
    """基于通达信公式的筛选规则"""

    name: str
    formula: str

    def enrich(self, df: pd.DataFrame) -> None:
        """公式规则不需要预先计算列"""
        pass

    def mask(self, df: pd.DataFrame) -> pd.Series:
        """对每只股票执行公式，返回筛选结果"""
        results = []
        for ts_code, group in df.groupby("ts_code"):
            try:
                # 对每只股票单独执行公式
                result = execute_formula(self.formula, group)
                # 只取最后一行的结果（当日）
                results.append((group.index[-1], bool(result.iloc[-1]) if len(result) > 0 else False))
            except Exception:
                # 公式执行失败，标记为不满足
                results.append((group.index[-1], False))

        # 构建完整的 mask
        mask = pd.Series(False, index=df.index)
        for idx, val in results:
            mask.loc[idx] = val
        return mask
