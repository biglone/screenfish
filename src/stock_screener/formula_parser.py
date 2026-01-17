"""
通达信公式解析器

支持的语法：
- 变量: OPEN/O, HIGH/H, LOW/L, CLOSE/C, VOL/V, AMOUNT
- 函数: MA, EMA, SMA, REF, HHV, LLV, CROSS, STD, SUM, ABS, MAX, MIN, IF, COUNT, EVERY, EXIST, BARSLAST, SLOPE
- 运算符: +, -, *, /, >, <, >=, <=, =, <>, AND, OR, NOT
- 赋值: VAR:=表达式;
- 输出: 表达式;

示例:
    MA5:=MA(CLOSE,5);
    MA10:=MA(CLOSE,10);
    CROSS(MA5,MA10);
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from stock_screener import indicators as ind


@dataclass
class FormulaContext:
    """公式执行上下文"""
    df: pd.DataFrame
    variables: dict[str, pd.Series] = field(default_factory=dict)

    def get_var(self, name: str) -> pd.Series:
        name_upper = name.upper()
        # 内置变量
        builtins = {
            'OPEN': 'open', 'O': 'open',
            'HIGH': 'high', 'H': 'high',
            'LOW': 'low', 'L': 'low',
            'CLOSE': 'close', 'C': 'close',
            'VOL': 'vol', 'V': 'vol',
            'VOLUME': 'vol',
            'AMOUNT': 'amount',
        }
        if name_upper in builtins:
            return self.df[builtins[name_upper]]
        if name_upper in self.variables:
            return self.variables[name_upper]
        raise ValueError(f"未知变量: {name}")

    def set_var(self, name: str, value: pd.Series) -> None:
        self.variables[name.upper()] = value


@dataclass(frozen=True)
class FormulaOutput:
    name: str | None
    series: pd.Series
    draw_attrs: list[str] = field(default_factory=list)


class FormulaParser:
    """通达信公式解析器"""
    FLOAT_EPS = 1e-6

    # 参数位置（0-based）需要为整数常量（N、M 等）
    INT_PARAMS: dict[str, set[int]] = {
        "MA": {1},
        "EMA": {1},
        "SMA": {1, 2},
        "REF": {1},
        "HHV": {1},
        "LLV": {1},
        "STD": {1},
        "SUM": {1},
        "COUNT": {1},
        "EVERY": {1},
        "EXIST": {1},
        "SLOPE": {1},
    }

    # 支持的函数
    FUNCTIONS: dict[str, Callable] = {
        'MA': ind.MA,
        'EMA': ind.EMA,
        'SMA': ind.SMA,
        'REF': ind.REF,
        'HHV': ind.HHV,
        'LLV': ind.LLV,
        'CROSS': ind.CROSS,
        'STD': ind.STD,
        'SUM': ind.SUM,
        'ABS': ind.ABS,
        'MAX': ind.MAX,
        'MIN': ind.MIN,
        'IF': ind.IF,
        'COUNT': ind.COUNT,
        'EVERY': ind.EVERY,
        'EXIST': ind.EXIST,
        'BARSLAST': ind.BARSLAST,
        'SLOPE': ind.SLOPE,
    }

    def __init__(self, formula: str):
        # Remove TongDaXin-style comments: { ... }
        self.formula = re.sub(r"\{.*?\}", "", formula, flags=re.DOTALL).strip()
        self.pos = 0
        self.length = len(self.formula)

    def parse(
        self,
        ctx: FormulaContext,
        *,
        prefer_output_name: str | None = None,
        output_selector: Literal["last", "first"] = "last",
    ) -> pd.Series:
        """
        解析并执行公式并返回结果。

        - 默认返回最后一个输出表达式（更适合筛选公式：通常最后一行是条件）。
        - 指标绘图场景可通过：
          - prefer_output_name：优先返回同名输出（例如公式名与首行输出名一致）
          - output_selector="first"：回退为第一个输出表达式
        """
        first_output: pd.Series | None = None
        last_output: pd.Series | None = None
        preferred_output: pd.Series | None = None
        statements = self._split_statements(self.formula)

        for stmt in statements:
            stmt = stmt.strip()
            if not stmt:
                continue

            # 检查是否是赋值语句 (VAR:=表达式)
            if ':=' in stmt:
                var_name, expr = stmt.split(':=', 1)
                var_name = var_name.strip()
                value = self._eval_expr(self._strip_draw_attrs(expr), ctx)
                ctx.set_var(var_name, value)
            # 检查是否是命名输出语句 (NAME:表达式，但不是 :=)
            elif ':' in stmt and not stmt.startswith(':'):
                # 找到第一个 : 的位置（排除括号内的）
                colon_pos = self._find_colon_outside_parens(stmt)
                if colon_pos > 0:
                    # TongDaXin 中 NAME:expr 也会产生变量（常用于参数/输出命名）
                    name = stmt[:colon_pos].strip()
                    expr = self._strip_draw_attrs(stmt[colon_pos + 1:])
                    value = self._eval_expr(expr, ctx)
                    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
                        ctx.set_var(name, value)
                    if first_output is None:
                        first_output = value
                    last_output = value
                    if prefer_output_name is not None and name == prefer_output_name:
                        preferred_output = value
                else:
                    # 普通输出语句
                    value = self._eval_expr(self._strip_draw_attrs(stmt), ctx)
                    if first_output is None:
                        first_output = value
                    last_output = value
            else:
                # 普通输出语句
                value = self._eval_expr(self._strip_draw_attrs(stmt), ctx)
                if first_output is None:
                    first_output = value
                last_output = value

        if preferred_output is not None:
            return preferred_output
        if output_selector == "first" and first_output is not None:
            return first_output
        if last_output is not None:
            return last_output
        return pd.Series(True, index=ctx.df.index)

    def _find_colon_outside_parens(self, expr: str) -> int:
        """找到括号外的第一个冒号位置"""
        paren_depth = 0
        for i, char in enumerate(expr):
            if char == '(':
                paren_depth += 1
            elif char == ')':
                paren_depth -= 1
            elif char == ':' and paren_depth == 0:
                return i
        return -1

    def _strip_draw_attrs(self, expr: str) -> str:
        """
        Strip TongDaXin drawing attributes from an output expression.

        Examples:
          - "EMA(C,10),COLORRED,LINETHICK2" -> "EMA(C,10)"
          - "LOW,COLORYELLOW,LINETHICK0" -> "LOW"
        """
        expr = expr.strip()
        paren_depth = 0
        for i, char in enumerate(expr):
            if char == '(':
                paren_depth += 1
            elif char == ')':
                paren_depth -= 1
            elif char == ',' and paren_depth == 0:
                return expr[:i].strip()
        return expr

    def _split_draw_attrs(self, expr: str) -> tuple[str, list[str]]:
        expr = expr.strip()
        paren_depth = 0
        for i, char in enumerate(expr):
            if char == '(':
                paren_depth += 1
            elif char == ')':
                paren_depth -= 1
            elif char == ',' and paren_depth == 0:
                main = expr[:i].strip()
                attrs = [a.strip().upper() for a in expr[i + 1 :].split(",") if a.strip()]
                return main, attrs
        return expr, []

    def parse_outputs(self, ctx: FormulaContext) -> list[FormulaOutput]:
        """解析并执行公式，返回所有输出表达式（含命名输出与绘图属性）。"""
        outputs: list[FormulaOutput] = []
        statements = self._split_statements(self.formula)

        for stmt in statements:
            stmt = stmt.strip()
            if not stmt:
                continue

            if ':=' in stmt:
                var_name, expr = stmt.split(':=', 1)
                var_name = var_name.strip()
                expr_main, _attrs = self._split_draw_attrs(expr)
                value = self._eval_expr(expr_main, ctx)
                ctx.set_var(var_name, value)
                continue

            if ':' in stmt and not stmt.startswith(':'):
                colon_pos = self._find_colon_outside_parens(stmt)
                if colon_pos > 0:
                    name = stmt[:colon_pos].strip()
                    expr_main, attrs = self._split_draw_attrs(stmt[colon_pos + 1 :])
                    value = self._eval_expr(expr_main, ctx)
                    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
                        ctx.set_var(name, value)
                    outputs.append(FormulaOutput(name=name or None, series=value, draw_attrs=attrs))
                    continue

            expr_main, attrs = self._split_draw_attrs(stmt)
            value = self._eval_expr(expr_main, ctx)
            outputs.append(FormulaOutput(name=None, series=value, draw_attrs=attrs))

        return outputs

    def _split_statements(self, formula: str) -> list[str]:
        """分割语句（按分号）"""
        statements = []
        current = []
        paren_depth = 0

        for char in formula:
            if char == '(':
                paren_depth += 1
                current.append(char)
            elif char == ')':
                paren_depth -= 1
                current.append(char)
            elif char == ';' and paren_depth == 0:
                statements.append(''.join(current))
                current = []
            else:
                current.append(char)

        if current:
            statements.append(''.join(current))

        return statements

    def _eval_expr(self, expr: str, ctx: FormulaContext) -> pd.Series:
        """计算表达式"""
        expr = expr.strip()

        # 处理 OR 运算符（最低优先级）
        if self._has_operator(expr, 'OR'):
            parts = self._split_by_operator(expr, 'OR')
            result = self._eval_expr(parts[0], ctx)
            for part in parts[1:]:
                result = result | self._eval_expr(part, ctx)
            return result

        # 处理 AND 运算符
        if self._has_operator(expr, 'AND'):
            parts = self._split_by_operator(expr, 'AND')
            result = self._eval_expr(parts[0], ctx)
            for part in parts[1:]:
                result = result & self._eval_expr(part, ctx)
            return result

        # 处理 NOT 运算符
        if expr.upper().startswith('NOT ') or expr.upper().startswith('NOT('):
            inner = expr[3:].strip()
            if inner.startswith('(') and inner.endswith(')'):
                inner = inner[1:-1]
            return ~self._eval_expr(inner, ctx).astype(bool)

        # 处理比较运算符
        for op, op_str in [('>=', '>='), ('<=', '<='), ('<>', '<>'), ('>', '>'), ('<', '<'), ('=', '=')]:
            if self._has_operator(expr, op):
                parts = self._split_by_operator(expr, op)
                if len(parts) == 2:
                    left = self._eval_expr(parts[0], ctx)
                    right = self._eval_expr(parts[1], ctx)
                    use_eps = self._is_numeric_series(left) and self._is_numeric_series(right)
                    eps = self.FLOAT_EPS
                    if op == '>=':
                        return left >= (right - eps) if use_eps else left >= right
                    elif op == '<=':
                        return left <= (right + eps) if use_eps else left <= right
                    elif op == '<>':
                        return (left - right).abs() > eps if use_eps else left != right
                    elif op == '>':
                        return left > (right - eps) if use_eps else left > right
                    elif op == '<':
                        return left < (right + eps) if use_eps else left < right
                    elif op == '=':
                        return (left - right).abs() <= eps if use_eps else left == right

        # 处理加减运算符
        if self._has_operator(expr, '+') or self._has_operator(expr, '-'):
            # 找到最后一个 + 或 - (从右向左，处理左结合)
            idx = self._find_last_operator(expr, ['+', '-'])
            if idx > 0:
                left = self._eval_expr(expr[:idx], ctx)
                op = expr[idx]
                right = self._eval_expr(expr[idx+1:], ctx)
                if op == '+':
                    return left + right
                else:
                    return left - right

        # 处理乘除运算符
        if self._has_operator(expr, '*') or self._has_operator(expr, '/'):
            idx = self._find_last_operator(expr, ['*', '/'])
            if idx > 0:
                left = self._eval_expr(expr[:idx], ctx)
                op = expr[idx]
                right = self._eval_expr(expr[idx+1:], ctx)
                if op == '*':
                    return left * right
                else:
                    return left / right

        # 处理括号
        if expr.startswith('(') and expr.endswith(')'):
            return self._eval_expr(expr[1:-1], ctx)

        # 处理函数调用
        func_match = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)$', expr, re.DOTALL)
        if func_match:
            func_name = func_match.group(1).upper()
            args_str = func_match.group(2)
            return self._call_function(func_name, args_str, ctx)

        # 处理数字
        try:
            num = float(expr)
            return pd.Series(num, index=ctx.df.index)
        except ValueError:
            pass

        # 处理变量
        return ctx.get_var(expr)

    def _has_operator(self, expr: str, op: str) -> bool:
        """检查表达式中是否有指定运算符（在括号外）"""
        paren_depth = 0
        op_len = len(op)
        expr_upper = expr.upper()
        op_upper = op.upper()

        i = 0
        while i < len(expr):
            if expr[i] == '(':
                paren_depth += 1
            elif expr[i] == ')':
                paren_depth -= 1
            elif paren_depth == 0:
                if expr_upper[i:i+op_len] == op_upper:
                    # 符号运算符（例如 >=, <=, <>）允许与变量/数字紧挨；单字符符号同理。
                    # 仅对单词运算符（AND/OR 等）做边界检查，避免误匹配变量名的一部分。
                    is_symbol_op = not op_upper.isalpha()
                    if (op_len == 1 and op in '+-*/<>=') or is_symbol_op:
                        return True
                    else:
                        before_ok = i == 0 or not expr[i-1].isalnum()
                        after_ok = i + op_len >= len(expr) or not expr[i+op_len].isalnum()
                        if before_ok and after_ok:
                            return True
            i += 1
        return False

    def _split_by_operator(self, expr: str, op: str) -> list[str]:
        """按运算符分割表达式"""
        parts = []
        current = []
        paren_depth = 0
        op_len = len(op)
        expr_upper = expr.upper()
        op_upper = op.upper()
        is_symbol_op = not op_upper.isalpha()

        i = 0
        while i < len(expr):
            if expr[i] == '(':
                paren_depth += 1
                current.append(expr[i])
            elif expr[i] == ')':
                paren_depth -= 1
                current.append(expr[i])
            elif paren_depth == 0 and expr_upper[i:i+op_len] == op_upper:
                # 对于单字符运算符不需要检查边界
                if (op_len == 1 and op in '+-*/<>=') or is_symbol_op:
                    parts.append(''.join(current))
                    current = []
                    i += op_len
                    continue
                else:
                    before_ok = i == 0 or not expr[i-1].isalnum()
                    after_ok = i + op_len >= len(expr) or not expr[i+op_len].isalnum()
                    if before_ok and after_ok:
                        parts.append(''.join(current))
                        current = []
                        i += op_len
                        continue
                    else:
                        current.append(expr[i])
            else:
                current.append(expr[i])
            i += 1

        if current:
            parts.append(''.join(current))

        return parts

    def _find_last_operator(self, expr: str, ops: list[str]) -> int:
        """找到最后一个运算符的位置（括号外）"""
        paren_depth = 0
        last_idx = -1

        for i, char in enumerate(expr):
            if char == '(':
                paren_depth += 1
            elif char == ')':
                paren_depth -= 1
            elif paren_depth == 0 and char in ops:
                # 排除负号（前面是运算符或开头）
                if char == '-' and (i == 0 or expr[i-1] in '(,+-*/<>='):
                    continue
                last_idx = i

        return last_idx

    def _is_numeric_series(self, series: pd.Series) -> bool:
        if not isinstance(series, pd.Series):
            return False
        if is_bool_dtype(series.dtype):
            return False
        return is_numeric_dtype(series.dtype)

    def _series_from_scalar(self, value: Any, ctx: FormulaContext) -> pd.Series:
        if isinstance(value, pd.Series):
            return value
        if isinstance(value, bool):
            return pd.Series(bool(value), index=ctx.df.index)
        if isinstance(value, (int, float)):
            return pd.Series(value, index=ctx.df.index)
        raise ValueError(f"无法将参数转换为序列: {value!r}")

    def _int_from_arg(self, value: Any) -> int:
        if isinstance(value, bool):
            raise ValueError("参数必须为整数")
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            if value.is_integer():
                return int(value)
            raise ValueError("参数必须为整数")
        raise ValueError("参数必须为整数")

    def _int_from_series(self, series: pd.Series) -> int:
        non_null = series.dropna()
        if non_null.empty:
            raise ValueError("参数必须为整数")
        uniq = pd.unique(non_null)
        if len(uniq) != 1:
            raise ValueError("参数必须为常量整数")
        raw = uniq[0]
        # numpy scalar to python primitive
        if hasattr(raw, "item"):
            raw = raw.item()
        return self._int_from_arg(raw)

    def _call_function(self, func_name: str, args_str: str, ctx: FormulaContext) -> pd.Series:
        """调用函数"""
        if func_name not in self.FUNCTIONS:
            raise ValueError(f"未知函数: {func_name}")

        func = self.FUNCTIONS[func_name]
        args = self._parse_args(args_str, ctx)

        int_positions = self.INT_PARAMS.get(func_name, set())
        coerced: list[Any] = []
        for i, arg in enumerate(args):
            if i in int_positions:
                if isinstance(arg, pd.Series):
                    coerced.append(self._int_from_series(arg))
                else:
                    coerced.append(self._int_from_arg(arg))
            else:
                # For vectorized functions, numeric literals should behave like constant series.
                coerced.append(self._series_from_scalar(arg, ctx))

        try:
            return func(*coerced)
        except Exception as e:
            raise ValueError(f"函数 {func_name} 调用失败: {e}")

    def _parse_args(self, args_str: str, ctx: FormulaContext) -> list[Any]:
        """解析函数参数"""
        args = []
        current = []
        paren_depth = 0

        for char in args_str:
            if char == '(':
                paren_depth += 1
                current.append(char)
            elif char == ')':
                paren_depth -= 1
                current.append(char)
            elif char == ',' and paren_depth == 0:
                arg_str = ''.join(current).strip()
                args.append(self._eval_arg(arg_str, ctx))
                current = []
            else:
                current.append(char)

        if current:
            arg_str = ''.join(current).strip()
            args.append(self._eval_arg(arg_str, ctx))

        return args

    def _eval_arg(self, arg_str: str, ctx: FormulaContext) -> Any:
        """计算参数值"""
        arg_str = arg_str.strip()

        # 尝试解析为数字
        try:
            if '.' in arg_str:
                return float(arg_str)
            return int(arg_str)
        except ValueError:
            pass

        # 否则作为表达式计算
        return self._eval_expr(arg_str, ctx)


def execute_formula(
    formula: str,
    df: pd.DataFrame,
    *,
    prefer_output_name: str | None = None,
    output_selector: Literal["last", "first"] = "last",
) -> pd.Series:
    """执行公式并返回结果"""
    ctx = FormulaContext(df=df)
    parser = FormulaParser(formula)
    return parser.parse(ctx, prefer_output_name=prefer_output_name, output_selector=output_selector)


def execute_formula_outputs(formula: str, df: pd.DataFrame) -> list[FormulaOutput]:
    """执行公式并返回所有输出（用于指标绘图等场景）。"""
    ctx = FormulaContext(df=df)
    parser = FormulaParser(formula)
    return parser.parse_outputs(ctx)


def validate_formula(formula: str) -> tuple[bool, str]:
    """验证公式语法是否正确"""
    try:
        # 创建一个小的测试 DataFrame
        test_df = pd.DataFrame({
            'open': [1.0, 2.0, 3.0, 4.0, 5.0],
            'high': [1.1, 2.1, 3.1, 4.1, 5.1],
            'low': [0.9, 1.9, 2.9, 3.9, 4.9],
            'close': [1.0, 2.0, 3.0, 4.0, 5.0],
            'vol': [100, 200, 300, 400, 500],
            'amount': [1000, 2000, 3000, 4000, 5000],
        })
        execute_formula(formula, test_df)
        return True, "公式语法正确"
    except Exception as e:
        return False, str(e)
