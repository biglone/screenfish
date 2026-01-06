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
from typing import Any, Callable

import pandas as pd

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


class FormulaParser:
    """通达信公式解析器"""

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
        self.formula = formula.strip()
        self.pos = 0
        self.length = len(self.formula)

    def parse(self, ctx: FormulaContext) -> pd.Series:
        """解析并执行公式，返回最后一个表达式的结果"""
        result = None
        statements = self._split_statements(self.formula)

        for stmt in statements:
            stmt = stmt.strip()
            if not stmt:
                continue

            # 检查是否是赋值语句 (VAR:=表达式)
            if ':=' in stmt:
                var_name, expr = stmt.split(':=', 1)
                var_name = var_name.strip()
                value = self._eval_expr(expr.strip(), ctx)
                ctx.set_var(var_name, value)
            # 检查是否是命名输出语句 (NAME:表达式，但不是 :=)
            elif ':' in stmt and not stmt.startswith(':'):
                # 找到第一个 : 的位置（排除括号内的）
                colon_pos = self._find_colon_outside_parens(stmt)
                if colon_pos > 0:
                    # 这是命名输出
                    expr = stmt[colon_pos + 1:].strip()
                    result = self._eval_expr(expr, ctx)
                else:
                    # 普通输出语句
                    result = self._eval_expr(stmt, ctx)
            else:
                # 普通输出语句
                result = self._eval_expr(stmt, ctx)

        return result if result is not None else pd.Series(True, index=ctx.df.index)

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
                    if op == '>=':
                        return left >= right
                    elif op == '<=':
                        return left <= right
                    elif op == '<>':
                        return left != right
                    elif op == '>':
                        return left > right
                    elif op == '<':
                        return left < right
                    elif op == '=':
                        return left == right

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
                    # 对于单字符运算符（+, -, *, /, <, >, =）不需要检查边界
                    # 对于多字符运算符（AND, OR, >=, <= 等）需要确保不是变量名的一部分
                    if op_len == 1 and op in '+-*/<>=':
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
                if op_len == 1 and op in '+-*/<>=':
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

    def _call_function(self, func_name: str, args_str: str, ctx: FormulaContext) -> pd.Series:
        """调用函数"""
        if func_name not in self.FUNCTIONS:
            raise ValueError(f"未知函数: {func_name}")

        func = self.FUNCTIONS[func_name]
        args = self._parse_args(args_str, ctx)

        # 将数字参数转换为 Series（用于需要 Series 作为第一个参数的函数）
        # 例如 REF(0, 1) 应该返回常量 0 的序列
        if args and isinstance(args[0], (int, float)) and not isinstance(args[0], bool):
            args[0] = pd.Series(args[0], index=ctx.df.index)

        try:
            return func(*args)
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


def execute_formula(formula: str, df: pd.DataFrame) -> pd.Series:
    """执行公式并返回结果"""
    ctx = FormulaContext(df=df)
    parser = FormulaParser(formula)
    return parser.parse(ctx)


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
