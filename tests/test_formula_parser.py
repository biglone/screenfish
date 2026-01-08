from stock_screener.formula_parser import validate_formula


def test_validate_formula_supports_tdx_params_and_window_vars() -> None:
    formula = """{KDJ超跌选股公式 - 筛选J值低于13的股票}
JT:13;
N:=9;
M1:=3;
M2:=3;

RSV:=(CLOSE-LLV(LOW,N))/(HHV(HIGH,N)-LLV(LOW,N))*100;
K:=SMA(RSV,M1,1);
D:=SMA(K,M2,1);
J:=3*K-2*D;

J<JT AND REF(J,1)<JT;
"""

    ok, msg = validate_formula(formula)
    assert ok, msg


def test_validate_formula_allows_numeric_constants_in_vector_funcs() -> None:
    formula = "IF(CLOSE>OPEN, 1, 0);"
    ok, msg = validate_formula(formula)
    assert ok, msg


def test_validate_formula_ignores_tdx_draw_attrs() -> None:
    formula = """
执行中期多空线:EMA(EMA(C,10),10),COLORFFFFFF,LINETHICK1;
最低价:LOW,COLORYELLOW,LINETHICK0;
"""
    ok, msg = validate_formula(formula)
    assert ok, msg
