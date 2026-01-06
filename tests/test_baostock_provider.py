from stock_screener.providers.baostock_provider import BaoStockProvider


class _DummyLoginResult:
    def __init__(self, error_code: str = "0", error_msg: str = "") -> None:
        self.error_code = error_code
        self.error_msg = error_msg


class _DummyResultSet:
    def __init__(
        self,
        *,
        error_code: str = "0",
        error_msg: str = "",
        fields: list[str] | None = None,
        rows: list[list[str]] | None = None,
    ) -> None:
        self.error_code = error_code
        self.error_msg = error_msg
        self.fields = fields or []
        self._rows = rows or []
        self._idx = -1

    def next(self) -> bool:
        self._idx += 1
        return self._idx < len(self._rows)

    def get_row_data(self) -> list[str]:
        return self._rows[self._idx]


class _DummyBS:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self._query_calls = 0

    def login(self) -> _DummyLoginResult:
        self.calls.append("login")
        return _DummyLoginResult("0", "login success")

    def logout(self) -> _DummyLoginResult:
        self.calls.append("logout")
        return _DummyLoginResult("0", "logout success")

    def query_history_k_data_plus(self, code: str, *_args, **_kwargs) -> _DummyResultSet:
        self._query_calls += 1
        if self._query_calls == 1:
            return _DummyResultSet(error_code="10001001", error_msg="用户未登录")
        return _DummyResultSet(
            error_code="0",
            fields=["date", "code", "open", "high", "low", "close", "volume", "amount"],
            rows=[["1990-12-19", code, "1", "2", "0.5", "1.5", "100", "1000"]],
        )


def test_fetch_daily_ranges_relogin_on_user_not_logged_in() -> None:
    p = BaoStockProvider()
    bs = _DummyBS()
    df = p._fetch_daily_ranges(bs=bs, bs_code="sh.600000", ranges=[("19901219", "19901219")])
    assert bs.calls == ["logout", "login"]
    assert df.shape[0] == 1
    assert df.loc[0, "ts_code"] == "600000.SH"
    assert df.loc[0, "trade_date"] == "19901219"
    assert float(df.loc[0, "close"]) == 1.5

