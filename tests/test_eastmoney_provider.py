import pandas as pd
import pytest

from stock_screener.providers.eastmoney_provider import EastmoneyProvider, ts_code_to_secid


def test_ts_code_to_secid() -> None:
    assert ts_code_to_secid("600000.SH") == "1.600000"
    assert ts_code_to_secid("000001.SZ") == "0.000001"
    assert ts_code_to_secid("920809.BJ") == "0.920809"
    with pytest.raises(ValueError):
        ts_code_to_secid("920809")


def test_fetch_daily_parses_klines(monkeypatch) -> None:
    provider = EastmoneyProvider()

    sample = {
        "rc": 0,
        "data": {
            "name": "安达科技",
            "klines": [
                "2023-03-23,12.00,11.23,12.06,10.79,314505,357125690.03",
                "2023-03-24,11.17,11.03,11.45,10.84,108435,120649591.35",
            ],
        },
    }

    def fake_get_json(self, *, params):  # noqa: ANN001
        assert params["secid"] == "0.920809"
        assert params["klt"] == "101"
        assert "fields1" in params
        assert "fields2" in params
        return sample

    monkeypatch.setattr(EastmoneyProvider, "_get_json", fake_get_json)

    df, name = provider.fetch_daily(ts_code="920809.BJ", start="20230323", end="20230324", adjust="none")
    assert name == "安达科技"
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"]
    assert df.iloc[0]["ts_code"] == "920809.BJ"
    assert df.iloc[0]["trade_date"] == "20230323"
    assert float(df.iloc[0]["open"]) == 12.0
