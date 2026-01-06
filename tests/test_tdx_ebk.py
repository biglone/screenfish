from pathlib import Path

from stock_screener.tdx import ts_code_to_ebk_code, write_ebk


def test_ts_code_to_ebk_code() -> None:
    assert ts_code_to_ebk_code("000001.SZ") == "0000001"
    assert ts_code_to_ebk_code("600000.SH") == "1600000"
    assert ts_code_to_ebk_code("920223.BJ") == "2920223"


def test_write_ebk_format(tmp_path: Path) -> None:
    out = tmp_path / "x.EBK"
    write_ebk(["000001.SZ", "600000.SH", "000001.SZ"], out)
    b = out.read_bytes()
    assert b.startswith(b"\r\n")
    assert b.decode("ascii") == "\r\n0000001\r\n1600000"

