from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TdxEbkFormat:
    """
    Observed from TongDaXin exported .EBK sample:
    - ASCII text with CRLF line terminators
    - starts with an empty line (leading CRLF)
    - each subsequent line is a 7-digit code:
        0 + 6-digit for SZ
        1 + 6-digit for SH
        2 + 6-digit for BJ
    - final line may omit trailing CRLF
    """

    leading_crlf: bool = True
    trailing_crlf: bool = False


def ts_code_to_ebk_code(ts_code: str) -> str:
    """
    Convert `000001.SZ` / `600000.SH` / `920001.BJ` to TDX EBK 7-digit code.
    """

    if "." not in ts_code:
        raise ValueError(f"invalid ts_code: {ts_code!r}")
    code, market = ts_code.split(".", 1)
    code = code.strip()
    market = market.strip().upper()
    if len(code) != 6 or not code.isdigit():
        raise ValueError(f"invalid ts_code code: {ts_code!r}")

    if market == "SZ":
        return "0" + code
    if market == "SH":
        return "1" + code
    if market == "BJ":
        return "2" + code
    raise ValueError(f"unsupported market for ts_code: {ts_code!r}")


def write_ebk(ts_codes: list[str], out_path: Path, *, fmt: TdxEbkFormat | None = None) -> None:
    fmt = fmt or TdxEbkFormat()
    seen: set[str] = set()
    ebk_lines: list[str] = []
    for ts_code in ts_codes:
        ebk = ts_code_to_ebk_code(ts_code)
        if ebk in seen:
            continue
        seen.add(ebk)
        ebk_lines.append(ebk)

    if not ebk_lines and fmt.leading_crlf:
        content = "\r\n"
    else:
        body = "\r\n".join(ebk_lines)
        content = ("\r\n" if fmt.leading_crlf else "") + body + ("\r\n" if fmt.trailing_crlf else "")
    out_path.write_bytes(content.encode("ascii"))

