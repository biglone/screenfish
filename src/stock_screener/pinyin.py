from __future__ import annotations

from functools import lru_cache

try:
    from pypinyin import Style, lazy_pinyin  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    lazy_pinyin = None  # type: ignore[assignment]
    Style = None  # type: ignore[assignment]


@lru_cache(maxsize=20000)
def pinyin_full(text: str) -> str | None:
    if lazy_pinyin is None:
        return None
    return "".join(lazy_pinyin(text)).lower()


@lru_cache(maxsize=20000)
def pinyin_initials(text: str) -> str | None:
    if lazy_pinyin is None:
        return None
    return "".join(lazy_pinyin(text, style=Style.FIRST_LETTER)).lower()  # type: ignore[union-attr]

