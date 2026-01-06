from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


class Rule(Protocol):
    name: str

    def enrich(self, df: pd.DataFrame) -> None: ...

    def mask(self, df: pd.DataFrame) -> pd.Series: ...


@dataclass(frozen=True)
class RuleColumns:
    ma60: str = "ma60"
    mid_bullbear: str = "mid_bullbear"
    j: str = "j"

