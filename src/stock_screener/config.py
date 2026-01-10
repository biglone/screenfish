from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class Settings(BaseModel):
    cache_dir: Path = Field(default=Path("./data"))
    data_backend: Literal["sqlite", "parquet"] = Field(default="sqlite")
    price_adjust: Literal["none", "qfq", "hfq"] = Field(default="none")

    @property
    def sqlite_path(self) -> Path:
        return self.cache_dir / "daily.sqlite3"

    @property
    def daily_table(self) -> str:
        if self.price_adjust == "qfq":
            return "daily_qfq"
        if self.price_adjust == "hfq":
            return "daily_hfq"
        return "daily"

    @property
    def update_log_table(self) -> str:
        if self.price_adjust == "qfq":
            return "update_log_qfq"
        if self.price_adjust == "hfq":
            return "update_log_hfq"
        return "update_log"

    @property
    def provider_stock_progress_table(self) -> str:
        if self.price_adjust == "qfq":
            return "provider_stock_progress_qfq"
        if self.price_adjust == "hfq":
            return "provider_stock_progress_hfq"
        return "provider_stock_progress"

    @property
    def baostock_adjustflag(self) -> str:
        # BaoStock: adjustflag -> 1: 后复权, 2: 前复权, 3: 不复权
        if self.price_adjust == "qfq":
            return "2"
        if self.price_adjust == "hfq":
            return "1"
        return "3"
