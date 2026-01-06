from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class Settings(BaseModel):
    cache_dir: Path = Field(default=Path("./data"))
    data_backend: Literal["sqlite", "parquet"] = Field(default="sqlite")

    @property
    def sqlite_path(self) -> Path:
        return self.cache_dir / "daily.sqlite3"

