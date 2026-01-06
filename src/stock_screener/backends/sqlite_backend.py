from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pandas as pd


@dataclass(frozen=True)
class SqliteBackend:
    path: Path

    def init(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS daily (
                  ts_code TEXT NOT NULL,
                  trade_date TEXT NOT NULL,
                  open REAL,
                  high REAL,
                  low REAL,
                  close REAL,
                  vol REAL,
                  amount REAL,
                  PRIMARY KEY (ts_code, trade_date)
                );
                CREATE INDEX IF NOT EXISTS idx_daily_trade_date ON daily (trade_date);

                CREATE TABLE IF NOT EXISTS update_log (
                  trade_date TEXT PRIMARY KEY,
                  updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS stock_basic (
                  ts_code TEXT PRIMARY KEY,
                  name TEXT,
                  updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS provider_stock_progress (
                  provider TEXT NOT NULL,
                  range_start TEXT NOT NULL,
                  range_end TEXT NOT NULL,
                  ts_code TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  PRIMARY KEY (provider, range_start, range_end, ts_code)
                );
                """
            )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.path))
        try:
            conn.row_factory = sqlite3.Row
            yield conn
            conn.commit()
        finally:
            conn.close()

    def get_updated_trade_dates(self, trade_dates: Sequence[str]) -> set[str]:
        if not trade_dates:
            return set()
        placeholders = ",".join(["?"] * len(trade_dates))
        query = f"SELECT trade_date FROM update_log WHERE trade_date IN ({placeholders})"
        with self.connect() as conn:
            rows = conn.execute(query, list(trade_dates)).fetchall()
        return {row["trade_date"] for row in rows}

    def mark_trade_date_updated(self, trade_date: str) -> None:
        with self.connect() as conn:
            self.mark_trade_date_updated_in_conn(conn, trade_date)

    def mark_trade_date_updated_in_conn(self, conn: sqlite3.Connection, trade_date: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT OR REPLACE INTO update_log (trade_date, updated_at) VALUES (?, ?)",
            (trade_date, now),
        )

    def upsert_daily_rows(self, rows: Iterable[tuple]) -> None:
        with self.connect() as conn:
            self.upsert_daily_rows_in_conn(conn, rows)

    def upsert_daily_rows_in_conn(self, conn: sqlite3.Connection, rows: Iterable[tuple]) -> None:
        conn.executemany(
            """
            INSERT OR REPLACE INTO daily
              (ts_code, trade_date, open, high, low, close, vol, amount)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def upsert_daily_df(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        with self.connect() as conn:
            self.upsert_daily_df_in_conn(conn, df)

    def upsert_daily_df_in_conn(self, conn: sqlite3.Connection, df: pd.DataFrame) -> None:
        if df.empty:
            return
        required = ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"missing columns in daily df: {missing}")
        rows = (
            df[required]
            .astype({"ts_code": "string", "trade_date": "string"})
            .itertuples(index=False, name=None)
        )
        self.upsert_daily_rows_in_conn(conn, rows)

    def load_daily_between(self, start: str, end: str) -> pd.DataFrame:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT ts_code, trade_date, open, high, low, close, vol, amount
                FROM daily
                WHERE trade_date >= ? AND trade_date <= ?
                ORDER BY ts_code ASC, trade_date ASC
                """,
                (start, end),
            ).fetchall()
        return pd.DataFrame(rows, columns=rows[0].keys() if rows else None)

    def load_daily_lookback(self, end: str, start: str) -> pd.DataFrame:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT ts_code, trade_date, open, high, low, close, vol, amount
                FROM daily
                WHERE trade_date >= ? AND trade_date <= ?
                ORDER BY ts_code ASC, trade_date ASC
                """,
                (start, end),
            ).fetchall()
        return pd.DataFrame(rows, columns=rows[0].keys() if rows else None)

    def get_progress_ts_codes(self, *, provider: str, range_start: str, range_end: str) -> set[str]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT ts_code
                FROM provider_stock_progress
                WHERE provider = ? AND range_start = ? AND range_end = ?
                """,
                (provider, range_start, range_end),
            ).fetchall()
        return {row["ts_code"] for row in rows}

    def upsert_stock_basic_df(self, df: pd.DataFrame) -> None:
        with self.connect() as conn:
            self.upsert_stock_basic_df_in_conn(conn, df)

    def upsert_stock_basic_df_in_conn(self, conn: sqlite3.Connection, df: pd.DataFrame) -> None:
        if df.empty:
            return
        required = ["ts_code", "name"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"missing columns in stock_basic df: {missing}")
        now = datetime.now(timezone.utc).isoformat()
        tmp = df[required].copy()
        tmp["ts_code"] = tmp["ts_code"].astype(str)
        tmp["name"] = tmp["name"].astype(object).where(tmp["name"].notna(), None)
        tmp["updated_at"] = now
        rows = tmp[["ts_code", "name", "updated_at"]].itertuples(index=False, name=None)
        conn.executemany(
            "INSERT OR REPLACE INTO stock_basic (ts_code, name, updated_at) VALUES (?, ?, ?)",
            rows,
        )

    def load_stock_names(self, ts_codes: Sequence[str]) -> dict[str, str]:
        if not ts_codes:
            return {}
        placeholders = ",".join(["?"] * len(ts_codes))
        query = f"SELECT ts_code, name FROM stock_basic WHERE ts_code IN ({placeholders})"
        with self.connect() as conn:
            rows = conn.execute(query, list(ts_codes)).fetchall()
        out: dict[str, str] = {}
        for r in rows:
            if r["name"] is None:
                continue
            out[str(r["ts_code"])] = str(r["name"])
        return out

    def max_trade_date_in_daily(self) -> str | None:
        with self.connect() as conn:
            row = conn.execute("SELECT MAX(trade_date) AS d FROM daily").fetchone()
        if not row:
            return None
        return row["d"]

    def max_trade_date_in_update_log(self) -> str | None:
        with self.connect() as conn:
            row = conn.execute("SELECT MAX(trade_date) AS d FROM update_log").fetchone()
        if not row:
            return None
        return row["d"]

    def mark_progress_ts_code_in_conn(
        self,
        conn: sqlite3.Connection,
        *,
        provider: str,
        range_start: str,
        range_end: str,
        ts_code: str,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """
            INSERT OR REPLACE INTO provider_stock_progress
              (provider, range_start, range_end, ts_code, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (provider, range_start, range_end, ts_code, now),
        )

    def distinct_ts_codes_in_range(self, *, start: str, end: str) -> set[str]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT ts_code
                FROM daily
                WHERE trade_date >= ? AND trade_date <= ?
                """,
                (start, end),
            ).fetchall()
        return {row["ts_code"] for row in rows}
