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
