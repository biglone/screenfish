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

                CREATE TABLE IF NOT EXISTS formulas (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL UNIQUE,
                  formula TEXT NOT NULL,
                  description TEXT,
                  enabled INTEGER NOT NULL DEFAULT 1,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );
                """
            )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        # Use a longer timeout to tolerate transient lock contention (e.g. concurrent readers/writers).
        conn = sqlite3.connect(str(self.path), timeout=30)
        try:
            conn.row_factory = sqlite3.Row
            # These PRAGMAs are per-connection; ensure consistent behavior across all connections.
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=30000;")
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

    def count_daily_rows_for_trade_date(self, trade_date: str) -> int:
        with self.connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM daily WHERE trade_date = ?", (trade_date,)).fetchone()
        return int(row["n"]) if row else 0

    def clear_progress(self, *, provider: str, range_start: str, range_end: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "DELETE FROM provider_stock_progress WHERE provider = ? AND range_start = ? AND range_end = ?",
                (provider, range_start, range_end),
            )

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

    # Formula CRUD methods

    def create_formula(
        self,
        *,
        name: str,
        formula: str,
        description: str | None = None,
        enabled: bool = True,
    ) -> dict:
        """创建新公式"""
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO formulas (name, formula, description, enabled, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (name, formula, description, 1 if enabled else 0, now, now),
            )
            formula_id = cursor.lastrowid
        return self.get_formula(formula_id)

    def get_formula(self, formula_id: int) -> dict | None:
        """获取单个公式"""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM formulas WHERE id = ?",
                (formula_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "name": row["name"],
            "formula": row["formula"],
            "description": row["description"],
            "enabled": bool(row["enabled"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def get_formula_by_name(self, name: str) -> dict | None:
        """根据名称获取公式"""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM formulas WHERE name = ?",
                (name,),
            ).fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "name": row["name"],
            "formula": row["formula"],
            "description": row["description"],
            "enabled": bool(row["enabled"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def list_formulas(self, *, enabled_only: bool = False) -> list[dict]:
        """列出所有公式"""
        with self.connect() as conn:
            if enabled_only:
                rows = conn.execute(
                    "SELECT * FROM formulas WHERE enabled = 1 ORDER BY id ASC"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM formulas ORDER BY id ASC"
                ).fetchall()
        return [
            {
                "id": row["id"],
                "name": row["name"],
                "formula": row["formula"],
                "description": row["description"],
                "enabled": bool(row["enabled"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def update_formula(
        self,
        formula_id: int,
        *,
        name: str | None = None,
        formula: str | None = None,
        description: str | None = None,
        enabled: bool | None = None,
    ) -> dict | None:
        """更新公式"""
        existing = self.get_formula(formula_id)
        if not existing:
            return None

        updates = []
        params = []
        if name is not None:
            updates.append("name = ?")
            params.append(name)
        if formula is not None:
            updates.append("formula = ?")
            params.append(formula)
        if description is not None:
            updates.append("description = ?")
            params.append(description)
        if enabled is not None:
            updates.append("enabled = ?")
            params.append(1 if enabled else 0)

        if not updates:
            return existing

        now = datetime.now(timezone.utc).isoformat()
        updates.append("updated_at = ?")
        params.append(now)
        params.append(formula_id)

        with self.connect() as conn:
            conn.execute(
                f"UPDATE formulas SET {', '.join(updates)} WHERE id = ?",
                params,
            )
        return self.get_formula(formula_id)

    def delete_formula(self, formula_id: int) -> bool:
        """删除公式"""
        with self.connect() as conn:
            cursor = conn.execute(
                "DELETE FROM formulas WHERE id = ?",
                (formula_id,),
            )
        return cursor.rowcount > 0
