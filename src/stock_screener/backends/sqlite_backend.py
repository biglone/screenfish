from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import re
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pandas as pd


@dataclass(frozen=True)
class SqliteBackend:
    path: Path
    daily_table: str = "daily"
    update_log_table: str = "update_log"
    provider_stock_progress_table: str = "provider_stock_progress"

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

                CREATE TABLE IF NOT EXISTS daily_qfq (
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
                CREATE INDEX IF NOT EXISTS idx_daily_qfq_trade_date ON daily_qfq (trade_date);

                CREATE TABLE IF NOT EXISTS daily_hfq (
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
                CREATE INDEX IF NOT EXISTS idx_daily_hfq_trade_date ON daily_hfq (trade_date);

                CREATE TABLE IF NOT EXISTS update_log (
                  trade_date TEXT PRIMARY KEY,
                  updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS update_log_qfq (
                  trade_date TEXT PRIMARY KEY,
                  updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS update_log_hfq (
                  trade_date TEXT PRIMARY KEY,
                  updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS stock_basic (
                  ts_code TEXT PRIMARY KEY,
                  name TEXT,
                  pinyin_initials TEXT,
                  pinyin_full TEXT,
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

                CREATE TABLE IF NOT EXISTS provider_stock_progress_qfq (
                  provider TEXT NOT NULL,
                  range_start TEXT NOT NULL,
                  range_end TEXT NOT NULL,
                  ts_code TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  PRIMARY KEY (provider, range_start, range_end, ts_code)
                );

                CREATE TABLE IF NOT EXISTS provider_stock_progress_hfq (
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
                  kind TEXT NOT NULL DEFAULT 'screen',
                  timeframe TEXT,
                  enabled INTEGER NOT NULL DEFAULT 1,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS watchlist_groups (
                  owner_id TEXT NOT NULL,
                  id TEXT NOT NULL,
                  name TEXT NOT NULL,
                  created_at INTEGER NOT NULL,
                  updated_at INTEGER NOT NULL,
                  PRIMARY KEY (owner_id, id)
                );
                CREATE INDEX IF NOT EXISTS idx_watchlist_groups_owner_updated_at
                  ON watchlist_groups (owner_id, updated_at DESC);

                CREATE TABLE IF NOT EXISTS watchlist_items (
                  owner_id TEXT NOT NULL,
                  group_id TEXT NOT NULL,
                  ts_code TEXT NOT NULL,
                  name TEXT,
                  created_at INTEGER NOT NULL,
                  updated_at INTEGER NOT NULL,
                  PRIMARY KEY (owner_id, group_id, ts_code)
                );
                CREATE INDEX IF NOT EXISTS idx_watchlist_items_owner_group_updated_at
                  ON watchlist_items (owner_id, group_id, updated_at DESC);

                CREATE TABLE IF NOT EXISTS users (
                  id TEXT PRIMARY KEY,
                  username TEXT NOT NULL UNIQUE,
                  email TEXT,
                  password_hash TEXT NOT NULL,
                  password_salt TEXT NOT NULL,
                  role TEXT NOT NULL DEFAULT 'user',
                  disabled INTEGER NOT NULL DEFAULT 0,
                  token_version INTEGER NOT NULL DEFAULT 0,
                  created_at INTEGER NOT NULL,
                  updated_at INTEGER NOT NULL,
                  last_login_at INTEGER,
                  last_login_ip TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_users_username ON users (username);

                CREATE TABLE IF NOT EXISTS email_verification_codes (
                  email TEXT PRIMARY KEY,
                  code_hash TEXT NOT NULL,
                  code_salt TEXT NOT NULL,
                  expires_at INTEGER NOT NULL,
                  created_at INTEGER NOT NULL,
                  updated_at INTEGER NOT NULL,
                  send_count INTEGER NOT NULL DEFAULT 0,
                  last_sent_at INTEGER NOT NULL DEFAULT 0,
                  attempts INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_email_verification_codes_expires_at
                  ON email_verification_codes (expires_at);

                CREATE TABLE IF NOT EXISTS auto_update_config (
                  id INTEGER PRIMARY KEY CHECK (id = 1),
                  enabled INTEGER NOT NULL DEFAULT 0,
                  interval_seconds INTEGER NOT NULL DEFAULT 600,
                  provider TEXT NOT NULL DEFAULT 'baostock',
                  repair_days INTEGER NOT NULL DEFAULT 30,

                  -- Optional: auto screen settings (run after successful auto update)
                  screen_enabled INTEGER NOT NULL DEFAULT 0,
                  screen_combo TEXT NOT NULL DEFAULT 'and',
                  screen_rules TEXT,
                  screen_lookback_days INTEGER NOT NULL DEFAULT 200,
                  screen_with_name INTEGER NOT NULL DEFAULT 0,
                  screen_exclude_st INTEGER NOT NULL DEFAULT 0,
                  screen_price_adjust TEXT NOT NULL DEFAULT 'qfq',
                  screen_owner_id TEXT,
                  screen_group_name TEXT NOT NULL DEFAULT '自动筛选',
                  screen_group_id TEXT,
                  screen_replace_group INTEGER NOT NULL DEFAULT 1,

                  -- Auto screen runtime status
                  screen_last_run_at INTEGER,
                  screen_last_trade_date TEXT,
                  screen_last_count INTEGER,
                  screen_last_error TEXT,

                  -- Auto update runtime status (best-effort; for UI visibility)
                  run_status TEXT NOT NULL DEFAULT 'idle',
                  run_started_at INTEGER,
                  run_target_trade_date TEXT,
                  run_mode TEXT,
                  run_message TEXT,

                  last_run_at INTEGER,
                  last_success_at INTEGER,
                  last_success_trade_date TEXT,
                  last_error TEXT,
                  updated_at INTEGER NOT NULL DEFAULT 0
                );
                INSERT OR IGNORE INTO auto_update_config
                  (id, enabled, interval_seconds, provider, repair_days, updated_at)
                VALUES (1, 0, 600, 'baostock', 30, 0);
                """
            )
            self._migrate(conn)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Best-effort schema migrations for existing databases."""

        def _has_column(table: str, column: str) -> bool:
            rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
            return any(r["name"] == column for r in rows)

        # Auto update config: add run-state columns for older DBs.
        if not _has_column("auto_update_config", "run_status"):
            conn.execute("ALTER TABLE auto_update_config ADD COLUMN run_status TEXT NOT NULL DEFAULT 'idle'")
        if not _has_column("auto_update_config", "run_started_at"):
            conn.execute("ALTER TABLE auto_update_config ADD COLUMN run_started_at INTEGER")
        if not _has_column("auto_update_config", "run_target_trade_date"):
            conn.execute("ALTER TABLE auto_update_config ADD COLUMN run_target_trade_date TEXT")
        if not _has_column("auto_update_config", "run_mode"):
            conn.execute("ALTER TABLE auto_update_config ADD COLUMN run_mode TEXT")
        if not _has_column("auto_update_config", "run_message"):
            conn.execute("ALTER TABLE auto_update_config ADD COLUMN run_message TEXT")

        # Formulas: add kind/timeframe columns for older DBs.
        if not _has_column("formulas", "kind"):
            conn.execute("ALTER TABLE formulas ADD COLUMN kind TEXT NOT NULL DEFAULT 'screen'")
        if not _has_column("formulas", "timeframe"):
            conn.execute("ALTER TABLE formulas ADD COLUMN timeframe TEXT")

        # Data migration: keep the built-in indicator aligned with TongDaXin's default display:
        # MA2 is an assignment (":=") and should not be plotted as an output line (":").
        try:
            row = conn.execute(
                "SELECT id, formula FROM formulas WHERE name = ? AND kind = 'indicator' LIMIT 1",
                ("执行中期多空线",),
            ).fetchone()
            if row:
                formula_id = int(row["id"])
                formula_raw = str(row["formula"] or "")

                has_ma2_assign = re.search(
                    r"\bMA2\s*:=\s*EMA\s*\(\s*CLOSE\s*,\s*13\s*\)\s*;?",
                    formula_raw,
                    flags=re.IGNORECASE,
                ) is not None
                ma13_mentions = re.findall(r"\bMA13\b", formula_raw, flags=re.IGNORECASE)
                has_ma13_alias_output = (
                    re.search(r"(?m)^\s*MA13\s*:\s*MA2\s*;\s*$", formula_raw, flags=re.IGNORECASE) is not None
                )

                if has_ma2_assign and has_ma13_alias_output and len(ma13_mentions) == 1:
                    updated_formula = re.sub(
                        r"(?m)^\s*MA13\s*:\s*MA2\s*;\s*\n?",
                        "",
                        formula_raw,
                        count=1,
                        flags=re.IGNORECASE,
                    ).rstrip() + "\n"
                    if updated_formula != formula_raw:
                        now = datetime.now(timezone.utc).isoformat()
                        conn.execute(
                            "UPDATE formulas SET formula = ?, updated_at = ? WHERE id = ?",
                            (updated_formula, now, formula_id),
                        )
        except Exception:
            # Best-effort only; never fail startup because of formula migrations.
            pass

        # Users: add email column and unique index.
        if _has_column("users", "id") and not _has_column("users", "email"):
            conn.execute("ALTER TABLE users ADD COLUMN email TEXT")
        if _has_column("users", "email"):
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email_unique
                  ON users (email)
                  WHERE email IS NOT NULL
                """
            )
        if _has_column("users", "id") and not _has_column("users", "token_version"):
            conn.execute("ALTER TABLE users ADD COLUMN token_version INTEGER NOT NULL DEFAULT 0")
        if _has_column("users", "id") and not _has_column("users", "last_login_at"):
            conn.execute("ALTER TABLE users ADD COLUMN last_login_at INTEGER")
        if _has_column("users", "id") and not _has_column("users", "last_login_ip"):
            conn.execute("ALTER TABLE users ADD COLUMN last_login_ip TEXT")

        # stock_basic: add pinyin cache columns for fast search.
        if _has_column("stock_basic", "ts_code") and not _has_column("stock_basic", "pinyin_initials"):
            conn.execute("ALTER TABLE stock_basic ADD COLUMN pinyin_initials TEXT")
        if _has_column("stock_basic", "ts_code") and not _has_column("stock_basic", "pinyin_full"):
            conn.execute("ALTER TABLE stock_basic ADD COLUMN pinyin_full TEXT")
        if _has_column("stock_basic", "pinyin_initials"):
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_stock_basic_pinyin_initials ON stock_basic (pinyin_initials)"
            )
        if _has_column("stock_basic", "pinyin_full"):
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_stock_basic_pinyin_full ON stock_basic (pinyin_full)"
            )

        # auto_update_config: add runtime status columns for older DBs.
        if _has_column("auto_update_config", "id") and not _has_column("auto_update_config", "last_run_at"):
            conn.execute("ALTER TABLE auto_update_config ADD COLUMN last_run_at INTEGER")
        if _has_column("auto_update_config", "id") and not _has_column("auto_update_config", "last_success_at"):
            conn.execute("ALTER TABLE auto_update_config ADD COLUMN last_success_at INTEGER")
        if _has_column("auto_update_config", "id") and not _has_column("auto_update_config", "last_success_trade_date"):
            conn.execute("ALTER TABLE auto_update_config ADD COLUMN last_success_trade_date TEXT")
        if _has_column("auto_update_config", "id") and not _has_column("auto_update_config", "last_error"):
            conn.execute("ALTER TABLE auto_update_config ADD COLUMN last_error TEXT")
        if _has_column("auto_update_config", "id") and not _has_column("auto_update_config", "updated_at"):
            conn.execute("ALTER TABLE auto_update_config ADD COLUMN updated_at INTEGER NOT NULL DEFAULT 0")

        # auto_update_config: auto screen settings + status columns.
        if _has_column("auto_update_config", "id") and not _has_column("auto_update_config", "screen_enabled"):
            conn.execute("ALTER TABLE auto_update_config ADD COLUMN screen_enabled INTEGER NOT NULL DEFAULT 0")
        if _has_column("auto_update_config", "id") and not _has_column("auto_update_config", "screen_combo"):
            conn.execute("ALTER TABLE auto_update_config ADD COLUMN screen_combo TEXT NOT NULL DEFAULT 'and'")
        if _has_column("auto_update_config", "id") and not _has_column("auto_update_config", "screen_rules"):
            conn.execute("ALTER TABLE auto_update_config ADD COLUMN screen_rules TEXT")
        if _has_column("auto_update_config", "id") and not _has_column("auto_update_config", "screen_lookback_days"):
            conn.execute("ALTER TABLE auto_update_config ADD COLUMN screen_lookback_days INTEGER NOT NULL DEFAULT 200")
        if _has_column("auto_update_config", "id") and not _has_column("auto_update_config", "screen_with_name"):
            conn.execute("ALTER TABLE auto_update_config ADD COLUMN screen_with_name INTEGER NOT NULL DEFAULT 0")
        if _has_column("auto_update_config", "id") and not _has_column("auto_update_config", "screen_exclude_st"):
            conn.execute("ALTER TABLE auto_update_config ADD COLUMN screen_exclude_st INTEGER NOT NULL DEFAULT 0")
        if _has_column("auto_update_config", "id") and not _has_column("auto_update_config", "screen_price_adjust"):
            conn.execute("ALTER TABLE auto_update_config ADD COLUMN screen_price_adjust TEXT NOT NULL DEFAULT 'qfq'")
        if _has_column("auto_update_config", "id") and not _has_column("auto_update_config", "screen_owner_id"):
            conn.execute("ALTER TABLE auto_update_config ADD COLUMN screen_owner_id TEXT")
        if _has_column("auto_update_config", "id") and not _has_column("auto_update_config", "screen_group_name"):
            conn.execute("ALTER TABLE auto_update_config ADD COLUMN screen_group_name TEXT NOT NULL DEFAULT '自动筛选'")
        if _has_column("auto_update_config", "id") and not _has_column("auto_update_config", "screen_group_id"):
            conn.execute("ALTER TABLE auto_update_config ADD COLUMN screen_group_id TEXT")
        if _has_column("auto_update_config", "id") and not _has_column("auto_update_config", "screen_replace_group"):
            conn.execute("ALTER TABLE auto_update_config ADD COLUMN screen_replace_group INTEGER NOT NULL DEFAULT 1")
        if _has_column("auto_update_config", "id") and not _has_column("auto_update_config", "screen_last_run_at"):
            conn.execute("ALTER TABLE auto_update_config ADD COLUMN screen_last_run_at INTEGER")
        if _has_column("auto_update_config", "id") and not _has_column("auto_update_config", "screen_last_trade_date"):
            conn.execute("ALTER TABLE auto_update_config ADD COLUMN screen_last_trade_date TEXT")
        if _has_column("auto_update_config", "id") and not _has_column("auto_update_config", "screen_last_count"):
            conn.execute("ALTER TABLE auto_update_config ADD COLUMN screen_last_count INTEGER")
        if _has_column("auto_update_config", "id") and not _has_column("auto_update_config", "screen_last_error"):
            conn.execute("ALTER TABLE auto_update_config ADD COLUMN screen_last_error TEXT")

        try:
            from stock_screener.pinyin import pinyin_full as _pinyin_full
            from stock_screener.pinyin import pinyin_initials as _pinyin_initials
        except Exception:
            return

        if _pinyin_initials("平安银行") is None or _pinyin_full("平安银行") is None:
            return

        try:
            rows = conn.execute(
                """
                SELECT ts_code, name
                FROM stock_basic
                WHERE name IS NOT NULL
                  AND (pinyin_initials IS NULL OR pinyin_full IS NULL)
                """,
            ).fetchall()
        except sqlite3.OperationalError:
            return

        updates: list[tuple[str | None, str | None, str]] = []
        for row in rows:
            name = row["name"]
            if name is None:
                continue
            name_str = str(name)
            updates.append(
                (
                    _pinyin_initials(name_str),
                    _pinyin_full(name_str),
                    str(row["ts_code"]),
                )
            )
        if updates:
            conn.executemany(
                "UPDATE stock_basic SET pinyin_initials = ?, pinyin_full = ? WHERE ts_code = ?",
                updates,
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
        query = f"SELECT trade_date FROM {self.update_log_table} WHERE trade_date IN ({placeholders})"
        with self.connect() as conn:
            rows = conn.execute(query, list(trade_dates)).fetchall()
        return {row["trade_date"] for row in rows}

    def mark_trade_date_updated(self, trade_date: str) -> None:
        with self.connect() as conn:
            self.mark_trade_date_updated_in_conn(conn, trade_date)

    def mark_trade_date_updated_in_conn(self, conn: sqlite3.Connection, trade_date: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            f"INSERT OR REPLACE INTO {self.update_log_table} (trade_date, updated_at) VALUES (?, ?)",
            (trade_date, now),
        )

    def upsert_daily_rows(self, rows: Iterable[tuple]) -> None:
        with self.connect() as conn:
            self.upsert_daily_rows_in_conn(conn, rows)

    def upsert_daily_rows_in_conn(self, conn: sqlite3.Connection, rows: Iterable[tuple]) -> None:
        conn.executemany(
            f"""
            INSERT OR REPLACE INTO {self.daily_table}
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
                f"""
                SELECT ts_code, trade_date, open, high, low, close, vol, amount
                FROM {self.daily_table}
                WHERE trade_date >= ? AND trade_date <= ?
                ORDER BY ts_code ASC, trade_date ASC
                """,
                (start, end),
            ).fetchall()
        return pd.DataFrame(rows, columns=rows[0].keys() if rows else None)

    def load_daily_lookback(self, end: str, start: str) -> pd.DataFrame:
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT ts_code, trade_date, open, high, low, close, vol, amount
                FROM {self.daily_table}
                WHERE trade_date >= ? AND trade_date <= ?
                ORDER BY ts_code ASC, trade_date ASC
                """,
                (start, end),
            ).fetchall()
        return pd.DataFrame(rows, columns=rows[0].keys() if rows else None)

    def get_progress_ts_codes(self, *, provider: str, range_start: str, range_end: str) -> set[str]:
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT ts_code
                FROM {self.provider_stock_progress_table}
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
        try:
            from stock_screener.pinyin import pinyin_full as _pinyin_full
            from stock_screener.pinyin import pinyin_initials as _pinyin_initials
        except Exception:
            _pinyin_full = None  # type: ignore[assignment]
            _pinyin_initials = None  # type: ignore[assignment]

        now = datetime.now(timezone.utc).isoformat()
        tmp = df[required].copy()
        tmp["ts_code"] = tmp["ts_code"].astype(str)
        tmp["name"] = tmp["name"].astype(object).where(tmp["name"].notna(), None)
        if _pinyin_initials is None or _pinyin_full is None:
            tmp["pinyin_initials"] = None
            tmp["pinyin_full"] = None
        else:
            tmp["pinyin_initials"] = tmp["name"].apply(lambda x: _pinyin_initials(str(x)) if x else None)
            tmp["pinyin_full"] = tmp["name"].apply(lambda x: _pinyin_full(str(x)) if x else None)
        tmp["updated_at"] = now
        rows = tmp[["ts_code", "name", "pinyin_initials", "pinyin_full", "updated_at"]].itertuples(index=False, name=None)
        conn.executemany(
            """
            INSERT INTO stock_basic (ts_code, name, pinyin_initials, pinyin_full, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(ts_code) DO UPDATE SET
              name = COALESCE(excluded.name, stock_basic.name),
              pinyin_initials = COALESCE(excluded.pinyin_initials, stock_basic.pinyin_initials),
              pinyin_full = COALESCE(excluded.pinyin_full, stock_basic.pinyin_full),
              updated_at = excluded.updated_at
            """,
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
            row = conn.execute(f"SELECT MAX(trade_date) AS d FROM {self.daily_table}").fetchone()
        if not row:
            return None
        return row["d"]

    def max_trade_date_in_update_log(self) -> str | None:
        with self.connect() as conn:
            row = conn.execute(f"SELECT MAX(trade_date) AS d FROM {self.update_log_table}").fetchone()
        if not row:
            return None
        return row["d"]

    def count_daily_rows_for_trade_date(self, trade_date: str) -> int:
        with self.connect() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) AS n FROM {self.daily_table} WHERE trade_date = ?",
                (trade_date,),
            ).fetchone()
        return int(row["n"]) if row else 0

    def clear_progress(self, *, provider: str, range_start: str, range_end: str) -> None:
        with self.connect() as conn:
            conn.execute(
                f"DELETE FROM {self.provider_stock_progress_table} WHERE provider = ? AND range_start = ? AND range_end = ?",
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
            f"""
            INSERT OR REPLACE INTO {self.provider_stock_progress_table}
              (provider, range_start, range_end, ts_code, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (provider, range_start, range_end, ts_code, now),
        )

    def distinct_ts_codes_in_range(self, *, start: str, end: str) -> set[str]:
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT DISTINCT ts_code
                FROM {self.daily_table}
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
        kind: str = "screen",
        timeframe: str | None = None,
        enabled: bool = True,
    ) -> dict:
        """创建新公式"""
        if kind != "indicator":
            timeframe = None
        if kind == "indicator" and timeframe is None:
            timeframe = "D"
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO formulas (name, formula, description, kind, timeframe, enabled, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (name, formula, description, kind, timeframe, 1 if enabled else 0, now, now),
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
            "kind": row["kind"],
            "timeframe": row["timeframe"],
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
            "kind": row["kind"],
            "timeframe": row["timeframe"],
            "enabled": bool(row["enabled"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def list_formulas(self, *, enabled_only: bool = False, kind: str | None = None) -> list[dict]:
        """列出所有公式"""
        with self.connect() as conn:
            query = "SELECT * FROM formulas"
            params: list[object] = []
            where: list[str] = []
            if enabled_only:
                where.append("enabled = 1")
            if kind is not None:
                where.append("kind = ?")
                params.append(kind)
            if where:
                query += " WHERE " + " AND ".join(where)
            query += " ORDER BY id ASC"
            rows = conn.execute(query, params).fetchall()
        return [
            {
                "id": row["id"],
                "name": row["name"],
                "formula": row["formula"],
                "description": row["description"],
                "kind": row["kind"],
                "timeframe": row["timeframe"],
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
        kind: str | None = None,
        timeframe: str | None = None,
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
        if kind is not None:
            updates.append("kind = ?")
            params.append(kind)
            # Keep timeframe consistent when switching kind.
            if kind != "indicator":
                timeframe = None
            elif timeframe is None and existing.get("timeframe") is None:
                timeframe = "D"
        if timeframe is not None or (kind is not None and kind != "indicator"):
            updates.append("timeframe = ?")
            params.append(timeframe)
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
