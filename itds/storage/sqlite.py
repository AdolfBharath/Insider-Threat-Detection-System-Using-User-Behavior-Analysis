from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

from dateutil import parser as dtparser


class SqliteStore:
    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    ts_unix INTEGER NOT NULL,
                    user TEXT NOT NULL,
                    source TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    action TEXT NOT NULL,
                    status TEXT NOT NULL,
                    ip TEXT,
                    host TEXT,
                    resource TEXT,
                    bytes INTEGER,
                    norm_json TEXT NOT NULL,
                    raw_line TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_events_user_ts ON events(user, ts);
                CREATE INDEX IF NOT EXISTS idx_events_user_ts_unix ON events(user, ts_unix);

                CREATE TABLE IF NOT EXISTS user_stats_daily (
                    day TEXT NOT NULL,
                    user TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    value REAL NOT NULL,
                    PRIMARY KEY(day, user, metric)
                );

                CREATE TABLE IF NOT EXISTS user_sets (
                    user TEXT NOT NULL,
                    set_name TEXT NOT NULL,
                    item TEXT NOT NULL,
                    first_seen_ts TEXT NOT NULL,
                    PRIMARY KEY(user, set_name, item)
                );
                """
            )

            # Lightweight migration for older DBs created before ts_unix existed.
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(events)").fetchall()}
            if "ts_unix" not in cols:
                conn.execute("ALTER TABLE events ADD COLUMN ts_unix INTEGER NOT NULL DEFAULT 0")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_events_user_ts_unix ON events(user, ts_unix)")

    def insert_event(self, raw_event: Any, norm_event: dict[str, Any]) -> None:
        ts_unix = 0
        try:
            ts_unix = int(dtparser.isoparse(str(norm_event.get("ts"))).timestamp())
        except Exception:
            ts_unix = 0

        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO events(ts, ts_unix, user, source, event_type, action, status, ip, host, resource, bytes, norm_json, raw_line)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    norm_event.get("ts"),
                    ts_unix,
                    norm_event.get("user"),
                    norm_event.get("source"),
                    norm_event.get("event_type"),
                    norm_event.get("action"),
                    norm_event.get("status"),
                    norm_event.get("ip"),
                    norm_event.get("host"),
                    norm_event.get("resource"),
                    norm_event.get("bytes"),
                    json.dumps(norm_event, ensure_ascii=False),
                    getattr(raw_event, "line", str(raw_event)),
                ),
            )

    def upsert_daily_metric(self, day: str, user: str, metric: str, value: float) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO user_stats_daily(day, user, metric, value)
                VALUES(?,?,?,?)
                ON CONFLICT(day, user, metric) DO UPDATE SET value=excluded.value
                """,
                (day, user, metric, value),
            )

    def get_daily_metric(self, day: str, user: str, metric: str) -> Optional[float]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value FROM user_stats_daily WHERE day=? AND user=? AND metric=?",
                (day, user, metric),
            ).fetchone()
        return float(row["value"]) if row else None

    def remember_set_item(self, user: str, set_name: str, item: str, first_seen_ts: str) -> bool:
        """Returns True if item is new (not previously seen)."""
        with self._conn() as conn:
            try:
                conn.execute(
                    "INSERT INTO user_sets(user, set_name, item, first_seen_ts) VALUES(?,?,?,?)",
                    (user, set_name, item, first_seen_ts),
                )
                return True
            except sqlite3.IntegrityError:
                return False
