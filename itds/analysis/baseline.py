from __future__ import annotations

from datetime import datetime

from itds.storage.sqlite import SqliteStore


class BaselineModel:
    """Student-scale baselines: rolling per-user averages + "first-seen" sets (IP/resource)."""

    def __init__(self, cfg: dict, store: SqliteStore):
        self._cfg = cfg
        self._store = store

    def update_with_event(self, e: dict) -> None:
        # Keep it lightweight: update daily counters for a few metrics.
        ts = e["ts"]
        day = ts[:10]  # YYYY-MM-DD
        user = e["user"]

        # Metrics written as daily totals; anomaly uses recent history window (computed in detector).
        # For demo simplicity, these metrics are updated via incremental fetch-free approach is omitted.
        # We rely on detector computing its window stats from events table.

        # Placeholder daily metrics (set to 0 if not used yet)
        for metric in ("events_total", "deny_total", "download_bytes_total"):
            if self._store.get_daily_metric(day, user, metric) is None:
                self._store.upsert_daily_metric(day, user, metric, 0.0)
