from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from dateutil import parser as dtparser

from itds.storage.sqlite import SqliteStore


@dataclass(frozen=True)
class Finding:
    kind: str  # rule | anomaly
    name: str
    weight: float
    evidence: dict


class Detector:
    def __init__(self, cfg: dict, store: SqliteStore, baseline):
        self._cfg = cfg
        self._store = store
        self._baseline = baseline

    def score_and_maybe_alert(self, e: dict) -> dict | None:
        findings: list[Finding] = []

        findings.extend(self._apply_rules(e))
        findings.extend(self._apply_simple_anomaly_checks(e))

        risk = self._aggregate_risk(e, findings)
        if risk["level"] == "low":
            return None

        return {
            "ts": e["ts"],
            "user": e["user"],
            "level": risk["level"],
            "score": risk["score"],
            "event": e,
            "explanations": [
                {
                    "kind": f.kind,
                    "name": f.name,
                    "weight": f.weight,
                    "evidence": f.evidence,
                }
                for f in findings
            ],
        }

    def _apply_rules(self, e: dict) -> list[Finding]:
        rules_cfg = self._cfg.get("rules", {})
        out: list[Finding] = []

        # R1: after-hours activity
        if self._is_after_hours(e["ts"], rules_cfg.get("working_hours", {})):
            out.append(
                Finding(
                    kind="rule",
                    name="after_hours_activity",
                    weight=float(rules_cfg.get("after_hours_weight", 10)),
                    evidence={"ts": e["ts"], "working_hours": rules_cfg.get("working_hours")},
                )
            )

        # R2: privilege escalation / sensitive command
        if e.get("event_type") == "sudo" and e.get("resource"):
            cmd = str(e["resource"]).lower()
            if "/etc/shadow" in cmd or "user=root" in cmd or "chmod" in cmd:
                out.append(
                    Finding(
                        kind="rule",
                        name="privilege_escalation_or_sensitive_sudo",
                        weight=float(rules_cfg.get("privilege_escalation_weight", 30)),
                        evidence={"command": e.get("resource")},
                    )
                )

        # R3: large download
        if e.get("action") == "download" and isinstance(e.get("bytes"), int):
            mb = e["bytes"] / (1024 * 1024)
            if mb >= float(rules_cfg.get("large_data_threshold_mb", 200)):
                out.append(
                    Finding(
                        kind="rule",
                        name="large_download",
                        weight=float(rules_cfg.get("large_download_weight", 35)),
                        evidence={"bytes": e["bytes"], "mb": round(mb, 2), "resource": e.get("resource")},
                    )
                )

        # R4: repeated denies (burst) - evaluated per event using last N minutes window
        if e.get("status") == "fail":
            deny_count_10m = self._count_recent_denies(e["user"], e["ts"], minutes=10)
            if deny_count_10m >= int(rules_cfg.get("deny_burst_threshold", 10)):
                out.append(
                    Finding(
                        kind="rule",
                        name="deny_burst_10m",
                        weight=float(rules_cfg.get("deny_burst_weight", 25)),
                        evidence={"deny_count_10m": deny_count_10m, "window_minutes": 10},
                    )
                )

        # R5: new IP / new resource (first-seen)
        if e.get("ip"):
            is_new_ip = self._store.remember_set_item(e["user"], "ip", e["ip"], e["ts"])
            if is_new_ip:
                out.append(
                    Finding(
                        kind="rule",
                        name="new_ip_first_seen",
                        weight=float(rules_cfg.get("new_ip_weight", 20)),
                        evidence={"ip": e["ip"]},
                    )
                )

        if e.get("resource"):
            is_new_res = self._store.remember_set_item(e["user"], "resource", e["resource"], e["ts"])
            if is_new_res:
                out.append(
                    Finding(
                        kind="rule",
                        name="new_resource_first_seen",
                        weight=float(rules_cfg.get("new_resource_weight", 15)),
                        evidence={"resource": e["resource"]},
                    )
                )

        return out

    def _apply_simple_anomaly_checks(self, e: dict) -> list[Finding]:
        # Explainable: compute event-rate deviation for this user in last 60 minutes vs typical.
        # For student-scale demo: typical rate is estimated from last 7 days average.
        z_thr = float(self._cfg.get("anomaly", {}).get("zscore_threshold", 3.0))

        # skip if we don't have a usable timestamp
        try:
            now_dt = _parse_iso_z(e["ts"])
        except Exception:
            return []

        user = e["user"]
        rate_60m = self._count_events(user, e["ts"], minutes=60)
        mean, std = self._estimate_user_hourly_rate_stats(user, now_dt)

        if std <= 1e-6:
            return []

        z = (rate_60m - mean) / std
        if z >= z_thr:
            return [
                Finding(
                    kind="anomaly",
                    name="spike_in_activity_60m",
                    weight=min(30.0, 10.0 * float(z)),
                    evidence={"rate_60m": rate_60m, "mean": mean, "std": std, "z": round(z, 2)},
                )
            ]

        return []

    def _aggregate_risk(self, e: dict, findings: list[Finding]) -> dict:
        base = sum(f.weight for f in findings)

        # simple decay based on time since event (for streaming it would reference last alert time)
        half_life_h = float(self._cfg.get("risk", {}).get("decay_half_life_hours", 24))
        decay = 1.0
        try:
            dt = _parse_iso_z(e["ts"])  # event time
            age_h = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0)
            decay = 0.5 ** (age_h / half_life_h)
        except Exception:
            decay = 1.0

        score = max(0.0, min(100.0, base * decay))
        hi = float(self._cfg.get("risk", {}).get("threshold_high", 70))
        med = float(self._cfg.get("risk", {}).get("threshold_medium", 45))

        if score >= hi:
            level = "high"
        elif score >= med:
            level = "medium"
        else:
            level = "low"

        return {"score": round(score, 2), "level": level}

    def _is_after_hours(self, ts: str, wh: dict) -> bool:
        # assumes ts is Z
        dt = _parse_iso_z(ts)
        hhmm = dt.strftime("%H:%M")
        start = wh.get("start", "09:00")
        end = wh.get("end", "18:00")
        return hhmm < start or hhmm > end

    def _count_recent_denies(self, user: str, ts: str, minutes: int) -> int:
        end_unix = _to_unix(ts)
        start_unix = max(0, end_unix - minutes * 60)
        with self._store._conn() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS c
                FROM events
                WHERE user=?
                  AND status='fail'
                  AND ts_unix >= ?
                  AND ts_unix <= ?
                """,
                (user, start_unix, end_unix),
            ).fetchone()
        return int(row["c"]) if row else 0

    def _count_events(self, user: str, ts: str, minutes: int) -> int:
        end_unix = _to_unix(ts)
        start_unix = max(0, end_unix - minutes * 60)
        with self._store._conn() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS c
                FROM events
                WHERE user=?
                  AND ts_unix >= ?
                  AND ts_unix <= ?
                """,
                (user, start_unix, end_unix),
            ).fetchone()
        return int(row["c"]) if row else 0

    def _estimate_user_hourly_rate_stats(self, user: str, now_dt: datetime) -> tuple[float, float]:
        # Approximation: last 7 days, count events per hour bucket, compute mean/std.
        cutoff_unix = int(now_dt.timestamp()) - 7 * 24 * 3600

        with self._store._conn() as conn:
            rows = conn.execute(
                """
                SELECT (ts_unix / 3600) AS hour_bucket, COUNT(*) AS c
                FROM events
                WHERE user=?
                  AND ts_unix >= ?
                GROUP BY hour_bucket
                """,
                (user, cutoff_unix),
            ).fetchall()

        counts = [int(r["c"]) for r in rows] if rows else []
        if len(counts) < 3:
            return (5.0, 3.0)  # safe demo defaults

        mean = sum(counts) / len(counts)
        var = sum((x - mean) ** 2 for x in counts) / max(1, len(counts) - 1)
        return (round(mean, 2), round(math.sqrt(var), 2))


def _parse_iso_z(ts: str) -> datetime:
    # '2026-01-27T10:20:30Z'
    if ts.endswith("Z"):
        ts = ts.replace("Z", "+00:00")
    return datetime.fromisoformat(ts).astimezone(timezone.utc)


def _to_unix(ts: str) -> int:
    try:
        return int(dtparser.isoparse(ts).timestamp())
    except Exception:
        # fallback for already-normalized timestamps that end with Z
        try:
            return int(_parse_iso_z(ts).timestamp())
        except Exception:
            return 0
