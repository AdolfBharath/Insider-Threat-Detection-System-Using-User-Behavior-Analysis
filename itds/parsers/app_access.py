from __future__ import annotations

from dateutil import parser as dtparser

from itds.collectors.batch import RawEvent


def parse_app_access(raw: RawEvent) -> dict:
    # Format: '2026-01-27T10:20:30Z user=alice action=download resource=/x bytes=123 ip=1.2.3.4 status=200'
    tokens = raw.line.split()
    if not tokens:
        return {"event_type": "unknown", "raw": raw.line}

    ts = dtparser.parse(tokens[0]).isoformat().replace("+00:00", "Z")
    fields = {"ts": ts, "event_type": "app_access", "action": "unknown", "status": "unknown"}

    for t in tokens[1:]:
        if "=" not in t:
            continue
        k, v = t.split("=", 1)
        fields[k] = v

    user = fields.get("user", "unknown")
    status_code = str(fields.get("status", "unknown"))
    status = "success" if status_code.startswith("2") else "fail"

    return {
        "ts": ts,
        "user": user,
        "ip": fields.get("ip"),
        "event_type": "app_access",
        "action": fields.get("action", "unknown"),
        "status": status,
        "resource": fields.get("resource"),
        "bytes": int(fields["bytes"]) if "bytes" in fields and fields["bytes"].isdigit() else None,
        "extra": {"http_status": status_code},
    }
