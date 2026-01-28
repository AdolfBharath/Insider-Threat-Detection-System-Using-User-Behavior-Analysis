from __future__ import annotations

from itds.collectors.batch import RawEvent
from itds.normalization.schema import NormalizedEvent
from itds.parsers.app_access import parse_app_access
from itds.parsers.linux_auth import parse_linux_auth


def normalize_event(raw: RawEvent) -> dict:
    if raw.source_format == "linux_auth":
        parsed = parse_linux_auth(raw)
    elif raw.source_format == "app_access":
        parsed = parse_app_access(raw)
    else:
        parsed = {"event_type": "unknown", "action": "unknown", "status": "unknown", "raw": raw.line}

    # unify keys and validate with pydantic (for predictable downstream logic)
    base = {
        "ts": parsed.get("ts", "1970-01-01T00:00:00Z"),
        "user": parsed.get("user", "unknown"),
        "host": parsed.get("host"),
        "source": raw.source_name,
        "event_type": parsed.get("event_type", "unknown"),
        "action": parsed.get("action", "unknown"),
        "status": parsed.get("status", "unknown"),
        "ip": parsed.get("ip"),
        "resource": parsed.get("resource"),
        "bytes": parsed.get("bytes"),
        "extra": parsed.get("extra", {}),
    }

    evt = NormalizedEvent(**base)
    return evt.model_dump()
