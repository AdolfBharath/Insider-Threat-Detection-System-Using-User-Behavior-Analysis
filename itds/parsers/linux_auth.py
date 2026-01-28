from __future__ import annotations

import re
from datetime import datetime, timezone

from dateutil import parser as dtparser

from itds.collectors.batch import RawEvent


_SSH_ACCEPT = re.compile(r"Accepted \w+ for (?P<user>\S+) from (?P<ip>\S+)")
_SSH_FAIL = re.compile(r"Failed \w+ for (invalid user )?(?P<user>\S+) from (?P<ip>\S+)")
_SUDO = re.compile(r"sudo:\s+(?P<user>\S+)\s*:.*COMMAND=(?P<cmd>.+)$")


def parse_linux_auth(raw: RawEvent) -> dict:
    # Sample: 'Jan 27 10:05:12 host1 sshd[1011]: Accepted password for alice from 10.0.0.10 ...'
    parts = raw.line.split()
    if len(parts) < 5:
        return {"event_type": "unknown", "raw": raw.line}

    month, day, time_str = parts[0], parts[1], parts[2]
    host = parts[3]
    rest = " ".join(parts[4:])

    # year inferred for demo purposes
    year = datetime.now().year
    ts = dtparser.parse(f"{year} {month} {day} {time_str}").replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")

    m = _SSH_ACCEPT.search(rest)
    if m:
        return {
            "ts": ts,
            "host": host,
            "user": m.group("user"),
            "ip": m.group("ip"),
            "event_type": "auth",
            "action": "login_success",
            "status": "success",
        }

    m = _SSH_FAIL.search(rest)
    if m:
        return {
            "ts": ts,
            "host": host,
            "user": m.group("user"),
            "ip": m.group("ip"),
            "event_type": "auth",
            "action": "login_failed",
            "status": "fail",
        }

    m = _SUDO.search(rest)
    if m:
        return {
            "ts": ts,
            "host": host,
            "user": m.group("user"),
            "event_type": "sudo",
            "action": "sudo_command",
            "status": "success",
            "resource": m.group("cmd").strip(),
        }

    return {"ts": ts, "host": host, "event_type": "unknown", "action": "unknown", "status": "unknown", "raw": raw.line}
