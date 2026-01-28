from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class NormalizedEvent(BaseModel):
    # required
    ts: str = Field(description="ISO-8601 UTC timestamp")
    user: str
    host: str | None = None
    source: str
    event_type: Literal[
        "auth",
        "sudo",
        "app_access",
        "file_access",
        "db_access",
        "unknown",
    ]
    action: str
    status: str

    # optional enrichments
    ip: Optional[str] = None
    resource: Optional[str] = None
    bytes: Optional[int] = None
    extra: dict[str, Any] = Field(default_factory=dict)
