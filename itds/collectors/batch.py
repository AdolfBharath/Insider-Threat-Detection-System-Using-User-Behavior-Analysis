from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True)
class RawEvent:
    source_name: str
    source_format: str
    line: str


class BatchCollector:
    def __init__(self, cfg: dict):
        self._cfg = cfg

    def collect(self) -> Iterator[RawEvent]:
        for src in self._cfg.get("ingestion", {}).get("sources", []):
            if src.get("type") != "file":
                continue
            path = Path(src["path"])
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                yield RawEvent(source_name=src["name"], source_format=src["format"], line=line)
