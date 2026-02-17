from __future__ import annotations

import json
from pathlib import Path

from rich import print


class AlertSink:
    def __init__(self, cfg: dict):
        self._cfg = cfg

    def emit(self, alert: dict) -> None:
        sink = self._cfg.get("alerting", {}).get("sink", "console")
        out_path = Path(self._cfg.get("alerting", {}).get("json_out", "data/out/alerts.jsonl"))
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Always append JSONL for audit/demo even when printing to console.
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(alert, ensure_ascii=False) + "\n")

        if sink == "json":
            return

        # console: human-readable, explainable
        print(f"[bold red]ALERT[/bold red] level={alert['level']} score={alert['score']} user={alert['user']} ts={alert['ts']}")
        for ex in alert.get("explanations", []):
            evidence = ex.get("evidence") or {}
            if ex.get("kind") == "dl" and isinstance(evidence, dict) and evidence.get("reason"):
                short = evidence.get("reason_short")
                if short:
                    print(f"  - (dl) {short}")
                else:
                    print(f"  - (dl) {evidence['reason']}")
            else:
                print(f"  - ({ex['kind']}) {ex['name']} weight={ex['weight']} evidence={evidence}")
