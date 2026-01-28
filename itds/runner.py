from __future__ import annotations

from pathlib import Path

import json

from itds.utils.config import load_config
from itds.storage.sqlite import SqliteStore
from itds.collectors.batch import BatchCollector
from itds.normalization.normalizer import normalize_event
from itds.analysis.baseline import BaselineModel
from itds.analysis.detector import Detector
from itds.alerting.sink import AlertSink


def run_pipeline(config_path: Path) -> None:
    cfg = load_config(config_path)

    normalized_path = Path(cfg["paths"].get("normalized_jsonl", "data/out/normalized.jsonl"))
    normalized_path.parent.mkdir(parents=True, exist_ok=True)

    store = SqliteStore(Path(cfg["paths"]["sqlite_db"]))
    store.init_schema()

    collector = BatchCollector(cfg)
    baseline = BaselineModel(cfg, store)
    detector = Detector(cfg, store, baseline)
    alerts = AlertSink(cfg)

    for raw_event in collector.collect():
        norm = normalize_event(raw_event)
        store.insert_event(raw_event=raw_event, norm_event=norm)

        # Persist normalized stream for easy demo + inspection
        with normalized_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(norm, ensure_ascii=False) + "\n")

        baseline.update_with_event(norm)
        alert = detector.score_and_maybe_alert(norm)
        if alert:
            alerts.emit(alert)
