from __future__ import annotations

from pathlib import Path

import json

from itds.utils.config import load_config
from itds.storage.sqlite import SqliteStore
from itds.collectors.batch import BatchCollector
from itds.normalization.normalizer import normalize_event
from itds.alerting.sink import AlertSink
from itds.ml.pipeline import score_sequences_from_jsonl


def run_pipeline(config_path: Path) -> None:
    cfg = load_config(config_path)

    normalized_path = Path(cfg["paths"].get("normalized_jsonl", "data/out/normalized.jsonl"))
    normalized_path.parent.mkdir(parents=True, exist_ok=True)

    # Optional: clear output JSONL files on each run so the dashboard shows fresh results.
    if bool(cfg.get("paths", {}).get("clear_normalized_on_run", True)):
        normalized_path.write_text("", encoding="utf-8")

    alert_out = Path(cfg.get("alerting", {}).get("json_out", "data/out/alerts.jsonl"))
    if bool(cfg.get("alerting", {}).get("clear_on_run", True)):
        alert_out.parent.mkdir(parents=True, exist_ok=True)
        alert_out.write_text("", encoding="utf-8")

    store = SqliteStore(Path(cfg["paths"]["sqlite_db"]))
    store.init_schema()

    collector = BatchCollector(cfg)
    alerts = AlertSink(cfg)

    for raw_event in collector.collect():
        norm = normalize_event(raw_event)
        store.insert_event(raw_event=raw_event, norm_event=norm)

        # Persist normalized stream for easy demo + inspection
        with normalized_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(norm, ensure_ascii=False) + "\n")

    # Unsupervised LSTM Autoencoder scoring over per-user sequences
    for alert in score_sequences_from_jsonl(cfg, normalized_path):
        alerts.emit(alert)
