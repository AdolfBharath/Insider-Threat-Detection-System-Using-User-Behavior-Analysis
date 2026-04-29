from __future__ import annotations

import cgi
import json
import threading
import time
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import yaml

from itds.runner import run_pipeline
from itds.utils.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = REPO_ROOT / "data" / "uploads"
OUT_DIR = REPO_ROOT / "data" / "out"
BASE_CONFIG = REPO_ROOT / "configs" / "itds.yml"
OVERRIDE_CONFIG = OUT_DIR / "upload_config.yml"
ALERTS_PATH = OUT_DIR / "alerts.jsonl"

STATE_LOCK = threading.Lock()
STATE: dict[str, Any] = {
    "state": "idle",
    "message": "Waiting for upload",
    "uploaded": None,
    "started_at": None,
    "finished_at": None,
    "alerts_count": 0,
}
RUN_THREAD: threading.Thread | None = None


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _set_state(**updates: Any) -> None:
    with STATE_LOCK:
        STATE.update(updates)


def _count_alerts() -> int:
    if not ALERTS_PATH.exists():
        return 0
    count = 0
    with ALERTS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _run_default_analysis() -> None:
    _set_state(state="running", message="Default analysis running", started_at=_now_iso(), finished_at=None)
    try:
        run_pipeline(BASE_CONFIG)
        alerts_count = _count_alerts()
        _set_state(
            state="done",
            message="Default analysis complete",
            finished_at=_now_iso(),
            alerts_count=alerts_count,
        )
    except Exception as exc:  # noqa: BLE001 - surface error in UI
        _set_state(state="error", message=str(exc), finished_at=_now_iso())


def _maybe_start_default_analysis() -> None:
    if STATE.get("state") in {"running", "queued"}:
        return
    if STATE.get("uploaded"):
        return
    if ALERTS_PATH.exists() and ALERTS_PATH.stat().st_size > 0:
        return

    global RUN_THREAD
    if RUN_THREAD is not None and RUN_THREAD.is_alive():
        return

    _set_state(
        state="queued",
        message="Starting default analysis",
        uploaded=None,
        started_at=None,
        finished_at=None,
        alerts_count=0,
    )
    RUN_THREAD = threading.Thread(target=_run_default_analysis, daemon=True)
    RUN_THREAD.start()


def _write_override_config(upload_path: Path, source_format: str) -> Path:
    cfg = load_config(BASE_CONFIG)
    cfg.setdefault("ingestion", {})
    cfg["ingestion"]["sources"] = [
        {
            "name": source_format,
            "type": "file",
            "format": source_format,
            "path": str(upload_path),
        }
    ]
    cfg.setdefault("paths", {})
    cfg["paths"]["sqlite_db"] = str(OUT_DIR / "itds.sqlite")
    cfg["paths"]["normalized_jsonl"] = str(OUT_DIR / "normalized.jsonl")
    cfg["paths"]["clear_normalized_on_run"] = True
    cfg.setdefault("alerting", {})
    cfg["alerting"]["json_out"] = str(ALERTS_PATH)
    cfg["alerting"]["clear_on_run"] = True

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OVERRIDE_CONFIG.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    return OVERRIDE_CONFIG


def _run_analysis(upload_path: Path, source_format: str) -> None:
    _set_state(state="running", message="Analysis running", started_at=_now_iso(), finished_at=None)
    try:
        override_path = _write_override_config(upload_path, source_format)
        run_pipeline(override_path)
        alerts_count = _count_alerts()
        _set_state(
            state="done",
            message="Analysis complete",
            finished_at=_now_iso(),
            alerts_count=alerts_count,
        )
    except Exception as exc:  # noqa: BLE001 - surface error in UI
        _set_state(state="error", message=str(exc), finished_at=_now_iso())


def _safe_filename(raw_name: str | None) -> str:
    if not raw_name:
        return "upload.log"
    return Path(raw_name).name


class DashboardHandler(SimpleHTTPRequestHandler):
    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text: str, status: int = 200) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - match base class signature
        if self.path.startswith("/api/status"):
            _maybe_start_default_analysis()
            with STATE_LOCK:
                payload = dict(STATE)
            self._send_json(payload)
            return

        if self.path.startswith("/api/alerts"):
            _maybe_start_default_analysis()
            alerts: list[dict[str, Any]] = []
            if ALERTS_PATH.exists():
                with ALERTS_PATH.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            alerts.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            self._send_json({"alerts": alerts})
            return

        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802 - match base class signature
        if self.path != "/api/upload":
            self._send_text("Not Found", status=404)
            return

        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self._send_text("Invalid content type", status=400)
            return

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": content_type},
        )
        file_field = form["logfile"] if "logfile" in form else None
        if isinstance(file_field, list):
            file_field = file_field[0] if file_field else None
        if file_field is None or getattr(file_field, "file", None) is None:
            self._send_text("Missing file", status=400)
            return

        source_format = "app_access"
        if "source_format" in form and form["source_format"].value:
            source_format = str(form["source_format"].value).strip()
        if source_format not in {"app_access", "linux_auth"}:
            source_format = "app_access"

        with STATE_LOCK:
            if STATE.get("state") == "running":
                self._send_text("Analysis already running", status=409)
                return

        filename = _safe_filename(file_field.filename)
        timestamp = int(time.time())
        stored_name = f"{timestamp}_{filename}"
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        stored_path = UPLOAD_DIR / stored_name

        with stored_path.open("wb") as handle:
            handle.write(file_field.file.read())

        _set_state(
            state="queued",
            message="Upload received",
            uploaded={
                "name": filename,
                "stored_name": stored_name,
                "bytes": stored_path.stat().st_size,
                "source_format": source_format,
                "uploaded_at": _now_iso(),
            },
        )

        global RUN_THREAD
        RUN_THREAD = threading.Thread(
            target=_run_analysis, args=(stored_path, source_format), daemon=True
        )
        RUN_THREAD.start()

        self._send_json({"ok": True, "stored_name": stored_name})


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    server_address = (host, port)
    handler = lambda *args, **kwargs: DashboardHandler(*args, directory=str(REPO_ROOT), **kwargs)
    httpd = ThreadingHTTPServer(server_address, handler)
    print(f"Serving dashboard on http://{host}:{port}/web/dashboard.html")
    httpd.serve_forever()


if __name__ == "__main__":
    run()
