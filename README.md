# Insider Threat Detection System Using User Behavior Analysis (ITDS)

A lightweight, explainable, demo-ready insider threat detection system for **small/medium organizations**.

This project monitors **post-login user activity** by ingesting host/app logs, normalizing events, building per-user behavior baselines, scoring anomalies + rules, and generating **transparent alerts**.

## Quick Demo (Windows-friendly)

1) Create venv + install deps:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Run demo on included sample logs:

```powershell
python -m itds --config .\configs\itds.yml
```

## View output in a webpage (Dashboard)

Option A (recommended): serve locally and auto-load `data/out/*`:

```powershell
python -m http.server 8000
```

Open:
- `http://localhost:8000/web/dashboard.html`

Option B: open `web/dashboard.html` and upload `data/out/alerts.jsonl` manually.

Outputs:
- SQLite DB: `data/out/itds.sqlite`
- Alerts JSONL: `data/out/alerts.jsonl`
- Normalized events JSONL: `data/out/normalized.jsonl`

## What This Repo Contains
- `itds/` Implementation modules (collector, parser, baseline, scoring, alerts)
- `configs/` Runtime configuration (sources, thresholds, weights)
- `data/sample_logs/` Sample Linux + application logs for demo
- `docs/` Architecture + viva-ready project blueprint

See [docs/PROJECT_BLUEPRINT.md](docs/PROJECT_BLUEPRINT.md) for the complete design and implementation details.
