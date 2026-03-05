# Insider Threat Detection System Using User Behavior Analysis (ITDS)

A lightweight, explainable, demo-ready insider threat detection system for **small/medium organizations**.

This project monitors **post-login user activity** by ingesting host/app logs, normalizing events, building **per-user behavior sequences**, scoring with an **unsupervised LSTM Autoencoder**, and generating **transparent alerts**.

## Recent Changes (Latest Update)

- **Single-model ML**: kept the system to **one unsupervised deep learning model** (LSTM Autoencoder) and uses reconstruction error for anomaly scoring.
- **Role-aware behavior modeling**: users can be mapped to roles via `configs/itds.yml`, enabling role-sensitive baselines.
- **Adaptive thresholds**: supports adaptive, running statistics (EWMA-style) for thresholds (role/global) so alerting adjusts as behavior drifts.
- **Risk scoring + prioritization**: alerts include a priority score and critical tags; output is sorted to surface highest-risk items first.
- **Performance improvements**: scoring can stream from `data/out/normalized.jsonl` (reduces memory usage vs loading everything at once).
- **Robustness**: detects feature-schema/model shape mismatches and automatically forces a retrain (also clears stale adaptive stats when needed).
- **Dashboard UI/UX**: `web/dashboard.html` was upgraded to display **Role** and **Priority**, show critical tags, and provide a cleaner details panel.

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

What you get:
- Alerts: `data/out/alerts.jsonl`
- Normalized events: `data/out/normalized.jsonl`
- SQLite DB: `data/out/itds.sqlite`

## View output in a webpage (Dashboard)

Serve the repo locally and the dashboard will auto-load `data/out/alerts.jsonl`:

```powershell
python -m http.server 8000
```

Open:
- `http://localhost:8000/web/dashboard.html`

Tip: click **Refresh** to reload the latest output.

## How to read "Top reason"
Each alert includes two human-readable explanations:
- `reason_short`: short tags (example: `After-hours + sudo + large download`)
- `reason`: detailed explanation (threshold exceeded + main drivers + top event)

## What This Repo Contains
- `itds/` Implementation modules (collector, parser, normalization, LSTM-AE scoring, alerts)
- `configs/` Runtime configuration (sources, thresholds, weights)
- `data/sample_logs/` Sample Linux + application logs for demo
- `docs/` Architecture + viva-ready project blueprint

See [docs/PROJECT_BLUEPRINT.md](docs/PROJECT_BLUEPRINT.md) for the complete design and implementation details.
