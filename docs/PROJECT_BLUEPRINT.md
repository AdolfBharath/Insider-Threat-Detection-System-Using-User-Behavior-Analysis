# Insider Threat Detection System Using User Behavior Analysis (ITDS) — Project Blueprint

## 1) Problem statement (real-world, SMB scope)
Small/medium organizations often lack a SOC/enterprise SIEM, yet face insider risks:
- A legitimate employee account is used to access data outside normal duties.
- A staff member abuses admin tools after login (sudo, sensitive file reads).
- Credential-sharing causes access from new IPs or unusual times.

**Goal:** Detect suspicious *post-login* behaviors from logs using a **single unsupervised deep-learning model** (LSTM Autoencoder) and produce **human-readable reasons** for each alert.

---

## 2) System architecture (components + data flow)

### Components
1. **Log Sources** (Linux + app): raw text logs.
2. **Collector**: reads logs (batch for demo; tail for live).
3. **Parser(s)**: source-specific extraction (auth.log, app access).
4. **Normalizer**: maps parsed fields → one unified JSON schema.
5. **Storage**: SQLite tables for events + baseline memory (first-seen IP/resource).
6. **Feature + Sequence Builder**: converts normalized events into per-user sliding windows.
7. **Detector (single model)**: **Unsupervised LSTM Autoencoder** → reconstruction error.
8. **Thresholding + Risk**: thresholds from training percentiles + simple risk aggregation.
9. **Alert Sink**: console output + JSONL for the dashboard.

### Data flow (pipeline)
Raw log line → Collector → Parser → Normalizer → SQLite → Feature vectors → Per-user sequences → LSTM-AE reconstruction error → Threshold → Risk → Alert

---

## 3) Log sources used (practical, post-login)

### Linux (host-level)
- `/var/log/auth.log` (Debian/Ubuntu) or `/var/log/secure` (RHEL):
  - SSH success/fail, sudo commands, user switching.
- Optional (future/bonus): `/var/log/audit/audit.log` (auditd): file access, execve.

### Application access logs (your web/internal app)
- Example: `user=<id> action=<view|download> resource=<path> bytes=<n> ip=<addr> status=<code>`
- Captures post-login data access patterns and denied access.

---

## 4) Log collection + normalization

### Collection approach
- **Batch mode (demo-ready):** read complete files and process line-by-line.
- **Tail mode (practical deployment):** tail files continuously (can be added with a lightweight Python tailer or watchdog).

### Normalization schema (single format for all sources)
Normalized JSON fields:
- `ts` (ISO-8601 UTC), `user`, `host`, `source`
- `event_type` (auth/sudo/app_access/unknown)
- `action` (login_success, sudo_command, download, view, ...)
- `status` (success/fail)
- optional: `ip`, `resource`, `bytes`, `extra`

**Why normalization matters:** all detection logic runs on one schema (easier to explain + extend).

---

## 5) Behavior analysis logic

### A) Feature engineering (explainable inputs)
Each normalized event is converted into a numeric vector that mixes:
- one-hot: `action:*`, `resource:*`
- numeric (robust-scaled): `bytes_log`, `success`, `privilege`, `hour_sin`, `hour_cos`

### B) Sequence modeling (single unsupervised model)
For each user, build sequences using a sliding window:
- `seq_len`: number of events in a window
- `stride`: step size

Train an **LSTM Autoencoder** on early sequences (assumed normal). At inference time:
- reconstruct each sequence
- compute reconstruction error (MSE)

Large reconstruction error means the sequence deviates from learned “normal”.

---

## 6) Risk scoring per user (mechanism)

### Risk score formula (simple weighted sum + decay)
Let findings be $f_i$ with weights $w_i$.
- Base score: $S = \sum_i w_i$
- Time decay (half-life $H$ hours): $S' = S \cdot 0.5^{(age/H)}$
- Clamp to 0..100.

### Alert thresholds
- `high` if score ≥ 70
- `medium` if score ≥ 45
- else `low` (no alert)

**Explainability:** alerts include short human tags plus detailed evidence.

---

## 7) Alert generation logic (not black-box)
An alert is generated when either:
- sequence reconstruction error exceeds the learned threshold (medium/high), OR
- user risk crosses `threshold_medium` / `threshold_high`.

Alert includes:
- `user`, `ts`, `score`, `level`
- `event`: sequence metadata (`seq_start`, `seq_end`, `seq_len`, `anomaly_score`, thresholds)
- `explanations` with:
  - `kind: dl`
  - `name: lstm_autoencoder_reconstruction_error`
  - `evidence.reason_short`: short tags (example: `After-hours + sudo + large download`)
  - `evidence.reason`: detailed readable reason (threshold exceeded + drivers + top event)
  - `top_features`: most contributing features
  - `top_events`: most contributing events within the sequence

---

## 8) Technology stack (open-source, affordable)
- **Python 3.10+** (main implementation)
- **SQLite** (single-file DB, easy demo)
- **PyYAML** (config)
- **Pydantic** (schema validation)
- **python-dateutil** (timestamp parsing)
- **rich** (clean alert output)

Deep learning (single model):
- **TensorFlow/Keras** (LSTM Autoencoder)
- **NumPy** (arrays)

No enterprise SIEM, no paid components.

---

## 9) Project folder structure

```
insider threat/
  itds/
    __main__.py
    cli.py
    runner.py
    collectors/
      batch.py
    parsers/
      linux_auth.py
      app_access.py
    normalization/
      schema.py
      normalizer.py
    storage/
      sqlite.py
    analysis/
      baseline.py
      detector.py
    alerting/
      sink.py
    ml/
      feature_engineering.py
      sequence_builder.py
      lstm_autoencoder.py
      pipeline.py
    utils/
      config.py
  configs/
    itds.yml
  data/
    sample_logs/
      linux/auth.log
      app/app_access.log
    out/
  docs/
    PROJECT_BLUEPRINT.md
  requirements.txt
  README.md
```

---

## 10) Pseudocode / logic (for viva)

### A) Log parsing
```
for each line in log_file:
  if format == linux_auth:
     if "Accepted" in line: emit {user, ip, ts, action=login_success}
     if "Failed" in line:   emit {user, ip, ts, action=login_failed}
     if "sudo:" in line:    emit {user, ts, action=sudo_command, resource=cmd}
  if format == app_access:
     split tokens by space; parse key=value fields
     emit {user, ts, action, resource, bytes, ip, status}
```

### B) Build sequences + score with LSTM-AE
```
events = normalize(all_raw_logs)
vectors = event_to_vector(events)

seqs = sliding_windows_per_user(events, vectors, seq_len, stride)
model = train_lstm_autoencoder(early_seqs)

for each seq in seqs:
  recon = model(seq)
  err = mse(seq, recon)
  if err >= t_high or risk >= high_threshold:
    emit HIGH alert
  elif err >= t_medium or risk >= medium_threshold:
    emit MEDIUM alert
```

---

## 11) Execution flow (step-by-step)
1. Load config `configs/itds.yml`.
2. Initialize SQLite schema.
3. Collect raw log lines from configured sources.
4. Parse each line → structured dict.
5. Normalize to unified schema (validated).
6. Store normalized + raw in SQLite.
7. Convert normalized events into feature vectors.
8. Build per-user sequences (sliding windows).
9. Train/load the LSTM Autoencoder + thresholds.
10. Score sequences, compute risk, emit alerts with `reason_short` and detailed evidence.

---

## 12) Limitations (honest, reasonable)
- The demo model is trained on bundled sample logs; real deployments need more history.
- Limited parsers (demo covers linux auth + app access). Adding auditd/Windows requires extra parsing.
- Thresholds are percentile-based; tuning may be needed to control alert volume.
- SQLite is fine for SMB/demo; not intended for high-volume enterprise ingestion.

---

## 13) Future enhancements (AI/ML upgrades without deep learning)
- Add role/peer baselines (compare against similar users).
- Add model retraining schedule (daily/weekly) with drift checks.
- Add log tailing service + systemd unit for real deployment.
- Add richer explainability: per-feature per-timestep contributions.
