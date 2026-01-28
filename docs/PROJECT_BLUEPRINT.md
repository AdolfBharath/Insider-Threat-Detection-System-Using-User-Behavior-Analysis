# Insider Threat Detection System Using User Behavior Analysis (ITDS) — Project Blueprint

## 1) Problem statement (real-world, SMB scope)
Small/medium organizations often lack a SOC/enterprise SIEM, yet face insider risks:
- A legitimate employee account is used to access data outside normal duties.
- A staff member abuses admin tools after login (sudo, sensitive file reads).
- Credential-sharing causes access from new IPs or unusual times.

**Goal:** Detect suspicious *post-login* behaviors from logs using **transparent rules + simple statistics** (no deep learning, no paid tools).

---

## 2) System architecture (components + data flow)

### Components
1. **Log Sources** (Linux + app): raw text logs.
2. **Collector**: reads logs (batch for demo; tail for live).
3. **Parser(s)**: source-specific extraction (auth.log, app access).
4. **Normalizer**: maps parsed fields → one unified JSON schema.
5. **Storage**: SQLite tables for events + baseline memory (first-seen IP/resource).
6. **Behavior Baseline**: rolling “normal” behavior per user.
7. **Detector**:
   - **Rule Engine**: deterministic, explainable triggers.
   - **Anomaly Checks**: z-score spike checks on event rates.
8. **Risk Scoring**: weighted aggregation + decay.
9. **Alert Sink**: console output (and optional JSONL file).

### Data flow (pipeline)
Raw log line → Collector → Parser → Normalizer → SQLite → Baseline Update → Detection → Risk Score → Alert

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

### A) Rule-based detection (explainable)
Rules implemented (weights configurable in `configs/itds.yml`):
1. **After-hours activity**: event timestamp outside working hours.
2. **Sensitive sudo**: sudo command touches sensitive targets (e.g., `/etc/shadow`).
3. **Large download**: download bytes ≥ threshold.
4. **Deny burst**: too many denied actions in 10 minutes (possible probing).
5. **New IP / new resource**: first-seen IP/resource for that user.

Each rule emits: `name`, `weight`, and `evidence`.

### B) Simple anomaly detection (transparent)
- **Spike in activity** (per user):
  - Compute `rate_60m` = events in last 60 minutes.
  - Estimate typical `mean/std` from last 7 days hourly buckets.
  - If z-score $z = (rate_60m - mean) / std$ exceeds threshold → anomaly.

This is explainable (shows `rate_60m`, `mean`, `std`, and `z`).

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

**Explainability:** alert includes the exact findings (rules/anomalies) and evidence.

---

## 7) Alert generation logic (not black-box)
Alert is generated when:
- risk level is `medium` or `high`.

Alert includes:
- `user`, `ts`, `score`, `level`
- the triggering event
- a list of explanations:
  - kind: `rule` or `anomaly`
  - name: `large_download`, `new_ip_first_seen`, etc.
  - weight
  - evidence (bytes, ip, command, deny_count_10m, z-score…)

---

## 8) Technology stack (open-source, affordable)
- **Python 3.10+** (main implementation)
- **SQLite** (single-file DB, easy demo)
- **PyYAML** (config)
- **Pydantic** (schema validation)
- **python-dateutil** (timestamp parsing)
- **rich** (clean alert output)

No enterprise SIEM, no deep learning, no paid components.

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

### B) Baseline creation
```
for each normalized event e:
  remember first-seen sets:
     if e.ip: add to set(user_ips)
     if e.resource: add to set(user_resources)
  (optional) update daily counters per user
```

### C) Anomaly detection (z-score spike)
```
rate_60m = count(events for user in last 60 minutes)
counts_by_hour = [count events for each hour bucket in last 7 days]
mean = avg(counts_by_hour)
std = stdev(counts_by_hour)
if std > 0:
   z = (rate_60m - mean) / std
   if z >= z_threshold: flag anomaly with evidence
```

### D) Alert generation
```
findings = []
for each rule:
  if condition(e): findings.append({rule_name, weight, evidence})
for each anomaly:
  if abnormal: findings.append({anomaly_name, weight, evidence})
score = sum(weights)
score = score * decay(age)
if score >= medium_threshold:
   emit alert(level, score, explanations=findings)
```

---

## 11) Execution flow (step-by-step)
1. Load config `configs/itds.yml`.
2. Initialize SQLite schema.
3. Collect raw log lines from configured sources.
4. Parse each line → structured dict.
5. Normalize to unified schema (validated).
6. Store normalized + raw in SQLite.
7. Update baseline memory (first-seen IP/resource).
8. Run rules + anomaly checks.
9. Compute risk score + level.
10. If medium/high → output alert with explanations.

---

## 12) Limitations (honest, reasonable)
- Baselines are simple; may need more days of data for stable statistics.
- Limited parsers (demo covers linux auth + app access). Adding auditd/Windows requires extra parsing.
- Activity-rate anomaly is coarse; it won’t catch low-and-slow exfiltration reliably.
- SQLite is fine for SMB/demo; not intended for high-volume enterprise ingestion.

---

## 13) Future enhancements (AI/ML upgrades without deep learning)
- Add feature vectors per user/day and use:
  - Isolation Forest (sklearn) for multivariate anomaly detection.
  - One-Class SVM for user-specific profiles.
- Add sequence-based rules (Markov chain on actions) for explainable transition anomalies.
- Add role-based baselines (compare against peers in same department).
- Add lightweight dashboard (Flask) for alerts + timelines.
- Add log tailing service + systemd unit for real deployment.
